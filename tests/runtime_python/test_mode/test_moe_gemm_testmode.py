"""Test-mode coverage for the routed-expert extension tasks
(moe_route_layer + moe_gemm_layer): routing, W13 with the fused gated
activation, W2, against PyTorch.

    python tests/runtime_python/test_mode/test_moe_gemm_testmode.py \
        [--batch 5] [--bf16] [--experts 32] [--topk 4] [--shared]

--bf16 runs bf16 expert weights (SiLU(gate) * up, no bias, the Qwen3.5 form)
instead of MXFP4 ones (clamped SwiGLU with biases, the GPT-OSS form);
--shared adds a shared expert every token reaches through a sigmoid gate.
"""

import argparse
import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk import moe_pack


def bfr(x):
    return x.to(torch.bfloat16).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--shared", action="store_true")
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--inter", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skew", type=float, default=0.0,
                    help="bias the router towards the first experts")
    ap.add_argument("--ksplit", type=int, default=1,
                    help="parts of a unit's K steps (at most its steps)")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev, bf = "cuda", torch.bfloat16
    B, E, TOPK, H, I = a.batch, a.experts, a.topk, a.hidden, a.inter
    mx = not a.bf16
    shared = 1 if a.shared else 0
    E_TOT, SLOTS = E + shared, TOPK + shared
    nt = moe_pack.moe_tokens_per_pass(B)
    if mx:
        limit, alpha, off = 7.0, 1.702, 1.0
    else:
        limit, alpha, off = float("inf"), 1.0, 0.0

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    if os.environ.get("TEST_WORKERS"):  # "<workers>,<schedulers>"
        num_workers, num_schedulers = (
            int(v) for v in os.environ["TEST_WORKERS"].split(","))
    params = PersistentKernel.get_default_init_parameters()
    params.update(test_mode=True, num_workers=num_workers,
                  num_local_schedulers=num_schedulers,
                  max_num_batched_tokens=B, max_num_batched_requests=1,
                  max_num_pages=16, page_size=max(64, B))
    pk = PersistentKernel(**params)

    # ---- data
    # the router's E logits, then (with --shared) the shared expert's gate logit
    logits_all = torch.cat(
        [torch.randn(B, E, device=dev) -
         a.skew * torch.arange(E, device=dev) / E,
         torch.randn(B, 8, device=dev)], dim=1).to(bf)
    logits, shared_logit = logits_all[:, :E], logits_all[:, E:E + 1]
    x = (torch.randn(B, H, device=dev) * 0.5).to(bf)
    if mx:
        blk13 = torch.randint(0, 256, (E_TOT, 2 * I, H // 32, 16),
                              dtype=torch.uint8, device=dev)
        sc13 = torch.randint(115, 123, (E_TOT, 2 * I, H // 32),
                             dtype=torch.uint8, device=dev)
        blk2 = torch.randint(0, 256, (E_TOT, H, I // 32, 16),
                             dtype=torch.uint8, device=dev)
        sc2 = torch.randint(115, 123, (E_TOT, H, I // 32),
                            dtype=torch.uint8, device=dev)
        w13f = moe_pack.unpack_mxfp4(blk13, sc13)   # rows: gate 0, up 0, ...
        w2f = moe_pack.unpack_mxfp4(blk2, sc2)
        w13p = moe_pack.pack_mxfp4(blk13, sc13)
        w2p = moe_pack.pack_mxfp4(blk2, sc2)
        b13 = (torch.randn(E_TOT, 2 * I, device=dev) * 0.2).to(bf)
        b2 = (torch.randn(E_TOT, H, device=dev) * 0.2).to(bf)
    else:
        # gate rows first, as the HF checkpoints store them
        w13 = (torch.randn(E_TOT, 2 * I, H, device=dev) * 0.05).to(bf)
        w2 = (torch.randn(E_TOT, H, I, device=dev) * 0.05).to(bf)
        w13i = moe_pack.interleave_gate_up(w13)
        w13f, w2f = w13i.float(), w2.float()
        w13p = moe_pack.pack_bf16_tiles(w13i)
        w2p = moe_pack.pack_bf16_tiles(w2)
        b13 = b2 = None

    weight, rlist, rmeta = moe_pack.moe_plan_tensors(B, E_TOT, SLOTS, nt)
    act = torch.zeros(B, SLOTS, I, dtype=bf, device=dev)
    out = torch.zeros(B, SLOTS, H, dtype=bf, device=dev)

    # ---- graph. Unless TEST_NO_HOLD is set the workers keep their TMEM
    # from their first GEMM task to a release wave at the end.
    held = os.environ.get("TEST_NO_HOLD") is None
    tm_token = torch.zeros(num_workers, dtype=torch.int32, device=dev)
    tm_done = torch.zeros(num_workers, dtype=torch.int32, device=dev)
    d_logits = pk.attach_input(logits_all, name="logits")
    d_tok = pk.attach_input(tm_token, name="tmem_token") if held else None
    d_w, d_l, d_m = (pk.attach_input(weight, name="route_weight"),
                     pk.attach_input(rlist, name="route_list"),
                     pk.attach_input(rmeta, name="route_meta"))
    d_act = pk.attach_input(act, name="act")
    # ---- split K: every layer's units in up to --ksplit parts, one scratch
    DR = 2 * H - 64                      # the dense layer's rows
    kb_mx = (lambda k: moe_pack.mxfp4_step(k)) if mx else (lambda k: None)
    ks13 = min(a.ksplit, pk.gemm_steps(H, B, kb_mx(H)))
    ks2 = min(a.ksplit, pk.gemm_steps(I, B, kb_mx(I)))
    ksd = min(a.ksplit, pk.gemm_steps(H, B))
    scratch = scratch_d = None
    if max(ks13, ks2, ksd) > 1:
        cap = moe_pack.moe_plan_cap(B, E_TOT, SLOTS, nt)
        # (one scratch for W13 and W2, which run one after the other; the
        # dense layer depends on neither and runs next to them: its own)
        sizes = [pk.gemm_scratch_size(cap * ((2 * I + 127) // 128), ks13, nt),
                 pk.gemm_scratch_size(cap * ((H + 127) // 128), ks2, nt)]
        sized = pk.gemm_scratch_size(((B + nt - 1) // nt) * ((DR + 127) // 128),
                                     ksd, nt)
        ks_parts = torch.zeros(max(1, max(v[0] for v in sizes)),
                               dtype=torch.float32, device=dev)
        ks_counts = torch.zeros(max(1, max(v[1] for v in sizes)),
                                dtype=torch.int32, device=dev)
        scratch = (pk.attach_input(ks_parts, name="gemm_parts"),
                   pk.attach_input(ks_counts, name="gemm_part_counts"))
        scratch_d = (
            pk.attach_input(torch.zeros(max(1, sized[0]), dtype=torch.float32,
                                        device=dev), name="gemm_parts_dense"),
            pk.attach_input(torch.zeros(max(1, sized[1]), dtype=torch.int32,
                                        device=dev),
                            name="gemm_part_counts_dense"))
        print(f"[ksplit] w13 {ks13}, w2 {ks2}, dense {ksd}")
    pk.moe_route_layer(
        logits=d_logits, weight=d_w,
        route_list=d_l, route_meta=d_m, nt=nt, num_experts=E,
        shared=bool(shared))
    pk.moe_gemm_layer(
        input=pk.attach_input(x, name="x"),
        weight=pk.attach_input(w13p, name="w13"), route_list=d_l,
        route_meta=d_m, output=d_act, nt=nt, mxfp4=mx, rows=2 * I,
        act=(limit, alpha, off),
        bias=pk.attach_input(b13, name="b13") if mx else None, tmem=d_tok,
        ksplit=ks13, scratch=scratch)
    pk.moe_gemm_layer(
        input=d_act, weight=pk.attach_input(w2p, name="w2"), route_list=d_l,
        route_meta=d_m, output=(d_out := pk.attach_input(out, name="out")),
        nt=nt,
        mxfp4=mx, rows=H,
        bias=pk.attach_input(b2, name="b2") if mx else None, tmem=d_tok,
        ksplit=ks2, scratch=scratch)

    residual = (torch.randn(B, H, device=dev) * 0.5).to(bf)
    comb = torch.zeros(B, H, dtype=bf, device=dev)
    pk.moe_combine_layer(
        input=d_out, weight=d_w,
        residual=pk.attach_input(residual, name="residual"),
        output=(d_comb := pk.attach_input(comb, name="comb")),
        shared=bool(shared))

    # ---- a dense linear layer on the same task: bias, residual, and the
    # workers holding their TMEM across the wave
    # (DR rows: not a multiple of 128)
    wd = (torch.randn(1, DR, H, device=dev) * 0.05).to(bf)
    bd = (torch.randn(DR, device=dev) * 0.2).to(bf)
    rd = (torch.randn(B, DR + 64, device=dev) * 0.5).to(bf)
    yd = torch.zeros(B, DR + 64, dtype=bf, device=dev)
    d_x2 = pk.attach_input(x.clone(), name="x_dense")
    d_yd = pk.attach_input(yd, name="y_dense")
    pk.dense_gemm_layer(
        input=d_x2, weight=pk.attach_input(moe_pack.pack_bf16_tiles(wd), name="w_dense"),
        output=d_yd, rows=DR, bias=pk.attach_input(bd, name="b_dense"),
        residual=pk.attach_input(rd, name="r_dense"), tmem=d_tok,
        ksplit=ksd, scratch=scratch_d if scratch is not None else None)
    # ---- a small linear layer summed by threads
    ws = (torch.randn(40, H, device=dev) * 0.05).to(bf)
    bs = (torch.randn(40, device=dev) * 0.2).to(bf)
    ys = torch.zeros(B, 48, dtype=bf, device=dev)
    pk.small_linear_layer(input=d_x2, weight=pk.attach_input(ws, name="w_small"),
                          bias=pk.attach_input(bs, name="b_small"),
                          output=pk.attach_input(ys, name="y_small"))
    if held:
        pk.tmem_release_layer([d_comb, d_yd], d_tok,
                              pk.attach_input(tm_done, name="tmem_done"))

    print("Compiling test kernel...")
    pk.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))
    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    # ---- reference
    lf = logits.float()
    # equal logits: the lower expert first (the kernel's rule)
    order = torch.sort(lf, dim=1, descending=True, stable=True).indices
    sel = order[:, :TOPK]
    w_ref = bfr(torch.softmax(torch.gather(lf, 1, sel), dim=1))
    if shared:
        sel = torch.cat([sel, torch.full((B, 1), E, device=dev)], dim=1)
        w_ref = torch.cat(
            [w_ref, bfr(torch.sigmoid(shared_logit.float()))], dim=1)
    xf = x.float()
    y = torch.einsum("bk,bsrk->bsr", xf, w13f[sel])
    y = bfr(y)
    if mx:
        y = bfr(y + b13.float()[sel])
    gate, up = y[..., 0::2], y[..., 1::2]
    g = gate.clamp(max=limit)
    u = up.clamp(-limit, limit)
    act_ref = bfr((u + off) * g * torch.sigmoid(alpha * g))
    o = bfr(torch.einsum("bsk,bsrk->bsr", act_ref, w2f[sel]))
    out_ref = bfr(o + b2.float()[sel]) if mx else o

    # the combine, from the kernel's own W2 output: bf16 products, their
    # bf16 sum (+ the shared expert's product), + the residual
    prod = bfr(out.float() * w_ref[..., None])
    csum = bfr(prod[:, :TOPK].sum(1))
    if shared:
        csum = bfr(csum + prod[:, TOPK])
    comb_ref = bfr(residual.float() + csum)

    ok = True

    def check(name, got, ref, rel):
        nonlocal ok
        got = got.float()
        err = (got - ref).abs()
        tol = rel * ref.abs() + rel * ref.abs().mean()
        bad = int((err > tol).sum())
        exact = float((got == ref).float().mean())
        print(f"[{name}] max |kernel - reference| = {err.max().item():.5f}, "
              f"|ref| max {ref.abs().max().item():.3f}, exact {exact:.4f}, "
              f"outside tolerance {bad} of {ref.numel()}")
        if bad or not torch.isfinite(got).all():
            ok = False
            print(f"[{name}] FAILED")

    # routing: the plan must hold every (token, slot) under its expert
    n_ep = int(rmeta[0])
    cap = moe_pack.moe_plan_cap(B, E_TOT, SLOTS, nt)
    counts = rmeta[1 + cap:1 + cap + E_TOT]
    want = torch.bincount(sel.flatten(), minlength=E_TOT)
    plan_ok = bool((counts == want).all())
    seen = torch.full((B, SLOTS), -1, dtype=torch.long, device=dev)
    for e in range(E_TOT):
        ent = rlist[e, :int(counts[e])].long()
        seen[ent >> 8, ent & 0xff] = e
    plan_ok = plan_ok and bool((seen == sel).all())
    pairs = rmeta[1:1 + n_ep].long()
    want_pairs = [(e << 16) | p for e in range(E_TOT)
                  for p in range((int(want[e]) + nt - 1) // nt)]
    plan_ok = plan_ok and pairs.tolist() == want_pairs
    print(f"[route] plan {'ok' if plan_ok else 'WRONG'}: {n_ep} (expert, pass) "
          f"pairs, {int((want > 0).sum())} experts active, most tokens on one "
          f"expert {int(want.max())}, NT {nt}")
    ok = ok and plan_ok
    check("route weight", weight, w_ref, 1e-6)
    check("w13 + activation", act, act_ref, 0.02)
    check("w2", out, out_ref, 0.02)
    check("combine", comb, comb_ref, 0.02)
    yd_ref = bfr(bfr(x.float() @ wd[0].float().T + bd.float()) +
                 rd.float()[:, :DR])
    check("dense + bias + residual", yd[:, :DR], yd_ref, 0.02)
    check("small linear", ys[:, :40],
          bfr(x.float() @ ws.float().T + bs.float()), 0.02)
    if float(yd[:, DR:].abs().max()) != 0:
        ok = False
        print("[dense] FAILED: wrote past its rows")

    pk.finalize()
    if not ok:
        sys.exit(1)
    print("\nPASSED")


if __name__ == "__main__":
    main()
