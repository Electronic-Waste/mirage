"""Test-mode coverage for Qwen3.5's extension tasks (qwen35_sm100.cuh):
norm1p, qsa_prep, gate_mul and several decode steps of gdn_step, against
PyTorch written after transformers' Qwen3_5Moe modules.

    python tests/runtime_python/test_mode/test_qwen35_tasks_testmode.py \
        [--batch 5] [--steps 6]
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

H, HD, NKV, QPK = 2048, 256, 2, 8          # hidden, attention head, heads
NKH, D = 16, 128                           # GDN key heads, head dim
NVH = 2 * NKH
C = 4 * NKH * D                            # conv channels
PW_A = NKV * (QPK + 2) * HD + NKV * QPK * HD   # q k v per KV head, then gates
PW_G = C + NVH * D + 2 * NVH + 64          # q k v z b a, padded
EPS = 1e-6


def bfr(x):
    return x.to(torch.bfloat16).float()


def norm1p_ref(x, w):
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    return bfr(y * (1.0 + w.float()))


def gdn_ref(proj, convw, hist, S, A, dt_bias, wnorm, fresh):
    """One step for every request; hist [B, C, 3] bf16-valued float, S
    [B, NVH, D, D] float. Returns (out [B, NVH * D], hist, S)."""
    B = proj.shape[0]
    p = proj.float()
    mixed = p[:, :C]
    z = p[:, C:C + NVH * D].reshape(B, NVH, D)
    b = p[:, C + NVH * D:C + NVH * D + NVH]
    a = p[:, C + NVH * D + NVH:C + NVH * D + 2 * NVH]
    if fresh:
        hist = torch.zeros_like(hist)
        S = torch.zeros_like(S)
    new = torch.cat([hist, mixed[..., None]], dim=-1)
    hist = new[..., 1:]
    y = bfr(F.silu(bfr((new * convw.float()[None]).sum(-1))))
    q = y[:, :NKH * D].reshape(B, NKH, D).repeat_interleave(2, dim=1)
    k = y[:, NKH * D:2 * NKH * D].reshape(B, NKH, D).repeat_interleave(2, dim=1)
    v = y[:, 2 * NKH * D:].reshape(B, NVH, D)
    beta = bfr(torch.sigmoid(b))
    g = A[None] * F.softplus(a + dt_bias[None])
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) / (D ** 0.5)
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    S = S * g.exp()[..., None, None]
    kv = (S * k[..., None]).sum(-2)
    delta = (v - kv) * beta[..., None]
    S = S + k[..., None] * delta[..., None, :]
    o = bfr((S * q[..., None]).sum(-2))
    n = bfr(o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + EPS))
    out = bfr(bfr(wnorm[None, None] * n) * F.silu(z))
    return out.reshape(B, NVH * D), hist, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gdn-tasks", type=int, default=None,
                    help="tasks of a gated delta net wave (fewer: more "
                         "(token, value head) units a task)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev, bf, B = "cuda", torch.bfloat16, a.batch

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
    x = (torch.randn(B, H, device=dev) * 2).to(bf)
    w_in = (torch.randn(H, device=dev) * 0.1).to(bf)
    y_norm = torch.zeros(B, H, dtype=bf, device=dev)

    proj_a = torch.randn(B, PW_A, device=dev).to(bf)
    wq = (torch.randn(HD, device=dev) * 0.1).to(bf)
    wk = (torch.randn(HD, device=dev) * 0.1).to(bf)
    qkv = torch.zeros(B, NKV * (QPK + 2) * HD, dtype=bf, device=dev)
    attn = torch.randn(B, NKV * QPK * HD, device=dev).to(bf)
    gated = torch.zeros_like(attn)

    # gate values of every size, small ones included
    proj_g = [(torch.randn(B, PW_G, device=dev) *
               (0.3 + 2.0 * torch.rand(B, 1, device=dev))).to(bf)
              for _ in range(a.steps)]
    convw = (torch.randn(C, 4, device=dev) * 0.5).to(bf)
    cstate = torch.zeros(B + 2, 4, C, dtype=bf, device=dev)
    sstate = torch.zeros(B + 2, NVH, D, D, dtype=torch.float32, device=dev)
    A = -(torch.rand(NVH, device=dev) * 2 - 1).exp()
    dt_bias = torch.randn(NVH, device=dev).to(bf).float()
    wnorm = (1 + torch.randn(D, device=dev) * 0.1).to(bf).float()
    par = torch.cat([A, dt_bias, wnorm]).contiguous()
    # requests in a shuffled order, two of them unused
    rid = torch.randperm(B + 2, device=dev)[:B].to(torch.int32)
    rmap = [torch.stack([rid, torch.full_like(rid, st)], dim=1).contiguous()
            for st in range(a.steps)]
    gout = [torch.zeros(B, NVH * D, dtype=bf, device=dev)
            for _ in range(a.steps)]

    # ---- graph
    pk.norm1p_layer(pk.attach_input(x, name="x"), pk.attach_input(w_in, name="w_in"),
                    pk.attach_input(y_norm, name="y_norm"), eps=EPS)
    d_proj_a = pk.attach_input(proj_a, name="proj_a")
    pk.qsa_prep_layer(d_proj_a, pk.attach_input(wq, name="wq"),
                      pk.attach_input(wk, name="wk"),
                      pk.attach_input(qkv, name="qkv"), NKV, QPK, HD, eps=EPS)
    pk.gate_mul_layer(pk.attach_input(attn, name="attn"), d_proj_a,
                      NKV * (QPK + 2) * HD, pk.attach_input(gated, name="gated"))
    # the steps of a generation as a chain of layers on the same states
    d_cw, d_cs = pk.attach_input(convw, name="convw"), pk.attach_input(cstate, name="cstate")
    d_ss, d_par = pk.attach_input(sstate, name="sstate"), pk.attach_input(par, name="par")
    d_prev = None
    for st in range(a.steps):
        d_out = pk.attach_input(gout[st], name=f"gout{st}")
        pk.gdn_step_layer(pk.attach_input(proj_g[st], name=f"proj_g{st}"),
                          d_cw, d_cs, d_ss, d_par, d_out, NKH, D, eps=EPS,
                          request_map=pk.attach_input(rmap[st], name=f"rmap{st}"),
                          after=d_prev, num_tasks=a.gdn_tasks)
        d_prev = d_out
    print("Compiling test kernel...")
    pk.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))

    ok = True

    def check(name, got, ref, rel=0.02):
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

    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()
    check("norm1p", y_norm, norm1p_ref(x, w_in))
    pa = proj_a.float()
    heads = pa[:, :NKV * (QPK + 2) * HD].reshape(B, NKV, QPK + 2, HD)
    ref = heads.clone()
    ref[:, :, :QPK] = norm1p_ref(heads[:, :, :QPK], wq)
    ref[:, :, QPK] = norm1p_ref(heads[:, :, QPK], wk)
    check("qsa_prep", qkv, ref.reshape(B, -1))
    gates = pa[:, NKV * (QPK + 2) * HD:]
    check("gate_mul", gated, bfr(attn.float() * bfr(torch.sigmoid(gates))))

    hist = torch.zeros(B, C, 3, device=dev)
    S = torch.zeros(B, NVH, D, D, device=dev)
    for step in range(a.steps):
        ref, hist, S = gdn_ref(proj_g[step], convw, hist, S, A, dt_bias, wnorm,
                               fresh=(step == 0))
        check(f"gdn step {step}", gout[step], ref)
    last = a.steps - 1
    serr = (sstate[rid.long()] - S).abs().max().item()
    herr = max((cstate[rid.long(), (last - j) % 4].float() -
                hist[..., 2 - j]).abs().max().item()
               for j in range(min(3, a.steps)))
    print(f"[gdn states] recurrent max |kernel - reference| = {serr:.6f} "
          f"(|ref| max {S.abs().max().item():.3f}), conv history {herr:.6f}")
    if serr > 1e-3 * max(1.0, S.abs().max().item()) or herr > 0:
        ok = False
        print("[gdn states] FAILED")

    pk.finalize()
    if not ok:
        sys.exit(1)
    print("\nPASSED")


if __name__ == "__main__":
    main()
