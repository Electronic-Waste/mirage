"""Test-mode coverage for wider tiles of the SM100 linear tasks.

PersistentKernel(linear_mma_n=N) sets how many tokens one tile of a linear
task covers (16 by default). A wider tile changes how often a task reads its
weights, not what it computes: every output element is the same chain of MMAs
into an fp32 accumulator. So the plain, the residual and the bias variant must
give the SAME BITS for every N, and N = 16 must agree with torch.

The split-K variant adds its partial sums to the output in bf16, in the order
the tasks happen to finish, so it is only compared with torch.

The batch is 200: 13, 7, 4 and 2 tiles for N = 16, 32, 64, 128, the last one
partial each time. One kernel is built per N, each in its own process.
"""

import os
import subprocess
import sys
import tempfile

import torch

BATCH = 200
HIDDEN = 1024
WIDTHS = (16, 32, 64, 128)

# (name, kind, out_features, grid)
CONFIGS = [
    ("plain", "plain", 640, (5, 1, 1)),        # 128 columns per task
    ("residual", "residual", 512, (8, 1, 1)),  # 64 columns per task
    ("bias", "bias", 320, (4, 1, 1)),          # 80 columns per task
    ("splitk", "splitk", 256, (2, 4, 1)),      # 128 columns, K / 4 per task
]


def make_inputs():
    g = torch.Generator().manual_seed(0)
    data = {"x": torch.randn(BATCH, HIDDEN, generator=g) * 0.1}
    for name, _, out_features, _ in CONFIGS:
        data[name + "_w"] = torch.randn(out_features, HIDDEN, generator=g) * 0.05
        data[name + "_r"] = torch.randn(BATCH, out_features, generator=g)
        data[name + "_b"] = torch.randn(1, out_features, generator=g)
    return {k: v.to(torch.bfloat16) for k, v in data.items()}


def reference(data, name, kind):
    ref = data["x"].float() @ data[name + "_w"].float().T
    if kind in ("residual", "splitk"):
        ref = ref + data[name + "_r"].float()
    if kind == "bias":
        ref = ref + data[name + "_b"].float()
    return ref


def run_one(width, path):
    import mirage
    from mirage.mpk.persistent_kernel import PersistentKernel

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        max_num_batched_tokens=BATCH,
        max_num_batched_requests=1,
        # (test mode traps on an empty KV page pool)
        max_num_pages=16,
        page_size=max(64, BATCH),
        linear_mma_n=width,
    )
    pk = PersistentKernel(**params)

    data = {k: v.cuda() for k, v in make_inputs().items()}
    x_dt = pk.attach_input(data["x"], name="x")
    outs = {}
    for name, kind, out_features, grid in CONFIGS:
        w_dt = pk.attach_input(data[name + "_w"], name=name + "_w")
        out = torch.zeros(BATCH, out_features, dtype=torch.bfloat16,
                          device="cuda")
        if kind == "splitk":
            out.copy_(data[name + "_r"])
        out_dt = pk.attach_input(out, name=name + "_out")
        if kind == "plain":
            pk.linear_layer(input=x_dt, weight=w_dt, output=out_dt,
                            grid_dim=grid, block_dim=(256, 1, 1))
        elif kind == "bias":
            pk.linear_layer(
                input=x_dt, weight=w_dt, output=out_dt, grid_dim=grid,
                block_dim=(256, 1, 1),
                bias=pk.attach_input(data[name + "_b"], name=name + "_b"))
        elif kind == "residual":
            pk.linear_with_residual_layer(
                input=x_dt, weight=w_dt,
                residual=pk.attach_input(data[name + "_r"], name=name + "_r"),
                output=out_dt, grid_dim=grid, block_dim=(256, 1, 1))
        else:
            pk.splitk_linear_layer(input=x_dt, weight=w_dt, output=out_dt,
                                   grid_dim=grid, block_dim=(256, 1, 1))
        outs[name] = out

    workdir = tempfile.mkdtemp(prefix=f"linear_mma_n_{width}_")
    pk.compile(output_dir=workdir)
    pk()
    torch.cuda.synchronize()
    torch.save({k: v.cpu() for k, v in outs.items()}, path)
    pk.finalize()


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--one":
        run_one(int(sys.argv[2]), sys.argv[3])
        return

    tmp = tempfile.mkdtemp(prefix="linear_mma_n_")
    results = {}
    for width in WIDTHS:
        path = os.path.join(tmp, f"n{width}.pt")
        print(f"Building and running the test kernel with linear_mma_n={width}...",
              flush=True)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--one",
                            str(width), path], capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(path):
            print(r.stdout[-3000:])
            print(r.stderr[-3000:])
            print(f"FAILED: linear_mma_n={width} did not build or run")
            sys.exit(1)
        results[width] = torch.load(path)

    data = make_inputs()
    ok = True
    for name, kind, out_features, grid in CONFIGS:
        ref = reference(data, name, kind)
        # (the split-K sum is rounded to bf16 once per K slice)
        tol = 0.1 if kind == "splitk" else 0.05
        line = [f"[{name}]"]
        for width in WIDTHS:
            out = results[width][name]
            diff = (out.float() - ref).abs().max().item()
            same = torch.equal(out, results[WIDTHS[0]][name])
            line.append(f"N={width}: max diff {diff:.4f}"
                        + ("" if width == WIDTHS[0] else
                           (", same bits" if same else ", OTHER BITS")))
            if diff >= tol:
                print(f"[{name}] FAILED: N={width} disagrees with torch")
                ok = False
            if kind != "splitk" and not same:
                print(f"[{name}] FAILED: N={width} is not bit-identical to "
                      f"N={WIDTHS[0]}")
                ok = False
        print("  ".join(line))

    if not ok:
        sys.exit(1)
    print("\nPASSED: the SM100 linear tasks give the same results with tiles "
          "of 16, 32, 64 and 128 tokens")


if __name__ == "__main__":
    main()
