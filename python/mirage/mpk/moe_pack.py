"""Expert-weight layouts and plan sizes for moe_route_layer / moe_gemm_layer
(include/mirage/persistent_kernel/tasks/blackwell/moe_gemm_sm100.cuh)."""

import torch


def mxfp4_step(k: int) -> int:
    """Columns of one MXFP4 weight block (one bulk copy, one step of the
    kernel) for K columns: every step costs the kernel a quarter microsecond
    of waits whatever its size, so the widest that divides K and still leaves
    two A stages in TMEM (160 columns each at 320)."""
    for kb in (320, 192, 64):
        if k % kb == 0:
            return kb
    raise ValueError(f"K = {k} is not a multiple of 64")

# Block scales the kernel converts exactly: every e2m1 value times
# 2^(s - 127) and 2^(s - 15) must be normal bf16 numbers.
MXFP4_SCALE_MIN, MXFP4_SCALE_MAX = 2, 142


def moe_tokens_per_pass(batch: int) -> int:
    """Tokens of one expert that a GEMM pass takes (the kernel's NT)."""
    for nt in (16, 32, 64, 128):
        if batch <= nt:
            return nt
    return 128


def moe_plan_cap(batch: int, num_experts: int, slots: int, nt: int) -> int:
    """Most (expert, pass) pairs a routing can have."""
    return num_experts + (batch * slots + nt - 1) // nt


def moe_plan_tensors(batch: int, num_experts: int, slots: int, nt: int,
                     device="cuda"):
    """(weights, route_list, route_meta) buffers of one routing."""
    cap = moe_plan_cap(batch, num_experts, slots, nt)
    return (torch.zeros(batch, slots, dtype=torch.float32, device=device),
            torch.zeros(num_experts, batch, dtype=torch.int32, device=device),
            torch.zeros(1 + cap + num_experts, dtype=torch.int32,
                        device=device))


def pack_mxfp4(blocks: torch.Tensor, scales: torch.Tensor,
               kb: int = None) -> torch.Tensor:
    """MXFP4 expert weights as released -> the kernel's blocks.

    blocks uint8 [E, R, K / 32, 16] (32 e2m1 values a group, the low nibble
    first), scales uint8 [E, R, K / 32] (the value is 2^(scale - 127)): row r
    of expert e holds the K weights of output r. Returns uint8
    [E, ceil(R / 128), K / kb, bytes]; a block is [kb / 32 groups][128 rows]
    [16 B] followed by [128 rows][kb / 32] scales. Rows past R are zero.
    """
    assert blocks.dtype == torch.uint8 and scales.dtype == torch.uint8
    e, r, groups, nbytes = blocks.shape
    assert nbytes == 16 and scales.shape == (e, r, groups)
    k = groups * 32
    if kb is None:
        kb = mxfp4_step(k)
    assert kb % 64 == 0 and k % kb == 0, (k, kb)
    lo, hi = int(scales.min()), int(scales.max())
    if lo < MXFP4_SCALE_MIN or hi > MXFP4_SCALE_MAX:
        raise ValueError(
            f"MXFP4 block scales {lo} .. {hi} are outside {MXFP4_SCALE_MIN} .. "
            f"{MXFP4_SCALE_MAX}: the bf16 conversion would not be exact")
    nch, nkb, rb = kb // 32, k // kb, (r + 127) // 128
    pad = rb * 128 - r
    if pad:
        blocks = torch.cat(
            [blocks, blocks.new_zeros(e, pad, groups, 16)], dim=1)
        scales = torch.cat(
            [scales, scales.new_full((e, pad, groups), 127)], dim=1)
    b = (blocks.reshape(e, rb, 128, nkb, nch, 16).permute(0, 1, 3, 4, 2, 5)
         .reshape(e, rb, nkb, nch * 128 * 16))
    s = (scales.reshape(e, rb, 128, nkb, nch).permute(0, 1, 3, 2, 4)
         .reshape(e, rb, nkb, 128 * nch))
    return torch.cat([b, s], dim=-1).contiguous()


def unpack_mxfp4(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """The values of MXFP4 weights, float32 [E, R, K] (exact)."""
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                       dtype=torch.float32, device=blocks.device)
    e, r, groups, _ = blocks.shape
    v = torch.stack([lut[(blocks & 0xF).long()], lut[(blocks >> 4).long()]],
                    dim=-1).reshape(e, r, groups, 32)
    v = torch.ldexp(v, (scales.to(torch.int32) - 127)[..., None])
    return v.reshape(e, r, groups * 32)


def interleave_gate_up(w: torch.Tensor) -> torch.Tensor:
    """[E, 2 I, ...] with the gate rows first -> rows (gate 0, up 0, gate 1,
    ...), the order moe_gemm_layer's activation expects."""
    half = w.shape[1] // 2
    return torch.stack([w[:, :half], w[:, half:]], dim=2).reshape(w.shape)


def pack_bf16_tiles(w: torch.Tensor) -> torch.Tensor:
    """bf16 expert weights [E, R, K] -> [E, ceil(R / 128), K / 64, 128 * 64]:
    tiles of 128 rows x 64 columns in the MMA's K-major SWIZZLE_128B order
    (the 16-byte chunk c of row m sits at chunk c ^ (m % 8))."""
    assert w.dtype == torch.bfloat16
    e, r, k = w.shape
    assert k % 64 == 0
    rb = (r + 127) // 128
    pad = rb * 128 - r
    if pad:
        w = torch.cat([w, w.new_zeros(e, pad, k)], dim=1)
    t = w.reshape(e, rb, 128, k // 64, 8, 8).permute(0, 1, 3, 2, 4, 5)
    m = torch.arange(128, device=w.device)[:, None]
    c = torch.arange(8, device=w.device)[None, :]
    t = t[:, :, :, m, c ^ (m & 7)]
    return t.reshape(e, rb, k // 64, 128 * 64).contiguous()
