// qwen35_sm100.cuh -- extension tasks for Qwen3.5's layers (SM100):
//
//   norm1p_sm100      RMSNorm with the weight 1 + w (the "zero-centered" norm)
//   gdn_step_sm100    one decode step of a gated delta net layer: causal
//                     conv + SiLU, the delta-rule state update, the gated norm
//   qsa_prep_sm100    full attention: q / k norm (1 + w), into the attention
//                     task's packed q / k / v layout
//   gate_mul_sm100    full attention: output * sigmoid(gate)
//
// The arithmetic follows the reference model (transformers' Qwen3_5Moe*) step
// by step, including where it rounds to bf16, so the results differ from it
// only by the order of fp32 sums.
//
// The tasks see whole tensors; a wave of G tasks deals tokens (or
// (token, head) units) round-robin over its tasks.
#pragma once
#include "moe_gemm_sm100.cuh"

namespace kernel {
namespace q35 {

using mxg::bf2f;
using mxg::bfround;
using mxg::f2bf;
using mxg::u64;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, off);
  }
  return v;
}
__device__ __forceinline__ float silu(float x) {
  return x / (1.f + __expf(fminf(-x, 40.f)));
}
__device__ __forceinline__ float sigmoid(float x) {
  return 1.f / (1.f + __expf(fminf(-x, 40.f)));
}

} // namespace q35

// ---------------------------------------------------------------------------
// y = (x * rsqrt(mean(x^2) + eps)) * (1 + w), in fp32, rounded to bf16.
// inputs: [0] x bf16 [BATCH][H]; [1] w bf16 [H].
// outputs: [0] y bf16 [BATCH][H].
// WIDE 0: a warp per token, lane l holds the 16-byte chunks l, l + 32, ...
// WIDE 1 (at most a token a task of the wave: small batches): the whole
// block on one token, a thread a chunk -- a warp alone with a token took
// 5 us of the layer twice.
// ---------------------------------------------------------------------------
template <int BATCH, int H, int G, int EPS_BITS, int WIDE, class TD, class RC>
__device__ __forceinline__ void norm1p_sm100(TD const *task_desc,
                                             RC const &runtime_config) {
  using namespace q35;
  uint4 const *x = static_cast<uint4 const *>(task_desc->input_ptrs[0]);
  uint4 const *w = static_cast<uint4 const *>(task_desc->input_ptrs[1]);
  uint4 *y = static_cast<uint4 *>(task_desc->output_ptrs[0]);
  int const lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int const nwarps = blockDim.x >> 5;
  int const task_i = static_cast<int>(task_desc->task_metadata.ext_bid_x);
  float const eps = __int_as_float(EPS_BITS);
  // one chunk (8 values): its sum of squares
  auto squares = [](uint4 const &v) {
    const unsigned *p = reinterpret_cast<const unsigned *>(&v);
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float a = bf2f(p[j] & 0xffff), b = bf2f(p[j] >> 16);
      ss += a * a + b * b;
    }
    return ss;
  };
  auto scaled = [](uint4 const &v, uint4 const &wv, float rs) {
    const unsigned *p = reinterpret_cast<const unsigned *>(&v);
    const unsigned *q = reinterpret_cast<const unsigned *>(&wv);
    uint4 o;
    unsigned *r = reinterpret_cast<unsigned *>(&o);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float a = (bf2f(p[j] & 0xffff) * rs) * (1.f + bf2f(q[j] & 0xffff));
      const float b = (bf2f(p[j] >> 16) * rs) * (1.f + bf2f(q[j] >> 16));
      r[j] = static_cast<unsigned>(f2bf(a)) |
             (static_cast<unsigned>(f2bf(b)) << 16);
    }
    return o;
  };
  if (WIDE) {
    constexpr int CPT = H / 2048; // chunks of 8 values a thread
    static_assert(H % 2048 == 0 && CPT >= 1 && CPT <= 4,
                  "H / 2048 chunks a thread");
    extern __shared__ char shared_memory[];
    float *red = reinterpret_cast<float *>(
        (reinterpret_cast<uintptr_t>(shared_memory) + 15) / 16 * 16);
    int const tid = threadIdx.x, nthreads = blockDim.x;
#pragma unroll 1
    for (int t = task_i; t < BATCH; t += G) {
      uint4 const *xr = x + static_cast<size_t>(t) * (H / 8);
      uint4 *yr = y + static_cast<size_t>(t) * (H / 8);
      uint4 v[CPT], wv[CPT];
      float ss = 0.f;
#pragma unroll
      for (int i = 0; i < CPT; ++i) {
        v[i] = xr[tid + nthreads * i];
        wv[i] = w[tid + nthreads * i];
        ss += squares(v[i]);
      }
      ss = warp_sum(ss);
      if (lane == 0) {
        red[warp] = ss;
      }
      __syncthreads();
      float tot = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        tot += red[i];
      }
      float const rs = rsqrtf(tot / H + eps);
#pragma unroll
      for (int i = 0; i < CPT; ++i) {
        yr[tid + nthreads * i] = scaled(v[i], wv[i], rs);
      }
      __syncthreads();
    }
    return;
  }
  constexpr int CPL = H / 256; // chunks of 8 values a lane
  static_assert(H % 256 == 0 && CPL >= 1 && CPL <= 8, "H / 256 chunks a lane");
  int const gw = task_i * nwarps + warp;
#pragma unroll 1
  for (int t = gw; t < BATCH; t += G * nwarps) {
    uint4 const *xr = x + static_cast<size_t>(t) * (H / 8);
    uint4 *yr = y + static_cast<size_t>(t) * (H / 8);
    uint4 v[CPL];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < CPL; ++i) {
      v[i] = xr[lane + 32 * i];
      ss += squares(v[i]);
    }
    float const rs = rsqrtf(warp_sum(ss) / H + eps);
#pragma unroll
    for (int i = 0; i < CPL; ++i) {
      yr[lane + 32 * i] = scaled(v[i], w[lane + 32 * i], rs);
    }
  }
}

// ---------------------------------------------------------------------------
// Full attention, before the attention task. inputs: [0] proj bf16
// [BATCH][PW]: per KV head its QPK query heads, its key, its value (HD each),
// then the gates; [1] q norm w, [2] k norm w, bf16 [HD].
// outputs: [0] qkv bf16 [BATCH][NKV * (QPK + 2) * HD]: the same layout with
// q and k normalized (weight 1 + w), v copied.
// A warp per (token, head): lane l holds the chunks l (HD = 256: one chunk).
// ---------------------------------------------------------------------------
template <int BATCH,
          int PW,
          int NKV,
          int QPK,
          int HD,
          int G,
          int EPS_BITS,
          class TD,
          class RC>
__device__ __forceinline__ void qsa_prep_sm100(TD const *task_desc,
                                               RC const &runtime_config) {
  using namespace q35;
  constexpr int HPK = QPK + 2;  // heads of a KV head in the layout
  constexpr int NH = NKV * HPK; // rows of a token
  constexpr int CPL = HD / 256;
  static_assert(HD % 256 == 0 && CPL >= 1 && CPL <= 4,
                "HD / 256 chunks a lane");
  uint4 const *proj = static_cast<uint4 const *>(task_desc->input_ptrs[0]);
  uint4 const *wq = static_cast<uint4 const *>(task_desc->input_ptrs[1]);
  uint4 const *wk = static_cast<uint4 const *>(task_desc->input_ptrs[2]);
  uint4 *out = static_cast<uint4 *>(task_desc->output_ptrs[0]);
  int const lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int const nwarps = blockDim.x >> 5;
  int const gw =
      static_cast<int>(task_desc->task_metadata.ext_bid_x) * nwarps + warp;
  float const eps = __int_as_float(EPS_BITS);
#pragma unroll 1
  for (int u = gw; u < BATCH * NH; u += G * nwarps) {
    int const t = u / NH, h = u % NH, role = h % HPK; // < QPK q, QPK k, else v
    uint4 const *xr = proj + (static_cast<size_t>(t) * PW + h * HD) / 8;
    uint4 *yr = out + (static_cast<size_t>(t) * NH + h) * (HD / 8);
    uint4 v[CPL];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < CPL; ++i) {
      v[i] = xr[lane + 32 * i];
      unsigned const *p = reinterpret_cast<unsigned const *>(&v[i]);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float const a = bf2f(p[j] & 0xffff), b = bf2f(p[j] >> 16);
        ss += a * a + b * b;
      }
    }
    float const rs = rsqrtf(warp_sum(ss) / HD + eps);
    uint4 const *wn = role == QPK ? wk : wq;
#pragma unroll
    for (int i = 0; i < CPL; ++i) {
      uint4 o = v[i];
      if (role <= QPK) {
        const uint4 wv = wn[lane + 32 * i];
        unsigned const *p = reinterpret_cast<unsigned const *>(&v[i]);
        unsigned const *q = reinterpret_cast<unsigned const *>(&wv);
        unsigned *r = reinterpret_cast<unsigned *>(&o);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          float const a =
              (bf2f(p[j] & 0xffff) * rs) * (1.f + bf2f(q[j] & 0xffff));
          float const b = (bf2f(p[j] >> 16) * rs) * (1.f + bf2f(q[j] >> 16));
          r[j] = static_cast<unsigned>(f2bf(a)) |
                 (static_cast<unsigned>(f2bf(b)) << 16);
        }
      }
      yr[lane + 32 * i] = o;
    }
  }
}

// ---------------------------------------------------------------------------
// Full attention, after the attention task: y = a * sigmoid(gate), both
// factors bf16 as in the reference. inputs: [0] a bf16 [BATCH][W]; [1] proj
// bf16 [BATCH][PW], the gates at column GOFF. outputs: [0] y bf16 [BATCH][W].
// Unit = (token, 16-byte chunk of 8 columns); the wave's lanes take the units
// round-robin, two at a time (a warp alone with a token walked its chunks one
// after another, each waiting for its loads: 7.5 us a token).
// ---------------------------------------------------------------------------
template <int BATCH, int W, int PW, int GOFF, int G, class TD, class RC>
__device__ __forceinline__ void gate_mul_sm100(TD const *task_desc,
                                               RC const &runtime_config) {
  using namespace q35;
  constexpr int NCHUNK = W / 8;
  static_assert(W % 8 == 0 && GOFF % 8 == 0 && PW % 8 == 0, "16 B chunks");
  uint4 const *a = static_cast<uint4 const *>(task_desc->input_ptrs[0]);
  uint4 const *proj = static_cast<uint4 const *>(task_desc->input_ptrs[1]);
  uint4 *y = static_cast<uint4 *>(task_desc->output_ptrs[0]);
  int const nthreads = blockDim.x;
  int const stride = G * nthreads;
  int const gid =
      static_cast<int>(task_desc->task_metadata.ext_bid_x) * nthreads +
      threadIdx.x;
#pragma unroll 1
  for (int u0 = gid; u0 < BATCH * NCHUNK; u0 += 2 * stride) {
    uint4 av[2], gv[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      int const u = u0 + i * stride;
      if (u < BATCH * NCHUNK) {
        int const t = u / NCHUNK, c = u % NCHUNK;
        av[i] = a[static_cast<size_t>(t) * NCHUNK + c];
        gv[i] = proj[(static_cast<size_t>(t) * PW + GOFF) / 8 + c];
      }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      int const u = u0 + i * stride;
      if (u < BATCH * NCHUNK) {
        unsigned const *p = reinterpret_cast<unsigned const *>(&av[i]);
        unsigned const *q = reinterpret_cast<unsigned const *>(&gv[i]);
        uint4 o;
        unsigned *r = reinterpret_cast<unsigned *>(&o);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          float const lo =
              bf2f(p[j] & 0xffff) * bfround(sigmoid(bf2f(q[j] & 0xffff)));
          float const hi =
              bf2f(p[j] >> 16) * bfround(sigmoid(bf2f(q[j] >> 16)));
          r[j] = static_cast<unsigned>(f2bf(lo)) |
                 (static_cast<unsigned>(f2bf(hi)) << 16);
        }
        y[u] = o;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// One decode step of a gated delta net layer, a unit = (token, value head).
// inputs: [0] proj bf16 [BATCH][PW]: q | k | v | z | b | a (q, k: NKH heads of
// D; v, z: 2 NKH heads of D; b, a: 2 NKH) -- the in_proj_qkv / z / b / a
// outputs of the normalized hidden state; [1] conv weight bf16 [C][4] (C =
// 4 NKH D conv channels: q | k | v; a channel's taps, oldest first); [2] conv
// state bf16 [NREQ][4][C], the last inputs of a request as a ring: input t in
// slot t % 4; [3] recurrent state fp32 [NREQ][2 NKH][D][D] (key dim, value
// dim); [4] fp32 [2 NKH + 2 NKH + D]: -exp(A_log) | dt_bias | the gated
// norm's w. outputs: [0] bf16 [BATCH][2 NKH D], the layer's input to out_proj.
// The request and its position come from the runtime (request_ids, step) or,
// with XMAP (tests), from input [5], int32 [BATCH][2]: position 0 starts from
// zero states.
//
// Per value head (the reference's torch_recurrent_gated_delta_rule):
//   S <- S exp(g); kv = S^T k; delta = (v - kv) beta; S <- S + k delta^T;
//   o = S^T q, with k, q L2-normalized and q scaled by 1 / sqrt(D).
// The head's state (D x D fp32) is copied into shared memory and read twice
// there (S^T k and S^T q, then the update); the new state goes straight to
// global memory. A warp owns 16 rows of the state, a lane 4 columns: a row is
// one 16-byte access of the warp. The two value heads of a key head are two
// units (two tasks at small batch): both do the key head's q / k conv, the
// even one writes its conv state -- hence the ring of 4 slots, which input t
// never reads the slot it writes.
// ---------------------------------------------------------------------------
template <int BATCH,
          int NREQ,
          int NKH,
          int D,
          int PW,
          int G,
          int EPS_BITS,
          int XMAP,
          class TD,
          class RC>
__device__ __forceinline__ void gdn_step_sm100(TD const *task_desc,
                                               RC const &runtime_config) {
  using namespace q35;
  using namespace mxg;
  constexpr int NVH = 2 * NKH;
  constexpr int C = 4 * NKH * D; // conv channels: q, k, v
  constexpr int SBYTES = D * D * 4;
  constexpr int RPW = D / 8; // state rows a warp
  static_assert(D == 128, "8 warps of 16 rows, 32 lanes of 4 columns");
  static_assert(PW >= C + NVH * D + 2 * NVH && PW % 4 == 0, "proj width");
  unsigned short const *proj =
      static_cast<unsigned short const *>(task_desc->input_ptrs[0]);
  unsigned short const *convw =
      static_cast<unsigned short const *>(task_desc->input_ptrs[1]);
  unsigned short *cstate =
      static_cast<unsigned short *>(task_desc->input_ptrs[2]);
  float *sglob = static_cast<float *>(task_desc->input_ptrs[3]);
  float const *par = static_cast<float const *>(task_desc->input_ptrs[4]);
  unsigned short *out =
      static_cast<unsigned short *>(task_desc->output_ptrs[0]);
  int const task_i = static_cast<int>(task_desc->task_metadata.ext_bid_x);
  float const eps = __int_as_float(EPS_BITS);

  extern __shared__ char shared_memory[];
  char *base = reinterpret_cast<char *>(
      (reinterpret_cast<uintptr_t>(shared_memory) + 127) / 128 * 128);
  float *S = reinterpret_cast<float *>(base); // [D][D]
  float *kn = S + D * D;                      // [D] normalized k
  float *qn = kn + D;                         // [D] normalized, scaled q
  float *vv = qn + D;                         // [D] the value column
  float *pm = vv + D;                         // [8 warps][2][D] partial sums
  float *red = pm + 16 * D;                   // [16] reductions

  int const tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  int const half = tid >> 7, j = tid & 127; // conv: q + v (half 0), k (half 1)
#pragma unroll 1
  for (int u = task_i; u < BATCH * NVH; u += G) {
    int const t = u / NVH, v = u % NVH, hk = v / 2;
    int rid, pos = 0;
    if (XMAP) {
      int const *xm = static_cast<int const *>(task_desc->input_ptrs[5]);
      rid = xm[2 * t];
      pos = xm[2 * t + 1];
    } else {
      rid = runtime_config.request_ids[t];
      if (rid >= 0) {
        pos = runtime_config.step[rid];
      }
    }
    if (rid < 0) {
      continue; // an empty request slot (uniform over the block)
    }
    bool const fresh = pos == 0;
    unsigned short const *pr = proj + static_cast<size_t>(t) * PW;
    unsigned short *cs = cstate + static_cast<size_t>(rid) * 4 * C;
    float *sg = sglob + (static_cast<size_t>(rid) * NVH + v) * D * D;
    int const sw = pos & 3, s0 = (pos + 1) & 3, s1 = (pos + 2) & 3,
              s2 = (pos + 3) & 3;

    // ---- this thread's inputs first: their loads are in flight while the
    // state's copies are issued
    int const c0 = (half * NKH + hk) * D + j; // q or k channel
    int const c1 = 2 * NKH * D + v * D + j;   // value channel (half 0)
    unsigned short const xin0 = pr[c0];
    const uint2 w0 = reinterpret_cast<uint2 const *>(convw)[c0];
    unsigned short xin1 = 0;
    uint2 w1 = make_uint2(0, 0);
    if (half == 0) {
      xin1 = pr[c1];
      w1 = reinterpret_cast<uint2 const *>(convw)[c1];
    }
    unsigned short h0[3] = {0, 0, 0}, h1[3] = {0, 0, 0};
    if (!fresh) {
      h0[0] = cs[s0 * C + c0];
      h0[1] = cs[s1 * C + c0];
      h0[2] = cs[s2 * C + c0];
      if (half == 0) {
        h1[0] = cs[s0 * C + c1];
        h1[1] = cs[s1 * C + c1];
        h1[2] = cs[s2 * C + c1];
      }
    }
    unsigned short const bg_raw = pr[C + NVH * D + v];
    unsigned short const ag_raw = pr[C + NVH * D + NVH + v];
    float const p_a = par[v], p_dt = par[NVH + v];
    uint2 zraw = make_uint2(0, 0);
    float4 wn = make_float4(0.f, 0.f, 0.f, 0.f);
    if (warp == 0) {
      zraw = *reinterpret_cast<uint2 const *>(pr + C + v * D + 4 * lane);
      wn = *reinterpret_cast<float4 const *>(par + 2 * NVH + 4 * lane);
    }
    if (!fresh) {
#pragma unroll 1
      for (int i = tid; i < SBYTES / 16; i += blockDim.x) {
        cp_async16(reinterpret_cast<char *>(S) + 16 * i,
                   reinterpret_cast<char const *>(sg) + 16 * i);
      }
      asm volatile("cp.async.commit_group;" ::: "memory");
    }
    // (nothing that uses a loaded value before here: the warp would wait for
    // it with the copies still to issue)
    float const bg = bf2f(bg_raw), ag = bf2f(ag_raw);

    // ---- causal conv + SiLU: taps w0 x(t-3), w1 x(t-2), w2 x(t-1), w3 x(t)
    float qk, vcol = 0.f;
    {
      float acc = bf2f(w0.y >> 16) * bf2f(xin0);
      if (!fresh) {
        acc += bf2f(w0.x & 0xffff) * bf2f(h0[0]) +
               bf2f(w0.x >> 16) * bf2f(h0[1]) +
               bf2f(w0.y & 0xffff) * bf2f(h0[2]);
      }
      qk = bfround(silu(bfround(acc)));
      if ((v & 1) == 0) { // the key head's q / k conv state: the even head's
        cs[sw * C + c0] = xin0;
        if (fresh) {
          cs[s0 * C + c0] = 0;
          cs[s1 * C + c0] = 0;
          cs[s2 * C + c0] = 0;
        }
      }
    }
    if (half == 0) {
      float acc = bf2f(w1.y >> 16) * bf2f(xin1);
      if (!fresh) {
        acc += bf2f(w1.x & 0xffff) * bf2f(h1[0]) +
               bf2f(w1.x >> 16) * bf2f(h1[1]) +
               bf2f(w1.y & 0xffff) * bf2f(h1[2]);
      }
      vcol = bfround(silu(bfround(acc)));
      cs[sw * C + c1] = xin1;
      if (fresh) {
        cs[s0 * C + c1] = 0;
        cs[s1 * C + c1] = 0;
        cs[s2 * C + c1] = 0;
      }
      vv[j] = vcol;
    }
    // ---- L2 norms of q (warps 0 .. 3) and k (warps 4 .. 7)
    float ss = warp_sum(qk * qk);
    if (lane == 0) {
      red[warp] = ss;
    }
    __syncthreads();
    {
      float const tot = red[4 * half] + red[4 * half + 1] + red[4 * half + 2] +
                        red[4 * half + 3];
      float const rs = rsqrtf(tot + 1e-6f);
      if (half == 0) {
        qn[j] = (qk * rs) * 0.08838834764831845f; // 1 / sqrt(128)
      } else {
        kn[j] = qk * rs;
      }
    }
    float const beta = bfround(sigmoid(bg));
    float const sp_in = ag + p_dt;
    float const sp = sp_in > 20.f ? sp_in : log1pf(__expf(sp_in));
    float const decay = __expf(p_a * sp);
    if (!fresh) {
      asm volatile("cp.async.wait_group 0;" ::: "memory");
    }
    __syncthreads();
    // ---- k . q, and the partial S^T k, S^T q of this warp's rows
    if (half == 0) {
      float const kq = warp_sum(qn[j] * kn[j]);
      if (lane == 0) {
        red[8 + warp] = kq;
      }
    }
    float4 const *row =
        reinterpret_cast<float4 const *>(S + (RPW * warp) * D) + lane;
    float4 const *kn4 = reinterpret_cast<float4 const *>(kn) + (RPW / 4) * warp;
    float4 const *qn4 = reinterpret_cast<float4 const *>(qn) + (RPW / 4) * warp;
    {
      float4 a = make_float4(0.f, 0.f, 0.f, 0.f), b = a;
      if (!fresh) {
#pragma unroll
        for (int r4 = 0; r4 < RPW / 4; ++r4) {
          const float4 kv = kn4[r4], qv = qn4[r4];
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const float4 s = row[(4 * r4 + q) * (D / 4)];
            float const kk = q == 0   ? kv.x
                             : q == 1 ? kv.y
                             : q == 2 ? kv.z
                                      : kv.w;
            float const qq = q == 0   ? qv.x
                             : q == 1 ? qv.y
                             : q == 2 ? qv.z
                                      : qv.w;
            a.x += s.x * kk;
            a.y += s.y * kk;
            a.z += s.z * kk;
            a.w += s.w * kk;
            b.x += s.x * qq;
            b.y += s.y * qq;
            b.z += s.z * qq;
            b.w += s.w * qq;
          }
        }
      }
      reinterpret_cast<float4 *>(pm + (2 * warp) * D)[lane] = a;
      reinterpret_cast<float4 *>(pm + (2 * warp + 1) * D)[lane] = b;
    }
    __syncthreads();
    float4 m1 = make_float4(0.f, 0.f, 0.f, 0.f), m2 = m1;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      const float4 a = reinterpret_cast<float4 const *>(pm + (2 * w) * D)[lane];
      const float4 b =
          reinterpret_cast<float4 const *>(pm + (2 * w + 1) * D)[lane];
      m1.x += a.x;
      m1.y += a.y;
      m1.z += a.z;
      m1.w += a.w;
      m2.x += b.x;
      m2.y += b.y;
      m2.z += b.z;
      m2.w += b.w;
    }
    float const kq = red[8] + red[9] + red[10] + red[11];
    const float4 vc = reinterpret_cast<float4 const *>(vv)[lane];
    float4 dl;
    dl.x = (vc.x - decay * m1.x) * beta;
    dl.y = (vc.y - decay * m1.y) * beta;
    dl.z = (vc.z - decay * m1.z) * beta;
    dl.w = (vc.w - decay * m1.w) * beta;
    // ---- the new state, straight to global memory
    {
      float4 *grow = reinterpret_cast<float4 *>(sg + (RPW * warp) * D) + lane;
#pragma unroll
      for (int r4 = 0; r4 < RPW / 4; ++r4) {
        const float4 kv = kn4[r4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
          if (!fresh) {
            s = row[(4 * r4 + q) * (D / 4)];
          }
          float const kk = q == 0 ? kv.x : q == 1 ? kv.y : q == 2 ? kv.z : kv.w;
          s.x = decay * s.x + kk * dl.x;
          s.y = decay * s.y + kk * dl.y;
          s.z = decay * s.z + kk * dl.z;
          s.w = decay * s.w + kk * dl.w;
          grow[(4 * r4 + q) * (D / 4)] = s;
        }
      }
    }
    // ---- the output with its gated norm (mean over the D columns): warp 0
    if (warp == 0) {
      float4 o;
      o.x = bfround(decay * m2.x + dl.x * kq);
      o.y = bfround(decay * m2.y + dl.y * kq);
      o.z = bfround(decay * m2.z + dl.z * kq);
      o.w = bfround(decay * m2.w + dl.w * kq);
      float const tot = warp_sum(o.x * o.x + o.y * o.y + o.z * o.z + o.w * o.w);
      float const rn = rsqrtf(tot / D + eps);
      float const z[4] = {bf2f(zraw.x & 0xffff),
                          bf2f(zraw.x >> 16),
                          bf2f(zraw.y & 0xffff),
                          bf2f(zraw.y >> 16)};
      float const ov[4] = {o.x, o.y, o.z, o.w};
      float const wv[4] = {wn.x, wn.y, wn.z, wn.w};
      unsigned short y[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        float const n = bfround(ov[q] * rn);
        y[q] = f2bf(bfround(wv[q] * n) * silu(z[q]));
      }
      *reinterpret_cast<uint2 *>(out + static_cast<size_t>(t) * NVH * D +
                                 v * D + 4 * lane) =
          make_uint2(y[0] | (static_cast<unsigned>(y[1]) << 16),
                     y[2] | (static_cast<unsigned>(y[3]) << 16));
    }
    __syncthreads();
  }
}

} // namespace kernel
