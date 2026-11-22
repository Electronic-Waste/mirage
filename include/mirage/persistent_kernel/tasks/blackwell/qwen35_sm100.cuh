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
// shared -> global bulk store of this thread, and its completion
__device__ __forceinline__ void
    bulk_store(void *dst, const void *src, unsigned bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::
                   "l"(reinterpret_cast<u64>(dst)),
               "r"(mxg::sm_u32(src)),
               "r"(bytes)
               : "memory");
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
__device__ __forceinline__ void bulk_store_wait() {
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}
__device__ __forceinline__ void fence_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

} // namespace q35

// ---------------------------------------------------------------------------
// y = (x * rsqrt(mean(x^2) + eps)) * (1 + w), in fp32, rounded to bf16.
// inputs: [0] x bf16 [BATCH][H]; [1] w bf16 [H]. outputs: [0] y bf16 [BATCH][H].
// WIDE 0: a warp per token, lane l holds the 16-byte chunks l, l + 32, ...
// WIDE 1 (at most a token a task of the wave: small batches): the whole
// block on one token, a thread a chunk -- a warp alone with a token took
// 5 us of the layer twice.
// ---------------------------------------------------------------------------
template <int BATCH, int H, int G, int EPS_BITS, int WIDE, class TD, class RC>
__device__ __forceinline__ void
    norm1p_sm100(TD const *task_desc, RC const &runtime_config) {
  using namespace q35;
  const uint4 *x = static_cast<const uint4 *>(task_desc->input_ptrs[0]);
  const uint4 *w = static_cast<const uint4 *>(task_desc->input_ptrs[1]);
  uint4 *y = static_cast<uint4 *>(task_desc->output_ptrs[0]);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int nwarps = blockDim.x >> 5;
  const int task_i = static_cast<int>(task_desc->task_metadata.ext_bid_x);
  const float eps = __int_as_float(EPS_BITS);
  // one chunk (8 values): its sum of squares
  auto squares = [](const uint4 &v) {
    const unsigned *p = reinterpret_cast<const unsigned *>(&v);
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float a = bf2f(p[j] & 0xffff), b = bf2f(p[j] >> 16);
      ss += a * a + b * b;
    }
    return ss;
  };
  auto scaled = [](const uint4 &v, const uint4 &wv, float rs) {
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
    static_assert(H % 2048 == 0 && CPT >= 1 && CPT <= 4, "H / 2048 chunks a thread");
    extern __shared__ char shared_memory[];
    float *red = reinterpret_cast<float *>(
        (reinterpret_cast<uintptr_t>(shared_memory) + 15) / 16 * 16);
    const int tid = threadIdx.x, nthreads = blockDim.x;
#pragma unroll 1
    for (int t = task_i; t < BATCH; t += G) {
      const uint4 *xr = x + static_cast<size_t>(t) * (H / 8);
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
      const float rs = rsqrtf(tot / H + eps);
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
  const int gw = task_i * nwarps + warp;
#pragma unroll 1
  for (int t = gw; t < BATCH; t += G * nwarps) {
    const uint4 *xr = x + static_cast<size_t>(t) * (H / 8);
    uint4 *yr = y + static_cast<size_t>(t) * (H / 8);
    uint4 v[CPL];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < CPL; ++i) {
      v[i] = xr[lane + 32 * i];
      ss += squares(v[i]);
    }
    const float rs = rsqrtf(warp_sum(ss) / H + eps);
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
__device__ __forceinline__ void
    qsa_prep_sm100(TD const *task_desc, RC const &runtime_config) {
  using namespace q35;
  constexpr int HPK = QPK + 2;       // heads of a KV head in the layout
  constexpr int NH = NKV * HPK;      // rows of a token
  constexpr int CPL = HD / 256;
  static_assert(HD % 256 == 0 && CPL >= 1 && CPL <= 4, "HD / 256 chunks a lane");
  const uint4 *proj = static_cast<const uint4 *>(task_desc->input_ptrs[0]);
  const uint4 *wq = static_cast<const uint4 *>(task_desc->input_ptrs[1]);
  const uint4 *wk = static_cast<const uint4 *>(task_desc->input_ptrs[2]);
  uint4 *out = static_cast<uint4 *>(task_desc->output_ptrs[0]);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int nwarps = blockDim.x >> 5;
  const int gw = static_cast<int>(task_desc->task_metadata.ext_bid_x) * nwarps +
                 warp;
  const float eps = __int_as_float(EPS_BITS);
#pragma unroll 1
  for (int u = gw; u < BATCH * NH; u += G * nwarps) {
    const int t = u / NH, h = u % NH, role = h % HPK; // < QPK q, QPK k, else v
    const uint4 *xr = proj + (static_cast<size_t>(t) * PW + h * HD) / 8;
    uint4 *yr = out + (static_cast<size_t>(t) * NH + h) * (HD / 8);
    uint4 v[CPL];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < CPL; ++i) {
      v[i] = xr[lane + 32 * i];
      const unsigned *p = reinterpret_cast<const unsigned *>(&v[i]);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float a = bf2f(p[j] & 0xffff), b = bf2f(p[j] >> 16);
        ss += a * a + b * b;
      }
    }
    const float rs = rsqrtf(warp_sum(ss) / HD + eps);
    const uint4 *wn = role == QPK ? wk : wq;
#pragma unroll
    for (int i = 0; i < CPL; ++i) {
      uint4 o = v[i];
      if (role <= QPK) {
        const uint4 wv = wn[lane + 32 * i];
        const unsigned *p = reinterpret_cast<const unsigned *>(&v[i]);
        const unsigned *q = reinterpret_cast<const unsigned *>(&wv);
        unsigned *r = reinterpret_cast<unsigned *>(&o);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float a =
              (bf2f(p[j] & 0xffff) * rs) * (1.f + bf2f(q[j] & 0xffff));
          const float b = (bf2f(p[j] >> 16) * rs) * (1.f + bf2f(q[j] >> 16));
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
// A warp per token, W / 256 chunks a lane.
// ---------------------------------------------------------------------------
template <int BATCH, int W, int PW, int GOFF, int G, class TD, class RC>
__device__ __forceinline__ void
    gate_mul_sm100(TD const *task_desc, RC const &runtime_config) {
  using namespace q35;
  constexpr int CPL = W / 256;
  static_assert(W % 256 == 0 && GOFF % 8 == 0 && PW % 8 == 0, "16 B chunks");
  const uint4 *a = static_cast<const uint4 *>(task_desc->input_ptrs[0]);
  const uint4 *proj = static_cast<const uint4 *>(task_desc->input_ptrs[1]);
  uint4 *y = static_cast<uint4 *>(task_desc->output_ptrs[0]);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int nwarps = blockDim.x >> 5;
  const int gw = static_cast<int>(task_desc->task_metadata.ext_bid_x) * nwarps +
                 warp;
#pragma unroll 1
  for (int t = gw; t < BATCH; t += G * nwarps) {
    const uint4 *ar = a + static_cast<size_t>(t) * (W / 8);
    const uint4 *gr = proj + (static_cast<size_t>(t) * PW + GOFF) / 8;
    uint4 *yr = y + static_cast<size_t>(t) * (W / 8);
#pragma unroll 1
    for (int i = 0; i < CPL; ++i) {
      const uint4 av = ar[lane + 32 * i], gv = gr[lane + 32 * i];
      const unsigned *p = reinterpret_cast<const unsigned *>(&av);
      const unsigned *q = reinterpret_cast<const unsigned *>(&gv);
      uint4 o;
      unsigned *r = reinterpret_cast<unsigned *>(&o);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float lo = bf2f(p[j] & 0xffff) * bfround(sigmoid(bf2f(q[j] & 0xffff)));
        const float hi = bf2f(p[j] >> 16) * bfround(sigmoid(bf2f(q[j] >> 16)));
        r[j] = static_cast<unsigned>(f2bf(lo)) |
               (static_cast<unsigned>(f2bf(hi)) << 16);
      }
      yr[lane + 32 * i] = o;
    }
  }
}

// ---------------------------------------------------------------------------
// One decode step of a gated delta net layer, one token a request.
// inputs:
//   [0] proj bf16 [BATCH][PW]: q (NKH x D), k (NKH x D), v (2 NKH x D),
//       z (2 NKH x D), b (2 NKH), a (2 NKH) -- the in_proj_qkv / z / b / a
//       outputs of the normalized hidden state;
//   [1] conv weight bf16 [4 NKH D][4] (the taps of a channel, oldest first);
//   [2] conv state bf16 [NREQ][3][4 NKH D]: the last three inputs of a
//       request, input t in slot t % 3;
//   [3] recurrent state fp32 [NREQ][2 NKH][D][D] (key dim, value dim);
//   [4] fp32 [2 NKH + 2 NKH + D]: -exp(A_log), dt_bias, the gated norm's w.
// outputs: [0] bf16 [BATCH][2 NKH D], the layer's input to out_proj.
// Unit = (token, key head): the head's q and k, its two value heads and their
// states (two value heads share a key head). The request and its position come
// from the runtime (request_ids, step) or, with XMAP (tests), from input [5],
// int32 [BATCH][2]: position 0 starts from zero states.
// Per value head (the reference's torch_recurrent_gated_delta_rule):
//   S <- S exp(g); kv = S^T k; delta = (v - kv) beta; S <- S + k delta^T;
//   o = S^T q, with k, q L2-normalized and q scaled by 1 / sqrt(D).
// The two states (D x D fp32 each) are worked on in shared memory: one bulk
// copy in, the update, one bulk store out.
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
__device__ __forceinline__ void
    gdn_step_sm100(TD const *task_desc, RC const &runtime_config) {
  using namespace q35;
  using namespace mxg;
  constexpr int NVH = 2 * NKH;
  constexpr int C = 4 * NKH * D; // conv channels: q, k, v
  constexpr int SBYTES = D * D * 4;
  static_assert(D == 128, "a value head's D columns: 4 warps, a lane a column");
  static_assert(PW >= C + NVH * D + 2 * NVH, "proj width");
  const unsigned short *proj =
      static_cast<const unsigned short *>(task_desc->input_ptrs[0]);
  const unsigned short *convw =
      static_cast<const unsigned short *>(task_desc->input_ptrs[1]);
  unsigned short *cstate = static_cast<unsigned short *>(task_desc->input_ptrs[2]);
  float *sglob = static_cast<float *>(task_desc->input_ptrs[3]);
  const float *par = static_cast<const float *>(task_desc->input_ptrs[4]);
  unsigned short *out = static_cast<unsigned short *>(task_desc->output_ptrs[0]);
  const int task_i = static_cast<int>(task_desc->task_metadata.ext_bid_x);
  const float eps = __int_as_float(EPS_BITS);

  extern __shared__ char shared_memory[];
  char *base = reinterpret_cast<char *>(
      (reinterpret_cast<uintptr_t>(shared_memory) + 127) / 128 * 128);
#ifdef Q35_GDN_BAR_FIRST
  u64 *bar = reinterpret_cast<u64 *>(base);        // the states' arrival
  float *S = reinterpret_cast<float *>(base + 128); // [2][D][D]
  float *kn = S + 2 * D * D;                       // [D] normalized k
  float *qn = kn + D;                              // [D] normalized, scaled q
  float *red = qn + D;                             // [16] reductions
#else
  float *S = reinterpret_cast<float *>(base);      // [2][D][D]
  float *kn = S + 2 * D * D;                       // [D] normalized k
  float *qn = kn + D;                              // [D] normalized, scaled q
  float *red = qn + D;                             // [16] reductions
  u64 *bar = reinterpret_cast<u64 *>(red + 16);    // the states' arrival
#endif

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  // thread = (value head vh of the pair, column vd)
  const int vh = tid >> 7, vd = tid & 127;
#ifdef Q35_GDN_BULK_LOAD
  // (Not the default: the barrier sits among this task's data, where the SM
  // may later write back the state it holds of it -- see moe_gemm_sm100's
  // barrier window -- and its first initialisation can lose to the states
  // the SM still holds from the worker's last task, hence twice.)
  if (tid == 0) {
    mbar_init(bar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_init(bar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
#endif
  int loads = 0; // state copies so far: the barrier's phase
#ifdef Q35_GDN_DEBUG
  // tuning aid: input [5] is an int32 [workers][4] progress record
  int *dbg = static_cast<int *>(task_desc->input_ptrs[5]) + 4 * blockIdx.x;
#define Q35_DBG(stage)                                                         \
  do {                                                                         \
    if (tid == 0) {                                                            \
      dbg[0] = task_i;                                                         \
      dbg[1] = u;                                                              \
      dbg[2] = (stage);                                                        \
      dbg[3] = loads;                                                          \
    }                                                                          \
  } while (0)
#else
#define Q35_DBG(stage)
#endif
#pragma unroll 1
  for (int u = task_i; u < BATCH * NKH; u += G) {
    const int t = u / NKH, hk = u % NKH;
    int rid, pos = 0;
    if (XMAP) {
      const int *xm = static_cast<const int *>(task_desc->input_ptrs[5]);
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
    const bool fresh = pos == 0;
    const unsigned short *pr = proj + static_cast<size_t>(t) * PW;
    unsigned short *cs = cstate + static_cast<size_t>(rid) * 3 * C;
    float *sg = sglob + (static_cast<size_t>(rid) * NVH + 2 * hk) * D * D;

    // ---- the states on their way in
#ifdef Q35_GDN_DEBUG
    if (tid == 0 && !fresh && mxg::mbar_test(bar, loads & 1)) {
      dbg[0] = task_i;
      dbg[1] = u;
      dbg[2] = 9; // the barrier's phase was over before this unit's copy
      dbg[3] = static_cast<int>(*bar >> 32);
      while (true) {
      }
    }
#endif
#ifdef Q35_GDN_BULK_LOAD
    if (tid == 0 && !fresh) {
      mbar_expect(bar, 2 * SBYTES);
      bulk_copy(S, sg, 2 * SBYTES, bar);
    }
#else
    // (16-byte copies of every thread, waited for by the thread itself)
    if (!fresh) {
#pragma unroll 1
      for (int i = tid; i < 2 * SBYTES / 16; i += blockDim.x) {
        cp_async16(reinterpret_cast<char *>(S) + 16 * i,
                   reinterpret_cast<const char *>(sg) + 16 * i);
      }
      asm volatile("cp.async.commit_group;" ::: "memory");
    }
#endif

    // ---- causal conv + SiLU of this thread's channels: q or k (vh 0 / 1),
    // then its value column
    float qk, vv;
#pragma unroll
    for (int which = 0; which < 2; ++which) {
      // channel: which 0 -> q (vh 0) or k (vh 1) of the key head, column vd;
      // which 1 -> the value head 2 hk + vh, column vd
      const int c = which == 0 ? (vh * NKH + hk) * D + vd
                               : 2 * NKH * D + (2 * hk + vh) * D + vd;
      const unsigned short xin = pr[c];
      const unsigned wv[2] = {
          reinterpret_cast<const unsigned *>(convw)[2 * c],
          reinterpret_cast<const unsigned *>(convw)[2 * c + 1]};
      // taps: w0 x(t-3), w1 x(t-2), w2 x(t-1), w3 x(t); x(t-3) is in the slot
      // input t is about to take
      float acc = bf2f(wv[1] >> 16) * bf2f(xin);
      if (!fresh) {
        const int s0 = pos % 3, s1 = (pos + 1) % 3, s2 = (pos + 2) % 3;
        acc += bf2f(wv[0] & 0xffff) * bf2f(cs[s0 * C + c]) +
               bf2f(wv[0] >> 16) * bf2f(cs[s1 * C + c]) +
               bf2f(wv[1] & 0xffff) * bf2f(cs[s2 * C + c]);
        cs[s0 * C + c] = xin;
      } else {
        cs[0 * C + c] = xin;
        cs[1 * C + c] = 0;
        cs[2 * C + c] = 0;
      }
      const float y = bfround(silu(bfround(acc)));
      if (which == 0) {
        qk = y;
      } else {
        vv = y;
      }
    }
    // ---- L2 norms of q (warps 0 .. 3) and k (warps 4 .. 7)
    float ss = warp_sum(qk * qk);
    if (lane == 0) {
      red[warp] = ss;
    }
    __syncthreads();
    {
      const float tot = red[4 * vh] + red[4 * vh + 1] + red[4 * vh + 2] +
                        red[4 * vh + 3];
      const float rs = rsqrtf(tot + 1e-6f);
      if (vh == 0) {
        qn[vd] = (qk * rs) * 0.08838834764831845f; // 1 / sqrt(128)
      } else {
        kn[vd] = qk * rs;
      }
    }
    // gates of this thread's value head
    const float bg = bf2f(pr[C + NVH * D + 2 * hk + vh]);
    const float ag = bf2f(pr[C + NVH * D + NVH + 2 * hk + vh]);
    const float beta = bfround(sigmoid(bg));
    const float sp_in = ag + par[NVH + 2 * hk + vh];
    const float sp = sp_in > 20.f ? sp_in : log1pf(__expf(sp_in));
    const float decay = __expf(par[2 * hk + vh] * sp);
    const float zg = bf2f(pr[C + (2 * hk + vh) * D + vd]);
    const float wnorm = par[2 * NVH + vd];
    Q35_DBG(1);
    if (!fresh) {
#ifdef Q35_GDN_BULK_LOAD
      mbar_wait(bar, loads & 1);
#else
      asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
      ++loads;
    }
    Q35_DBG(2);
    __syncthreads();
    // ---- kq = k . q, and this column's S^T k, S^T q
    float kq = warp_sum(vh == 0 ? qn[vd] * kn[vd] : 0.f);
    if (lane == 0) {
      red[8 + warp] = kq;
    }
    float *col = S + vh * D * D + vd; // S[kd][vd] at col[kd * D]
    float m1 = 0.f, m2 = 0.f;
    if (!fresh) {
#pragma unroll 4
      for (int kd = 0; kd < D; ++kd) {
        const float s = col[kd * D];
        m1 += s * kn[kd];
        m2 += s * qn[kd];
      }
    }
    __syncthreads();
    kq = red[8] + red[9] + red[10] + red[11];
    const float delta = (vv - decay * m1) * beta;
    const float o = bfround(decay * m2 + delta * kq);
#pragma unroll 4
    for (int kd = 0; kd < D; ++kd) {
      const float s = fresh ? 0.f : col[kd * D];
      col[kd * D] = decay * s + kn[kd] * delta;
    }
    // ---- gated norm of the head's output (mean over its D columns)
    ss = warp_sum(o * o);
    if (lane == 0) {
      red[warp] = ss;
    }
    fence_async_shared();
    __syncthreads();
    {
      const float tot = red[4 * vh] + red[4 * vh + 1] + red[4 * vh + 2] +
                        red[4 * vh + 3];
      const float n = bfround(o * rsqrtf(tot / D + eps));
      const float y = bfround(wnorm * n) * silu(zg);
      out[static_cast<size_t>(t) * NVH * D + (2 * hk + vh) * D + vd] = f2bf(y);
    }
    Q35_DBG(3);
    if (tid == 0) {
      bulk_store(sg, S, 2 * SBYTES);
      bulk_store_wait();
    }
    Q35_DBG(4);
    __syncthreads();
  }
}

} // namespace kernel
