// moe_gemm_sm100.cuh -- routed-expert GEMMs for SM100 as extension tasks.
//
//   moe_route_sm100     top-k routing of one layer: the softmax weights and
//                       a route plan (per expert the tokens that reach it).
//   moe_gemm_sm100      one worker's share of an expert GEMM wave (W13 with
//                       the gated activation fused into the epilogue, or W2).
//                       DENSE: an ordinary linear layer on the same machinery
//                       (the builders' dense="gemm"; MPK's linear tasks are
//                       the default for dense layers).
//   moe_combine_sm100   the experts' outputs back into the residual stream.
//   small_linear_sm100  a small linear layer without the tensor core and
//   tmem_release_sm100  the end of an iteration's held TMEM (dense="gemm").
//
// The expert weights are stored in a layout made for streaming
// (python/mirage/mpk/moe_pack.py):
//   MX = 1  MXFP4 as released (e2m1 values, one ue8m0 scale per 32): blocks of
//           128 rows x KB columns, [KB/32 chunks][128 rows][16 B] followed by
//           [128 rows][KB/32] scale bytes. A block arrives with one bulk copy;
//           four warps convert it to bf16 (the value times its block scale,
//           exactly) and write it to TMEM, the MMA's A operand.
//   MX = 0  bf16: tiles of 128 rows x 64 columns already in the MMA's
//           K-major SWIZZLE_128B order; a bulk copy lands a tile in place.
// The activations are bf16 rows gathered by the route plan (cp.async, 16 B a
// thread). Nothing is quantized on the way: the products are the released
// weights times the bf16 activations, accumulated in fp32.
//
// Work split: a wave has G tasks. The route plan lists the layer's
// (expert, pass) pairs -- a pass is at most NT tokens of an expert -- and
// unit u = pair (u / RB), row block (u % RB) of 128 output rows. Task i runs
// the units i, i + G, i + 2 G, ...: the active experts' work is spread over
// all workers whatever the routing is.
//
// A unit runs in K steps of KB columns: one weight copy, one A stage, one B
// stage, KB / 16 MMAs and their two commits. The waits of a step are the
// loop's fixed cost, so the steps are few and the rings as deep as shared
// memory and TMEM allow (at small batch the whole unit's activations are in
// flight at once).
#pragma once
#include <cstdint>

namespace kernel {
namespace mxg {

typedef unsigned long long u64;

__device__ __forceinline__ unsigned sm_u32(void const *p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ float bf2f(unsigned short h) {
  return __uint_as_float(static_cast<unsigned>(h) << 16);
}
// round to nearest even
__device__ __forceinline__ unsigned short f2bf(float f) {
  unsigned u = __float_as_uint(f);
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<unsigned short>(u >> 16);
}
__device__ __forceinline__ float bfround(float f) {
  return bf2f(f2bf(f));
}

__device__ __forceinline__ void mbar_init(u64 *bar, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(sm_u32(bar)),
               "r"(count)
               : "memory");
}
__device__ __forceinline__ void mbar_arrive(u64 *bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(sm_u32(bar))
               : "memory");
}
__device__ __forceinline__ void mbar_expect(u64 *bar, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                   sm_u32(bar)),
               "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void mbar_wait(u64 *bar, unsigned phase) {
  asm volatile("{\n\t"
               ".reg .pred p;\n\t"
               "MXG_WAIT:\n\t"
               "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
               "@p bra MXG_DONE;\n\t"
               "bra MXG_WAIT;\n\t"
               "MXG_DONE:\n\t"
               "}" ::"r"(sm_u32(bar)),
               "r"(phase)
               : "memory");
}
// contiguous bytes global -> shared, completion on `bar`
__device__ __forceinline__ void
    bulk_copy(void *dst, void const *src, unsigned bytes, u64 *bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::"
               "bytes [%0], [%1], %2, [%3];" ::"r"(sm_u32(dst)),
               "l"(reinterpret_cast<u64>(src)),
               "r"(bytes),
               "r"(sm_u32(bar))
               : "memory");
}
__device__ __forceinline__ void cp_async16(void *dst, void const *src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sm_u32(dst)),
               "l"(reinterpret_cast<u64>(src))
               : "memory");
}
// arrives when this thread's earlier cp.async copies have landed
__device__ __forceinline__ void cp_async_arrive(u64 *bar) {
  asm volatile(
      "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(sm_u32(bar))
      : "memory");
}

// kind::f16 instruction descriptor: D fp32, A and B bf16 (the instruction
// takes one format for both), both K-major
__device__ __forceinline__ uint32_t idesc(int m, int n) {
  return (1u << 4) | (1u << 7) | (1u << 10) |
         (static_cast<unsigned>(n >> 3) << 17) |
         (static_cast<unsigned>(m >> 4) << 24);
}
// K-major SWIZZLE_128B descriptor of a [rows][64 x 16 bit] box
__device__ __forceinline__ u64 smem_desc(unsigned addr) {
  return ((addr & 0x3FFFFu) >> 4) | (static_cast<u64>(1024 >> 4) << 32) |
         (1ull << 46) | (2ull << 61);
}
// one 128 x n x 16 MMA; A in shared memory
__device__ __forceinline__ void
    umma_ss(unsigned td, u64 ad, u64 bd, uint32_t ides, int acc) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}" ::"r"(td),
      "l"(ad),
      "l"(bd),
      "r"(ides),
      "r"(acc)
      : "memory");
}
// A in TMEM (8 columns hold the 16 values of a row)
__device__ __forceinline__ void
    umma_ts(unsigned td, unsigned ta, u64 bd, uint32_t ides, int acc) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n\t}" ::"r"(
          td),
      "r"(ta),
      "l"(bd),
      "r"(ides),
      "r"(acc)
      : "memory");
}
// the MMAs issued so far arrive on `bar` once they have run
__device__ __forceinline__ void umma_commit(u64 *bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::"
               "cluster.b64 [%0];" ::"r"(sm_u32(bar))
               : "memory");
}
__device__ __forceinline__ void tmem_fence_before() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tmem_fence_after() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tmem_alloc512(unsigned *slot) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 "
               "[%0], 512;" ::"r"(sm_u32(slot))
               : "memory");
}
__device__ __forceinline__ void tmem_free512(unsigned base) {
  asm volatile(
      "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(base)
      : "memory");
}
__device__ __forceinline__ void tmem_ld16(unsigned addr, unsigned (&v)[16]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(v[0]),
                 "=r"(v[1]),
                 "=r"(v[2]),
                 "=r"(v[3]),
                 "=r"(v[4]),
                 "=r"(v[5]),
                 "=r"(v[6]),
                 "=r"(v[7]),
                 "=r"(v[8]),
                 "=r"(v[9]),
                 "=r"(v[10]),
                 "=r"(v[11]),
                 "=r"(v[12]),
                 "=r"(v[13]),
                 "=r"(v[14]),
                 "=r"(v[15])
               : "r"(addr)
               : "memory");
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tmem_st32(unsigned addr,
                                          unsigned const (&v)[32]) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], "
      "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,"
      "%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};" ::"r"(
          addr),
      "r"(v[0]),
      "r"(v[1]),
      "r"(v[2]),
      "r"(v[3]),
      "r"(v[4]),
      "r"(v[5]),
      "r"(v[6]),
      "r"(v[7]),
      "r"(v[8]),
      "r"(v[9]),
      "r"(v[10]),
      "r"(v[11]),
      "r"(v[12]),
      "r"(v[13]),
      "r"(v[14]),
      "r"(v[15]),
      "r"(v[16]),
      "r"(v[17]),
      "r"(v[18]),
      "r"(v[19]),
      "r"(v[20]),
      "r"(v[21]),
      "r"(v[22]),
      "r"(v[23]),
      "r"(v[24]),
      "r"(v[25]),
      "r"(v[26]),
      "r"(v[27]),
      "r"(v[28]),
      "r"(v[29]),
      "r"(v[30]),
      "r"(v[31])
      : "memory");
}
__device__ __forceinline__ void tmem_wait_st() {
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}
// one word = 8 e2m1 values -> 4 bf16 pairs times the block scale 2^k, all
// exact: v + 8 is in [2, 14] (no zero, no sign), its fp16 bits shifted right
// by 3 are the bf16 number (v + 8) 2^-112, and
// (v + 8) 2^-112 * 2^(k + 112) - 8 2^k = v 2^k is one fused multiply-add
// (s2 = 2^(k + 112), c2 = -8 2^k, each in both halves)
__device__ __forceinline__ void
    cvt4_scale(unsigned w, unsigned s2, unsigned c2, unsigned *v) {
  asm("{\n\t.reg .b8 b0, b1, b2, b3;\n\t.reg .b32 t0, t1, t2, t3;\n\t"
      "mov.b32 {b0, b1, b2, b3}, %4;\n\t"
      "cvt.rn.f16x2.e2m1x2 t0, b0;\n\tcvt.rn.f16x2.e2m1x2 t1, b1;\n\t"
      "cvt.rn.f16x2.e2m1x2 t2, b2;\n\tcvt.rn.f16x2.e2m1x2 t3, b3;\n\t"
      "add.rn.f16x2 t0, t0, %7;\n\tadd.rn.f16x2 t1, t1, %7;\n\t"
      "add.rn.f16x2 t2, t2, %7;\n\tadd.rn.f16x2 t3, t3, %7;\n\t"
      "shr.u32 t0, t0, 3;\n\tshr.u32 t1, t1, 3;\n\t"
      "shr.u32 t2, t2, 3;\n\tshr.u32 t3, t3, 3;\n\t"
      "fma.rn.bf16x2 %0, t0, %5, %6;\n\tfma.rn.bf16x2 %1, t1, %5, %6;\n\t"
      "fma.rn.bf16x2 %2, t2, %5, %6;\n\tfma.rn.bf16x2 %3, t3, %5, %6;\n\t}"
      : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3])
      : "r"(w), "r"(s2), "r"(c2), "r"(0x48004800u));
}
__device__ __forceinline__ bool mbar_test(u64 *bar, unsigned phase) {
  unsigned ok;
  asm volatile("{\n\t.reg .pred p;\n\t"
               "mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
               "selp.u32 %0, 1, 0, p;\n\t}"
               : "=r"(ok)
               : "r"(sm_u32(bar)), "r"(phase)
               : "memory");
  return ok != 0;
}
// Tuning aid: with -DMXG_TRACE the task's extra last input is an int64
// [workers][MXG_TRACE_N] buffer of %globaltimer stamps (ns).
#ifdef MXG_TRACE
#define MXG_TRACE_N 256
// unit 0's steps: slot 64 + 8 * step + k
#define MXG_STEP(kb, k)                                                        \
  do {                                                                         \
    if (j == 0 && (kb) < 24) {                                                 \
      mxg_tr[64 + 8 * (kb) + (k)] = mxg_now();                                 \
    }                                                                          \
  } while (0)
__device__ __forceinline__ long long mxg_now() {
  long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
#define MXG_T(slot)                                                            \
  do {                                                                         \
    if ((slot) < MXG_TRACE_N) {                                                \
      mxg_tr[(slot)] = mxg_now();                                              \
    }                                                                          \
  } while (0)
#define MXG_TADD(slot, t0)                                                     \
  do {                                                                         \
    if ((slot) < MXG_TRACE_N) {                                                \
      mxg_tr[(slot)] += mxg_now() - (t0);                                      \
    }                                                                          \
  } while (0)
#define MXG_TRACE_INPUTS 1
#else
#define MXG_T(slot)
#define MXG_TADD(slot, t0)
#define MXG_TRACE_INPUTS 0
#endif

// the four warps of rows 0 .. 127 (the epilogue's)
__device__ __forceinline__ void rows_sync() {
  asm volatile("bar.sync 9, 128;" ::: "memory");
}

} // namespace mxg

// ---------------------------------------------------------------------------
// Routing. inputs: [0] logits bf16 [BATCH][LSTRIDE]: the E router logits and,
// with SHARED, the shared expert's gate logit in column E. outputs: [0] weights
// fp32 [BATCH][TOPK + SHARED] (softmax over the chosen logits; sigmoid for the
// shared expert; both rounded to bf16 as the reference models do);
// [1] plan list int32 [E + SHARED][BATCH], entries (token << 8) | slot;
// [2] plan meta int32 [1 + CAP + E + SHARED]: [0] the number of (expert, pass)
// pairs, [1 ..] the pairs (expert << 16) | pass in expert order,
// [1 + CAP + e] the token count of expert e.
// One warp per token: every lane holds E / 32 logits, each as one unsigned
// key (the value's order in the high half, 0xffff - expert in the low one), so
// that the largest key is the largest logit, the lower expert among equals.
// G > 1: the wave's G tasks share the tokens (one task took 5 us for every
// eight tokens: two thirds of a millisecond a layer at batch 1024). The
// experts' counters are then global -- input [1], int32 [E + SHARED + 1],
// zeros between waves; its last element counts the tasks that are through
// their tokens -- and task 0 waits for the others before it writes the plan.
// The plan: warp w has the experts 32 w .. 32 w + 31.
// ---------------------------------------------------------------------------
template <int BATCH,
          int E,
          int TOPK,
          int NT,
          int SHARED,
          int CAP,
          int LSTRIDE,
          int G,
          class TD,
          class RC>
__device__ __forceinline__ void moe_route_sm100(TD const *task_desc,
                                                RC const &runtime_config) {
  using namespace mxg;
  constexpr int E_TOT = E + SHARED, SLOTS = TOPK + SHARED, VPL = E / 32;
  static_assert(E % 32 == 0 && VPL >= 1 && VPL <= 8, "E / 32 logits a lane");
  static_assert(TOPK + SHARED <= 32, "a lane per slot");
  static_assert(BATCH < (1 << 23), "token index in 24 bits");
  unsigned short const *logits =
      static_cast<unsigned short const *>(task_desc->input_ptrs[0]);
  float *weight = static_cast<float *>(task_desc->output_ptrs[0]);
  int *list = static_cast<int *>(task_desc->output_ptrs[1]);
  int *meta = static_cast<int *>(task_desc->output_ptrs[2]);

  extern __shared__ char shared_memory[];
  int const tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  int const nwarps = blockDim.x >> 5;
  int const task_i = static_cast<int>(task_desc->task_metadata.ext_bid_x);
  int *sm = reinterpret_cast<int *>(
      (reinterpret_cast<uintptr_t>(shared_memory) + 15) / 16 * 16);
  int *tot = sm; // [8] plan entries of a warp's experts
  int *cnt = G > 1 ? static_cast<int *>(task_desc->input_ptrs[1]) : sm + 8;
  if (G == 1) {
    for (int i = tid; i < E_TOT; i += blockDim.x) {
      cnt[i] = 0;
    }
    __syncthreads();
  }

  for (int t = task_i * nwarps + warp; t < BATCH; t += G * nwarps) {
    unsigned short const *lr = logits + static_cast<size_t>(t) * LSTRIDE;
    unsigned raw[VPL];
    if (VPL == 8 && LSTRIDE % 8 == 0) {
      const uint4 v = reinterpret_cast<uint4 const *>(lr)[lane];
      unsigned const w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int j = 0; j < VPL; ++j) {
        raw[j] = (w[(j / 2) & 3] >> (16 * (j & 1))) & 0xffff;
      }
    } else {
#pragma unroll
      for (int j = 0; j < VPL; ++j) {
        raw[j] = lr[lane * VPL + j];
      }
    }
    unsigned key[VPL];
#pragma unroll
    for (int j = 0; j < VPL; ++j) {
      unsigned b = raw[j] == 0x8000u ? 0u : raw[j]; // -0 is 0
      b = (b & 0x8000u) ? (~b & 0xffffu) : (b | 0x8000u);
      key[j] = (b << 16) | (0xffffu - static_cast<unsigned>(lane * VPL + j));
    }
    unsigned sel = 0;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      unsigned best = key[0];
#pragma unroll
      for (int j = 1; j < VPL; ++j) {
        best = max(best, key[j]);
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        best = max(best, __shfl_xor_sync(0xffffffffu, best, off));
      }
      if (lane == k) {
        sel = best;
      }
#pragma unroll
      for (int j = 0; j < VPL; ++j) {
        if (key[j] == best) {
          key[j] = 0u; // (below every key of a number)
        }
      }
    }
    int const sel_idx = static_cast<int>(0xffffu - (sel & 0xffffu));
    unsigned const so = sel >> 16;
    float const sel_val = bf2f(static_cast<unsigned short>(
        (so & 0x8000u) ? (so ^ 0x8000u) : (~so & 0xffffu)));
    float const top = __shfl_sync(0xffffffffu, sel_val, 0);
    float const ex = lane < TOPK ? __expf(sel_val - top) : 0.f;
    float sum = ex;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }
    if (lane < TOPK) {
      weight[static_cast<size_t>(t) * SLOTS + lane] = bfround(ex / sum);
      int const pos = atomicAdd(&cnt[sel_idx], 1);
      list[static_cast<size_t>(sel_idx) * BATCH + pos] = (t << 8) | lane;
    } else if (SHARED && lane == TOPK) {
      float const g = bf2f(lr[E]);
      weight[static_cast<size_t>(t) * SLOTS + TOPK] =
          bfround(1.f / (1.f + __expf(fminf(-g, 40.f))));
      list[static_cast<size_t>(E) * BATCH + t] = (t << 8) | TOPK;
    }
  }
  __syncthreads();
  if (G > 1) {
    if (tid == 0) {
      __threadfence();
      atomicAdd(&cnt[E_TOT], 1);
    }
    if (task_i != 0) {
      return;
    }
    if (tid == 0) {
      while (*reinterpret_cast<int volatile *>(&cnt[E_TOT]) != G) {
      }
    }
    __syncthreads();
  }

  // ---- the plan: the (expert, pass) pairs in expert order, the counts
  int c = 0, np = 0, pre = 0;
  if (warp < VPL) {
    int const e = 32 * warp + lane;
    c = *reinterpret_cast<int volatile *>(&cnt[e]);
    if (G > 1) {
      cnt[e] = 0; // (for the next wave)
    }
    np = (c + NT - 1) / NT;
    pre = np;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      int const o = __shfl_up_sync(0xffffffffu, pre, off);
      if (lane >= off) {
        pre += o;
      }
    }
    if (lane == 31) {
      tot[warp] = pre;
    }
  }
  __syncthreads();
  int base = 0, all = 0;
#pragma unroll
  for (int w = 0; w < VPL; ++w) {
    int const v = tot[w];
    all += v;
    if (w < warp) {
      base += v;
    }
  }
  if (warp < VPL) {
    int const e = 32 * warp + lane;
    int const first = base + pre - np;
    for (int p = 0; p < np; ++p) {
      meta[1 + first + p] = (e << 16) | p;
    }
    meta[1 + CAP + e] = c;
  }
  if (tid == 0) {
    int n = all;
    if (SHARED) { // the shared expert: every token, after the routed ones
      int const nps = (BATCH + NT - 1) / NT;
      for (int p = 0; p < nps; ++p) {
        meta[1 + all + p] = (E << 16) | p;
      }
      meta[1 + CAP + E] = BATCH;
      n += nps;
    }
    meta[0] = n;
    if (G > 1) {
      cnt[E_TOT] = 0;
    }
  }
}

// ---------------------------------------------------------------------------
// One task of an expert GEMM wave.
// inputs: [0] activations bf16, [BATCH][K] or (IN_SLOTTED) [BATCH][SLOTS][K];
// [1] the packed weights of [E_TOT][RB] row blocks; [2] plan list; [3] plan
// meta; [4] (HAS_BIAS) bias bf16 [E_TOT][ROWS].
// outputs: [0] bf16 [BATCH][SLOTS][ROWS], or with ACT [BATCH][SLOTS][ROWS / 2]:
// the rows are (gate, up) pairs and the output is
//   (clamp(up, -LIMIT, LIMIT) + OFF) * g * sigmoid(ALPHA * g),
//   g = min(gate, LIMIT)
// (LIMIT = inf, OFF = 0, ALPHA = 1: SiLU(gate) * up).
// DENSE: an ordinary linear layer on the same machinery -- one "expert", every
// token, in passes of NT; the inputs are [0] activations [BATCH][K], [1] the
// packed weight, then (HAS_BIAS) bias bf16 [ROWS] and (RES) a residual bf16
// [BATCH][OSTRIDE]; the output is bf16 [BATCH][OSTRIDE]. FUSE rounds
// product + bias once (a linear layer) instead of the product first (the
// experts' bmm, then the bias); the residual is added to the rounded value.
// HELD: the worker keeps its TMEM from task to task (tmem_release_sm100 at
// the end of the iteration) instead of this task allocating and freeing it;
// its last input but two is the token tensor.
// KS > 1 splits every unit's K steps into KS parts that are units of their
// own, so a wave with fewer units than workers runs chains of K / (16 KS)
// MMAs side by side instead of K / 16 on some workers and nothing on the
// rest. A part's accumulator goes to a scratch tensor (the last input but
// one, fp32 [units][KS][128][NT]); whichever part of a unit finishes last
// (a counter per unit and row quadrant, the last input, int32 [units][4])
// adds the parts in order and runs the epilogue.
// Warps: 0 .. 3 the 128 rows (dequantization, epilogue), 4 the weight copies,
// 5 the MMAs, 6 .. 7 the activation gather.
// ---------------------------------------------------------------------------
template <int MX,
          int ACT,
          int HAS_BIAS,
          int IN_SLOTTED,
          int BATCH,
          int E_TOT,
          int SLOTS,
          int NT,
          int K,
          int ROWS,
          int KB,
          int G,
          int CAP,
          int LIMIT_BITS,
          int ALPHA_BITS,
          int OFF_BITS,
          int DENSE,
          int RES,
          int FUSE,
          int HELD,
          int OSTRIDE,
          int KS,
          class TD,
          class RC>
__device__ __forceinline__ void moe_gemm_sm100(TD const *task_desc,
                                               RC const &runtime_config) {
  using namespace mxg;
  constexpr int RB = (ROWS + 127) / 128; // row blocks of an expert
  constexpr int NKB = K / KB;            // steps of a unit
  constexpr int TPB = KB / 64;           // 64-column boxes of a step
  constexpr int NCH = KB / 32;           // MX: scale groups of a row in a block
  // bytes of a weight copy
  constexpr int ABLK = MX ? NCH * 128 * 16 + 128 * NCH : TPB * 128 * 128;
  constexpr int ACOLS = KB / 2; // MX: TMEM columns of an A stage
  constexpr int AS = 2;         // accumulator stages
  // (An MMA costs the issuing thread 35 .. 45 ns whatever its N. Spreading
  // a unit's MMAs over NACC independent accumulators that the epilogue adds
  // up was tried, 4 at NT 16: no gain, so it is not a dependency chain.)
  // Two threads issuing at once do get more out of the SM while N is small
  // (measured on a B200, M = 128: one thread 32 ns an MMA at N = 16 .. 64,
  // two threads 21 ns of the SM at N = 16 and 26 at N = 64, four no better;
  // at N = 128 the MMA itself takes 35 ns and at N = 256 twice that, so
  // neither more threads nor weight rows on the N side help there). MMA2: a
  // second thread (warp 7) issues every other MMA, into an accumulator of
  // its own, and the gather makes do with warp 6. Not at NT 64 with MXFP4,
  // where the second accumulators would cost two of the four A stages.
  constexpr bool MMA2 = NT <= 32 || (!MX && NT <= 64);
  constexpr int NMT = MMA2 ? 2 : 1;   // threads that issue MMAs
  constexpr int GL = MMA2 ? 32 : 64;  // lanes of the gather
  constexpr int NACC = NMT;           // accumulators of a unit
  constexpr int ACC_COLS = NACC * NT; // TMEM columns of an accumulator stage
  // rows of a B box: the most tokens a pass can have
  constexpr int BROWS = ((NT < BATCH ? NT : BATCH) + 7) / 8 * 8;
  constexpr int BBOX = BROWS * 128; // bytes of a B box
  constexpr int BSTG = TPB * BBOX;  // bytes of a B stage
  constexpr int NTL = BROWS / 8;    // gather: tokens of a lane
  constexpr int MAXU = 64;          // units of a task
  constexpr int SMEM_TOTAL = 196 * 1024;
  // MX: up to 100 KB of B stages, the rest for weight copies (13 KB each).
  // bf16: two B stages reserved, weight copies (16 KB a 64-column tile) in
  // the rest, then B stages in what they leave.
  constexpr int STG_MAX = MX ? 8 : 6;
  constexpr int STGB_MX =
      (100 * 1024) / BSTG >= NKB
          ? NKB
          : ((100 * 1024) / BSTG < 2 ? 2 : (100 * 1024) / BSTG);
  constexpr int STG_BF = (SMEM_TOTAL - 8192 - 2 * BSTG) / ABLK;
  constexpr int STG_FIT = MX ? (SMEM_TOTAL - 8192 - STGB_MX * BSTG) / ABLK
                             : (STG_BF < 1 ? 1 : STG_BF);
  constexpr int STG_0 = STG_FIT > STG_MAX ? STG_MAX : STG_FIT; // weight copies
  constexpr int STGB_BF = (SMEM_TOTAL - 8192 - STG_0 * ABLK) / BSTG;
  constexpr int STGB_1 = MX ? STGB_MX : (STGB_BF >= NKB ? NKB : STGB_BF);
  // B2X: both of two consecutive units' activations fit (small batches). A
  // unit then has its own NKB stages by parity, the gather of unit j waits
  // only for unit j - 2's last MMA (acc_full), and the MMAs commit no B
  // barrier at all: 15 commits a unit less, and the next unit's activations
  // are in before its first weights are.
  constexpr int A_MIN = MX ? 4 * ABLK : 2 * ABLK;
  // (With KS parts a unit, a task's units must all be the same part -- G a
  // multiple of KS -- for a stage to be used once every other unit, which
  // the parity of its barrier counts on.)
  constexpr bool B2X = 2 * NKB * BSTG + A_MIN + 8192 <= SMEM_TOTAL &&
                       STGB_1 == NKB && (KS == 1 || G % KS == 0);
  constexpr int STGB = B2X ? 2 * NKB : STGB_1;
  constexpr int STG_B2X = (SMEM_TOTAL - 8192 - 2 * NKB * BSTG) / ABLK;
  constexpr int STG = B2X ? (STG_B2X > STG_MAX ? STG_MAX : STG_B2X) : STG_0;
  constexpr int SA_FIT = (512 - AS * ACC_COLS) / ACOLS;
  constexpr int SA = MX ? (SA_FIT > 4 ? 4 : SA_FIT) : STG; // A stages
  constexpr int OUT = ACT ? ROWS / 2 : ROWS;
  constexpr int AREG = (STG * ABLK + 1023) / 1024 * 1024;
  constexpr int BREG = STGB * BSTG;
  constexpr int NBAR = 2 * STG + 2 * SA + 2 * STGB + 2 * AS;
  constexpr int SMEM = AREG + BREG + 16 * MAXU + 8 * NT + 64;
  constexpr int BARWIN = 1024; // bytes of the barriers' window
  static_assert(8 * (NBAR + 8) + 8 <= BARWIN, "barriers");
  static_assert(SMEM_TOTAL + 1024 + BARWIN <=
                    mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE,
                "the stages stay clear of the barriers' window");
  static_assert(K % KB == 0 && KB % 64 == 0, "K steps");
  static_assert(NT % 16 == 0 && NT >= 16 && NT <= 128, "tokens of a pass");
  static_assert(STG >= 1 && SA >= 1 && STGB >= 1, "ring depths");
  static_assert(AS == 2, "B2X pairs units with accumulator stages");
  static_assert(AS * ACC_COLS <= 512, "TMEM: the accumulators");
  static_assert(!ACT || ROWS % 2 == 0, "gate / up row pairs");
  static_assert(SMEM + 2048 <= SMEM_TOTAL, "shared memory");
  static_assert(E_TOT < 32768 && SLOTS <= 256, "plan entry fields");

  char const *x = static_cast<char const *>(task_desc->input_ptrs[0]);
  char const *wpack = static_cast<char const *>(task_desc->input_ptrs[1]);
  constexpr int IDX_BIAS = DENSE ? 2 : 4;
  constexpr int IDX_RES = IDX_BIAS + HAS_BIAS;
  constexpr int OS = OSTRIDE ? OSTRIDE : OUT; // the output's row stride
  constexpr int NPASS = (BATCH + NT - 1) / NT;
  static_assert(!DENSE || (E_TOT == 1 && SLOTS == 1 && !IN_SLOTTED), "dense");
  static_assert(!RES || !ACT, "a residual goes with a plain output");
  static_assert(KS >= 1 && KS <= NKB, "parts of a unit's K steps");
  constexpr int IDX_TOK = IDX_RES + RES + MXG_TRACE_INPUTS;
  constexpr int IDX_PART = IDX_TOK + HELD;
  float *kpart =
      KS > 1 ? static_cast<float *>(task_desc->input_ptrs[IDX_PART]) : nullptr;
  int *kcnt = KS > 1 ? static_cast<int *>(task_desc->input_ptrs[IDX_PART + 1])
                     : nullptr;
  int const *rlist =
      DENSE ? nullptr : static_cast<int const *>(task_desc->input_ptrs[2]);
  int const *meta =
      DENSE ? nullptr : static_cast<int const *>(task_desc->input_ptrs[3]);
  unsigned short const *bias =
      HAS_BIAS
          ? static_cast<unsigned short const *>(task_desc->input_ptrs[IDX_BIAS])
          : nullptr;
  unsigned short const *resid =
      RES ? static_cast<unsigned short const *>(task_desc->input_ptrs[IDX_RES])
          : nullptr;
  unsigned short *out =
      static_cast<unsigned short *>(task_desc->output_ptrs[0]);
  int const task_i = static_cast<int>(task_desc->task_metadata.ext_bid_x);
  int const tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
#ifdef MXG_TRACE
  long long *mxg_tr =
      static_cast<long long *>(task_desc->input_ptrs[IDX_RES + RES]) +
      static_cast<size_t>(blockIdx.x) * MXG_TRACE_N;
  if (tid == 0) {
    for (int i = 0; i < MXG_TRACE_N; ++i) {
      mxg_tr[i] = 0;
    }
    MXG_T(0);
  }
#endif

  extern __shared__ char shared_memory[];
  char *areg = reinterpret_cast<char *>(
      (reinterpret_cast<uintptr_t>(shared_memory) + 1023) / 1024 * 1024);
  char *breg = areg + AREG;
  // (the MMA reads 16 rows of a B box at least: past an 8-row box it reads
  // the next box, or what follows the ring -- columns the epilogue ignores)
  // The barriers: the last BARWIN bytes of the worker's dynamic shared
  // memory, in every instance of this task, and nothing else is ever put
  // there. The SM keeps barrier states of its own and writes them back to
  // shared memory later -- when a barrier is next initialised, when the
  // context is switched out for another process -- at the addresses they
  // had, whatever the current task keeps there: with the barriers after
  // each instance's stages, one instance's old barriers landed in the
  // activations of another (a token's row wrong for one unit, on every
  // worker at once, whenever a second process shared the GPU).
  u64 *bar_flush = reinterpret_cast<u64 *>(
      (reinterpret_cast<uintptr_t>(shared_memory) +
       mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE - BARWIN + 7) /
      8 * 8);                   // [8]: a warp's own (see below)
  u64 *x_full = bar_flush + 8;  // a weight copy landed
  u64 *x_empty = x_full + STG;  // ... and was consumed
  u64 *a_ready = x_empty + STG; // MX: an A stage is in TMEM
  u64 *a_empty = a_ready + SA;  // MX: ... and the MMAs have read it
  u64 *b_full = a_empty + SA;
  u64 *b_empty = b_full + STGB;
  u64 *acc_full = b_empty + STGB;
  u64 *acc_empty = acc_full + AS;
  int *ut = reinterpret_cast<int *>(breg + BREG); // [MAXU][4]
  int *elist = ut + 4 * MAXU;                     // [2][NT]
  int *misc = elist + 2 * NT;                     // n_ep, tmem base

  int n_ep = NPASS;
  if (!DENSE) {
    if (tid == 0) {
      misc[0] = meta[0];
    }
    __syncthreads();
    n_ep = misc[0];
  }
  int const total = n_ep * RB * KS;
  int const n_my = total > task_i ? (total - task_i - 1) / G + 1 : 0;
  if (!DENSE && n_my > MAXU) {
    __trap();
  }
  if (n_my == 0) {
    return;
  }
#ifdef MXG_TRACE
  if (tid == 0) {
    MXG_T(1);
    mxg_tr[5] = n_my;
    mxg_tr[6] = task_i;
  }
#endif
  // HELD: the worker's TMEM stays allocated from its first GEMM task of the
  // iteration (which records the address, top bit set, in the token tensor)
  // to the release wave.
  int *held_tok =
      HELD ? static_cast<int *>(task_desc->input_ptrs[IDX_TOK]) + blockIdx.x
           : nullptr;
  // The window holds the last GEMM task's barriers, and the SM its states of
  // them: the first mbarrier.init of a task (a store, then an invalidation
  // of the SM's barrier states) lost to those, so that the barriers
  // initialised first kept the old task's phase -- an odd one let the weight
  // copies run a stage ahead of the MMAs (wrong weights, or an arrive on a
  // copy still in flight: an illegal instruction). So every warp first
  // initialises a barrier of its own that nothing uses, which takes that;
  // then the barriers proper, a thread each (one thread doing all of them
  // took 2 us of every task: up to 88 stores, half as many invalidations).
  // Nothing of this task is in shared memory before that.
  if (lane == 0) {
    mbar_init(&bar_flush[warp], 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  // (Only now this task's own data: that first initialisation is also when
  // the SM writes back what it holds of the last task's barriers, wherever
  // that task had them -- MPK's linear layers have theirs where the unit
  // table below is.)
  if (!DENSE && tid < n_my) {
    int const u = (task_i + tid * G) / KS;
    int const ent = meta[1 + u / RB];
    int const e = ent >> 16, p = ent & 0xffff;
    int const left = meta[1 + CAP + e] - p * NT;
    ut[tid * 4 + 0] = e;
    ut[tid * 4 + 1] = p;
    ut[tid * 4 + 2] = u % RB;
    ut[tid * 4 + 3] = left < NT ? left : NT;
  }
  static_assert(NBAR <= 224, "a barrier a thread of warps 0 .. 6");
  if (tid < NBAR) {
    int const i = tid;
    int const count = i < STG                                ? 1
                      : i < 2 * STG                          ? (MX ? 4 : NMT)
                      : i < 2 * STG + SA                     ? 4
                      : i < 2 * STG + 2 * SA                 ? NMT
                      : i < 2 * STG + 2 * SA + STGB          ? GL
                      : i < 2 * STG + 2 * SA + 2 * STGB      ? NMT
                      : i < 2 * STG + 2 * SA + 2 * STGB + AS ? NMT
                                                             : 4;
    mbar_init(&x_full[i], count);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  int const held = HELD ? *held_tok : 0; // (every thread reads it itself)
  bool const have_tmem = held < 0;
  if (!have_tmem && warp == 0) {
    tmem_alloc512(reinterpret_cast<unsigned *>(&misc[1]));
    tmem_fence_before();
  }
  __syncthreads();
  tmem_fence_after();
  if (!have_tmem && HELD && tid == 0) {
    *held_tok = misc[1] | static_cast<int>(0x80000000u);
  }
#ifdef MXG_TRACE
  if (tid == 0) {
    MXG_T(2);
  }
#endif
  // the accumulators
  unsigned const tb =
      static_cast<unsigned>(have_tmem ? (held & 0x7fffffff) : misc[1]);
  // unit j of this task: expert, pass, row block, tokens
  auto unit_at = [&](int j, int &e, int &p, int &rb, int &n) {
    if (DENSE) {
      const int u = (task_i + j * G) / KS;
      e = 0;
      p = u / RB;
      rb = u % RB;
      const int left = BATCH - p * NT;
      n = left < NT ? left : NT;
    } else {
      e = ut[j * 4 + 0];
      p = ut[j * 4 + 1];
      rb = ut[j * 4 + 2];
      n = ut[j * 4 + 3];
    }
  };
  // unit j is part q of unit uid: the K steps [kb0, kb1)
  auto part_at = [&](int j, int &q, int &uid, int &kb0, int &kb1) {
    const int su = task_i + j * G;
    q = su % KS;
    uid = su / KS;
    kb0 = q * NKB / KS;
    kb1 = (q + 1) * NKB / KS;
  };
  // entry ci of a pass's token list: (token << 8) | slot
  auto entry_at = [&](int const *el, int p, int ci) {
    return DENSE ? ((p * NT + ci) << 8) : el[ci];
  };
  unsigned const ta = tb + AS * ACC_COLS; // MX: A stages
  float const LIMIT = __int_as_float(LIMIT_BITS);
  float const ALPHA = __int_as_float(ALPHA_BITS);
  float const OFF = __int_as_float(OFF_BITS);

  if (warp < 4) {
    // ---- the 128 rows: dequantize (MX) and run the epilogues. A unit's
    // epilogue waits for its last MMA; the next unit's blocks are converted
    // meanwhile, and the epilogue runs between two of them.
    int const q = warp, row = tid;
    unsigned const qrow = static_cast<unsigned>(q * 32) << 16;
    float bias_v[2] = {0.f, 0.f}; // of the units in flight (by parity)
    int epi = 0;                  // the units whose epilogue has run
    auto epilogue = [&](int j) {
      int e_, p_, rb, n;
      unit_at(j, e_, p_, rb, n);
      const int grow = rb * 128 + row;
      const int *el = elist + (j & 1) * NT;
      const float bv = bias_v[j & 1];
      const int st = j % AS;
      tmem_fence_after();
      const int nr = n > 16 ? (n + 15) & ~15 : 16;
      int pq, uid, kb0_, kb1_;
      part_at(j, pq, uid, kb0_, kb1_);
      // this warp's rows of the unit's parts: [KS][32 rows][NT] in the scratch
      float *prow =
          KS > 1 ? kpart + ((static_cast<size_t>(uid) * KS) * 128 + row) * NT
                 : nullptr;
      bool finish = true;
      if (KS > 1) {
#pragma unroll 1
        for (int c0 = 0; c0 < nr; c0 += 16) {
          unsigned r[16];
          float pa[16];
          tmem_ld16(tb + st * ACC_COLS + c0 + qrow, r);
#pragma unroll
          for (int c = 0; c < 16; ++c) {
            pa[c] = __uint_as_float(r[c]);
          }
#pragma unroll
          for (int a = 1; a < NACC; ++a) {
            tmem_ld16(tb + st * ACC_COLS + a * NT + c0 + qrow, r);
#pragma unroll
            for (int c = 0; c < 16; ++c) {
              pa[c] += __uint_as_float(r[c]);
            }
          }
#pragma unroll
          for (int c4 = 0; c4 < 4; ++c4) {
            *reinterpret_cast<float4 *>(
                prow + static_cast<size_t>(pq) * 128 * NT + c0 + 4 * c4) =
                make_float4(
                    pa[4 * c4], pa[4 * c4 + 1], pa[4 * c4 + 2], pa[4 * c4 + 3]);
          }
        }
        __threadfence();
        int arrived = 0;
        if (lane == 0) {
          arrived = atomicAdd(&kcnt[uid * 4 + q], 1);
        }
        arrived = __shfl_sync(0xffffffffu, arrived, 0);
        finish = arrived == KS - 1;
        if (finish) {
          __threadfence();
          if (lane == 0) {
            kcnt[uid * 4 + q] = 0;
          }
        }
      }
#pragma unroll 1
      for (int c0 = 0; finish && c0 < nr; c0 += 16) {
        unsigned r[16];
        float acc[16];
        tmem_ld16(tb + st * ACC_COLS + c0 + qrow, r);
#pragma unroll
        for (int c = 0; c < 16; ++c) {
          acc[c] = __uint_as_float(r[c]);
        }
#pragma unroll
        for (int a = 1; a < NACC; ++a) {
          tmem_ld16(tb + st * ACC_COLS + a * NT + c0 + qrow, r);
#pragma unroll
          for (int c = 0; c < 16; ++c) {
            acc[c] += __uint_as_float(r[c]);
          }
        }
        if (KS > 1) {
          // the unit's parts in order, this one from its accumulator
          float sum[16];
#pragma unroll
          for (int c = 0; c < 16; ++c) {
            sum[c] = 0.f;
          }
#pragma unroll 1
          for (int o = 0; o < KS; ++o) {
            if (o == pq) {
#pragma unroll
              for (int c = 0; c < 16; ++c) {
                sum[c] += acc[c];
              }
            } else {
#pragma unroll
              for (int c4 = 0; c4 < 4; ++c4) {
                const float4 v = *reinterpret_cast<const float4 *>(
                    prow + static_cast<size_t>(o) * 128 * NT + c0 + 4 * c4);
                sum[4 * c4] += v.x;
                sum[4 * c4 + 1] += v.y;
                sum[4 * c4 + 2] += v.z;
                sum[4 * c4 + 3] += v.w;
              }
            }
          }
#pragma unroll
          for (int c = 0; c < 16; ++c) {
            acc[c] = sum[c];
          }
        }
        // (a loop, not 16 copies: the epilogue runs once a unit, and code
        // that runs once costs its fetch)
#pragma unroll 1
        for (int c = 0; c < 16; ++c) {
          const int ci = c0 + c;
          // as the reference: the product rounded to bf16, then the bias
          // (FUSE: a linear layer's product + bias, rounded once below)
          const float y = (FUSE ? acc[c] : bfround(acc[c])) + bv;
          if (ACT) {
            const float yr = bfround(y);
            const float o = __shfl_xor_sync(0xffffffffu, yr, 1);
            if ((lane & 1) == 0 && ci < n && grow < ROWS) {
              const float g = fminf(yr, LIMIT);
              const float u = fminf(fmaxf(o, -LIMIT), LIMIT);
              const float sg = 1.f / (1.f + __expf(fminf(-ALPHA * g, 40.f)));
              const int ent = entry_at(el, p_, ci);
              out[(static_cast<size_t>(ent >> 8) * SLOTS + (ent & 0xff)) * OS +
                  rb * 64 + (row >> 1)] = f2bf((u + OFF) * g * sg);
            }
          } else if (ci < n && grow < ROWS) {
            const int ent = entry_at(el, p_, ci);
            const size_t o =
                (static_cast<size_t>(ent >> 8) * SLOTS + (ent & 0xff)) * OS +
                grow;
            out[o] = f2bf(RES ? bfround(y) + bf2f(resid[o]) : y);
          }
        }
      }
      tmem_fence_before();
      __syncwarp();
      if (lane == 0) {
        mbar_arrive(&acc_empty[st]);
      }
#ifdef MXG_TRACE
      if (tid == 0 && j < 4) {
        MXG_T(8 + 12 * j + 4);
      }
#endif
    };
    int gs = 0;
#pragma unroll 1
    for (int j = 0; j < n_my; ++j) {
      int e, p, rb, n;
      unit_at(j, e, p, rb, n);
      int const grow = rb * 128 + row;
      // unit j takes over unit j - 2's list and accumulator stage
      if (epi + 2 <= j) {
        mbar_wait(&acc_full[epi % AS], (epi / AS) & 1);
        epilogue(epi);
        ++epi;
      }
      rows_sync();
      if (!DENSE && row < n) {
        elist[(j & 1) * NT + row] =
            rlist[static_cast<size_t>(e) * BATCH + p * NT + row];
      }
      bias_v[j & 1] = 0.f;
      if (HAS_BIAS && grow < ROWS) {
        bias_v[j & 1] = bf2f(bias[static_cast<size_t>(e) * ROWS + grow]);
      }
      // (a dense layer's e is 0: bias [ROWS])
      rows_sync();
#ifdef MXG_TRACE
      if (tid == 0 && j < 4) {
        MXG_T(8 + 12 * j + 0);
      }
      long long tr_x = 0, tr_e = 0;
#endif
      int pq, uid_, kb0, kb1;
      part_at(j, pq, uid_, kb0, kb1);
      if (MX) {
#pragma unroll 1
        for (int kb = kb0; kb < kb1; ++kb, ++gs) {
          if (epi < j && mbar_test(&acc_full[epi % AS], (epi / AS) & 1)) {
            epilogue(epi);
            ++epi;
          }
          int const s = gs % STG, sa = gs % SA;
#ifdef MXG_TRACE
          long long const tw0 = mxg_now();
#endif
          mbar_wait(&x_full[s], (gs / STG) & 1);
#ifdef MXG_TRACE
          long long const tw1 = mxg_now();
          tr_x += tw1 - tw0;
          if (tid == 0 && kb == kb0 && j < 4) {
            MXG_T(8 + 12 * j + 1);
          }
          if (tid == 0) {
            MXG_STEP(kb, 0);
          }
#endif
          if (gs >= SA) {
            mbar_wait(&a_empty[sa], ((gs / SA) - 1) & 1);
          }
#ifdef MXG_TRACE
          tr_e += mxg_now() - tw1;
          if (tid == 0) {
            MXG_STEP(kb, 1);
          }
#endif
          tmem_fence_after();
          unsigned char const *blk =
              reinterpret_cast<unsigned char const *>(areg) + s * ABLK;
          // (a loop over the block's tiles: as fast as its copies once
          // warm, and a third of the code to fetch on a task's first block)
#pragma unroll 1
          for (int h = 0; h < TPB; ++h) {
            unsigned v[32];
#pragma unroll
            for (int cc = 0; cc < 2; ++cc) {
              int const c = 2 * h + cc;
              unsigned const sb = blk[NCH * 128 * 16 + row * NCH + c];
              unsigned const s2 = ((sb + 112u) << 7) * 0x00010001u;
              unsigned const c2 = (0x8000u | ((sb + 3u) << 7)) * 0x00010001u;
              const uint4 w4 =
                  *reinterpret_cast<uint4 const *>(blk + (c * 128 + row) * 16);
              cvt4_scale(w4.x, s2, c2, &v[cc * 16 + 0]);
              cvt4_scale(w4.y, s2, c2, &v[cc * 16 + 4]);
              cvt4_scale(w4.z, s2, c2, &v[cc * 16 + 8]);
              cvt4_scale(w4.w, s2, c2, &v[cc * 16 + 12]);
            }
#ifdef MXG_TRACE
            if (tid == 0 && gs == 0) {
              MXG_T(40 + 2 * h);
            }
#endif
            tmem_st32(ta + sa * ACOLS + h * 32 + qrow, v);
#ifdef MXG_TRACE
            if (tid == 0 && gs == 0) {
              MXG_T(41 + 2 * h);
            }
#endif
          }
          __syncwarp(); // the warp has read the block: its slot is free
          if (lane == 0) {
            mbar_arrive(&x_empty[s]);
          }
#ifdef MXG_TRACE
          if (tid == 0 && gs == 0) {
            MXG_T(46);
          }
#endif
          tmem_wait_st();
#ifdef MXG_TRACE
          if (tid == 0 && gs == 0) {
            MXG_T(47);
          }
#endif
          tmem_fence_before();
          __syncwarp();
          if (lane == 0) {
            mbar_arrive(&a_ready[sa]);
          }
#ifdef MXG_TRACE
          if (tid == 0) {
            MXG_STEP(kb, 2);
          }
#endif
        }
      }
#ifdef MXG_TRACE
      if (tid == 0 && j < 4) {
        MXG_T(8 + 12 * j + 2);
        mxg_tr[8 + 12 * j + 8] = tr_x;
        mxg_tr[8 + 12 * j + 9] = tr_e;
      }
#endif
    }
#pragma unroll 1
    while (epi < n_my) {
      mbar_wait(&acc_full[epi % AS], (epi / AS) & 1);
#ifdef MXG_TRACE
      if (tid == 0 && epi < 4) {
        MXG_T(8 + 12 * epi + 3);
      }
#endif
      epilogue(epi);
      ++epi;
    }
  } else if (warp == 4) {
    // ---- the weight copies
    if (lane == 0) {
      int gs = 0;
#pragma unroll 1
      for (int j = 0; j < n_my; ++j) {
        int e, p_, rb, n_;
        unit_at(j, e, p_, rb, n_);
        char const *src = wpack + static_cast<size_t>(e * RB + rb) * NKB * ABLK;
        int pq, uid_, kb0, kb1;
        part_at(j, pq, uid_, kb0, kb1);
#pragma unroll 1
        for (int kb = kb0; kb < kb1; ++kb, ++gs) {
          int const s = gs % STG;
#ifdef MXG_TRACE
          if (j == 0 && kb < 24) {
            mxg_tr[64 + 8 * kb + 0] = mxg_now(); // asks for the stage
          }
#endif
          if (gs >= STG) {
            mbar_wait(&x_empty[s], ((gs / STG) - 1) & 1);
          }
          mbar_expect(&x_full[s], ABLK);
          bulk_copy(areg + s * ABLK,
                    src + static_cast<size_t>(kb) * ABLK,
                    ABLK,
                    &x_full[s]);
#ifdef MXG_TRACE
          if (j == 0 && kb < 24) {
            mxg_tr[64 + 8 * kb + 1] = mxg_now(); // its copy is out
          }
#endif
        }
      }
    }
  } else if (warp == 5 || (MMA2 && warp == 7)) {
    // ---- the MMAs (MMA2: this thread's half of them)
    if (lane == 0) {
      int const half = warp == 7 ? 1 : 0;
      int gs = 0;
#pragma unroll 1
      for (int j = 0; j < n_my; ++j) {
        int e_, p_, rb_, n;
        unit_at(j, e_, p_, rb_, n);
        int const nr = n > 16 ? (n + 15) & ~15 : 16;
        const uint32_t ides = idesc(128, nr);
        int const st = j % AS;
        if (j >= AS) {
          mbar_wait(&acc_empty[st], ((j / AS) - 1) & 1);
        }
        tmem_fence_after();
        unsigned const td = tb + st * ACC_COLS;
        int mi = 0; // MMAs of the unit so far
#ifdef MXG_TRACE
        long long tr_b = 0, tr_a = 0, tr_i = 0, tr_c = 0;
#endif
        int pq, uid_, kb0, kb1;
        part_at(j, pq, uid_, kb0, kb1);
#pragma unroll 1
        for (int kb = kb0; kb < kb1; ++kb, ++gs) {
          int const bs = B2X ? (j & 1) * NKB + kb : gs % STGB, sa = gs % SA;
#ifdef MXG_TRACE
          long long const tw0 = mxg_now();
#endif
          mbar_wait(&b_full[bs], (B2X ? j / 2 : gs / STGB) & 1);
#ifdef MXG_TRACE
          long long const tw1 = mxg_now();
#endif
          if (MX) {
            mbar_wait(&a_ready[sa], (gs / SA) & 1);
          } else {
            mbar_wait(&x_full[sa], (gs / SA) & 1);
          }
          tmem_fence_after();
#ifdef MXG_TRACE
          long long const tw2 = mxg_now();
#endif
          // (unrolled: a loop here costs the MMA thread a quarter more per
          // MMA while the conversion warps keep the SM busy)
#pragma unroll
          for (int b = 0; b < TPB; ++b) {
            const u64 bd = smem_desc(sm_u32(breg + bs * BSTG + b * BBOX));
#pragma unroll
            for (int k2 = 0; k2 < 4 / NMT; ++k2, ++mi) {
              static_assert((TPB * 4) % NACC == 0, "accumulator of an MMA");
              static_assert(!MMA2 || NACC == 2, "an accumulator a thread");
              int const kk = MMA2 ? 2 * k2 + half : k2;
              int const accf = MMA2
                                   ? ((kb > kb0 || b > 0 || k2 > 0) ? 1 : 0)
                                   : ((kb > kb0 || b * 4 + kk >= NACC) ? 1 : 0);
              unsigned const tdm =
                  td + (MMA2 ? half : (b * 4 + kk) % NACC) * NT;
              if (MX) {
                umma_ts(tdm,
                        ta + sa * ACOLS + (b * 4 + kk) * 8,
                        bd + 2 * kk,
                        ides,
                        accf);
              } else {
                umma_ss(tdm,
                        smem_desc(sm_u32(areg + sa * ABLK + b * 16384)) +
                            2 * kk,
                        bd + 2 * kk,
                        ides,
                        accf);
              }
            }
          }
#ifdef MXG_TRACE
          long long const tw3 = mxg_now();
#endif
          if (!B2X) {
            umma_commit(&b_empty[bs]);
          }
          umma_commit(MX ? &a_empty[sa] : &x_empty[sa]);
#ifdef MXG_TRACE
          long long const tw4 = mxg_now();
          tr_b += tw1 - tw0;
          tr_a += tw2 - tw1;
          tr_i += tw3 - tw2;
          tr_c += tw4 - tw3;
          if (j == 0 && kb < 24) {
            mxg_tr[64 + 8 * kb + 3] = tw1;
            mxg_tr[64 + 8 * kb + 4] = tw2;
            mxg_tr[64 + 8 * kb + 5] = tw3;
            mxg_tr[64 + 8 * kb + 6] = tw0;
          }
#endif
        }
        umma_commit(&acc_full[st]);
#ifdef MXG_TRACE
        if (j < 4) {
          mxg_tr[8 + 12 * j + 5] = tr_b;
          mxg_tr[8 + 12 * j + 6] = tr_a;
          mxg_tr[8 + 12 * j + 7] = mxg_now();
          mxg_tr[8 + 12 * j + 10] = tr_i;
          mxg_tr[8 + 12 * j + 11] = tr_c;
        }
#endif
      }
    }
  } else {
    // ---- the activation gather: GL lanes, a lane copies 64 / GL of the
    // eight 16 B chunks of every 8th token in every box (row r of a box,
    // chunk c: SWIZZLE_128B puts it at chunk c ^ (r & 7))
    constexpr int CPL = 64 / GL;
    int const gl = MMA2 ? lane : (warp - 6) * 32 + lane;
    int const tsub = gl / (8 / CPL), c0 = (gl % (8 / CPL)) * CPL;
    int gs = 0;
#pragma unroll 1
    for (int j = 0; j < n_my; ++j) {
      int e, p, rb_, n;
      unit_at(j, e, p, rb_, n);
      char const *src[NTL];
#pragma unroll
      for (int it = 0; it < NTL; ++it) {
        int const jt = tsub + 8 * it;
        src[it] = nullptr;
        if (jt < n) {
          int const ent =
              DENSE ? ((p * NT + jt) << 8)
                    : rlist[static_cast<size_t>(e) * BATCH + p * NT + jt];
          const size_t r =
              IN_SLOTTED ? static_cast<size_t>(ent >> 8) * SLOTS + (ent & 0xff)
                         : static_cast<size_t>(ent >> 8);
          src[it] = x + r * (K * 2);
        }
      }
      if (B2X && j >= 2) {
        // unit j - 2's MMAs are over: its stages are this unit's
        mbar_wait(&acc_full[j % AS], ((j - 2) / AS) & 1);
      }
      int pq, uid_, kb0, kb1;
      part_at(j, pq, uid_, kb0, kb1);
#pragma unroll 1
      for (int kb = kb0; kb < kb1; ++kb, ++gs) {
        int const bs = B2X ? (j & 1) * NKB + kb : gs % STGB;
        if (!B2X && gs >= STGB) {
          mbar_wait(&b_empty[bs], ((gs / STGB) - 1) & 1);
        }
        char *bst = breg + bs * BSTG;
#pragma unroll
        for (int it = 0; it < NTL; ++it) {
          int const jt = tsub + 8 * it;
          if (jt < n) {
#pragma unroll
            for (int b = 0; b < TPB; ++b) {
#pragma unroll
              for (int cc = 0; cc < CPL; ++cc) {
                int const c = c0 + cc;
                cp_async16(bst + b * BBOX + jt * 128 + ((c ^ (jt & 7)) << 4),
                           src[it] + c * 16 + (kb * KB + b * 64) * 2);
              }
            }
          }
        }
        cp_async_arrive(&b_full[bs]);
      }
    }
  }
  __syncthreads();
  // Every warp that used barriers drops what the SM holds of them (an
  // initialisation of a barrier of its own).
  if (lane == 0) {
    mbar_init(&bar_flush[warp], 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (!HELD && warp == 0) {
    tmem_free512(tb);
  }
#ifdef MXG_TRACE
  if (tid == 0) {
    MXG_T(3);
  }
#endif
}

// ---------------------------------------------------------------------------
// The experts' outputs back into the residual stream:
//   y = residual + sum_s w[s] * out[s]
// with the reference models' roundings: every product is a bf16 number, their
// sum is, and so is the sum with the residual; with SHARED the last slot (the
// shared expert) is added to the rounded sum of the routed ones.
// inputs: [0] out bf16 [BATCH][SLOTS][H]; [1] w fp32 [BATCH][SLOTS];
// [2] residual bf16 [BATCH][H]. outputs: [0] y bf16 [BATCH][H].
// Unit = (token, chunk of CW columns), CW 8 (16 bytes) or 2 (4 bytes: a lane
// has a quarter of the arithmetic, for the batches where there are lanes to
// spare -- a chunk of 8 is about 1.5 us of a lane); the wave's lanes take the
// units round-robin (a lane's loads of one unit are in flight together; a
// warp walking a token's chunks one after another would wait for each).
// ---------------------------------------------------------------------------
template <int BATCH,
          int SLOTS,
          int H,
          int G,
          int SHARED,
          int CW,
          class TD,
          class RC>
__device__ __forceinline__ void moe_combine_sm100(TD const *task_desc,
                                                  RC const &runtime_config) {
  using namespace mxg;
  constexpr int NCHUNK = H / CW;
  constexpr int NW = CW / 2; // 32-bit words (two columns) a chunk
  constexpr int ROUTED = SLOTS - SHARED;
  static_assert((CW == 8 || CW == 2) && H % 8 == 0 && SLOTS <= 16,
                "16 B or 4 B chunks, the slots in registers");
  unsigned const *in = static_cast<unsigned const *>(task_desc->input_ptrs[0]);
  float const *wt = static_cast<float const *>(task_desc->input_ptrs[1]);
  unsigned const *res = static_cast<unsigned const *>(task_desc->input_ptrs[2]);
  unsigned *y = static_cast<unsigned *>(task_desc->output_ptrs[0]);
  int const nthreads = blockDim.x;
  int const gid =
      static_cast<int>(task_desc->task_metadata.ext_bid_x) * nthreads +
      threadIdx.x;
  // a chunk's words: one load
  auto load = [](unsigned const *p, unsigned *w) {
    if (NW == 4) {
      const uint4 v = *reinterpret_cast<const uint4 *>(p);
      w[0] = v.x;
      w[1] = v.y;
      w[2] = v.z;
      w[3] = v.w;
    } else {
      w[0] = *p;
    }
  };
#pragma unroll 1
  for (int u = gid; u < BATCH * NCHUNK; u += G * nthreads) {
    int const t = u / NCHUNK, c = u % NCHUNK;
    unsigned const *ir = in + static_cast<size_t>(t) * SLOTS * (H / 2);
    unsigned v[SLOTS][NW], rv[NW];
    float w[SLOTS];
#pragma unroll
    for (int s = 0; s < SLOTS; ++s) {
      load(ir + s * (H / 2) + c * NW, v[s]);
      w[s] = wt[static_cast<size_t>(t) * SLOTS + s];
    }
    load(res + static_cast<size_t>(t) * (H / 2) + c * NW, rv);
    float acc[CW], sh[CW];
#pragma unroll
    for (int j = 0; j < CW; ++j) {
      acc[j] = 0.f;
      sh[j] = 0.f;
    }
#pragma unroll
    for (int s = 0; s < SLOTS; ++s) {
#pragma unroll
      for (int j = 0; j < NW; ++j) {
        float const lo = bfround(bf2f(v[s][j] & 0xffff) * w[s]);
        float const hi = bfround(bf2f(v[s][j] >> 16) * w[s]);
        if (s < ROUTED) {
          acc[2 * j] += lo;
          acc[2 * j + 1] += hi;
        } else {
          sh[2 * j] = lo;
          sh[2 * j + 1] = hi;
        }
      }
    }
    unsigned r[NW];
#pragma unroll
    for (int j = 0; j < NW; ++j) {
      float lo = bfround(acc[2 * j]), hi = bfround(acc[2 * j + 1]);
      if (SHARED) {
        lo = bfround(lo + sh[2 * j]);
        hi = bfround(hi + sh[2 * j + 1]);
      }
      r[j] = static_cast<unsigned>(f2bf(bf2f(rv[j] & 0xffff) + lo)) |
             (static_cast<unsigned>(f2bf(bf2f(rv[j] >> 16) + hi)) << 16);
    }
    unsigned *yr = y + static_cast<size_t>(t) * (H / 2) + c * NW;
    if (NW == 4) {
      *reinterpret_cast<uint4 *>(yr) = make_uint4(r[0], r[1], r[2], r[3]);
    } else {
      yr[0] = r[0];
    }
  }
}

// ---------------------------------------------------------------------------
// A small linear layer without the tensor core: y[t][r] = x[t] . W[r] (+ b[r]),
// fp32 sums of bf16 products rounded once to bf16. A K-long row costs a GEMM
// unit K / 16 MMAs in a chain whatever its size (about 7 us at K = 2880); a
// router's few hundred outputs are done sooner by threads.
// inputs: [0] x bf16 [BATCH][K]; [1] W bf16 [ROWS][K]; [2] (HAS_BIAS) b bf16
// [ROWS]. outputs: [0] y bf16 [BATCH][OSTRIDE].
// Unit = (token, row), one a warp at a time: lane l sums the 16-byte chunks
// l, l + 32, ... of the row.
// ---------------------------------------------------------------------------
template <int BATCH,
          int ROWS,
          int K,
          int OSTRIDE,
          int HAS_BIAS,
          int G,
          class TD,
          class RC>
__device__ __forceinline__ void small_linear_sm100(TD const *task_desc,
                                                   RC const &runtime_config) {
  using namespace mxg;
  constexpr int NCHUNK = K / 8;
  constexpr int CPL = (NCHUNK + 31) / 32; // chunks a lane
  static_assert(K % 8 == 0, "16 B chunks");
  uint4 const *x = static_cast<uint4 const *>(task_desc->input_ptrs[0]);
  uint4 const *w = static_cast<uint4 const *>(task_desc->input_ptrs[1]);
  unsigned short const *bias =
      HAS_BIAS ? static_cast<unsigned short const *>(task_desc->input_ptrs[2])
               : nullptr;
  unsigned short *y = static_cast<unsigned short *>(task_desc->output_ptrs[0]);
  int const lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int const nwarps = blockDim.x >> 5;
  int const gw =
      static_cast<int>(task_desc->task_metadata.ext_bid_x) * nwarps + warp;
#pragma unroll 1
  for (int u = gw; u < BATCH * ROWS; u += G * nwarps) {
    int const t = u / ROWS, r = u % ROWS;
    uint4 const *xr = x + static_cast<size_t>(t) * NCHUNK;
    uint4 const *wr = w + static_cast<size_t>(r) * NCHUNK;
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < CPL; ++i) {
      int const c = lane + 32 * i;
      if (c < NCHUNK) {
        const uint4 xv = xr[c], wv = wr[c];
        unsigned const *p = reinterpret_cast<unsigned const *>(&xv);
        unsigned const *q = reinterpret_cast<unsigned const *>(&wv);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          acc += bf2f(p[j] & 0xffff) * bf2f(q[j] & 0xffff) +
                 bf2f(p[j] >> 16) * bf2f(q[j] >> 16);
        }
      }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      acc += __shfl_xor_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
      float const b = HAS_BIAS ? bf2f(bias[r]) : 0.f;
      y[static_cast<size_t>(t) * OSTRIDE + r] = f2bf(acc + b);
    }
  }
}

// ---------------------------------------------------------------------------
// The end of an iteration's held TMEM: a wave of exactly num_workers tasks
// (one a worker: task k of an iteration runs on worker k % num_workers) frees
// what the workers' HELD GEMM tasks allocated and clears the token tensor
// (int32 [num_workers], input TOKEN_INPUT). Holding saves the allocation and
// the release in every GEMM task, about 1.5 us each. Every TMEM user of the
// iteration must be a HELD task: one that allocates on a worker that holds
// all 512 columns waits forever.
// ---------------------------------------------------------------------------
template <int TOKEN_INPUT, class TD, class RC>
__device__ __forceinline__ void tmem_release_sm100(TD const *task_desc,
                                                   RC const &runtime_config) {
  int *tok =
      static_cast<int *>(task_desc->input_ptrs[TOKEN_INPUT]) + blockIdx.x;
  int const v = *tok;
  if (v < 0) {
    if ((threadIdx.x >> 5) == 0) {
      mxg::tmem_free512(static_cast<unsigned>(v & 0x7fffffff));
    }
    if (threadIdx.x == 0) {
      *tok = 0;
    }
  }
}

} // namespace kernel
