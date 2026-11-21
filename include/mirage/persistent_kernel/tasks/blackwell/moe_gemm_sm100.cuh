// moe_gemm_sm100.cuh -- routed-expert GEMMs for SM100 as extension tasks.
//
//   moe_route_sm100   top-k routing of one layer: the softmax weights and a
//                     route plan (per expert the tokens that reach it).
//   moe_gemm_sm100    one worker's share of an expert GEMM wave (W13 with the
//                     gated activation fused into the epilogue, or W2).
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

__device__ __forceinline__ unsigned sm_u32(const void *p) {
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
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
          sm_u32(bar)),
      "r"(bytes)
      : "memory");
}
__device__ __forceinline__ void mbar_wait(u64 *bar, unsigned phase) {
#ifdef MXG_SPIN
  asm volatile("{\n\t"
               ".reg .pred p;\n\t"
               "MXG_SPIN_WAIT:\n\t"
               "mbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
               "@!p bra MXG_SPIN_WAIT;\n\t"
               "}" ::"r"(sm_u32(bar)),
               "r"(phase)
               : "memory");
  return;
#endif
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
    bulk_copy(void *dst, const void *src, unsigned bytes, u64 *bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::"
               "bytes [%0], [%1], %2, [%3];" ::"r"(sm_u32(dst)),
               "l"(reinterpret_cast<u64>(src)),
               "r"(bytes),
               "r"(sm_u32(bar))
               : "memory");
}
__device__ __forceinline__ void cp_async16(void *dst, const void *src) {
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
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}" ::
                   "r"(td),
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
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
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
__device__ __forceinline__ void
    tmem_st32(unsigned addr, const unsigned (&v)[32]) {
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
#else
#define MXG_T(slot)
#define MXG_TADD(slot, t0)
#endif

// the four warps of rows 0 .. 127 (the epilogue's)
__device__ __forceinline__ void rows_sync() {
  asm volatile("bar.sync 9, 128;" ::: "memory");
}

} // namespace mxg

// ---------------------------------------------------------------------------
// Routing. inputs: [0] logits bf16 [BATCH][LSTRIDE]: the E router logits and,
// with SHARED, the shared expert's gate logit in column E. outputs: [0] weights fp32
// [BATCH][TOPK + SHARED] (softmax over the chosen logits; sigmoid for the
// shared expert; both rounded to bf16 as the reference models do);
// [1] plan list int32 [E + SHARED][BATCH], entries (token << 8) | slot;
// [2] plan meta int32 [1 + CAP + E + SHARED]: [0] the number of (expert, pass)
// pairs, [1 ..] the pairs (expert << 16) | pass in expert order,
// [1 + CAP + e] the token count of expert e.
// One warp per token: every lane holds E / 32 logits.
// ---------------------------------------------------------------------------
template <int BATCH,
          int E,
          int TOPK,
          int NT,
          int SHARED,
          int CAP,
          int LSTRIDE,
          class TD,
          class RC>
__device__ __forceinline__ void
    moe_route_sm100(TD const *task_desc, RC const &runtime_config) {
  using namespace mxg;
  constexpr int E_TOT = E + SHARED, SLOTS = TOPK + SHARED, VPL = E / 32;
  static_assert(E % 32 == 0 && VPL >= 1 && VPL <= 8, "E / 32 logits a lane");
  static_assert(TOPK + SHARED <= 32, "a lane per slot");
  static_assert(BATCH < (1 << 23), "token index in 24 bits");
  const unsigned short *logits =
      static_cast<const unsigned short *>(task_desc->input_ptrs[0]);
  float *weight = static_cast<float *>(task_desc->output_ptrs[0]);
  int *list = static_cast<int *>(task_desc->output_ptrs[1]);
  int *meta = static_cast<int *>(task_desc->output_ptrs[2]);

  extern __shared__ char shared_memory[];
  int *cnt = reinterpret_cast<int *>(
      (reinterpret_cast<uintptr_t>(shared_memory) + 15) / 16 * 16);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int nwarps = blockDim.x >> 5;
  for (int i = tid; i < E_TOT; i += blockDim.x) {
    cnt[i] = 0;
  }
  __syncthreads();

  for (int t = warp; t < BATCH; t += nwarps) {
    float v[VPL];
#pragma unroll
    for (int j = 0; j < VPL; ++j) {
      v[j] = bf2f(logits[static_cast<size_t>(t) * LSTRIDE + lane * VPL + j]);
    }
    float sel_val = 0.f;
    int sel_idx = 0;
#pragma unroll 1
    for (int k = 0; k < TOPK; ++k) {
      float bv = v[0];
      int bi = lane * VPL;
#pragma unroll
      for (int j = 1; j < VPL; ++j) {
        if (v[j] > bv) {
          bv = v[j];
          bi = lane * VPL + j;
        }
      }
      // the largest of the warp; equal values: the lower expert
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) {
          bv = ov;
          bi = oi;
        }
      }
      if (lane == k) {
        sel_val = bv;
        sel_idx = bi;
      }
#pragma unroll
      for (int j = 0; j < VPL; ++j) {
        if (bi == lane * VPL + j) {
          v[j] = -3.0e38f;
        }
      }
    }
    const float top = __shfl_sync(0xffffffffu, sel_val, 0);
    const float ex = lane < TOPK ? __expf(sel_val - top) : 0.f;
    float sum = ex;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }
    if (lane < TOPK) {
      weight[static_cast<size_t>(t) * SLOTS + lane] = bfround(ex / sum);
      const int pos = atomicAdd(&cnt[sel_idx], 1);
      list[static_cast<size_t>(sel_idx) * BATCH + pos] = (t << 8) | lane;
    } else if (SHARED && lane == TOPK) {
      const float g = bf2f(logits[static_cast<size_t>(t) * LSTRIDE + E]);
      weight[static_cast<size_t>(t) * SLOTS + TOPK] =
          bfround(1.f / (1.f + __expf(fminf(-g, 40.f))));
      list[static_cast<size_t>(E) * BATCH + t] = (t << 8) | TOPK;
    }
  }
  __syncthreads();

  if (warp == 0) {
    int base = 0;
#pragma unroll 1
    for (int e0 = 0; e0 < E_TOT; e0 += 32) {
      const int e = e0 + lane;
      int c = 0;
      if (e < E) {
        c = cnt[e];
      } else if (e < E_TOT) {
        c = BATCH;
      }
      const int np = (c + NT - 1) / NT;
      int pre = np;
#pragma unroll
      for (int off = 1; off < 32; off <<= 1) {
        const int o = __shfl_up_sync(0xffffffffu, pre, off);
        if (lane >= off) {
          pre += o;
        }
      }
      const int first = base + pre - np;
      for (int p = 0; p < np; ++p) {
        meta[1 + first + p] = (e << 16) | p;
      }
      if (e < E_TOT) {
        meta[1 + CAP + e] = c;
      }
      base += __shfl_sync(0xffffffffu, pre, 31);
    }
    if (lane == 0) {
      meta[0] = base;
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
//   (clamp(up, -LIMIT, LIMIT) + OFF) * g * sigmoid(ALPHA * g), g = min(gate, LIMIT)
// (LIMIT = inf, OFF = 0, ALPHA = 1: SiLU(gate) * up).
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
          class TD,
          class RC>
__device__ __forceinline__ void
    moe_gemm_sm100(TD const *task_desc, RC const &runtime_config) {
  using namespace mxg;
  constexpr int RB = (ROWS + 127) / 128; // row blocks of an expert
  constexpr int NKB = K / KB;            // steps of a unit
  constexpr int TPB = KB / 64;           // 64-column boxes of a step
  constexpr int NCH = KB / 32;           // MX: scale groups of a row in a block
  // bytes of a weight copy
  constexpr int ABLK = MX ? NCH * 128 * 16 + 128 * NCH : TPB * 128 * 128;
  constexpr int ACOLS = KB / 2; // MX: TMEM columns of an A stage
  constexpr int AS = 2;         // accumulator stages
  // An MMA that adds to an accumulator waits for the one before it (about
  // 55 ns each, whatever its size): the K steps of a unit go round NACC
  // accumulators, which the epilogue adds up.
#ifdef MXG_NACC
  constexpr int NACC = MXG_NACC;
#else
  constexpr int NACC = NT <= 32 ? 4 : (NT <= 64 ? 2 : 1);
#endif
  constexpr int ACC_COLS = NACC * NT; // TMEM columns of an accumulator stage
  // rows of a B box: the most tokens a pass can have
  constexpr int BROWS = ((NT < BATCH ? NT : BATCH) + 7) / 8 * 8;
  constexpr int BBOX = BROWS * 128;  // bytes of a B box
  constexpr int BSTG = TPB * BBOX;   // bytes of a B stage
  constexpr int NTL = BROWS / 8;     // gather: tokens of a lane
  constexpr int MAXU = 64;           // units of a task
  constexpr int SMEM_TOTAL = 196 * 1024;
  constexpr int STGB_FIT = (100 * 1024) / BSTG;
  constexpr int STGB = STGB_FIT >= NKB ? NKB : (STGB_FIT < 2 ? 2 : STGB_FIT);
  constexpr int STG_FIT = (SMEM_TOTAL - 8192 - STGB * BSTG) / ABLK;
  constexpr int STG_MAX = MX ? 8 : 6;
  constexpr int STG = STG_FIT > STG_MAX ? STG_MAX : STG_FIT; // weight copies
  constexpr int SA_FIT = (512 - AS * ACC_COLS) / ACOLS;
  constexpr int SA = MX ? (SA_FIT > 4 ? 4 : SA_FIT) : STG; // A stages
  constexpr int OUT = ACT ? ROWS / 2 : ROWS;
  constexpr int AREG = (STG * ABLK + 1023) / 1024 * 1024;
  constexpr int BREG = STGB * BSTG;
  constexpr int NBAR = 2 * STG + 2 * SA + 2 * STGB + 2 * AS;
  constexpr int SMEM = AREG + BREG + 8 * NBAR + 16 * MAXU + 8 * NT + 64;
  static_assert(K % KB == 0 && KB % 64 == 0, "K steps");
  static_assert(NT % 16 == 0 && NT >= 16 && NT <= 128, "tokens of a pass");
  static_assert(STG >= 1 && SA >= 1 && STGB >= 1, "ring depths");
  static_assert(AS * ACC_COLS <= 512, "TMEM: the accumulators");
  static_assert(!ACT || ROWS % 2 == 0, "gate / up row pairs");
  static_assert(SMEM + 2048 <= SMEM_TOTAL, "shared memory");
  static_assert(E_TOT < 32768 && SLOTS <= 256, "plan entry fields");

  const char *x = static_cast<const char *>(task_desc->input_ptrs[0]);
  const char *wpack = static_cast<const char *>(task_desc->input_ptrs[1]);
  const int *rlist = static_cast<const int *>(task_desc->input_ptrs[2]);
  const int *meta = static_cast<const int *>(task_desc->input_ptrs[3]);
  const unsigned short *bias =
      HAS_BIAS ? static_cast<const unsigned short *>(task_desc->input_ptrs[4])
               : nullptr;
  unsigned short *out =
      static_cast<unsigned short *>(task_desc->output_ptrs[0]);
  const int task_i = static_cast<int>(task_desc->task_metadata.ext_bid_x);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
#ifdef MXG_TRACE
  long long *mxg_tr =
      static_cast<long long *>(task_desc->input_ptrs[4 + HAS_BIAS]) +
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
  u64 *x_full = reinterpret_cast<u64 *>(breg + BREG); // a weight copy landed
  u64 *x_empty = x_full + STG;                        // ... and was consumed
  u64 *a_ready = x_empty + STG; // MX: an A stage is in TMEM
  u64 *a_empty = a_ready + SA;  // MX: ... and the MMAs have read it
  u64 *b_full = a_empty + SA;
  u64 *b_empty = b_full + STGB;
  u64 *acc_full = b_empty + STGB;
  u64 *acc_empty = acc_full + AS;
  int *ut = reinterpret_cast<int *>(acc_empty + AS); // [MAXU][4]
  int *elist = ut + 4 * MAXU;                        // [2][NT]
  int *misc = elist + 2 * NT;                        // n_ep, tmem base

  if (tid == 0) {
    misc[0] = meta[0];
  }
  __syncthreads();
  const int total = misc[0] * RB;
  const int n_my = total > task_i ? (total - task_i - 1) / G + 1 : 0;
  if (n_my > MAXU) {
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
  if (tid < n_my) {
    const int u = task_i + tid * G;
    const int ent = meta[1 + u / RB];
    const int e = ent >> 16, p = ent & 0xffff;
    const int left = meta[1 + CAP + e] - p * NT;
    ut[tid * 4 + 0] = e;
    ut[tid * 4 + 1] = p;
    ut[tid * 4 + 2] = u % RB;
    ut[tid * 4 + 3] = left < NT ? left : NT;
  }
  if (tid == 0) {
#pragma unroll 1
    for (int i = 0; i < STG; ++i) {
      mbar_init(&x_full[i], 1);
      mbar_init(&x_empty[i], MX ? 4 : 1);
    }
#pragma unroll 1
    for (int i = 0; i < SA; ++i) {
      mbar_init(&a_ready[i], 4);
      mbar_init(&a_empty[i], 1);
    }
#pragma unroll 1
    for (int i = 0; i < STGB; ++i) {
      mbar_init(&b_full[i], 64);
      mbar_init(&b_empty[i], 1);
    }
#pragma unroll 1
    for (int i = 0; i < AS; ++i) {
      mbar_init(&acc_full[i], 1);
      mbar_init(&acc_empty[i], 4);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 0) {
    tmem_alloc512(reinterpret_cast<unsigned *>(&misc[1]));
    tmem_fence_before();
  }
  __syncthreads();
  tmem_fence_after();
#ifdef MXG_TRACE
  if (tid == 0) {
    MXG_T(2);
  }
#endif
  const unsigned tb = static_cast<unsigned>(misc[1]); // accumulators
  const unsigned ta = tb + AS * ACC_COLS;             // MX: A stages
  const float LIMIT = __int_as_float(LIMIT_BITS);
  const float ALPHA = __int_as_float(ALPHA_BITS);
  const float OFF = __int_as_float(OFF_BITS);

  if (warp < 4) {
    // ---- the 128 rows: dequantize (MX) and run the epilogues. A unit's
    // epilogue waits for its last MMA; the next unit's blocks are converted
    // meanwhile, and the epilogue runs between two of them.
    const int q = warp, row = tid;
    const unsigned qrow = static_cast<unsigned>(q * 32) << 16;
    float bias_v[2] = {0.f, 0.f}; // of the units in flight (by parity)
    int epi = 0;                  // the units whose epilogue has run
    auto epilogue = [&](int j) {
      const int rb = ut[j * 4 + 2], n = ut[j * 4 + 3];
      const int grow = rb * 128 + row;
      const int *el = elist + (j & 1) * NT;
      const float bv = bias_v[j & 1];
      const int st = j % AS;
      tmem_fence_after();
      const int nr = n > 16 ? (n + 15) & ~15 : 16;
#pragma unroll 1
      for (int c0 = 0; c0 < nr; c0 += 16) {
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
        // (a loop, not 16 copies: the epilogue runs once a unit, and code
        // that runs once costs its fetch)
#ifdef MXG_EPI_UNROLL
#pragma unroll
#else
#pragma unroll 1
#endif
        for (int c = 0; c < 16; ++c) {
          const int ci = c0 + c;
          // as the reference: the product rounded to bf16, then the bias
          const float y = bfround(acc[c]) + bv;
          if (ACT) {
            const float yr = bfround(y);
            const float o = __shfl_xor_sync(0xffffffffu, yr, 1);
            if ((lane & 1) == 0 && ci < n && grow < ROWS) {
              const float g = fminf(yr, LIMIT);
              const float u = fminf(fmaxf(o, -LIMIT), LIMIT);
              const float sg = 1.f / (1.f + __expf(fminf(-ALPHA * g, 40.f)));
              const int ent = el[ci];
              out[(static_cast<size_t>(ent >> 8) * SLOTS + (ent & 0xff)) * OUT +
                  rb * 64 + (row >> 1)] = f2bf((u + OFF) * g * sg);
            }
          } else if (ci < n && grow < ROWS) {
            const int ent = el[ci];
            out[(static_cast<size_t>(ent >> 8) * SLOTS + (ent & 0xff)) * OUT +
                grow] = f2bf(y);
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
      const int e = ut[j * 4 + 0], p = ut[j * 4 + 1], rb = ut[j * 4 + 2];
      const int n = ut[j * 4 + 3];
      const int grow = rb * 128 + row;
      // unit j takes over unit j - 2's list and accumulator stage
      if (epi + 2 <= j) {
        mbar_wait(&acc_full[epi % AS], (epi / AS) & 1);
        epilogue(epi);
        ++epi;
      }
      rows_sync();
      if (row < n) {
        elist[(j & 1) * NT + row] =
            rlist[static_cast<size_t>(e) * BATCH + p * NT + row];
      }
      bias_v[j & 1] = 0.f;
      if (HAS_BIAS && grow < ROWS) {
        bias_v[j & 1] = bf2f(bias[static_cast<size_t>(e) * ROWS + grow]);
      }
      rows_sync();
#ifdef MXG_TRACE
      if (tid == 0 && j < 4) {
        MXG_T(8 + 12 * j + 0);
      }
      long long tr_x = 0, tr_e = 0;
#endif
      if (MX) {
#pragma unroll 1
        for (int kb = 0; kb < NKB; ++kb, ++gs) {
          if (epi < j && mbar_test(&acc_full[epi % AS], (epi / AS) & 1)) {
            epilogue(epi);
            ++epi;
          }
          const int s = gs % STG, sa = gs % SA;
#ifdef MXG_TRACE
          const long long tw0 = mxg_now();
#endif
          mbar_wait(&x_full[s], (gs / STG) & 1);
#ifdef MXG_TRACE
          const long long tw1 = mxg_now();
          tr_x += tw1 - tw0;
          if (tid == 0 && kb == 0 && j < 4) {
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
          const unsigned char *blk =
              reinterpret_cast<const unsigned char *>(areg) + s * ABLK;
#pragma unroll
          for (int h = 0; h < TPB; ++h) {
            unsigned v[32];
#ifdef MXG_ABL_NO_DQ
            if (gs > 0) {
              continue;
            }
#endif
#pragma unroll
            for (int cc = 0; cc < 2; ++cc) {
              const int c = 2 * h + cc;
              const unsigned sb = blk[NCH * 128 * 16 + row * NCH + c];
              const unsigned s2 = ((sb + 112u) << 7) * 0x00010001u;
              const unsigned c2 = (0x8000u | ((sb + 3u) << 7)) * 0x00010001u;
              const uint4 w4 = *reinterpret_cast<const uint4 *>(
                  blk + (c * 128 + row) * 16);
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
        const int e = ut[j * 4 + 0], rb = ut[j * 4 + 2];
        const char *src =
            wpack + static_cast<size_t>(e * RB + rb) * NKB * ABLK;
#pragma unroll 1
        for (int kb = 0; kb < NKB; ++kb, ++gs) {
          const int s = gs % STG;
          if (gs >= STG) {
            mbar_wait(&x_empty[s], ((gs / STG) - 1) & 1);
          }
          mbar_expect(&x_full[s], ABLK);
          bulk_copy(areg + s * ABLK,
                    src + static_cast<size_t>(kb) * ABLK,
                    ABLK,
                    &x_full[s]);
        }
      }
    }
  } else if (warp == 5) {
    // ---- the MMAs
    if (lane == 0) {
      int gs = 0;
#pragma unroll 1
      for (int j = 0; j < n_my; ++j) {
        const int n = ut[j * 4 + 3];
        const int nr = n > 16 ? (n + 15) & ~15 : 16;
        const uint32_t ides = idesc(128, nr);
        const int st = j % AS;
        if (j >= AS) {
          mbar_wait(&acc_empty[st], ((j / AS) - 1) & 1);
        }
        tmem_fence_after();
        const unsigned td = tb + st * ACC_COLS;
        int mi = 0; // MMAs of the unit so far
#ifdef MXG_TRACE
        long long tr_b = 0, tr_a = 0, tr_i = 0, tr_c = 0;
#endif
#pragma unroll 1
        for (int kb = 0; kb < NKB; ++kb, ++gs) {
          const int bs = gs % STGB, sa = gs % SA;
#ifdef MXG_TRACE
          const long long tw0 = mxg_now();
#endif
          mbar_wait(&b_full[bs], (gs / STGB) & 1);
#ifdef MXG_TRACE
          const long long tw1 = mxg_now();
#endif
          if (MX) {
            mbar_wait(&a_ready[sa], (gs / SA) & 1);
          } else {
            mbar_wait(&x_full[sa], (gs / SA) & 1);
          }
          tmem_fence_after();
#ifdef MXG_TRACE
          const long long tw2 = mxg_now();
#endif
#pragma unroll
          for (int b = 0; b < TPB; ++b) {
            const u64 bd = smem_desc(sm_u32(breg + bs * BSTG + b * BBOX));
#pragma unroll
            for (int kk = 0; kk < 4; ++kk, ++mi) {
              const int accf = mi >= NACC ? 1 : 0;
              const unsigned tdm = td + (mi % NACC) * NT;
#ifdef MXG_ABL_NO_MMA
              if (mi > 0) {
                continue;
              }
#endif
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
          const long long tw3 = mxg_now();
#endif
          umma_commit(&b_empty[bs]);
          umma_commit(MX ? &a_empty[sa] : &x_empty[sa]);
#ifdef MXG_TRACE
          const long long tw4 = mxg_now();
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
    // ---- the activation gather: 64 lanes, a lane copies one 16 B chunk of
    // every 8th token in every box (row r of a box, chunk c: SWIZZLE_128B
    // puts it at chunk c ^ (r & 7))
    const int gl = (warp - 6) * 32 + lane;
    const int tsub = gl >> 3, c = gl & 7;
    int gs = 0;
#pragma unroll 1
    for (int j = 0; j < n_my; ++j) {
      const int e = ut[j * 4 + 0], p = ut[j * 4 + 1], n = ut[j * 4 + 3];
      const char *src[NTL];
#pragma unroll
      for (int it = 0; it < NTL; ++it) {
        const int jt = tsub + 8 * it;
        src[it] = nullptr;
        if (jt < n) {
          const int ent = rlist[static_cast<size_t>(e) * BATCH + p * NT + jt];
          const size_t r = IN_SLOTTED ? static_cast<size_t>(ent >> 8) * SLOTS +
                                            (ent & 0xff)
                                      : static_cast<size_t>(ent >> 8);
          src[it] = x + r * (K * 2) + c * 16;
        }
      }
#pragma unroll 1
      for (int kb = 0; kb < NKB; ++kb, ++gs) {
        const int bs = gs % STGB;
        if (gs >= STGB) {
          mbar_wait(&b_empty[bs], ((gs / STGB) - 1) & 1);
        }
        char *bst = breg + bs * BSTG;
#pragma unroll
        for (int it = 0; it < NTL; ++it) {
          const int jt = tsub + 8 * it;
          if (jt < n) {
#pragma unroll
            for (int b = 0; b < TPB; ++b) {
              cp_async16(bst + b * BBOX + jt * 128 + ((c ^ (jt & 7)) << 4),
                         src[it] + (kb * KB + b * 64) * 2);
            }
          }
        }
        cp_async_arrive(&b_full[bs]);
      }
    }
  }
  __syncthreads();
  if (warp == 0) {
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
// A warp per token; lane l holds the 16-byte chunks l, l + 32, ...
// ---------------------------------------------------------------------------
template <int BATCH, int SLOTS, int H, int G, int SHARED, class TD, class RC>
__device__ __forceinline__ void
    moe_combine_sm100(TD const *task_desc, RC const &runtime_config) {
  using namespace mxg;
  constexpr int NCHUNK = H / 8;
  constexpr int ROUTED = SLOTS - SHARED;
  static_assert(H % 8 == 0 && SLOTS <= 16, "16 B chunks, the slots in registers");
  const uint4 *in = static_cast<const uint4 *>(task_desc->input_ptrs[0]);
  const float *wt = static_cast<const float *>(task_desc->input_ptrs[1]);
  const uint4 *res = static_cast<const uint4 *>(task_desc->input_ptrs[2]);
  uint4 *y = static_cast<uint4 *>(task_desc->output_ptrs[0]);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int nwarps = blockDim.x >> 5;
  const int gw = static_cast<int>(task_desc->task_metadata.ext_bid_x) * nwarps +
                 warp;
#pragma unroll 1
  for (int t = gw; t < BATCH; t += G * nwarps) {
    float w[SLOTS];
#pragma unroll
    for (int s = 0; s < SLOTS; ++s) {
      w[s] = wt[static_cast<size_t>(t) * SLOTS + s];
    }
    const uint4 *ir = in + static_cast<size_t>(t) * SLOTS * NCHUNK;
#pragma unroll 1
    for (int c = lane; c < NCHUNK; c += 32) {
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      float sh[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int s = 0; s < SLOTS; ++s) {
        const uint4 v = ir[s * NCHUNK + c];
        const unsigned *p = reinterpret_cast<const unsigned *>(&v);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float lo = bfround(bf2f(p[j] & 0xffff) * w[s]);
          const float hi = bfround(bf2f(p[j] >> 16) * w[s]);
          if (s < ROUTED) {
            acc[2 * j] += lo;
            acc[2 * j + 1] += hi;
          } else {
            sh[2 * j] = lo;
            sh[2 * j + 1] = hi;
          }
        }
      }
      const uint4 rv = res[static_cast<size_t>(t) * NCHUNK + c];
      const unsigned *q = reinterpret_cast<const unsigned *>(&rv);
      uint4 o;
      unsigned *r = reinterpret_cast<unsigned *>(&o);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float lo = bfround(acc[2 * j]), hi = bfround(acc[2 * j + 1]);
        if (SHARED) {
          lo = bfround(lo + sh[2 * j]);
          hi = bfround(hi + sh[2 * j + 1]);
        }
        r[j] = static_cast<unsigned>(f2bf(bf2f(q[j] & 0xffff) + lo)) |
               (static_cast<unsigned>(f2bf(bf2f(q[j] >> 16) + hi)) << 16);
      }
      y[static_cast<size_t>(t) * NCHUNK + c] = o;
    }
  }
}

} // namespace kernel
