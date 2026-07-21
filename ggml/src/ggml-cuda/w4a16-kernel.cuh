#pragma once

// Device kernels and tuning configuration for the W4A16 (w4 x f16) GEMM.
// Included by w4a16.cu and by the standalone tuning harness; the W4A16_*
// macros can be overridden with -D for parameter sweeps.

#include "cp-async.cuh"
#include "mma.cuh"

using namespace ggml_cuda_mma;

namespace {


#ifndef W4A16_K_GROUP
#define W4A16_K_GROUP 32
#endif
#ifndef W4A16_N_TILE
#define W4A16_N_TILE 256
#endif
#ifndef W4A16_N_MAX
#define W4A16_N_MAX 65536
#endif
#ifndef W4A16_N_MAX_Q6
#define W4A16_N_MAX_Q6 262144
#endif
#ifndef W4A16_M_TILE
#define W4A16_M_TILE 32
#endif
#ifndef W4A16_M_WARP
#define W4A16_M_WARP 32
#endif
#ifndef W4A16_N_WARP
#define W4A16_N_WARP 64
#endif
#ifndef W4A16_STAGES
#define W4A16_STAGES 3
#endif
constexpr int W4A16_A_STRIDE      = 32;
constexpr int W4A16_N_WARPS       = W4A16_N_TILE / W4A16_N_WARP;
constexpr int W4A16_WARPS         = (W4A16_M_TILE / W4A16_M_WARP) * W4A16_N_WARPS;
constexpr int W4A16_Q_WORDS       = W4A16_N_TILE * 4;
constexpr int W4A16_META_WORDS    = W4A16_N_TILE;
constexpr int W4A16_CHUNK_BYTES   = W4A16_Q_WORDS * sizeof(uint32_t) + W4A16_META_WORDS * sizeof(uint32_t);
constexpr int W4A16_A_STAGE_BYTES = W4A16_M_TILE * W4A16_A_STRIDE * sizeof(half);
constexpr int W4A16_Q_STAGE_BYTES = W4A16_Q_WORDS * sizeof(uint32_t);
constexpr int W4A16_SHARED_BYTES =
    W4A16_STAGES * (W4A16_A_STAGE_BYTES + W4A16_Q_STAGE_BYTES + W4A16_META_WORDS * sizeof(uint32_t));



static __device__ __forceinline__ void get_scale_min_k4_w4a16(int             index,
                                                              const uint8_t * scales,
                                                              uint8_t &       scale,
                                                              uint8_t &       minimum) {
    if (index < 4) {
        scale   = scales[index] & 63;
        minimum = scales[index + 4] & 63;
    } else {
        scale   = (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4);
        minimum = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4);
    }
}

static __device__ __forceinline__ uint32_t q4_k_value(const block_q4_K & block, int subgroup, int index) {
    const uint8_t packed = block.qs[(subgroup / 2) * 32 + index];
    return subgroup & 1 ? packed >> 4 : packed & 0x0f;
}

// Weight-format traits for the sidecar layout. W=4: Q4_K, W=6: Q6_K.
// A chunk holds one 256-row tile for one 32-wide k group.
template <int W> struct w4a16_fmt;
template <> struct w4a16_fmt<4> {
    static constexpr int Q_WORDS     = W4A16_N_TILE * 4;
    static constexpr int META_WORDS  = W4A16_N_TILE;
    static constexpr int CHUNK_BYTES = (Q_WORDS + META_WORDS) * (int)sizeof(uint32_t);
};
template <> struct w4a16_fmt<6> {
    static constexpr int Q_WORDS     = W4A16_N_TILE * 8; // 4 words of low 4 bits + 4 words of high 2 bits
    static constexpr int META_WORDS  = W4A16_N_TILE;
    static constexpr int CHUNK_BYTES = (Q_WORDS + META_WORDS) * (int)sizeof(uint32_t);
};

static __global__ void repack_q4_k_w4a16(const block_q4_K * __restrict__ src,
                                         uint8_t * __restrict__ sidecar,
                                         int k,
                                         int n) {
    const int groups = k / W4A16_K_GROUP;
    const int index  = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n * groups) {
        return;
    }

    const int          row      = index / groups;
    const int          group    = index - row * groups;
    const int          subgroup = group % (QK_K / W4A16_K_GROUP);
    const block_q4_K & block    = src[row * (k / QK_K) + group / (QK_K / W4A16_K_GROUP)];

    const int  tile        = row / W4A16_N_TILE;
    const int  row_in_tile = row % W4A16_N_TILE;
    uint8_t *  chunk       = sidecar + (int64_t(tile) * groups + group) * W4A16_CHUNK_BYTES;
    uint32_t * q_out       = reinterpret_cast<uint32_t *>(chunk) + row_in_tile * 4;

#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
        const int k0     = 2 * lane;
        uint32_t  packed = 0;
        packed |= q4_k_value(block, subgroup, k0 + 0) << 0;
        packed |= q4_k_value(block, subgroup, k0 + 8) << 4;
        packed |= q4_k_value(block, subgroup, k0 + 16) << 8;
        packed |= q4_k_value(block, subgroup, k0 + 24) << 12;
        packed |= q4_k_value(block, subgroup, k0 + 1) << 16;
        packed |= q4_k_value(block, subgroup, k0 + 9) << 20;
        packed |= q4_k_value(block, subgroup, k0 + 17) << 24;
        packed |= q4_k_value(block, subgroup, k0 + 25) << 28;
        q_out[lane] = packed;
    }

    uint8_t scale_u8;
    uint8_t minimum_u8;
    get_scale_min_k4_w4a16(subgroup, block.scales, scale_u8, minimum_u8);

    union {
        half2    h2;
        uint32_t u32;
    } metadata;

    metadata.h2 = __halves2half2(__float2half_rn(__half2float(__low2half(block.dm)) * scale_u8),
                                 __float2half_rn(-__half2float(__high2half(block.dm)) * minimum_u8));
    reinterpret_cast<uint32_t *>(chunk + W4A16_Q_STAGE_BYTES)[row_in_tile] = metadata.u32;
}

// Extract the 6-bit value of block_q6_K at position k in [0, 256).
static __device__ __forceinline__ uint32_t q6_k_value(const block_q6_K & block, int index) {
    const int h = index / 128;
    const int l = index % 32;
    const int s = (index % 128) / 32;
    const uint8_t ql = block.ql[h * 64 + l + (s & 1) * 32];
    const uint8_t qh = block.qh[h * 32 + l];
    const int lo4 = (s & 2) ? (ql >> 4) : (ql & 0x0f);
    const int hi2 = (qh >> (2 * s)) & 3;
    return lo4 | (hi2 << 4);
}

static __global__ void repack_q6_k_w6a16(const block_q6_K * __restrict__ src,
                                         uint8_t * __restrict__ sidecar,
                                         int k,
                                         int n) {
    const int groups = k / W4A16_K_GROUP;
    const int index  = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n * groups) {
        return;
    }

    const int          row      = index / groups;
    const int          group    = index - row * groups;
    const block_q6_K & block    = src[row * (k / QK_K) + group / (QK_K / W4A16_K_GROUP)];

    const int  tile        = row / W4A16_N_TILE;
    const int  row_in_tile = row % W4A16_N_TILE;
    uint8_t *  chunk       = sidecar + (int64_t(tile) * groups + group) * w4a16_fmt<6>::CHUNK_BYTES;
    uint32_t * q_out       = reinterpret_cast<uint32_t *>(chunk) + row_in_tile * 8;

    const int base = group % (QK_K / W4A16_K_GROUP) * W4A16_K_GROUP;
#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
        const int k0      = 2 * lane;
        uint32_t  pack_ql = 0;
        uint32_t  pack_qh = 0;
#pragma unroll
        for (int np = 0; np < 8; ++np) {
            const int     idx = base + (np % 4) * 8 + k0 + np / 4;
            const uint8_t v   = q6_k_value(block, idx);
            pack_ql |= (uint32_t)(v & 0x0f) << (4 * np);
            pack_qh |= (uint32_t)(v >> 4) << (4 * np);
        }
        q_out[lane]     = pack_ql;
        q_out[lane + 4] = pack_qh;
    }

    union {
        half2    h2;
        uint32_t u32;
    } metadata;

    const float d = __half2float(block.d);
    metadata.h2   = __halves2half2(__float2half_rn(d * block.scales[2 * (base / 32)]),
                                   __float2half_rn(d * block.scales[2 * (base / 32) + 1]));
    reinterpret_cast<uint32_t *>(chunk + w4a16_fmt<6>::Q_WORDS * sizeof(uint32_t))[row_in_tile] = metadata.u32;
}

static __global__ void convert_f32_f16_w4a16(const float4 * __restrict__ src, half2 * __restrict__ dst, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const float4 v = src[index];
        dst[2 * index + 0] = __floats2half2_rn(v.x, v.y);
        dst[2 * index + 1] = __floats2half2_rn(v.z, v.w);
    }
}

// Kernel configuration. N_TILE is fixed at 256 because it determines the sidecar
// layout; M_TILE/M_WARP/N_WARP/STAGES can be selected per shape at launch time.
template <int M_TILE_P, int M_WARP_P, int N_WARP_P, int STAGES_P>
struct w4a16_cfg {
    static constexpr int M_TILE  = M_TILE_P;
    static constexpr int M_WARP  = M_WARP_P;
    static constexpr int N_WARP  = N_WARP_P;
    static constexpr int STAGES  = STAGES_P;
    static constexpr int N_WARPS = W4A16_N_TILE / N_WARP;
    static constexpr int WARPS   = (M_TILE / M_WARP) * N_WARPS;

    static constexpr int A_STAGE_BYTES = M_TILE * W4A16_A_STRIDE * (int)sizeof(half);
    static constexpr int SHARED_BYTES  =
        STAGES * (A_STAGE_BYTES + W4A16_Q_STAGE_BYTES + W4A16_META_WORDS * (int)sizeof(uint32_t));
};

using w4a16_default_cfg = w4a16_cfg<W4A16_M_TILE, W4A16_M_WARP, W4A16_N_WARP, W4A16_STAGES>;

template <typename Cfg, int W>
constexpr size_t w4a16_shared_bytes() {
    return Cfg::STAGES * (Cfg::A_STAGE_BYTES + w4a16_fmt<W>::Q_WORDS * sizeof(uint32_t) +
                          w4a16_fmt<W>::META_WORDS * sizeof(uint32_t));
}

#if defined(TURING_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)

static __device__ __forceinline__ void commit_w4a16_copies() {
    asm volatile("cp.async.commit_group;");
}

template <int groups> static __device__ __forceinline__ void wait_w4a16_copies() {
    if constexpr (groups == 0) {
        asm volatile("cp.async.wait_group 0;");
    } else {
        asm volatile("cp.async.wait_group 1;");
    }
}

static __device__ __forceinline__ int w4a16_a_chunk(int row, int chunk) {
    return chunk ^ ((row >> 1) & 3);
}

template <typename Cfg = w4a16_default_cfg, int W = 4>
static __device__ __forceinline__ void issue_w4a16_stage(const half * __restrict__ activation,
                                                         const uint8_t * __restrict__ sidecar,
                                                         half *     shared_a,
                                                         uint32_t * shared_q,
                                                         uint32_t * shared_meta,
                                                         int        k,
                                                         int        groups,
                                                         int        m0,
                                                         int        n_tile,
                                                         int        group,
                                                         int        stage) {
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;

#    pragma unroll
    for (int a_copy = tid; a_copy < Cfg::M_TILE * 4; a_copy += Cfg::WARPS * WARP_SIZE) {
        const int    a_row   = a_copy / 4;
        const int    a_chunk = a_copy % 4;
        const half * src_a   = activation + int64_t(m0 + a_row) * k + group * W4A16_K_GROUP + a_chunk * 8;
        half *       dst_a   = shared_a + stage * (Cfg::A_STAGE_BYTES / sizeof(half)) + a_row * W4A16_A_STRIDE +
                       w4a16_a_chunk(a_row, a_chunk) * 8;
        cp_async_cg_16<256>(ggml_cuda_cvta_generic_to_shared(dst_a), src_a);
    }

    constexpr int Q_STAGE_BYTES = w4a16_fmt<W>::Q_WORDS * (int)sizeof(uint32_t);
    const uint8_t * src_chunk = sidecar + (int64_t(n_tile) * groups + group) * w4a16_fmt<W>::CHUNK_BYTES;
    uint8_t *       dst_q     = reinterpret_cast<uint8_t *>(shared_q + stage * w4a16_fmt<W>::Q_WORDS);
#    pragma unroll
    for (int q_copy = tid; q_copy < Q_STAGE_BYTES / 16; q_copy += Cfg::WARPS * WARP_SIZE) {
        cp_async_cg_16<256>(ggml_cuda_cvta_generic_to_shared(dst_q + q_copy * 16), src_chunk + q_copy * 16);
    }

    uint8_t * dst_meta = reinterpret_cast<uint8_t *>(shared_meta + stage * w4a16_fmt<W>::META_WORDS);
    if (tid < w4a16_fmt<W>::META_WORDS * int(sizeof(uint32_t)) / 16) {
        cp_async_cg_16<256>(ggml_cuda_cvta_generic_to_shared(dst_meta + tid * 16),
                            src_chunk + Q_STAGE_BYTES + tid * 16);
    }
    commit_w4a16_copies();
}

static __device__ __forceinline__ void load_w4a16_a(tile<16, 8, half2> & values,
                                                    const half2 * __restrict__ shared_a,
                                                    int logical_chunk) {
    int *       output         = reinterpret_cast<int *>(values.x);
    const int   row            = threadIdx.x % 16;
    const int   chunk          = logical_chunk + threadIdx.x / 16;
    const int   physical_chunk = w4a16_a_chunk(row, chunk);
    const int * src = reinterpret_cast<const int *>(shared_a) + row * (W4A16_A_STRIDE / 2) + physical_chunk * 4;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(output[0]), "=r"(output[1]), "=r"(output[2]), "=r"(output[3])
                 : "l"(src));
}

static __device__ __forceinline__ half2 bits_to_half2_w4a16(uint32_t bits) {
    union {
        uint32_t u32;
        half2    h2;
    } value;

    value.u32 = bits;
    return value.h2;
}

static __device__ __forceinline__ uint32_t lop3_and_or_w4a16(uint32_t value, uint32_t mask, uint32_t bits) {
    uint32_t result;
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(result) : "r"(value), "r"(mask), "r"(bits));
    return result;
}

static __device__ __forceinline__ void dequant_w4a16_half(uint32_t            packed,
                                                          half2               scale,
                                                          half2               bias,
                                                          tile<8, 8, half2> & first,
                                                          tile<8, 8, half2> & second) {
    constexpr uint32_t LO_MASK  = 0x000f000f;
    constexpr uint32_t HI_MASK  = 0x00f000f0;
    constexpr uint32_t MAGIC    = 0x64006400;
    const half2        sub_1024 = bits_to_half2_w4a16(MAGIC);
    const half2        mul_1_16 = bits_to_half2_w4a16(0x2c002c00);
    const half2        sub_64   = bits_to_half2_w4a16(0xd400d400);

    uint32_t values = packed;
#    pragma unroll
    for (int k16 = 0; k16 < 2; ++k16) {
        const uint32_t low_bits  = lop3_and_or_w4a16(values, LO_MASK, MAGIC);
        const uint32_t high_bits = lop3_and_or_w4a16(values, HI_MASK, MAGIC);
        half2          low       = __hsub2(bits_to_half2_w4a16(low_bits), sub_1024);
        half2          high      = __hfma2(bits_to_half2_w4a16(high_bits), mul_1_16, sub_64);
        low                      = __hfma2(low, scale, bias);
        high                     = __hfma2(high, scale, bias);
        tile<8, 8, half2> & dst  = k16 == 0 ? first : second;
        dst.x[0]                 = low;
        dst.x[1]                 = high;
        values >>= 8;
    }
}

template <typename Cfg = w4a16_default_cfg>
static __device__ __forceinline__ void load_w4a16_b(const uint32_t *    shared_q,
                                                    const uint32_t *    shared_meta,
                                                    int                 warp_n,
                                                    int                 n_fragment,
                                                    tile<8, 8, half2> & first,
                                                    tile<8, 8, half2> & second) {
    const int      row    = warp_n * Cfg::N_WARP + n_fragment * 8 + threadIdx.x / 4;
    const int      lane   = threadIdx.x % 4;
    const uint32_t packed = shared_q[row * 4 + lane];

    union {
        uint32_t u32;
        half2    h2;
    } metadata;

    metadata.u32       = shared_meta[row];
    const half scale_h = __low2half(metadata.h2);
    const half bias_h  = __high2half(metadata.h2);
    dequant_w4a16_half(packed, __halves2half2(scale_h, scale_h), __halves2half2(bias_h, bias_h), first, second);
}

// Dequant one packed Q6_K word pair (low 4 bits + high 2 bits) into 4 half2 values.
// scale0 applies to the first 16 values of the group, scale1 to the last 16.
static __device__ __forceinline__ void dequant_w6a16_half(uint32_t            packed_ql,
                                                          uint32_t            packed_qh,
                                                          half2               scale0,
                                                          half2               scale1,
                                                          tile<8, 8, half2> & first,
                                                          tile<8, 8, half2> & second) {
    constexpr uint32_t LO_MASK  = 0x000f000f;
    constexpr uint32_t HI_MASK  = 0x00f000f0;
    constexpr uint32_t MAGIC    = 0x64006400;
    const half2        sub_1056 = bits_to_half2_w4a16(0x64206420);
    const half2        mul_16   = bits_to_half2_w4a16(0x4c004c00);
    const half2        sub_16384 = bits_to_half2_w4a16(0xf400f400);
    const half2        mul_1_16 = bits_to_half2_w4a16(0x2c002c00);
    const half2        sub_96   = bits_to_half2_w4a16(0xd600d600);

    uint32_t values_ql = packed_ql;
    uint32_t values_qh = packed_qh;
#    pragma unroll
    for (int k16 = 0; k16 < 2; ++k16) {
        const half2 scale = k16 == 0 ? scale0 : scale1;

        const uint32_t lo_bits  = lop3_and_or_w4a16(values_ql, LO_MASK, MAGIC);
        const uint32_t lo2_bits = lop3_and_or_w4a16(values_qh, 0x00030003, MAGIC);
        const half2    q_lo     = __hadd2(bits_to_half2_w4a16(lo_bits), __hfma2(bits_to_half2_w4a16(lo2_bits), mul_16, sub_16384));
        half2          w_lo     = __hsub2(q_lo, sub_1056);
        w_lo                    = __hfma2(w_lo, scale, __float2half2_rn(0.f));

        const uint32_t hi_bits  = lop3_and_or_w4a16(values_ql, HI_MASK, MAGIC);
        const uint32_t hi2_bits = lop3_and_or_w4a16(values_qh, 0x00300030, MAGIC);
        const half2    q_hi     = __hadd2(bits_to_half2_w4a16(hi_bits), __hfma2(bits_to_half2_w4a16(hi2_bits), mul_16, sub_16384));
        half2          w_hi     = __hfma2(q_hi, mul_1_16, sub_96);
        w_hi                    = __hfma2(w_hi, scale, __float2half2_rn(0.f));

        tile<8, 8, half2> & dst = k16 == 0 ? first : second;
        dst.x[0]                = w_lo;
        dst.x[1]                = w_hi;
        values_ql >>= 8;
        values_qh >>= 8;
    }
}

template <typename Cfg = w4a16_default_cfg>
static __device__ __forceinline__ void load_w6a16_b(const uint32_t *    shared_q,
                                                    const uint32_t *    shared_meta,
                                                    int                 warp_n,
                                                    int                 n_fragment,
                                                    tile<8, 8, half2> & first,
                                                    tile<8, 8, half2> & second) {
    const int      row       = warp_n * Cfg::N_WARP + n_fragment * 8 + threadIdx.x / 4;
    const int      lane      = threadIdx.x % 4;
    const uint32_t packed_ql = shared_q[row * 8 + lane];
    const uint32_t packed_qh = shared_q[row * 8 + 4 + lane];

    union {
        uint32_t u32;
        half2    h2;
    } metadata;

    metadata.u32      = shared_meta[row];
    const half2 scale0 = __halves2half2(__low2half(metadata.h2), __low2half(metadata.h2));
    const half2 scale1 = __halves2half2(__high2half(metadata.h2), __high2half(metadata.h2));
    dequant_w6a16_half(packed_ql, packed_qh, scale0, scale1, first, second);
}

template <typename Cfg = w4a16_default_cfg, int W = 4>
__launch_bounds__(Cfg::WARPS * WARP_SIZE, 1) static __global__ void mul_mat_w4a16(const uint8_t * __restrict__ sidecar,
                                                                                  const half * __restrict__ activation,
                                                                                  float * __restrict__ dst,
                                                                                  int k,
                                                                                  int n,
                                                                                  int m) {
    GGML_UNUSED(m);

    constexpr int M_TILE = Cfg::M_TILE;
    constexpr int M_WARP = Cfg::M_WARP;
    constexpr int N_WARP = Cfg::N_WARP;
    constexpr int STAGES = Cfg::STAGES;

    __align__(16) extern __shared__ uint8_t shared_raw[];
    half *                                  shared_a = reinterpret_cast<half *>(shared_raw);
    uint32_t * shared_q = reinterpret_cast<uint32_t *>(shared_raw + STAGES * Cfg::A_STAGE_BYTES);
    uint32_t * shared_meta =
        reinterpret_cast<uint32_t *>(shared_raw + STAGES * (Cfg::A_STAGE_BYTES + w4a16_fmt<W>::Q_WORDS * sizeof(uint32_t)));

    const int n_tile = blockIdx.x;
    const int m0     = blockIdx.y * M_TILE;
    const int groups = k / W4A16_K_GROUP;
    const int warp_m = threadIdx.y / Cfg::N_WARPS;
    const int warp_n = threadIdx.y % Cfg::N_WARPS;

    tile<16, 8, float> accum[M_WARP / 16][N_WARP / 8];

    issue_w4a16_stage<Cfg, W>(activation, sidecar, shared_a, shared_q, shared_meta, k, groups, m0, n_tile, 0, 0);
    if (groups > 1) {
        issue_w4a16_stage<Cfg, W>(activation, sidecar, shared_a, shared_q, shared_meta, k, groups, m0, n_tile, 1, 1);
    }

    for (int group = 0; group < groups; ++group) {
        if (group + 1 < groups) {
            wait_w4a16_copies<1>();
        } else {
            wait_w4a16_copies<0>();
        }
        __syncthreads();

        if (group + 2 < groups) {
            issue_w4a16_stage<Cfg, W>(activation, sidecar, shared_a, shared_q, shared_meta, k, groups, m0, n_tile,
                                      group + 2, (group + 2) % STAGES);
        }

        const int     stage   = group % STAGES;
        const half2 * a_stage = reinterpret_cast<const half2 *>(
            shared_a + stage * (Cfg::A_STAGE_BYTES / sizeof(half)) + warp_m * M_WARP * W4A16_A_STRIDE);
        const uint32_t * q_stage    = shared_q + stage * w4a16_fmt<W>::Q_WORDS;
        const uint32_t * meta_stage = shared_meta + stage * w4a16_fmt<W>::META_WORDS;

        tile<16, 8, half2> a[M_WARP / 16][2];
#    pragma unroll
        for (int m_fragment = 0; m_fragment < M_WARP / 16; ++m_fragment) {
            const half2 * a_fragment = a_stage + m_fragment * 16 * (W4A16_A_STRIDE / 2);
            load_w4a16_a(a[m_fragment][0], a_fragment, 0);
            load_w4a16_a(a[m_fragment][1], a_fragment, 2);
        }

        tile<8, 8, half2> b[3][2];
        if constexpr (W == 4) {
            load_w4a16_b<Cfg>(q_stage, meta_stage, warp_n, 0, b[0][0], b[0][1]);
            load_w4a16_b<Cfg>(q_stage, meta_stage, warp_n, 1, b[1][0], b[1][1]);
        } else {
            load_w6a16_b<Cfg>(q_stage, meta_stage, warp_n, 0, b[0][0], b[0][1]);
            load_w6a16_b<Cfg>(q_stage, meta_stage, warp_n, 1, b[1][0], b[1][1]);
        }
#    pragma unroll
        for (int fragment = 0; fragment < N_WARP / 8; ++fragment) {
            const int current = fragment % 3;
            if (fragment + 2 < N_WARP / 8) {
                const int next = (fragment + 2) % 3;
                if constexpr (W == 4) {
                    load_w4a16_b<Cfg>(q_stage, meta_stage, warp_n, fragment + 2, b[next][0], b[next][1]);
                } else {
                    load_w6a16_b<Cfg>(q_stage, meta_stage, warp_n, fragment + 2, b[next][0], b[next][1]);
                }
            }
#    pragma unroll
            for (int k16 = 0; k16 < 2; ++k16) {
#    pragma unroll
                for (int m_fragment = 0; m_fragment < M_WARP / 16; ++m_fragment) {
                    mma(accum[m_fragment][fragment], a[m_fragment][k16], b[current][k16]);
                }
            }
        }
    }

    const int row0 = m0 + warp_m * M_WARP + threadIdx.x / 4;
    const int col0 = n_tile * W4A16_N_TILE + warp_n * N_WARP + 2 * (threadIdx.x % 4);
#pragma unroll
    for (int m_fragment = 0; m_fragment < M_WARP / 16; ++m_fragment) {
#pragma unroll
        for (int fragment = 0; fragment < N_WARP / 8; ++fragment) {
            float2 * dst0 = reinterpret_cast<float2 *>(dst + int64_t(row0 + m_fragment * 16) * n + col0 + fragment * 8);
            float2 * dst1 =
                reinterpret_cast<float2 *>(dst + int64_t(row0 + m_fragment * 16 + 8) * n + col0 + fragment * 8);
            *dst0 = make_float2(accum[m_fragment][fragment].x[0], accum[m_fragment][fragment].x[1]);
            *dst1 = make_float2(accum[m_fragment][fragment].x[2], accum[m_fragment][fragment].x[3]);
        }
    }
}

// Fused pair GEMM: two weight tensors (same k, same activation) in one launch.
// Each block's 256-column tile lies entirely within one tensor, so the inner loop is
// identical to mul_mat_w4a16; only the sidecar and dst pointers are selected per block.
template <typename Cfg = w4a16_default_cfg, int W = 4>
__launch_bounds__(Cfg::WARPS * WARP_SIZE, 1) static __global__ void mul_mat_w4a16_pair(const uint8_t * __restrict__ sidecar0,
                                                                                      const uint8_t * __restrict__ sidecar1,
                                                                                      const half * __restrict__ activation,
                                                                                      float * __restrict__ dst0,
                                                                                      float * __restrict__ dst1,
                                                                                      int k, int n0, int n1, int m) {
    GGML_UNUSED(m);

    constexpr int M_TILE = Cfg::M_TILE;
    constexpr int M_WARP = Cfg::M_WARP;
    constexpr int N_WARP = Cfg::N_WARP;
    constexpr int STAGES = Cfg::STAGES;

    __align__(16) extern __shared__ uint8_t shared_raw[];
    half *                                  shared_a = reinterpret_cast<half *>(shared_raw);
    uint32_t * shared_q = reinterpret_cast<uint32_t *>(shared_raw + STAGES * Cfg::A_STAGE_BYTES);
    uint32_t * shared_meta =
        reinterpret_cast<uint32_t *>(shared_raw + STAGES * (Cfg::A_STAGE_BYTES + w4a16_fmt<W>::Q_WORDS * sizeof(uint32_t)));

    const int n0_tiles  = n0 / W4A16_N_TILE;
    const int n_tile    = blockIdx.x;
    const bool second   = n_tile >= n0_tiles;
    const int tile_idx  = second ? n_tile - n0_tiles : n_tile;
    const uint8_t * sidecar = second ? sidecar1 : sidecar0;
    const int dst_n     = second ? n1 : n0;
    float * dst         = second ? dst1 : dst0;

    const int m0     = blockIdx.y * M_TILE;
    const int groups = k / W4A16_K_GROUP;
    const int warp_m = threadIdx.y / Cfg::N_WARPS;
    const int warp_n = threadIdx.y % Cfg::N_WARPS;

    tile<16, 8, float> accum[M_WARP / 16][N_WARP / 8];

    issue_w4a16_stage<Cfg, W>(activation, sidecar, shared_a, shared_q, shared_meta, k, groups, m0, tile_idx, 0, 0);
    if (groups > 1) {
        issue_w4a16_stage<Cfg, W>(activation, sidecar, shared_a, shared_q, shared_meta, k, groups, m0, tile_idx, 1, 1);
    }

    for (int group = 0; group < groups; ++group) {
        if (group + 1 < groups) {
            wait_w4a16_copies<1>();
        } else {
            wait_w4a16_copies<0>();
        }
        __syncthreads();

        if (group + 2 < groups) {
            issue_w4a16_stage<Cfg, W>(activation, sidecar, shared_a, shared_q, shared_meta, k, groups, m0, tile_idx,
                                      group + 2, (group + 2) % STAGES);
        }

        const int     stage   = group % STAGES;
        const half2 * a_stage = reinterpret_cast<const half2 *>(
            shared_a + stage * (Cfg::A_STAGE_BYTES / sizeof(half)) + warp_m * M_WARP * W4A16_A_STRIDE);
        const uint32_t * q_stage    = shared_q + stage * w4a16_fmt<W>::Q_WORDS;
        const uint32_t * meta_stage = shared_meta + stage * w4a16_fmt<W>::META_WORDS;

        tile<16, 8, half2> a[M_WARP / 16][2];
#pragma unroll
        for (int m_fragment = 0; m_fragment < M_WARP / 16; ++m_fragment) {
            const half2 * a_fragment = a_stage + m_fragment * 16 * (W4A16_A_STRIDE / 2);
            load_w4a16_a(a[m_fragment][0], a_fragment, 0);
            load_w4a16_a(a[m_fragment][1], a_fragment, 2);
        }

        tile<8, 8, half2> b[3][2];
        if constexpr (W == 4) {
            load_w4a16_b<Cfg>(q_stage, meta_stage, warp_n, 0, b[0][0], b[0][1]);
            load_w4a16_b<Cfg>(q_stage, meta_stage, warp_n, 1, b[1][0], b[1][1]);
        } else {
            load_w6a16_b<Cfg>(q_stage, meta_stage, warp_n, 0, b[0][0], b[0][1]);
            load_w6a16_b<Cfg>(q_stage, meta_stage, warp_n, 1, b[1][0], b[1][1]);
        }
#pragma unroll
        for (int fragment = 0; fragment < N_WARP / 8; ++fragment) {
            const int current = fragment % 3;
            if (fragment + 2 < N_WARP / 8) {
                const int next = (fragment + 2) % 3;
                if constexpr (W == 4) {
                    load_w4a16_b<Cfg>(q_stage, meta_stage, warp_n, fragment + 2, b[next][0], b[next][1]);
                } else {
                    load_w6a16_b<Cfg>(q_stage, meta_stage, warp_n, fragment + 2, b[next][0], b[next][1]);
                }
            }
#pragma unroll
            for (int k16 = 0; k16 < 2; ++k16) {
#pragma unroll
                for (int m_fragment = 0; m_fragment < M_WARP / 16; ++m_fragment) {
                    mma(accum[m_fragment][fragment], a[m_fragment][k16], b[current][k16]);
                }
            }
        }
    }

    const int row0 = m0 + warp_m * M_WARP + threadIdx.x / 4;
    const int col0 = tile_idx * W4A16_N_TILE + warp_n * N_WARP + 2 * (threadIdx.x % 4);
#pragma unroll
    for (int m_fragment = 0; m_fragment < M_WARP / 16; ++m_fragment) {
#pragma unroll
        for (int fragment = 0; fragment < N_WARP / 8; ++fragment) {
            float2 * out0 = reinterpret_cast<float2 *>(dst + int64_t(row0 + m_fragment * 16) * dst_n + col0 + fragment * 8);
            float2 * out1 =
                reinterpret_cast<float2 *>(dst + int64_t(row0 + m_fragment * 16 + 8) * dst_n + col0 + fragment * 8);
            *out0 = make_float2(accum[m_fragment][fragment].x[0], accum[m_fragment][fragment].x[1]);
            *out1 = make_float2(accum[m_fragment][fragment].x[2], accum[m_fragment][fragment].x[3]);
        }
    }
}

#else

template <typename Cfg = w4a16_default_cfg, int W = 4>
static __global__ void mul_mat_w4a16(const uint8_t * __restrict__ sidecar,
                                     const half * __restrict__ activation,
                                     float * __restrict__ dst,
                                     int k,
                                     int n,
                                     int m) {
    GGML_UNUSED_VARS(sidecar, activation, dst, k, n, m);
    NO_DEVICE_CODE;
}

template <typename Cfg = w4a16_default_cfg, int W = 4>
static __global__ void mul_mat_w4a16_pair(const uint8_t * __restrict__ sidecar0,
                                          const uint8_t * __restrict__ sidecar1,
                                          const half * __restrict__ activation,
                                          float * __restrict__ dst0,
                                          float * __restrict__ dst1,
                                          int k, int n0, int n1, int m) {
    GGML_UNUSED_VARS(sidecar0, sidecar1, activation, dst0, dst1, k, n0, n1, m);
    NO_DEVICE_CODE;
}

#endif

}  // namespace
