#include "w4a16-kernel.cuh"
#include "w4a16.cuh"

#include <cstdlib>

namespace {

static bool w4a16_enabled() {
    const char * value = std::getenv("GGML_CUDA_W4A16");
    return value == nullptr || value[0] != '0';
}

template <typename cfg>
static void launch_w4a16_kernel(int W, const void * sidecar, const half * activation_f16, ggml_tensor * dst, int k,
                                int n, int m, cudaStream_t stream) {
    const dim3 block(WARP_SIZE, cfg::WARPS, 1);
    const dim3 grid(n / W4A16_N_TILE, m / cfg::M_TILE, 1);
    if (W == 6) {
        static bool raised6 = false;
        if (!raised6) {
            CUDA_CHECK(cudaFuncSetAttribute((mul_mat_w4a16<cfg, 6>), cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            (int)w4a16_shared_bytes<cfg, 6>()));
            raised6 = true;
        }
        mul_mat_w4a16<cfg, 6><<<grid, block, w4a16_shared_bytes<cfg, 6>(), stream>>>(
            static_cast<const uint8_t *>(sidecar), activation_f16, static_cast<float *>(dst->data), k, n, m);
    } else {
        static bool raised4 = false;
        if (!raised4) {
            CUDA_CHECK(cudaFuncSetAttribute((mul_mat_w4a16<cfg, 4>), cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            (int)w4a16_shared_bytes<cfg, 4>()));
            raised4 = true;
        }
        mul_mat_w4a16<cfg, 4><<<grid, block, w4a16_shared_bytes<cfg, 4>(), stream>>>(
            static_cast<const uint8_t *>(sidecar), activation_f16, static_cast<float *>(dst->data), k, n, m);
    }
}

static bool w4a16_weight_eligible(int device, const ggml_tensor * tensor) {
    if (!w4a16_enabled() || ggml_cuda_info().devices[device].cc != GGML_CUDA_CC_DGX_SPARK) {
        return false;
    }

    // The int8 kernel is still faster for Q6_K with large n and short k (e.g. output logits).
    if (tensor->type == GGML_TYPE_Q6_K && tensor->ne[1] >= 8192 && tensor->ne[0] <= 4096) {
        return false;
    }

    const int n_max = tensor->type == GGML_TYPE_Q6_K ? W4A16_N_MAX_Q6 : W4A16_N_MAX;
    return (tensor->type == GGML_TYPE_Q4_K || tensor->type == GGML_TYPE_Q6_K) && tensor->op == GGML_OP_NONE &&
           tensor->ne[0] % QK_K == 0 && tensor->ne[1] % W4A16_N_TILE == 0 && tensor->ne[1] <= n_max &&
           tensor->ne[2] == 1 && tensor->ne[3] == 1 && ggml_is_contiguous(tensor);
}

}  // namespace

size_t ggml_cuda_w4a16_sidecar_size(int device, const ggml_tensor * tensor) {
    if (!w4a16_weight_eligible(device, tensor)) {
        return 0;
    }
    const size_t bpw = tensor->type == GGML_TYPE_Q6_K ? 9 : 5;
    return size_t(tensor->ne[0]) * size_t(tensor->ne[1]) * bpw / 8;
}

void ggml_cuda_w4a16_repack(const ggml_tensor * tensor, void * sidecar, cudaStream_t stream) {
    GGML_ASSERT(tensor->ne[0] % QK_K == 0);
    GGML_ASSERT(tensor->ne[1] % W4A16_N_TILE == 0);

    const int     k       = tensor->ne[0];
    const int     n       = tensor->ne[1];
    const int     count   = n * (k / W4A16_K_GROUP);
    constexpr int threads = 256;
    if (tensor->type == GGML_TYPE_Q4_K) {
        repack_q4_k_w4a16<<<(count + threads - 1) / threads, threads, 0, stream>>>(
            static_cast<const block_q4_K *>(tensor->data), static_cast<uint8_t *>(sidecar), k, n);
    } else if (tensor->type == GGML_TYPE_Q6_K) {
        repack_q6_k_w6a16<<<(count + threads - 1) / threads, threads, 0, stream>>>(
            static_cast<const block_q6_K *>(tensor->data), static_cast<uint8_t *>(sidecar), k, n);
    } else {
        GGML_ABORT("unsupported type for w4a16 repack");
    }
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_should_use_w4a16(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
    if (!w4a16_enabled() || cc != GGML_CUDA_CC_DGX_SPARK ||
        (src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q6_K) ||
        src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    // The int8 kernel is still faster for Q6_K with large n and short k (e.g. output logits).
    if (src0->type == GGML_TYPE_Q6_K && src0->ne[1] >= 8192 && src0->ne[0] <= 4096) {
        return false;
    }

    const int n_max = src0->type == GGML_TYPE_Q6_K ? W4A16_N_MAX_Q6 : W4A16_N_MAX;
    return src0->ne[0] % QK_K == 0 && src0->ne[1] % W4A16_N_TILE == 0 && src0->ne[1] <= n_max &&
           src1->ne[1] % 32 == 0 && src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 &&
           src1->ne[3] == 1 && dst->ne[2] == 1 && dst->ne[3] == 1 && src1->ne[0] == src0->ne[0] &&
           dst->ne[0] == src0->ne[1] && dst->ne[1] == src1->ne[1] && ggml_is_contiguous(src0) &&
           ggml_is_contiguous(src1) && ggml_is_contiguous(dst);
}

void ggml_cuda_convert_w4a16_activation(ggml_backend_cuda_context & ctx,
                                        const ggml_tensor *         src1,
                                        half *                      activation_f16) {
    const int count = ggml_nelements(src1);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && count % 4 == 0 && ggml_is_contiguous(src1));

    constexpr int convert_threads = 256;
    const int     quads           = count / 4;
    convert_f32_f16_w4a16<<<(quads + convert_threads - 1) / convert_threads, convert_threads, 0, ctx.stream()>>>(
        static_cast<const float4 *>(src1->data), reinterpret_cast<half2 *>(activation_f16), quads);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_mul_mat_w4a16(ggml_backend_cuda_context & ctx,
                             const ggml_tensor *         src0,
                             const ggml_tensor *         src1,
                             ggml_tensor *               dst,
                             const void *                sidecar,
                             const half *                activation_f16) {
    const int k = src0->ne[0];
    const int n = src0->ne[1];
    const int m = src1->ne[1];
    GGML_ASSERT(k % QK_K == 0 && n % W4A16_N_TILE == 0 && m % W4A16_M_TILE == 0);

    ggml_cuda_pool_alloc<half> activation_storage(ctx.pool());
    if (activation_f16 == nullptr) {
        activation_f16 = activation_storage.alloc(size_t(k) * m);
        ggml_cuda_convert_w4a16_activation(ctx, src1, activation_storage.get());
    }

#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(sidecar);
    GGML_ABORT("W4A16 is only implemented for CUDA");
#else
    const int W = src0->type == GGML_TYPE_Q6_K ? 6 : 4;
    // Per-shape per-type config selection (see MMQ_PERF_LOG.md for the tuning data).
    if (W == 6) {
        if (m % 64 == 0) {
            using cfg = w4a16_cfg<64, 64, 64, 3>;
            launch_w4a16_kernel<cfg>(W, sidecar, activation_f16, dst, k, n, m, ctx.stream());
        } else {
            using cfg = w4a16_cfg<32, 32, 64, 3>;
            launch_w4a16_kernel<cfg>(W, sidecar, activation_f16, dst, k, n, m, ctx.stream());
        }
    } else if (n >= 8192 && k <= 4096 && m % 128 == 0) {
        using cfg = w4a16_cfg<128, 64, 64, 3>;
        launch_w4a16_kernel<cfg>(W, sidecar, activation_f16, dst, k, n, m, ctx.stream());
    } else if (k >= 8192) {
        using cfg = w4a16_cfg<32, 32, 64, 4>;
        launch_w4a16_kernel<cfg>(W, sidecar, activation_f16, dst, k, n, m, ctx.stream());
    } else {
        using cfg = w4a16_cfg<32, 32, 64, 3>;
        launch_w4a16_kernel<cfg>(W, sidecar, activation_f16, dst, k, n, m, ctx.stream());
    }
    CUDA_CHECK(cudaGetLastError());
#endif
}
