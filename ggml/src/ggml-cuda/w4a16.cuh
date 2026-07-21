#pragma once

#include "common.cuh"

size_t ggml_cuda_w4a16_sidecar_size(int device, const ggml_tensor * tensor);

void ggml_cuda_w4a16_repack(
        const ggml_tensor * tensor, void * sidecar, cudaStream_t stream);

bool ggml_cuda_should_use_w4a16(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

void ggml_cuda_convert_w4a16_activation(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src1, half * activation_f16);

void ggml_cuda_mul_mat_w4a16(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst,
        const void * sidecar,
        const half * activation_f16);

bool ggml_cuda_w4a16_pair_compatible(
        const ggml_tensor * node, const ggml_tensor * partner, int cc);

void ggml_cuda_mul_mat_w4a16_pair(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0_a,
        const ggml_tensor * src0_b,
        const ggml_tensor * src1,
        ggml_tensor * dst_a,
        ggml_tensor * dst_b,
        const void * sidecar_a,
        const void * sidecar_b,
        const half * activation_f16);
