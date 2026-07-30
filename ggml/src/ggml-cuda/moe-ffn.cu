#include "moe-ffn.cuh"
#include "mmid.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "topk-moe.cuh"

static __global__ void moe_ffn_iota(int32_t * dst, const int n) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = i;
    }
}

// ids_dst maps a sorted slot to the flat weight index it*n_expert_used + iex, which is also
// the index into the weights tensor; split it into the token row and its routing weight
static __global__ void moe_ffn_sorted_scales(
        const int32_t * __restrict__ ids_dst, const float * __restrict__ weights,
        int32_t * __restrict__ ids_token, float * __restrict__ scales_sorted,
        const int n, const int n_expert_used) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int slot = ids_dst[i];
    ids_token[i]     = slot/n_expert_used;
    scales_sorted[i] = weights[slot];
}

void ggml_cuda_moe_ffn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x         = dst->src[0];
    const ggml_tensor * gate_inp  = dst->src[1];
    const ggml_tensor * up_exps   = dst->src[2];
    const ggml_tensor * gate_exps = dst->src[3];
    const ggml_tensor * down_exps = dst->src[4];

    const bool merged = gate_exps == nullptr;

    GGML_ASSERT(x->type        == GGML_TYPE_F32);
    GGML_ASSERT(gate_inp->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type      == GGML_TYPE_F32);
    GGML_ASSERT(merged || up_exps->type == gate_exps->type);
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(gate_inp));
    GGML_ASSERT(ggml_is_contiguous(up_exps));
    GGML_ASSERT(merged || ggml_is_contiguous(gate_exps));
    GGML_ASSERT(ggml_is_contiguous(down_exps));

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t n_expert = gate_inp->ne[1];
    const int64_t n_ff     = merged ? up_exps->ne[1]/2 : up_exps->ne[1];

    const int n_expert_used = ggml_get_op_params_i32(dst, 0);

    const int64_t ne_sorted = n_tokens*n_expert_used;

    if (n_tokens == 0) {
        return;
    }

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const ggml_type type_up   = up_exps->type;
    const ggml_type type_down = down_exps->type;

    // router logits = gate_inp^T @ x -> [n_expert, n_tokens]
    ggml_cuda_pool_alloc<float> logits(ctx.pool(), n_expert*n_tokens);
    {
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), stream));
        // full fp32: with TF32 the top-k selection can flip between nearly-tied experts
        CUBLAS_CHECK(cublasSetMathMode(ctx.cublas_handle(), CUBLAS_PEDANTIC_MATH));
        CUBLAS_CHECK(cublasSgemm(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
            n_expert, n_tokens, n_embd,
            &alpha, (const float *) gate_inp->data, gate_inp->nb[1]/sizeof(float),
                    (const float *) x->data,        x->nb[1]/sizeof(float),
            &beta,  logits.get(), n_expert));
        CUBLAS_CHECK(cublasSetMathMode(ctx.cublas_handle(), CUBLAS_TF32_TENSOR_OP_MATH));
    }

    // fused softmax + top-k + weight normalization; ids are written with a row stride of n_expert
    ggml_cuda_pool_alloc<float>   weights(ctx.pool(), ne_sorted);
    ggml_cuda_pool_alloc<int32_t> ids(ctx.pool(), n_expert*n_tokens);

    // matches the clamp in llm_graph_context::build_moe_ffn
    ggml_cuda_topk_moe_softmax_norm(ctx, logits.get(), weights.get(), ids.get(), n_tokens, n_expert, n_expert_used, 6.103515625e-5f);

    // expert-sorted row mapping, shared by all three GEMMs
    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_sorted);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_sorted);
    ggml_cuda_pool_alloc<int32_t> ids_iota(ctx.pool(), ne_sorted);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), n_expert + 1);

    ggml_cuda_launch_mm_ids_helper(ids.get(), ids_src1.get(), ids_dst.get(), expert_bounds.get(),
        n_expert, n_tokens, n_expert_used, /*nchannels_y=*/1, /*si1=*/n_expert, /*sis1=*/1, stream);
    CUDA_CHECK(cudaGetLastError());

    {
        const int block = 256;
        moe_ffn_iota<<<(ne_sorted + block - 1)/block, block, 0, stream>>>(ids_iota.get(), ne_sorted);
        CUDA_CHECK(cudaGetLastError());
    }

    // quantize x once in expert-sorted order, shared by the up and gate GEMM
    const int64_t n_embd_padded = GGML_PAD(n_embd, MATRIX_ROW_PADDING);
    const int64_t n_ff_padded   = GGML_PAD(n_ff,   MATRIX_ROW_PADDING);

    // gate and up outputs always share one buffer with gate first, so the swiglu-quantize
    // reads both halves with a single row stride regardless of how the weights are stored
    const int64_t stride_upgate = 2*n_ff;

    const bool fb_upgate = n_ff   % 128 != 0;
    const bool fb_down   = n_embd % 128 != 0;

    const size_t nbytes_x_q = ne_sorted*n_embd_padded*sizeof(block_q8_1_mmq)/QK8_1_MMQ +
        ggml_cuda_mmq_get_J_max(type_up, fb_upgate, cc, n_tokens)*sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> x_q(ctx.pool(), nbytes_x_q);

    quantize_mmq_q8_1_cuda((const float *) x->data, ids_src1.get(), x_q.get(), type_up,
        n_embd, x->nb[1]/sizeof(float), x->nb[1]/sizeof(float), n_embd*n_tokens,
        n_embd_padded, ne_sorted, 1, 1, stream);
    CUDA_CHECK(cudaGetLastError());

    ggml_cuda_pool_alloc<int32_t> ids_token(ctx.pool(), ne_sorted);
    ggml_cuda_pool_alloc<float>   scales_sorted(ctx.pool(), ne_sorted);
    {
        const int block = 256;
        moe_ffn_sorted_scales<<<(ne_sorted + block - 1)/block, block, 0, stream>>>(
            ids_dst.get(), weights.get(), ids_token.get(), scales_sorted.get(), ne_sorted, n_expert_used);
        CUDA_CHECK(cudaGetLastError());
    }

    ggml_cuda_pool_alloc<float> upgate_s(ctx.pool(), stride_upgate*ne_sorted);

    // strides of the quantized activations; unused by the kernel when ids_dst is present
    const int64_t s12_x_q = n_embd_padded*sizeof(block_q8_1)/(QK8_1*sizeof(int));
    const int64_t s13_x_q = n_tokens*s12_x_q;

    const size_t ts_upgate = ggml_type_size(type_up);

    // up/gate GEMM outputs stay in expert-sorted order (identity ids_dst)
    auto launch_upgate = [&](const ggml_tensor * exps, float * out, int64_t nrows) {
        const mmq_args args = {
            (const char *) exps->data, type_up, (const int *) x_q.get(), ids_iota.get(), expert_bounds.get(), out,
            n_embd, nrows, ne_sorted, (int64_t)(exps->nb[1]/ts_upgate), ne_sorted, stride_upgate,
            n_expert, n_expert, (int64_t)(exps->nb[2]/ts_upgate), s12_x_q, 0,
            1, 1, (int64_t)(exps->nb[2]/ts_upgate)*n_expert, s13_x_q, 0,
            n_tokens, /*col_scales =*/ nullptr};
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
    };

    if (merged) {
        launch_upgate(up_exps, upgate_s.get(), stride_upgate);
    } else {
        launch_upgate(gate_exps, upgate_s.get(),        n_ff);
        launch_upgate(up_exps,   upgate_s.get() + n_ff, n_ff);
    }

    const size_t nbytes_act_q = ne_sorted*n_ff_padded*sizeof(block_q8_1_mmq)/QK8_1_MMQ +
        ggml_cuda_mmq_get_J_max(type_down, fb_down, cc, n_tokens)*sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> act_q(ctx.pool(), nbytes_act_q);

    // silu(gate)*up is applied while quantizing, the product is never written out
    quantize_mmq_q8_1_swiglu_cuda(upgate_s.get() + n_ff, upgate_s.get(), act_q.get(), type_down,
        n_ff, stride_upgate, n_ff_padded, ne_sorted, stream);
    CUDA_CHECK(cudaGetLastError());

    // the down GEMM accumulates weight*result directly into the token rows of dst
    CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), stream));

    const int64_t s12_act_q = n_ff_padded*sizeof(block_q8_1)/(QK8_1*sizeof(int));
    const int64_t s13_act_q = n_tokens*s12_act_q;

    const size_t ts_down = ggml_type_size(type_down);

    const mmq_args args_down = {
        (const char *) down_exps->data, type_down, (const int *) act_q.get(), ids_token.get(), expert_bounds.get(), (float *) dst->data,
        n_ff, n_embd, ne_sorted, (int64_t)(down_exps->nb[1]/ts_down), ne_sorted, n_embd,
        n_expert, n_expert, (int64_t)(down_exps->nb[2]/ts_down), s12_act_q, 0,
        1, 1, (int64_t)(down_exps->nb[2]/ts_down)*n_expert, s13_act_q, 0,
        n_tokens, scales_sorted.get()};
    ggml_cuda_mul_mat_q_switch_type(ctx, args_down, stream);
}
