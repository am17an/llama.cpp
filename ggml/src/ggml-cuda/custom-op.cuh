#pragma once

#include "common.cuh"

// kernels of the user for GGML_OP_CUSTOM nodes, see ggml_custom_op_init in ggml.h

// op has a kernel for this backend and the device can load it
bool ggml_cuda_custom_op_supported(int device, const ggml_tensor * op);

void ggml_cuda_op_custom(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// async custom ops (see ggml_custom_sched_set_async): the kernel runs on a side stream, next to the nodes after it,
// until the first node that takes it as an argument

// the node that node i of the graph runs until, -1 if node i is not async here: it does not ask for it, or the first node that takes it
// is not a custom op with a kernel for this backend (other nodes can be fused with the ones before them, see ggml_cuda_try_fuse)
int ggml_cuda_custom_op_async_until(const ggml_cgraph * cgraph, int i);

// before node i runs: if it is async and a side stream is free, makes that side stream the current one. The caller restores it after the node
void ggml_cuda_custom_op_fork(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);

// before a node runs: the stream of the graph waits for the async nodes that the node takes
void ggml_cuda_custom_op_join(ggml_backend_cuda_context & ctx, const ggml_tensor * node);

// an async node did not join
bool ggml_cuda_custom_op_async_open(const ggml_backend_cuda_context & ctx);

// a CUDA graph captures the kernel of a custom op and its params: in the node properties of the graph (node is a copy)
// the address of the params becomes a hash of their bytes and of the uid of the kernel, so that a change of either updates the graph
void ggml_cuda_custom_op_graph_props(ggml_tensor * node);
