#pragma once

#include "common.cuh"

// kernels of the user for GGML_OP_CUSTOM nodes, see ggml_custom_op_init in ggml.h

// op has a kernel for this backend and the device can load it
bool ggml_cuda_custom_op_supported(int device, const ggml_tensor * op);

void ggml_cuda_op_custom(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// a CUDA graph captures the kernel of a custom op and its params: in the node properties of the graph (node is a copy)
// the address of the params becomes a hash of their bytes and of the uid of the kernel, so that a change of either updates the graph
void ggml_cuda_custom_op_graph_props(ggml_tensor * node);
