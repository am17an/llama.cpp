#pragma once

#include "common.cuh"

// kernels of the user for GGML_OP_CUSTOM nodes, see ggml_custom_op in ggml.h

// op has a kernel for this backend and the device can load it
bool ggml_cuda_custom_op_supported(int device, const ggml_tensor * op);

void ggml_cuda_op_custom(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// a CUDA graph captures the params of a custom op with its kernel: in the node properties of the graph (node is a copy)
// the address of the params becomes a hash of their bytes, so that a change of the params updates the graph
void ggml_cuda_custom_op_graph_props(ggml_tensor * node);
