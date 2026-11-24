#include "custom-op.cuh"

#include "ggml-cuda.h"

#include <iterator>
#include <map>
#include <mutex>
#include <string>
#include <utility>

// the modules are loaded with the driver API, as linked for the VMM pool
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && !defined(GGML_CUDA_NO_VMM)
#define GGML_CUDA_CUSTOM_OP_AVAILABLE
#endif

#ifdef GGML_CUDA_CUSTOM_OP_AVAILABLE
static ggml_custom_op_params ggml_cuda_custom_op_params(const ggml_tensor * op) {
    ggml_custom_op_params p;
    memcpy(&p, op->op_params, sizeof(p));
    return p;
}

// the kernel that the node has for this backend, nullptr if it has none
static const ggml_custom_kernel * ggml_cuda_custom_kernel(const ggml_tensor * op) {
    const ggml_custom_kernel * k = ggml_custom_op_kernel(op, GGML_CUDA_NAME);
    return k != nullptr && k->code != nullptr && k->name != nullptr ? k : nullptr;
}

// the kernel on the device, nullptr if the module or the kernel cannot be loaded
static CUfunction ggml_cuda_custom_function(int device, const ggml_custom_kernel & k) {
    static std::mutex mutex;
    static std::map<std::pair<int, uint64_t>, CUfunction> functions;
    static std::map<std::pair<int, std::string>, CUmodule> modules;

    std::lock_guard<std::mutex> lock(mutex);
    const auto key = std::make_pair(device, k.uid);
    if (const auto it = functions.find(key); it != functions.end()) {
        return it->second;
    }

    // the kernels with the same code share its module
    const auto [it, is_new] = modules.try_emplace({ device, std::string((const char *) k.code, k.code_size) }, nullptr);
    if (is_new) {
        ggml_cuda_set_device(device);
        // the driver API loads the module into the current context: make it the one of the runtime
        size_t mem_free  = 0;
        size_t mem_total = 0;
        CUDA_CHECK(cudaMemGetInfo(&mem_free, &mem_total));
        const CUresult err = cuModuleLoadData(&it->second, k.code);
        if (err != CUDA_SUCCESS) {
            GGML_LOG_WARN("%s: device %d cannot load the module of kernel %s: %s\n", __func__, device, k.name, cu_get_error_str(err));
            it->second = nullptr;
        }
    }

    CUfunction fun = nullptr;
    if (it->second != nullptr) {
        const CUresult err = cuModuleGetFunction(&fun, it->second, k.name);
        if (err != CUDA_SUCCESS) {
            GGML_LOG_WARN("%s: kernel %s is not in its module: %s\n", __func__, k.name, cu_get_error_str(err));
            fun = nullptr;
        }
    }
    functions[key] = fun;
    return fun;
}

// the kernel takes the data of the node, the data of each argument, then the params
static bool ggml_cuda_custom_function_matches(CUfunction fun, const ggml_tensor * op, const ggml_custom_op_params & p) {
#if CUDA_VERSION >= 12040
    size_t n_args = 1;
    for (int i = 0; i < GGML_MAX_SRC && op->src[i] != nullptr; ++i) {
        n_args++;
    }
    size_t offs = 0;
    size_t size = 0;
    for (size_t i = 0; i < n_args; ++i) {
        if (cuFuncGetParamInfo(fun, i, &offs, &size) != CUDA_SUCCESS || size != sizeof(void *)) {
            return false;
        }
    }
    if (p.params_size > 0 && (cuFuncGetParamInfo(fun, n_args, &offs, &size) != CUDA_SUCCESS || size != p.params_size)) {
        return false;
    }
    return cuFuncGetParamInfo(fun, n_args + (p.params_size > 0), &offs, &size) != CUDA_SUCCESS;
#else
    GGML_UNUSED_VARS(fun, op, p);
    return true;
#endif // CUDA_VERSION >= 12040
}
#endif // GGML_CUDA_CUSTOM_OP_AVAILABLE

bool ggml_cuda_custom_op_supported(int device, const ggml_tensor * op) {
#ifdef GGML_CUDA_CUSTOM_OP_AVAILABLE
    const ggml_custom_op_params p = ggml_cuda_custom_op_params(op);
    const ggml_custom_kernel * k = ggml_cuda_custom_kernel(op);
    if (k == nullptr || !ggml_is_contiguous(op)) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC && op->src[i] != nullptr; ++i) {
        if (!ggml_is_contiguous(op->src[i])) {
            return false;
        }
    }
    const CUfunction fun = ggml_cuda_custom_function(device, *k);
    if (fun == nullptr) {
        return false;
    }
    if (!ggml_cuda_custom_function_matches(fun, op, p)) {
        GGML_LOG_WARN("%s: kernel %s of %s must take the data of the node, the data of each argument, then the params\n", __func__, k->name, p.op->name);
        return false;
    }
    return true;
#else
    GGML_UNUSED_VARS(device, op);
    return false;
#endif // GGML_CUDA_CUSTOM_OP_AVAILABLE
}

int ggml_cuda_custom_op_async_until(const ggml_cgraph * cgraph, int i) {
#ifdef GGML_CUDA_CUSTOM_OP_AVAILABLE
    const ggml_tensor * node = cgraph->nodes[i];
    const ggml_custom_kernel * k = ggml_cuda_custom_kernel(node);
    if (k == nullptr || !ggml_custom_op_sched(node, k).async) {
        return -1;
    }
    for (int j = i + 1; j < cgraph->n_nodes; ++j) {
        for (int s = 0; s < GGML_MAX_SRC && cgraph->nodes[j]->src[s] != nullptr; ++s) {
            if (cgraph->nodes[j]->src[s] == node) {
                return ggml_cuda_custom_kernel(cgraph->nodes[j]) != nullptr ? j : -1;
            }
        }
    }
    return -1;
#else
    GGML_UNUSED_VARS(cgraph, i);
    return -1;
#endif // GGML_CUDA_CUSTOM_OP_AVAILABLE
}

#ifdef GGML_CUDA_CUSTOM_OP_AVAILABLE
static cudaEvent_t ggml_cuda_custom_op_event(cudaEvent_t & event) {
    if (event == nullptr) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
    return event;
}

// side stream i of the async custom ops
static int ggml_cuda_custom_op_side(int i) {
    return GGML_CUDA_MAX_STREAMS - 1 - i;
}
#endif // GGML_CUDA_CUSTOM_OP_AVAILABLE

void ggml_cuda_custom_op_fork(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
#ifdef GGML_CUDA_CUSTOM_OP_AVAILABLE
    if (cgraph->nodes[i]->op != GGML_OP_CUSTOM || ggml_cuda_custom_op_async_until(cgraph, i) < 0) {
        return;
    }
    for (int s = 0; s < (int) std::size(ctx.custom_op_async); ++s) {
        if (ctx.custom_op_async[s] != nullptr) {
            continue;
        }
        // the side stream goes on after the nodes before this one
        const cudaStream_t main = ctx.stream();
        const cudaStream_t side = ctx.stream(ctx.device, ggml_cuda_custom_op_side(s));
        const cudaEvent_t  fork = ggml_cuda_custom_op_event(ctx.custom_op_events[0]);
        CUDA_CHECK(cudaEventRecord(fork, main));
        CUDA_CHECK(cudaStreamWaitEvent(side, fork, 0));
        ctx.custom_op_async[s] = cgraph->nodes[i];
        ctx.curr_stream_no     = ggml_cuda_custom_op_side(s);
        return;
    }
    // no side stream is free: on the stream of the graph, as without the hint
#else
    GGML_UNUSED_VARS(ctx, cgraph, i);
#endif // GGML_CUDA_CUSTOM_OP_AVAILABLE
}

void ggml_cuda_custom_op_join(ggml_backend_cuda_context & ctx, const ggml_tensor * node) {
#ifdef GGML_CUDA_CUSTOM_OP_AVAILABLE
    for (int s = 0; s < (int) std::size(ctx.custom_op_async); ++s) {
        if (ctx.custom_op_async[s] == nullptr) {
            continue;
        }
        for (int j = 0; j < GGML_MAX_SRC && node->src[j] != nullptr; ++j) {
            if (node->src[j] != ctx.custom_op_async[s]) {
                continue;
            }
            const cudaEvent_t join = ggml_cuda_custom_op_event(ctx.custom_op_events[1 + s]);
            CUDA_CHECK(cudaEventRecord(join, ctx.stream(ctx.device, ggml_cuda_custom_op_side(s))));
            CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), join, 0));
            ctx.custom_op_async[s] = nullptr;
            break;
        }
    }
#else
    GGML_UNUSED_VARS(ctx, node);
#endif // GGML_CUDA_CUSTOM_OP_AVAILABLE
}

bool ggml_cuda_custom_op_async_open(const ggml_backend_cuda_context & ctx) {
    for (const ggml_tensor * node : ctx.custom_op_async) {
        if (node != nullptr) {
            return true;
        }
    }
    return false;
}

void ggml_cuda_custom_op_graph_props(ggml_tensor * node) {
    if (node->op != GGML_OP_CUSTOM) {
        return;
    }
    ggml_custom_op_params p;
    memcpy(&p, node->op_params, sizeof(p));
    if (p.op == nullptr) {
        return;
    }
    uint64_t hash = 0xcbf29ce484222325ull; // FNV-1a
    const auto add = [&hash](const void * data, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            hash = (hash ^ ((const uint8_t *) data)[i])*0x100000001b3ull;
        }
    };
    add(p.userdata, p.params_size);
    if (const ggml_custom_kernel * k = ggml_custom_op_kernel(node, GGML_CUDA_NAME); k != nullptr) {
        add(&k->uid, sizeof(k->uid));
    }
    p.userdata = (void *) (uintptr_t) hash;
    memcpy(node->op_params, &p, sizeof(p));
}

void ggml_cuda_op_custom(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#ifdef GGML_CUDA_CUSTOM_OP_AVAILABLE
    const ggml_custom_op_params p = ggml_cuda_custom_op_params(dst);
    const ggml_custom_kernel * k = ggml_cuda_custom_kernel(dst);
    const CUfunction fun = ggml_cuda_custom_function(ctx.device, *k);
    GGML_ASSERT(fun != nullptr);

    // the kernel tells its grid and its blocks
    const ggml_custom_sched sched = ggml_custom_op_sched(dst, k);

    void * args[GGML_MAX_SRC + 2];
    int n_args = 0;
    args[n_args++] = &dst->data;
    for (int i = 0; i < GGML_MAX_SRC && dst->src[i] != nullptr; ++i) {
        args[n_args++] = &dst->src[i]->data;
    }
    if (p.params_size > 0) {
        args[n_args++] = p.userdata;
    }

    CU_CHECK(cuLaunchKernel(fun, sched.n_groups[0], sched.n_groups[1], sched.n_groups[2], sched.group_size[0], sched.group_size[1], sched.group_size[2],
        0, (CUstream) ctx.stream(), args, nullptr));
#else
    GGML_UNUSED_VARS(ctx, dst); // not reached, see ggml_cuda_custom_op_supported
#endif // GGML_CUDA_CUSTOM_OP_AVAILABLE
}
