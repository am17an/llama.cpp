#include "ggml-vulkan-common.h"

#include <cstring>

// kernels of the user for GGML_OP_CUSTOM nodes, see ggml_custom_op_init in ggml.h
//
// The code of a kernel is SPIR-V, its name is the entry point of the compute shader:
// - set 0, binding 0: the data of the node, binding 1 + i: the data of argument i (storage buffers)
// - the push constants: the params
// - the specialization constants 0, 1 and 2: the size of a workgroup that the kernel tells (local_size_x_id = 0, ...)

// every device has push constants of this size
#define GGML_VK_CUSTOM_PARAMS_SIZE_MAX 128

struct vk_op_custom_push_constants {
    uint8_t params[GGML_VK_CUSTOM_PARAMS_SIZE_MAX]; // the params, then zeros
};

static_assert(GGML_MAX_SRC + 1 <= MAX_PARAMETER_COUNT, "a binding for the node and each argument");

static ggml_custom_op_params ggml_vk_custom_op_params(const ggml_tensor * op) {
    ggml_custom_op_params p;
    memcpy(&p, op->op_params, sizeof(p));
    return p;
}

// the kernel that the node has for this backend, nullptr if it has none
static const ggml_custom_kernel * ggml_vk_custom_kernel(const ggml_tensor * op) {
    const ggml_custom_kernel * k = ggml_custom_op_kernel(op, GGML_VK_NAME);
    return k != nullptr && k->code != nullptr && k->name != nullptr ? k : nullptr;
}

// the code is SPIR-V with a compute shader that has the name of the kernel
static bool ggml_vk_custom_has_entry_point(const ggml_custom_kernel & k) {
    const uint32_t * words   = (const uint32_t *) k.code;
    const size_t     n_words = k.code_size/sizeof(uint32_t);
    if (k.code_size % sizeof(uint32_t) != 0 || n_words < 5 || words[0] != spv::MagicNumber) {
        return false;
    }
    for (size_t pos = 5; pos < n_words;) {
        const uint32_t opcode = words[pos] & spv::OpCodeMask;
        const uint32_t len    = words[pos] >> spv::WordCountShift;
        if (len == 0 || pos + len > n_words) {
            return false;
        }
        if (opcode == spv::OpFunction) {
            break; // the entry points are before the functions
        }
        // OpEntryPoint: execution model, id of the function, name
        if (opcode == spv::OpEntryPoint && len > 3 && words[pos + 1] == spv::ExecutionModelGLCompute) {
            const char * name     = (const char *) &words[pos + 3];
            const size_t name_max = (len - 3)*sizeof(uint32_t);
            if (strnlen(name, name_max) < name_max && strcmp(name, k.name) == 0) {
                return true;
            }
        }
        pos += len;
    }
    return false;
}

// the pipeline of the kernel for a size of workgroup, nullptr if it cannot be made
static vk_pipeline ggml_vk_custom_pipeline(vk_device & device, const ggml_custom_kernel & k, const ggml_custom_sched & sched) {
    std::lock_guard<std::recursive_mutex> guard(device->mutex);

    const std::array<uint32_t, 3> group_size = { (uint32_t) sched.group_size[0], (uint32_t) sched.group_size[1], (uint32_t) sched.group_size[2] };

    const auto key = std::make_pair(k.uid, group_size);
    if (const auto it = device->pipeline_custom.find(key); it != device->pipeline_custom.end()) {
        return it->second;
    }

    // the kernels with the same code share its pipelines
    const auto [it, is_new] = device->pipeline_custom_code.try_emplace(
        std::make_tuple(std::string((const char *) k.code, k.code_size), std::string(k.name), group_size), nullptr);
    if (is_new && !ggml_vk_custom_has_entry_point(k)) {
        GGML_LOG_WARN("%s: the code of kernel %s is not SPIR-V with a compute shader of this name\n", __func__, k.name);
    } else if (is_new) {
        vk_pipeline pipeline = std::make_shared<vk_pipeline_struct>();
        pipeline->name               = std::string("custom_") + k.name;
        pipeline->parameter_count    = MAX_PARAMETER_COUNT;
        pipeline->push_constant_size = sizeof(vk_op_custom_push_constants);
        pipeline->wg_denoms          = { 1, 1, 1 };
        pipeline->align              = 1;
        pipeline->initialized        = true;
        try {
            ggml_vk_create_pipeline_func(device, pipeline, k.code_size, k.code, k.name, pipeline->parameter_count, pipeline->wg_denoms,
                { group_size[0], group_size[1], group_size[2] }, false, false, 0);
            it->second = pipeline;
        } catch (const vk::SystemError & e) {
            GGML_LOG_WARN("%s: device %s cannot make the pipeline of kernel %s: %s\n", __func__, device->name.c_str(), k.name, e.what());
            device->device.destroyPipelineLayout(pipeline->layout);
            device->device.destroyShaderModule(pipeline->shader_module);
        }
    }

    device->pipeline_custom[key] = it->second;
    return it->second;
}

bool ggml_vk_custom_op_supported(const vk_device & device, const ggml_tensor * op) {
    const ggml_custom_op_params p = ggml_vk_custom_op_params(op);
    const ggml_custom_kernel * k = ggml_vk_custom_kernel(op);
    if (k == nullptr || p.params_size > GGML_VK_CUSTOM_PARAMS_SIZE_MAX) {
        return false;
    }

    // the kernel indexes each tensor from 0: contiguous, at an offset that a binding can have
    const auto & limits = device->properties.limits;
    const auto bindable = [&limits](const ggml_tensor * t) {
        return ggml_is_contiguous(t) && t->view_offs % limits.minStorageBufferOffsetAlignment == 0;
    };
    if (!bindable(op)) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC && op->src[i] != nullptr; ++i) {
        if (!bindable(op->src[i])) {
            return false;
        }
    }

    // the workgroups that the kernel tells
    const ggml_custom_sched sched = ggml_custom_op_sched(op, k);
    uint64_t group_threads = 1;
    for (int i = 0; i < 3; ++i) {
        if ((uint32_t) sched.n_groups[i] > limits.maxComputeWorkGroupCount[i] || (uint32_t) sched.group_size[i] > limits.maxComputeWorkGroupSize[i]) {
            return false;
        }
        group_threads *= sched.group_size[i];
    }
    if (group_threads > limits.maxComputeWorkGroupInvocations) {
        return false;
    }

    vk_device dev = device;
    return ggml_vk_custom_pipeline(dev, *k, sched) != nullptr;
}

void ggml_vk_custom_op(ggml_backend_vk_context * ctx, vk_context & subctx, ggml_tensor * dst) {
    const ggml_custom_op_params p = ggml_vk_custom_op_params(dst);
    const ggml_custom_kernel * k = ggml_vk_custom_kernel(dst);
    const ggml_custom_sched sched = ggml_custom_op_sched(dst, k);

    vk_pipeline pipeline = ggml_vk_custom_pipeline(ctx->device, *k, sched);
    GGML_ASSERT(pipeline != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    // the node, then its arguments. The bindings that are left get the node
    vk_subbuffer bufs[MAX_PARAMETER_COUNT];
    uint32_t n_bufs = 0;
    bufs[n_bufs++] = ggml_vk_tensor_subbuffer(ctx, dst);
    for (int i = 0; i < GGML_MAX_SRC && dst->src[i] != nullptr; ++i) {
        bufs[n_bufs++] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }
    for (; n_bufs < MAX_PARAMETER_COUNT; ++n_bufs) {
        bufs[n_bufs] = bufs[0];
    }

    vk_op_custom_push_constants pc = {};
    if (p.params_size > 0) {
        memcpy(pc.params, p.userdata, p.params_size);
    }

    static_assert(MAX_PARAMETER_COUNT == 12);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        { bufs[0], bufs[1], bufs[2], bufs[3], bufs[4], bufs[5], bufs[6], bufs[7], bufs[8], bufs[9], bufs[10], bufs[11] }, pc,
        { (uint32_t) sched.n_groups[0], (uint32_t) sched.n_groups[1], (uint32_t) sched.n_groups[2] });
}
