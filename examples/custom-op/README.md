# custom-op

A custom operator with implementations for several backends: `saxpy`, `dst = a*x + y`.

The op is described once (see `ggml_custom_op_init()` in `ggml.h`): a list of kernels, one for each backend that can run it, by the name of the backend.

- `CPU`: a function, as for `ggml_custom_4d()`
- `CUDA`: PTX, here compiled from [custom-op.cu](custom-op.cu) with `nvcc -ptx`
- `MTL`: the Metal source of the kernel
- `Vulkan`: SPIR-V, here compiled from [custom-op.comp](custom-op.comp) with `glslc`

Each kernel also has a callback that tells how it runs a node, from the node and its arguments: the number of threads for the CPU, the number and size of the workgroups for the others. A Vulkan shader gets the size of its workgroup in specialization constants (`local_size_x_id = 0`).

A node is made with `ggml_custom_op_4d()`. It gets the arguments and a small params struct (copied, every kernel receives it).

The scheduler gives the node to the first backend that can run it. A backend without a kernel leaves it to the CPU.

The example computes the op with the best backend, then with the CPU only, and compares the results:

```
$ ./custom-op
saxpy computed by CUDA0:
[ 12.00 24.00 36.00 48.00 60.00 72.00 84.00 96.00 ]
saxpy computed by CPU:
[ 12.00 24.00 36.00 48.00 60.00 72.00 84.00 96.00 ]
```
