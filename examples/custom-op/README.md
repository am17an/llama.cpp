# custom-op

A custom operator with implementations for several backends: `saxpy`, `dst = a*x + y`.

The op is described once, with `struct ggml_custom_op` (see `ggml.h`):

- a CPU function, as for `ggml_custom_4d()`
- one kernel for each backend that can run the op, by the name of the backend:
  - `CUDA`: PTX, here compiled from [custom-op.cu](custom-op.cu) with `nvcc -ptx`
  - `MTL`: the Metal source of the kernel

A node is made with `ggml_custom_op_4d()`. It gets the arguments, a small params struct (copied, every implementation receives it) and how to run it: the number of CPU threads, and the number and size of the workgroups for the kernels.

The scheduler gives the node to the first backend that can run it. A backend without a kernel leaves it to the CPU.

The example computes the op with the best backend, then with the CPU only, and compares the results:

```
$ ./custom-op
saxpy computed by CUDA0:
[ 12.00 24.00 36.00 48.00 60.00 72.00 84.00 96.00 ]
saxpy computed by CPU:
[ 12.00 24.00 36.00 48.00 60.00 72.00 84.00 96.00 ]
```
