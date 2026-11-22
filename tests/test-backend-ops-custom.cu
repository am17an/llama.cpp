// the CUDA kernel of test_custom_op in test-backend-ops.cpp, CMake compiles it to PTX (nvcc -ptx)

struct test_custom_op_params {
    float a;
    int   n;
};

extern "C" __global__ void test_custom_op(float * dst, const float * x, const float * y, const test_custom_op_params p) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < p.n) {
        dst[i] = p.a*x[i] + y[i];
    }
}
