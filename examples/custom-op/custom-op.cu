// the CUDA kernel of the saxpy op in custom-op.cpp, CMake compiles it to PTX (nvcc -ptx)

struct saxpy_params {
    float a;
    int   n;
};

// the data of the node, the data of its arguments, then the params
extern "C" __global__ void saxpy(float * dst, const float * x, const float * y, const saxpy_params p) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < p.n) {
        dst[i] = p.a*x[i] + y[i];
    }
}
