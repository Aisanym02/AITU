#include <cuda_runtime.h>
#include <iostream>

__global__ void reduce_sum(float* input, float* output, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

__global__ void prefix_scan(float* input, float* output, int n) {
    extern __shared__ float temp[];
    int tid = threadIdx.x;

    temp[tid] = input[tid];
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        float val = 0;
        if (tid >= offset)
            val = temp[tid - offset];
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }

    output[tid] = temp[tid];
}

float cpu_sum(float* data, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++)
        sum += data[i];
    return sum;
}
