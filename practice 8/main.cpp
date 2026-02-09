#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <cuda_runtime.h>

#define N 1000000
#define THREADS_PER_BLOCK 256

// ================= GPU kernel =================
__global__ void multiply_gpu(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= 2.0f;
    }
}

// ================= CPU function =================
void multiply_cpu(float* data, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        data[i] *= 2.0f;
    }
}

// ================= MAIN =================
int main() {
    std::vector<float> data(N, 1.0f);

    // -------- CPU --------
    auto start_cpu = std::chrono::high_resolution_clock::now();
    multiply_cpu(data.data(), N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::cout << "CPU time: "
              << std::chrono::duration<double>(end_cpu - start_cpu).count()
              << " sec\n";

    // -------- GPU --------
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemcpy(d_data, data.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    multiply_gpu<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
                   THREADS_PER_BLOCK>>>(d_data, N);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();

    cudaMemcpy(data.data(), d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    std::cout << "GPU time: "
              << std::chrono::duration<double>(end_gpu - start_gpu).count()
              << " sec\n";

    // -------- HYBRID --------
    float* d_half;
    int half = N / 2;
    cudaMalloc(&d_half, half * sizeof(float));
    cudaMemcpy(d_half, data.data() + half, half * sizeof(float),
               cudaMemcpyHostToDevice);

    auto start_hybrid = std::chrono::high_resolution_clock::now();

    #pragma omp parallel sections
    {
        #pragma omp section
        multiply_cpu(data.data(), half);

        #pragma omp section
        {
            multiply_gpu<<<(half + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
                           THREADS_PER_BLOCK>>>(d_half, half);
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(data.data() + half, d_half,
               half * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_half);

    auto end_hybrid = std::chrono::high_resolution_clock::now();
    std::cout << "Hybrid time: "
              << std::chrono::duration<double>(end_hybrid - start_hybrid).count()
              << " sec\n";

    return 0;
}
