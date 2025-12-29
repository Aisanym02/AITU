#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

int rand_int(int l, int r) {
    static random_device rd;
    static mt19937 gen(rd());
    uniform_int_distribution<int> dist(l, r);
    return dist(gen);
}

bool is_sorted_ok(const vector<int>& a) {
    for (size_t i = 1; i < a.size(); ++i)
        if (a[i-1] > a[i]) return false;
    return true;
}

// ===== Kernel 1: сортировка подмассива внутри блока (bitonic-like bubble внутри shared) =====
// Для учебной работы: простой O(k^2) внутри блока (на подмассиве блока).
// Подмассив блока: [blockIdx.x * blockSize .. +blockSize)
__global__ void block_sort_kernel(int* d, int n, int blockSize) {
    extern __shared__ int s[];

    int blockStart = blockIdx.x * blockSize;
    int tid = threadIdx.x;

    // загрузка в shared
    int idx = blockStart + tid;
    if (tid < blockSize) {
        s[tid] = (idx < n) ? d[idx] : INT_MAX; // хвост заполняем INT_MAX
    }
    __syncthreads();

    // простая сортировка пузырьком в shared (учебно, не самая быстрая)
    for (int i = 0; i < blockSize; ++i) {
        for (int j = tid; j < blockSize - 1; j += blockDim.x) {
            if (s[j] > s[j+1]) {
                int tmp = s[j]; s[j] = s[j+1]; s[j+1] = tmp;
            }
        }
        __syncthreads();
    }

    // выгрузка обратно
    if (idx < n && tid < blockSize) d[idx] = s[tid];
}

// ===== Kernel 2: слияние пар отсортированных отрезков =====
// merge size = width. Сливаем [left..left+width) и [left+width..left+2*width)
__global__ void merge_pass_kernel(const int* src, int* dst, int n, int width) {
    int pairStart = (blockIdx.x * blockDim.x + threadIdx.x) * (2 * width);
    if (pairStart >= n) return;

    int left = pairStart;
    int mid  = min(left + width, n);
    int right= min(left + 2*width, n);

    int i = left, j = mid, k = left;

    while (i < mid && j < right) {
        if (src[i] <= src[j]) dst[k++] = src[i++];
        else dst[k++] = src[j++];
    }
    while (i < mid)   dst[k++] = src[i++];
    while (j < right) dst[k++] = src[j++];
}

// GPU merge sort (host function)
void gpu_merge_sort(vector<int>& a, int blockSize) {
    int n = (int)a.size();
    int *d1 = nullptr, *d2 = nullptr;

    cudaMalloc(&d1, n * sizeof(int));
    cudaMalloc(&d2, n * sizeof(int));
    cudaMemcpy(d1, a.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    // 1) сортировка каждого блока
    int numBlocks = (n + blockSize - 1) / blockSize;
    int threads = blockSize; // условие: blockSize <= 1024
    size_t shmem = blockSize * sizeof(int);

    block_sort_kernel<<<numBlocks, threads, shmem>>>(d1, n, blockSize);
    cudaDeviceSynchronize();

    // 2) итеративные merge-pass: width = blockSize, 2*blockSize, 4*...
    int width = blockSize;
    bool toggle = false;

    while (width < n) {
        int pairs = (n + (2*width) - 1) / (2*width);
        int tpb = 128;
        int bpg = (pairs + tpb - 1) / tpb;

        const int* src = toggle ? d2 : d1;
        int* dst       = toggle ? d1 : d2;

        merge_pass_kernel<<<bpg, tpb>>>(src, dst, n, width);
        cudaDeviceSynchronize();

        toggle = !toggle;
        width *= 2;
    }

    // копируем результат
    const int* finalDev = toggle ? d2 : d1;
    cudaMemcpy(a.data(), finalDev, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d1);
    cudaFree(d2);
}

long long time_gpu(vector<int>& a, int blockSize) {
    auto t1 = chrono::high_resolution_clock::now();
    gpu_merge_sort(a, blockSize);
    auto t2 = chrono::high_resolution_clock::now();
    return chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
}

int main() {
    vector<int> sizes = {10000, 100000};
    int blockSize = 256; // подблок сортировки (<=1024)

    for (int n : sizes) {
        vector<int> a(n);
        for (int i = 0; i < n; ++i) a[i] = rand_int(1, 1'000'000);

        long long ms = time_gpu(a, blockSize);
        cout << "N=" << n << " GPU time=" << ms << " ms, sorted=" << (is_sorted_ok(a) ? "OK" : "BAD") << "\n";
    }
    return 0;
}
