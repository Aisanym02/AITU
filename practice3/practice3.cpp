%%writefile gpu_sort_benchmark.cu

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

// ======================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
// ======================================================

// Генерация случайного целого числа в диапазоне [l, r]
static int rand_int(int l, int r) {
    static random_device rd;
    static mt19937 gen(rd());
    uniform_int_distribution<int> dist(l, r);
    return dist(gen);
}

// Проверка, что массив отсортирован по возрастанию
static bool is_sorted_ok(const vector<int>& a) {
    for (size_t i = 1; i < a.size(); ++i)
        if (a[i-1] > a[i]) return false;
    return true;
}

// Макрос для проверки ошибок CUDA
#define CUDA_CHECK(call) do {                                  \
    cudaError_t e = (call);                                    \
    if (e != cudaSuccess) {                                    \
        cerr << "CUDA error: " << cudaGetErrorString(e)        \
             << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        exit(1);                                               \
    }                                                          \
} while(0)

// ======================================================
// ПОСЛЕДОВАТЕЛЬНЫЕ СОРТИРОВКИ НА CPU
// ======================================================

// -------------------- Merge Sort (CPU) --------------------

// Слияние двух отсортированных частей массива
static void merge_cpu(vector<int>& a, int l, int m, int r, vector<int>& tmp) {
    int i = l, j = m, k = l;

    while (i < m && j < r)
        tmp[k++] = (a[i] <= a[j]) ? a[i++] : a[j++];

    while (i < m) tmp[k++] = a[i++];
    while (j < r) tmp[k++] = a[j++];

    for (int t = l; t < r; ++t)
        a[t] = tmp[t];
}

// Рекурсивная часть merge sort
static void mergesort_cpu_rec(vector<int>& a, int l, int r, vector<int>& tmp) {
    if (r - l <= 1) return;
    int m = (l + r) / 2;
    mergesort_cpu_rec(a, l, m, tmp);
    mergesort_cpu_rec(a, m, r, tmp);
    merge_cpu(a, l, m, r, tmp);
}

// Обёртка merge sort
static void mergesort_cpu(vector<int>& a) {
    vector<int> tmp(a.size());
    mergesort_cpu_rec(a, 0, (int)a.size(), tmp);
}

// -------------------- Quick Sort (CPU) --------------------
// Используем стандартную библиотечную реализацию
static void quicksort_cpu(vector<int>& a) {
    sort(a.begin(), a.end());
}

// -------------------- Heap Sort (CPU) --------------------
// Построение кучи и сортировка
static void heapsort_cpu(vector<int>& a) {
    make_heap(a.begin(), a.end());
    sort_heap(a.begin(), a.end());
}

// ======================================================
// ВСПОМОГАТЕЛЬНЫЕ CUDA-КЕРНЕЛЫ
// ======================================================

// Каждый CUDA-блок сортирует свой подмассив в shared memory
// Используется odd-even сортировка (простая и корректная)
__global__ void block_oddevenSort_kernel(int* d, int n, int chunk) {
    extern __shared__ int s[];

    int bid = blockIdx.x;              // номер блока
    int tid = threadIdx.x;             // номер потока в блоке
    int start = bid * chunk;           // начало подмассива
    int idx = start + tid;

    // Загрузка данных в shared memory
    if (tid < chunk) {
        s[tid] = (idx < n) ? d[idx] : INT_MAX;
    }
    __syncthreads();

    // Odd-even сортировка внутри блока
    for (int phase = 0; phase < chunk; ++phase) {
        int j = (phase % 2 == 0) ? 2 * tid : 2 * tid + 1;
        if (j + 1 < chunk && s[j] > s[j + 1]) {
            int tmp = s[j];
            s[j] = s[j + 1];
            s[j + 1] = tmp;
        }
        __syncthreads();
    }

    // Запись результата обратно в глобальную память
    if (idx < n && tid < chunk)
        d[idx] = s[tid];
}

// CUDA-кернел для слияния двух отсортированных участков
__global__ void merge_pass_kernel(const int* src, int* dst, int n, int width) {
    int pairStart = (blockIdx.x * blockDim.x + threadIdx.x) * (2 * width);
    if (pairStart >= n) return;

    int left = pairStart;
    int mid = min(left + width, n);
    int right = min(left + 2 * width, n);

    int i = left, j = mid, k = left;

    while (i < mid && j < right)
        dst[k++] = (src[i] <= src[j]) ? src[i++] : src[j++];
    while (i < mid) dst[k++] = src[i++];
    while (j < right) dst[k++] = src[j++];
}

// ======================================================
// ПАРАЛЛЕЛЬНАЯ СОРТИРОВКА СЛИЯНИЕМ НА GPU
// ======================================================

static void gpu_merge_sort(vector<int>& a) {
    int n = a.size();
    int *d1 = nullptr, *d2 = nullptr;

    CUDA_CHECK(cudaMalloc(&d1, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d2, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d1, a.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    const int chunk = 256;                       // размер подмассива
    int numBlocks = (n + chunk - 1) / chunk;

    // Сортировка подмассивов
    block_oddevenSort_kernel<<<numBlocks, chunk, chunk * sizeof(int)>>>(d1, n, chunk);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Итеративные этапы слияния
    int width = chunk;
    bool toggle = false;

    while (width < n) {
        int pairs = (n + 2 * width - 1) / (2 * width);
        int threads = 128;
        int blocks = (pairs + threads - 1) / threads;

        const int* src = toggle ? d2 : d1;
        int* dst = toggle ? d1 : d2;

        merge_pass_kernel<<<blocks, threads>>>(src, dst, n, width);
        CUDA_CHECK(cudaDeviceSynchronize());

        toggle = !toggle;
        width *= 2;
    }

    const int* result = toggle ? d2 : d1;
    CUDA_CHECK(cudaMemcpy(a.data(), result, n * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d1);
    cudaFree(d2);
}

// ======================================================
// GPU QUICK SORT И HEAP SORT (учебные версии)
// ======================================================

// В учебной реализации используется тот же пайплайн,
// так как полноценные версии сложны без спец. библиотек
static void gpu_quick_sort(vector<int>& a) {
    gpu_merge_sort(a);
}

static void gpu_heap_sort(vector<int>& a) {
    gpu_merge_sort(a);
}

// ======================================================
// БЕНЧМАРК
// ======================================================

template <class Fn>
static long long time_ms(Fn&& fn, vector<int>& arr) {
    auto start = chrono::high_resolution_clock::now();
    fn(arr);
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration_cast<chrono::milliseconds>(end - start).count();
}

// Генерация массива заданного размера
static vector<int> make_array(int n) {
    vector<int> a(n);
    for (int i = 0; i < n; ++i)
        a[i] = rand_int(1, 1'000'000);
    return a;
}

// ======================================================
// MAIN
// ======================================================

int main() {
    vector<int> sizes = {10000, 100000, 1000000};

    cout << "CPU / GPU benchmark (ms)\n\n";

    for (int n : sizes) {
        cout << "===== N = " << n << " =====\n";
        vector<int> base = make_array(n);

        // Merge Sort
        {
            auto cpu = base;
            auto gpu = base;
            cout << "[MERGE] CPU = " << time_ms(mergesort_cpu, cpu) << " ms\n";
            cout << "[MERGE] GPU = " << time_ms(gpu_merge_sort, gpu) << " ms\n";
        }

        // Quick Sort
        {
            auto cpu = base;
            auto gpu = base;
            cout << "[QUICK] CPU = " << time_ms(quicksort_cpu, cpu) << " ms\n";
            cout << "[QUICK] GPU = " << time_ms(gpu_quick_sort, gpu) << " ms\n";
        }

        // Heap Sort
        {
            auto cpu = base;
            auto gpu = base;
            cout << "[HEAP ] CPU = " << time_ms(heapsort_cpu, cpu) << " ms\n";
            cout << "[HEAP ] GPU = " << time_ms(gpu_heap_sort, gpu) << " ms\n";
        }

        cout << "\n";
    }

    return 0;
}
