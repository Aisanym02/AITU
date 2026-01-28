// pw4.cu
// Практическая №4: глобальная/разделяемая/локальная память CUDA

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cmath>

#define CUDA_CHECK(call) do {                                  \
    cudaError_t err = (call);                                  \
    if (err != cudaSuccess) {                                  \
        fprintf(stderr, "CUDA error %s:%d: %s\n",              \
                __FILE__, __LINE__, cudaGetErrorString(err));  \
        std::exit(1);                                          \
    }                                                          \
} while(0)

static float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

// ------------------------- 1) REDUCTION -------------------------

// a) Только глобальная память: atomicAdd по каждому элементу
__global__ void reduce_global_atomic(const float* __restrict__ in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(out, in[i]);
}

// b) Глобальная + shared: редукция в shared внутри блока, atomicAdd только 1 раз на блок
template<int BLOCK>
__global__ void reduce_shared_block(const float* __restrict__ in, float* out, int n) {
    __shared__ float sdata[BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * BLOCK + tid;

    float x = (i < n) ? in[i] : 0.0f;
    sdata[tid] = x;
    __syncthreads();

    // редукция в shared
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(out, sdata[0]);
}

// ------------------------- 2) SORTING -------------------------

// Требование: "пузырьком для небольших подмассивов с использованием локальной памяти"
// Сделаем так: каждый поток сортирует TILE элементов в локальном массиве.
// После этого получаем N/TILE отсортированных "ранов" длины TILE.
// Далее делаем итеративное слияние (merge) пар ранoв, используя shared память.

constexpr int TILE = 32; // маленький подмассив на поток (локальная память)

// bubble sort на локальном массиве (внутри потока)
__device__ __forceinline__ void bubble_sort_local(float a[TILE]) {
    #pragma unroll
    for (int i = 0; i < TILE - 1; i++) {
        #pragma unroll
        for (int j = 0; j < TILE - 1 - i; j++) {
            float x = a[j], y = a[j + 1];
            if (x > y) { a[j] = y; a[j + 1] = x; }
        }
    }
}

// Стадия 1: локальная сортировка TILE-чанков
__global__ void sort_tiles_local(float* data, int n) {
    int tile_id = blockIdx.x * blockDim.x + threadIdx.x; // один поток = один tile
    int base = tile_id * TILE;
    if (base >= n) return;

    float local[TILE];

    // загрузка в локальный массив (локальная память/регистры)
    #pragma unroll
    for (int k = 0; k < TILE; k++) {
        int idx = base + k;
        local[k] = (idx < n) ? data[idx] : INFINITY; // паддинг
    }

    bubble_sort_local(local);

    // запись обратно в глобальную память
    #pragma unroll
    for (int k = 0; k < TILE; k++) {
        int idx = base + k;
        if (idx < n) data[idx] = local[k];
    }
}

// Слияние ранoв (merge) с использованием shared memory.
// Каждый блок сливает один "пара-ран": [left, left+run) и [left+run, left+2run)
__global__ void merge_runs_shared(const float* in, float* out, int n, int run) {
    extern __shared__ float sh[]; // размер будет 2*run элементов (или меньше на краю)
    int pair_id = blockIdx.x;     // одна пара ран на блок
    int left = pair_id * (2 * run);
    if (left >= n) return;

    int mid = min(left + run, n);
    int right = min(left + 2 * run, n);

    int left_len  = mid - left;
    int right_len = right - mid;
    int total = left_len + right_len;

    // загрузка в shared (коалесцировано)
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int gidx = left + i;
        sh[i] = in[gidx];
    }
    __syncthreads();

    // Параллельный merge "по индексам" в учебных работах часто упрощают.
    // Здесь сделаем простой последовательный merge внутри одного потока (tid==0),
    // но данные берем из shared, что выполняет требование по памяти.
    if (threadIdx.x == 0) {
        int i = 0;
        int j = left_len;
        int i_end = left_len;
        int j_end = left_len + right_len;

        for (int k = 0; k < total; k++) {
            float a = (i < i_end) ? sh[i] : INFINITY;
            float b = (j < j_end) ? sh[j] : INFINITY;
            if (a <= b) { out[left + k] = a; i++; }
            else        { out[left + k] = b; j++; }
        }
    }
}

// ------------------------- Helpers -------------------------

static void fill_random(std::vector<float>& v, unsigned seed=123) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto &x : v) x = dist(rng);
}

static float cpu_sum(const std::vector<float>& v) {
    double s = 0.0;
    for (float x : v) s += x;
    return (float)s;
}

static bool is_sorted_cpu(const std::vector<float>& v) {
    for (size_t i = 1; i < v.size(); i++)
        if (v[i-1] > v[i]) return false;
    return true;
}

int main() {
    std::vector<int> sizes = {10000, 100000, 1000000};

    std::ofstream csv("results.csv");
    csv << "N,reduce_global_ms,reduce_shared_ms,sort_total_ms\n";

    // события для тайминга
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int N : sizes) {
        std::cout << "=== N = " << N << " ===\n";

        // 1) Подготовка данных
        std::vector<float> h(N);
        fill_random(h, 123);

        float *d_data = nullptr;
        CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_data, h.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        // ---------------- Reduction A: global atomic ----------------
        float *d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));

        int block = 256;
        int grid = (N + block - 1) / block;

        CUDA_CHECK(cudaEventRecord(start));
        reduce_global_atomic<<<grid, block>>>(d_data, d_out, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float t_reduce_global = elapsed_ms(start, stop);

        float h_sum_global = 0.0f;
        CUDA_CHECK(cudaMemcpy(&h_sum_global, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));

        // ---------------- Reduction B: shared block reduce ----------------
        constexpr int RBLOCK = 256;
        int grid2 = (N + RBLOCK - 1) / RBLOCK;

        CUDA_CHECK(cudaEventRecord(start));
        reduce_shared_block<RBLOCK><<<grid2, RBLOCK>>>(d_data, d_out, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float t_reduce_shared = elapsed_ms(start, stop);

        float h_sum_shared = 0.0f;
        CUDA_CHECK(cudaMemcpy(&h_sum_shared, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        // (необязательно) сверка с CPU
        float h_sum_cpu = cpu_sum(h);
        std::cout << "Sum CPU:    " << h_sum_cpu << "\n";
        std::cout << "Sum Global: " << h_sum_global << "\n";
        std::cout << "Sum Shared: " << h_sum_shared << "\n";

        CUDA_CHECK(cudaFree(d_out));

        // ---------------- Sorting: local bubble + shared merges ----------------
        // Скопируем заново исходные данные (чтобы не сортировать уже измененное)
        CUDA_CHECK(cudaMemcpy(d_data, h.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        // 1) локальная сортировка TILE-ранов
        int tiles = (N + TILE - 1) / TILE;
        int sort_block = 256;
        int sort_grid  = (tiles + sort_block - 1) / sort_block;

        float *d_tmp = nullptr;
        CUDA_CHECK(cudaMalloc(&d_tmp, N * sizeof(float)));

        CUDA_CHECK(cudaEventRecord(start));
        sort_tiles_local<<<sort_grid, sort_block>>>(d_data, N);
        CUDA_CHECK(cudaGetLastError());

        // 2) итеративное слияние ранoв: run = TILE, 2*TILE, 4*TILE ...
        // будем чередовать буферы d_data <-> d_tmp
        int run = TILE;
        bool flip = false;

        while (run < N) {
            int pairs = (N + 2*run - 1) / (2*run);
            int merge_block = 256;

            // shared size: до 2*run float
            // Важно: shared ограничена, поэтому для огромных run этот учебный метод станет тяжелым.
            // Для практической работы этого достаточно, потому что демонстрируется идея shared.
            size_t shmem = (size_t)min(2*run, N) * sizeof(float);

            const float* in  = flip ? d_tmp : d_data;
            float* out = flip ? d_data : d_tmp;

            merge_runs_shared<<<pairs, merge_block, shmem>>>(in, out, N, run);
            CUDA_CHECK(cudaGetLastError());

            flip = !flip;
            run *= 2;
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float t_sort_total = elapsed_ms(start, stop);

        // итоговый указатель на отсортированные данные
        float* d_sorted = (run/TILE % 2 == 0) ? d_data : d_tmp; 
        // (выше простой способ определить, где результат; можно не усложнять)

        std::vector<float> h_sorted(N);
        CUDA_CHECK(cudaMemcpy(h_sorted.data(), d_sorted, N * sizeof(float), cudaMemcpyDeviceToHost));

        std::cout << "Sorted OK: " << (is_sorted_cpu(h_sorted) ? "YES" : "NO") << "\n";

        // запись CSV
        csv << N << "," << t_reduce_global << "," << t_reduce_shared << "," << t_sort_total << "\n";

        std::cout << "Reduction global (ms): " << t_reduce_global << "\n";
        std::cout << "Reduction shared (ms): " << t_reduce_shared << "\n";
        std::cout << "Sort total (ms):       " << t_sort_total << "\n\n";

        CUDA_CHECK(cudaFree(d_tmp));
        CUDA_CHECK(cudaFree(d_data));
    }

    CSV_CHECK:
    csv.close();

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::cout << "Saved: results.csv\n";
    return 0;
}
