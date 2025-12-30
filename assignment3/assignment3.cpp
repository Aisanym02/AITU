#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>

using namespace std;

// ------------------------------
// Макрос для проверки ошибок CUDA
// ------------------------------
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        cerr << "CUDA ERROR: " << cudaGetErrorString(err) \
             << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        exit(1); \
    } \
} while(0)

// -----------------------------------------------------
// Утилита: генерируем вектор float случайными значениями
// -----------------------------------------------------
static void fill_random(vector<float>& a, float lo = -1.0f, float hi = 1.0f) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dist(lo, hi);
    for (auto &x : a) x = dist(gen);
}

// ----------------------------------------
// Утилита: измерение времени CUDA-кернелов
// ----------------------------------------
// Возвращает время в миллисекундах (ms)
static float time_kernel_ms(void (*launcher)(), int iters = 50) {
    // launcher() должен запускать kernel, но НЕ делать cudaDeviceSynchronize()
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // "Прогрев" (чтобы первые запуски не искажали время)
    launcher();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) launcher();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / iters; // среднее время за 1 запуск
}

// =====================================================
// ЗАДАНИЕ 1: поэлементная обработка массива (умножение)
// 1) только глобальная память
// 2) использование разделяемой памяти (shared)
// =====================================================

// (1) Глобальная память: читаем из global, пишем в global
__global__ void mul_global(const float* __restrict__ in, float* __restrict__ out,
                           float k, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] * k;
}

// (2) Shared memory: сначала грузим кусок в shared, умножаем, пишем обратно
// ВНИМАНИЕ: для простой операции это обычно НЕ быстрее, но задание просит
// именно показать вариант с shared.
__global__ void mul_shared(const float* __restrict__ in, float* __restrict__ out,
                           float k, int n) {
    extern __shared__ float s[]; // shared массив размером blockDim.x
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // 1) читаем из global в shared
    float val = 0.0f;
    if (idx < n) val = in[idx];
    s[tid] = val;
    __syncthreads();

    // 2) делаем вычисление в shared
    s[tid] = s[tid] * k;
    __syncthreads();

    // 3) пишем результат из shared в global
    if (idx < n) out[idx] = s[tid];
}

// =====================================================
// ЗАДАНИЕ 2: поэлементное сложение двух массивов
// + исследование влияния размера блока (3 варианта)
// =====================================================
__global__ void add_arrays(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c,
                           int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

// =====================================================
// ЗАДАНИЕ 3: коалесцированный и некоалесцированный доступ
// Идея:
//  - coalesced: поток i читает in[i] (соседние потоки читают соседние адреса)
//  - uncoalesced: поток i читает in[(i*stride) % n] при stride=32
//    (получается "скачущий" доступ, хуже для памяти)
// =====================================================
__global__ void access_coalesced(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] * 1.000001f; // простая операция, чтобы было чтение+запись
}

__global__ void access_uncoalesced(const float* __restrict__ in,
                                   float* __restrict__ out,
                                   int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int j = (idx * stride) % n;     // "перемешанный" индекс -> некоалесцированный доступ
        out[idx] = in[j] * 1.000001f;
    }
}

// =====================================================
// ЗАДАНИЕ 4: подбор оптимальных параметров сетки/блоков
// Сделаем на примере сложения массивов (Задание 2):
//   - "неоптимально": block=64 (часто хуже)
//   - "оптимально":  block=256 (часто лучше на T4)
// Плюс можно быстро сравнить с 128/512.
// =====================================================

// -------------------------------
// Основная программа / Бенчмарки
// -------------------------------
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // Размер массивов согласно заданию: 1 000 000
    const int N = 1'000'000;
    const size_t bytes = (size_t)N * sizeof(float);

    cout << "Assignment 3 (CUDA). N = " << N << "\n";

    // --------------------------
    // Подготовка данных на CPU
    // --------------------------
    vector<float> h_a(N), h_b(N), h_out(N);
    fill_random(h_a);
    fill_random(h_b);

    // --------------------------
    // Выделяем память на GPU
    // --------------------------
    float *d_a=nullptr, *d_b=nullptr, *d_c=nullptr, *d_tmp=nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMalloc(&d_tmp, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // =====================================================
    // ЗАДАНИЕ 1: умножение массива на число
    // =====================================================
    cout << "\n=== Task 1: element-wise multiply (global vs shared) ===\n";
    const float K = 2.5f;

    // Выбираем типичный размер блока
    const int block1 = 256;
    const int grid1  = (N + block1 - 1) / block1;

    // Лаунчеры (нужны для time_kernel_ms)
    auto launch_mul_global = [&]() {
        mul_global<<<grid1, block1>>>(d_a, d_c, K, N);
    };
    auto launch_mul_shared = [&]() {
        // shared memory: block1 * sizeof(float)
        mul_shared<<<grid1, block1, block1 * (int)sizeof(float)>>>(d_a, d_c, K, N);
    };

    // Измеряем
    float t_mul_global = time_kernel_ms((void(*)())+[](){}, 1); // заглушка, не используем
    // В C++ так напрямую лямбду в функцию не приведём без обёртки,
    // поэтому ниже делаем измерение вручную через события:

    // Функция измерения для конкретного запуска лямбды
    auto measure = [&](auto&& launch, int iters = 50) -> float {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        // прогрев
        launch();
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; ++i) launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        return ms / iters;
    };

    t_mul_global = measure(launch_mul_global);
    float t_mul_shared = measure(launch_mul_shared);

    cout << "Multiply (global memory) avg: " << t_mul_global << " ms\n";
    cout << "Multiply (shared memory) avg: " << t_mul_shared << " ms\n";
    cout << "Note: for simple ops shared can be slower; it is shown for educational purposes.\n";

    // =====================================================
    // ЗАДАНИЕ 2: сложение двух массивов + влияние block size
    // =====================================================
    cout << "\n=== Task 2: element-wise add, block size impact ===\n";

    auto test_add_block = [&](int block) {
        int grid = (N + block - 1) / block;
        auto launch = [&]() { add_arrays<<<grid, block>>>(d_a, d_b, d_c, N); };
        float t = measure(launch);
        cout << "Block=" << block << " -> avg: " << t << " ms\n";
        return t;
    };

    // Минимум 3 различных размера блока (как в задании)
    float t_add_128 = test_add_block(128);
    float t_add_256 = test_add_block(256);
    float t_add_512 = test_add_block(512);

    // =====================================================
    // ЗАДАНИЕ 3: coalesced vs uncoalesced доступ к global памяти
    // =====================================================
    cout << "\n=== Task 3: coalesced vs uncoalesced memory access ===\n";
    const int block3 = 256;
    const int grid3  = (N + block3 - 1) / block3;

    auto launch_coalesced = [&]() { access_coalesced<<<grid3, block3>>>(d_a, d_tmp, N); };
    auto launch_uncoal = [&]() { access_uncoalesced<<<grid3, block3>>>(d_a, d_tmp, N, 32); };

    float t_coal = measure(launch_coalesced);
    float t_uncoal = measure(launch_uncoal);

    cout << "Coalesced access avg:     " << t_coal << " ms\n";
    cout << "Uncoalesced access avg:   " << t_uncoal << " ms\n";
    cout << "Usually uncoalesced is slower because memory transactions are less efficient.\n";

    // =====================================================
    // ЗАДАНИЕ 4: "неоптимальная" и "оптимальная" конфигурации
    // Берём программу из Task 2 (сложение массивов)
    // =====================================================
    cout << "\n=== Task 4: compare bad vs optimized launch configuration (Task 2) ===\n";

    // Неоптимально (пример): маленький блок 64
    float t_bad = test_add_block(64);

    // Оптимальнее (часто): 256 (на T4 часто хорош)
    float t_opt = test_add_block(256);

    cout << "\nSummary (Task 4):\n";
    cout << "Bad config (block=64):   " << t_bad << " ms\n";
    cout << "Optimized (block=256):   " << t_opt << " ms\n";
    if (t_opt > 0.0f) {
        cout << "Speedup ~ " << (t_bad / t_opt) << "x\n";
    }

    // -----------------------------------------------------
    // Небольшая проверка корректности для сложения (Task 2)
    // -----------------------------------------------------
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    // Проверим несколько элементов (не все, чтобы не тормозить)
    bool ok = true;
    for (int i = 0; i < 10; ++i) {
        float cpu = h_a[i] + h_b[i];
        if (fabs(h_out[i] - cpu) > 1e-5f) { ok = false; break; }
    }
    cout << "\nCorrectness check (first 10 elements of add): " << (ok ? "OK" : "BAD") << "\n";

    // Освобождаем память
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_tmp);

    cout << "\nDone.\n";
    return 0;
}
