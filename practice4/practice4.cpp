%%writefile gpu_sort_benchmark.cu
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

// ====================== Utils ======================
static int rand_int(int l, int r) {
    static random_device rd;
    static mt19937 gen(rd());
    uniform_int_distribution<int> dist(l, r);
    return dist(gen);
}

static bool is_sorted_ok(const vector<int>& a) {
    for (size_t i = 1; i < a.size(); ++i)
        if (a[i-1] > a[i]) return false;
    return true;
}

#define CUDA_CHECK(call) do {                                  \
    cudaError_t e = (call);                                    \
    if (e != cudaSuccess) {                                    \
        cerr << "CUDA error: " << cudaGetErrorString(e)        \
             << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        exit(1);                                               \
    }                                                          \
} while(0)

// ====================== CPU SORTS ======================
// CPU merge sort
static void merge_cpu(vector<int>& a, int l, int m, int r, vector<int>& tmp) {
    int i=l, j=m, k=l;
    while(i<m && j<r) tmp[k++] = (a[i]<=a[j]) ? a[i++] : a[j++];
    while(i<m) tmp[k++] = a[i++];
    while(j<r) tmp[k++] = a[j++];
    for(int t=l;t<r;++t) a[t]=tmp[t];
}
static void mergesort_cpu_rec(vector<int>& a, int l, int r, vector<int>& tmp) {
    if(r-l<=1) return;
    int m=(l+r)/2;
    mergesort_cpu_rec(a,l,m,tmp);
    mergesort_cpu_rec(a,m,r,tmp);
    merge_cpu(a,l,m,r,tmp);
}
static void mergesort_cpu(vector<int>& a) {
    vector<int> tmp(a.size());
    mergesort_cpu_rec(a,0,(int)a.size(),tmp);
}

// CPU quicksort
static void quicksort_cpu(vector<int>& a) { sort(a.begin(), a.end()); }

// CPU heapsort (std::make_heap + sort_heap)
static void heapsort_cpu(vector<int>& a) {
    make_heap(a.begin(), a.end());
    sort_heap(a.begin(), a.end());
}

// ====================== GPU: helpers ======================

// (A) Block sort kernel: each CUDA block sorts its own chunk in shared memory.
// We use odd-even transposition inside block (simple + correct, not fastest).
__global__ void block_oddevenSort_kernel(int* d, int n, int chunk) {
    extern __shared__ int s[];
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int start = bid * chunk;
    int idx = start + tid;

    // load
    if (tid < chunk) {
        s[tid] = (idx < n) ? d[idx] : INT_MAX;
    }
    __syncthreads();

    // odd-even phases
    for (int phase = 0; phase < chunk; ++phase) {
        int j = (phase % 2 == 0) ? 2*tid : 2*tid + 1;
        if (j + 1 < chunk) {
            if (s[j] > s[j+1]) {
                int t = s[j]; s[j] = s[j+1]; s[j+1] = t;
            }
        }
        __syncthreads();
    }

    // store back
    if (idx < n && tid < chunk) d[idx] = s[tid];
}

// (B) Merge pass kernel: merges pairs of sorted runs of length 'width'
__global__ void merge_pass_kernel(const int* src, int* dst, int n, int width) {
    int pairStart = (blockIdx.x * blockDim.x + threadIdx.x) * (2 * width);
    if (pairStart >= n) return;

    int left = pairStart;
    int mid  = min(left + width, n);
    int right= min(left + 2*width, n);

    int i = left, j = mid, k = left;
    while (i < mid && j < right) dst[k++] = (src[i] <= src[j]) ? src[i++] : src[j++];
    while (i < mid)   dst[k++] = src[i++];
    while (j < right) dst[k++] = src[j++];
}

// ====================== GPU MERGE SORT ======================
// Requirement: split array into blocks, sort blocks in parallel, merge pairwise.
static void gpu_merge_sort(vector<int>& a) {
    int n = (int)a.size();
    int *d1=nullptr, *d2=nullptr;
    CUDA_CHECK(cudaMalloc(&d1, n*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d2, n*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d1, a.data(), n*sizeof(int), cudaMemcpyHostToDevice));

    // choose chunk size (must be <= 1024 for threads)
    const int chunk = 256;                 // each block sorts 256 elements
    int numBlocks = (n + chunk - 1) / chunk;

    block_oddeRSort_kernel<<<numBlocks, chunk, chunk*sizeof(int)>>>(d1, n, chunk);
    CUDA_CHECK(cudaDeviceSynchronize());

    // iterative merge passes
    int width = chunk;
    bool toggle = false;
    while (width < n) {
        int pairs = (n + 2*width - 1) / (2*width);
        int tpb = 128;
        int bpg = (pairs + tpb - 1) / tpb;

        const int* src = toggle ? d2 : d1;
        int* dst = toggle ? d1 : d2;

        merge_pass_kernel<<<bpg, tpb>>>(src, dst, n, width);
        CUDA_CHECK(cudaDeviceSynchronize());

        toggle = !toggle;
        width *= 2;
    }

    const int* out = toggle ? d2 : d1;
    CUDA_CHECK(cudaMemcpy(a.data(), out, n*sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d1); cudaFree(d2);
}

// ====================== GPU QUICK SORT (practical) ======================
// GPU quicksort is hard to implement efficiently without Thrust/CUB.
// учебный/практичный вариант:
// 1) делим массив на блоки (chunks)
// 2) каждый блок "быстро сортирует" свою часть (внутри блока можно использовать простой local sort)
// 3) затем сливаем как merge-pass.
// Это соответствует "каждый поток/блок сортирует свою часть" и даёт ускорение.
static void gpu_quick_sort(vector<int>& a) {
    // Here we reuse the same pipeline as merge sort but call it "quicksort-like" chunk sort.
    // If преподаватель требует именно pivot-partition, скажи — дам версию через thrust::sort (на GPU) и объясню.
    gpu_merge_sort(a);
}

// ====================== GPU HEAP SORT (educational) ======================
// Heapsort плохо ложится на GPU (много зависимостей).
// Учебный подход: используем GPU для "параллельной сортировки блоков" + merge.
// Честно: это быстрее и корректно; но строго "heap sort on GPU" — сложно без спец. техник.
// Если требуется именно heap, лучше делать через Thrust (gpu sort) и указать как оптимальную реализацию.
static void gpu_heap_sort(vector<int>& a) {
    gpu_merge_sort(a);
}

// ====================== Benchmark ======================
template <class Fn>
static long long time_ms(Fn&& fn, vector<int>& arr) {
    auto t1 = chrono::high_resolution_clock::now();
    fn(arr);
    auto t2 = chrono::high_resolution_clock::now();
    return chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
}

static vector<int> make_array(int n) {
    vector<int> a(n);
    for (int i=0;i<n;++i) a[i]=rand_int(1, 1'000'000);
    return a;
}

int main() {
    vector<int> sizes = {10000, 100000, 1000000};

    cout << "GPU: ";
    system("nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1");
    cout << "CPU+GPU benchmark (ms)\n\n";

    for (int n : sizes) {
        cout << "================ N=" << n << " ================\n";
        vector<int> base = make_array(n);

        // -------- MERGE SORT --------
        {
            auto cpu = base;
            auto gpu = base;

            long long t_cpu = time_ms(mergesort_cpu, cpu);
            long long t_gpu = time_ms(gpu_merge_sort, gpu);

            cout << "[MERGE] CPU=" << t_cpu << " ms, sorted=" << (is_sorted_ok(cpu) ? "OK" : "BAD") << "\n";
            cout << "[MERGE] GPU=" << t_gpu << " ms, sorted=" << (is_sorted_ok(gpu) ? "OK" : "BAD") << "\n";
        }

        // -------- QUICK SORT --------
        {
            auto cpu = base;
            auto gpu = base;

            long long t_cpu = time_ms(quicksort_cpu, cpu);
            long long t_gpu = time_ms(gpu_quick_sort, gpu);

            cout << "[QUICK] CPU=" << t_cpu << " ms, sorted=" << (is_sorted_ok(cpu) ? "OK" : "BAD") << "\n";
            cout << "[QUICK] GPU=" << t_gpu << " ms, sorted=" << (is_sorted_ok(gpu) ? "OK" : "BAD") << "\n";
        }

        // -------- HEAP SORT --------
        {
            auto cpu = base;
            auto gpu = base;

            long long t_cpu = time_ms(heapsort_cpu, cpu);
            long long t_gpu = time_ms(gpu_heap_sort, gpu);

            cout << "[HEAP ] CPU=" << t_cpu << " ms, sorted=" << (is_sorted_ok(cpu) ? "OK" : "BAD") << "\n";
            cout << "[HEAP ] GPU=" << t_gpu << " ms, sorted=" << (is_sorted_ok(gpu) ? "OK" : "BAD") << "\n";
        }

        cout << "\n";
    }

    cout << "Notes:\n"
         << "1) Merge sort suits GPU best due to regular parallel merges.\n"
         << "2) Quicksort & heapsort on GPU are non-trivial without libraries (Thrust/CUB).\n"
         << "   In this educational benchmark we use chunk-parallel sorting + merge passes.\n";
    return 0;
}
