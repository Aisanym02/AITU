#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <limits>
#include <omp.h>

using namespace std;

// Генерация случайных чисел
int rand_int(int l, int r) {
    static random_device rd;
    static mt19937 gen(rd());
    uniform_int_distribution<int> dist(l, r);
    return dist(gen);
}

// Последовательный min/max
pair<int,int> minmax_seq(const vector<int>& a) {
    int mn = numeric_limits<int>::max();
    int mx = numeric_limits<int>::min();
    for (int x : a) {
        if (x < mn) mn = x;
        if (x > mx) mx = x;
    }
    return {mn, mx};
}

// Параллельный min/max (через local min/max + объединение)
pair<int,int> minmax_omp(const vector<int>& a) {
    int global_min = numeric_limits<int>::max();
    int global_max = numeric_limits<int>::min();

    #pragma omp parallel
    {
        int local_min = numeric_limits<int>::max();
        int local_max = numeric_limits<int>::min();

        #pragma omp for
        for (int i = 0; i < (int)a.size(); ++i) {
            int x = a[i];
            if (x < local_min) local_min = x;
            if (x > local_max) local_max = x;
        }

        #pragma omp critical
        {
            if (local_min < global_min) global_min = local_min;
            if (local_max > global_max) global_max = local_max;
        }
    }

    return {global_min, global_max};
}

int main() {
    const int N = 10000;

    // 1) массив из 10 000 случайных чисел
    vector<int> a(N);
    for (int i = 0; i < N; ++i) a[i] = rand_int(1, 1'000'000);

    // 2) последовательная реализация
    auto t1 = chrono::high_resolution_clock::now();
    auto seq = minmax_seq(a);
    auto t2 = chrono::high_resolution_clock::now();

    // 2) OpenMP реализация
    auto t3 = chrono::high_resolution_clock::now();
    auto par = minmax_omp(a);
    auto t4 = chrono::high_resolution_clock::now();

    auto seq_us = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
    auto par_us = chrono::duration_cast<chrono::microseconds>(t4 - t3).count();

    cout << "Threads: " << omp_get_max_threads() << "\n";
    cout << "SEQ: min=" << seq.first << " max=" << seq.second
         << " time=" << seq_us << " us\n";
    cout << "OMP: min=" << par.first << " max=" << par.second
         << " time=" << par_us << " us\n";

    // 3) выводы: (можно переписать в отчёт)
    if (par_us < seq_us) cout << "Conclusion: OpenMP faster on this run.\n";
    else cout << "Conclusion: Overheads may dominate; not always faster on small N.\n";

    return 0;
}
