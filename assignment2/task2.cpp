#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <omp.h>

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

// Последовательная сортировка выбором
void selection_sort_seq(vector<int>& a) {
    int n = (int)a.size();
    for (int i = 0; i < n - 1; ++i) {
        int minIdx = i;
        for (int j = i + 1; j < n; ++j)
            if (a[j] < a[minIdx]) minIdx = j;
        swap(a[i], a[minIdx]);
    }
}

// OpenMP: параллельный поиск минимума на каждом шаге i
void selection_sort_omp(vector<int>& a) {
    int n = (int)a.size();

    for (int i = 0; i < n - 1; ++i) {
        int globalMinVal = a[i];
        int globalMinIdx = i;

        #pragma omp parallel
        {
            int localMinVal = globalMinVal;
            int localMinIdx = globalMinIdx;

            #pragma omp for nowait
            for (int j = i + 1; j < n; ++j) {
                if (a[j] < localMinVal) {
                    localMinVal = a[j];
                    localMinIdx = j;
                }
            }

            #pragma omp critical
            {
                if (localMinVal < globalMinVal) {
                    globalMinVal = localMinVal;
                    globalMinIdx = localMinIdx;
                }
            }
        }

        swap(a[i], a[globalMinIdx]);
    }
}

template <typename F>
long long time_us(F&& fn, vector<int>& a) {
    auto t1 = chrono::high_resolution_clock::now();
    fn(a);
    auto t2 = chrono::high_resolution_clock::now();
    return chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
}

vector<int> make_array(int n) {
    vector<int> a(n);
    for (int i = 0; i < n; ++i) a[i] = rand_int(1, 1'000'000);
    return a;
}

int main() {
    cout << "Threads: " << omp_get_max_threads() << "\n\n";
    vector<int> sizes = {1000, 10000};

    for (int n : sizes) {
        cout << "==== N=" << n << " ====\n";
        vector<int> base = make_array(n);

        vector<int> a1 = base, a2 = base;
        long long t_seq = time_us(selection_sort_seq, a1);
        long long t_omp = time_us(selection_sort_omp, a2);

        cout << "SEQ time: " << t_seq << " us, sorted=" << (is_sorted_ok(a1) ? "OK" : "BAD") << "\n";
        cout << "OMP time: " << t_omp << " us, sorted=" << (is_sorted_ok(a2) ? "OK" : "BAD") << "\n\n";
    }

    return 0;
}
