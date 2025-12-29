#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <limits>
#include <omp.h>

using namespace std;
// Функция возвращает случайное целое число в диапазоне [l, r]
static int rand_int(int l, int r) {
    static random_device rd;
    static mt19937 gen(rd());
    uniform_int_distribution<int> dist(l, r);
    return dist(gen);
}

// Проверка: отсортирован ли массив по возрастанию
static bool is_sorted_ok(const vector<int>& a) {
    for (size_t i = 1; i < a.size(); ++i)
        if (a[i-1] > a[i]) return false;
    return true;
}

// ======================== 1) SEQUENTIAL SORTS ========================

// Bubble sort O(n^2)
void bubble_sort_seq(vector<int>& a) {
    int n = (int)a.size();
    for (int i = 0; i < n - 1; ++i) {
        bool swapped = false;
        for (int j = 0; j < n - 1 - i; ++j) {
            if (a[j] > a[j + 1]) { swap(a[j], a[j + 1]); swapped = true; }
        }
        if (!swapped) break;
    }
}

// Selection sort O(n^2)
void selection_sort_seq(vector<int>& a) {
    int n = (int)a.size();
    for (int i = 0; i < n - 1; ++i) {
        int minIdx = i;
        for (int j = i + 1; j < n; ++j)
            if (a[j] < a[minIdx]) minIdx = j;
        swap(a[i], a[minIdx]);
    }
}

// Insertion sort O(n^2)
void insertion_sort_seq(vector<int>& a) {
    int n = (int)a.size();
    for (int i = 1; i < n; ++i) {
        int key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            --j;
        }
        a[j + 1] = key;
    }
}

// ======================== 2) PARALLEL SORTS (OpenMP) ========================

// Parallel bubble: Odd-Even Transposition Sort (корректно параллелится)
// Идея: n фаз; на чётной фазе сравниваем (0,1)(2,3)..., на нечётной (1,2)(3,4)...
void bubble_sort_omp_oddeven(vector<int>& a) {
    int n = (int)a.size();
    for (int phase = 0; phase < n; ++phase) {
        int start = (phase % 2 == 0) ? 0 : 1;
        #pragma omp parallel for
        for (int j = start; j < n - 1; j += 2) {
            if (a[j] > a[j + 1]) swap(a[j], a[j + 1]);
        }
    }
}

// Parallel selection: внешний цикл зависим (i идёт последовательно),
// но поиск минимума на [i..n) делаем параллельно (локальные minima + объединение).
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

// Insertion: параллельная версия через блоки
// 1) Делим на T блоков, каждый поток сортирует свой блок insertion_sort_seq
// 2) Последовательно сливаем блоки (merge) — корректно и реально ускоряет на больших N.
static void insertion_sort_range(vector<int>& a, int L, int R) {
    // сортируем a[L..R) вставками
    for (int i = L + 1; i < R; ++i) {
        int key = a[i];
        int j = i - 1;
        while (j >= L && a[j] > key) {
            a[j + 1] = a[j];
            --j;
        }
        a[j + 1] = key;
    }
}

static vector<int> merge_two_sorted(const vector<int>& left, const vector<int>& right) {
    vector<int> out;
    out.reserve(left.size() + right.size());
    size_t i = 0, j = 0;
    while (i < left.size() && j < right.size()) {
        if (left[i] <= right[j]) out.push_back(left[i++]);
        else out.push_back(right[j++]);
    }
    while (i < left.size()) out.push_back(left[i++]);
    while (j < right.size()) out.push_back(right[j++]);
    return out;
}

void insertion_sort_omp_blocks(vector<int>& a) {
    int n = (int)a.size();
    int T = omp_get_max_threads();
    if (T <= 1 || n < 2000) { // на маленьких массивах параллельность часто хуже
        insertion_sort_seq(a);
        return;
    }

    vector<pair<int,int>> blocks;
    blocks.reserve(T);

    // разбиение на T блоков
    for (int t = 0; t < T; ++t) {
        int L = (long long)t * n / T;
        int R = (long long)(t + 1) * n / T;
        blocks.push_back({L, R});
    }

    // параллельно сортируем блоки вставками
    #pragma omp parallel for
    for (int t = 0; t < T; ++t) {
        auto [L, R] = blocks[t];
        insertion_sort_range(a, L, R);
    }

    // сливаем блоки (пока последовательно для простоты и стабильности)
    // переносим каждый блок в отдельный вектор и merge
    vector<int> merged;
    {
        auto [L0, R0] = blocks[0];
        merged.assign(a.begin() + L0, a.begin() + R0);
    }

    for (int t = 1; t < T; ++t) {
        auto [L, R] = blocks[t];
        vector<int> block(a.begin() + L, a.begin() + R);
        merged = merge_two_sorted(merged, block);
    }

    a.swap(merged);
}

// ======================== 3) BENCHMARK ========================

template <typename F>
long long time_us(F&& fn, vector<int>& a) {
    auto t1 = chrono::high_resolution_clock::now();
    fn(a);
    auto t2 = chrono::high_resolution_clock::now();
    return chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
}

static vector<int> make_random_array(int n) {
    vector<int> a(n);
    for (int i = 0; i < n; ++i) a[i] = rand_int(1, 1'000'000);
    return a;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cout << "OpenMP max threads: " << omp_get_max_threads() << "\n\n";

    // ⚠️ O(n^2) на 100000 может быть очень долго на пузырьке/вставках/выбором.
    // Если будет “висеть” — уменьшай sizes или отключай bubble для 100000.
    vector<int> sizes = {1000, 10000, 100000};

    for (int n : sizes) {
        cout << "==================== N = " << n << " ====================\n";
        vector<int> base = make_random_array(n);

        // ---- Bubble ----
        {
            vector<int> a1 = base, a2 = base;
            long long t_seq = time_us(bubble_sort_seq, a1);
            long long t_par = time_us(bubble_sort_omp_oddeven, a2);

            cout << "Bubble (seq): " << t_seq << " us | sorted=" << (is_sorted_ok(a1) ? "OK" : "BAD") << "\n";
            cout << "Bubble (omp odd-even): " << t_par << " us | sorted=" << (is_sorted_ok(a2) ? "OK" : "BAD") << "\n";
        }

        // ---- Selection ----
        {
            vector<int> a1 = base, a2 = base;
            long long t_seq = time_us(selection_sort_seq, a1);
            long long t_par = time_us(selection_sort_omp, a2);

            cout << "Selection (seq): " << t_seq << " us | sorted=" << (is_sorted_ok(a1) ? "OK" : "BAD") << "\n";
            cout << "Selection (omp min-search): " << t_par << " us | sorted=" << (is_sorted_ok(a2) ? "OK" : "BAD") << "\n";
        }

        // ---- Insertion ----
        {
            vector<int> a1 = base, a2 = base;
            long long t_seq = time_us(insertion_sort_seq, a1);
            long long t_par = time_us(insertion_sort_omp_blocks, a2);

            cout << "Insertion (seq): " << t_seq << " us | sorted=" << (is_sorted_ok(a1) ? "OK" : "BAD") << "\n";
            cout << "Insertion (omp blocks+merge): " << t_par << " us | sorted=" << (is_sorted_ok(a2) ? "OK" : "BAD") << "\n";
        }

        cout << "\n";
    }

    cout << "Вывод: O(n^2) сортировки на 100000 могут выполняться очень долго.\n"
         << "Параллельность помогает не всегда: накладные расходы + синхронизация.\n";
    return 0;
}
