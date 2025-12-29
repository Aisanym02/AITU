#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <limits>
#include <omp.h>

using namespace std;

int rand_int(int l, int r) {
    static random_device rd;
    static mt19937 gen(rd());
    uniform_int_distribution<int> dist(l, r);
    return dist(gen);
}

void print_array(const vector<int>& a) {
    for (size_t i = 0; i < a.size(); ++i) {
        cout << a[i] << (i + 1 == a.size() ? '\n' : ' ');
    }
}

// ======================= Часть 1: Массивы =======================

pair<int,int> minmax_sequential(const vector<int>& a) {
    int mn = numeric_limits<int>::max();
    int mx = numeric_limits<int>::min();
    for (int x : a) {
        if (x < mn) mn = x;
        if (x > mx) mx = x;
    }
    return {mn, mx};
}

// Вариант через parallel for + критические секции (просто и понятно)
pair<int,int> minmax_parallel(const vector<int>& a) {
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

void part1_arrays(int N) {
    cout << "===== ЧАСТЬ 1: МАССИВЫ =====\n";
    vector<int> a(N);
    for (int i = 0; i < N; ++i) a[i] = rand_int(1, 100);

    cout << "Массив:\n";
    print_array(a);

    // Последовательно
    auto t1 = chrono::high_resolution_clock::now();
    auto seq = minmax_sequential(a);
    auto t2 = chrono::high_resolution_clock::now();

    // Параллельно
    auto t3 = chrono::high_resolution_clock::now();
    auto par = minmax_parallel(a);
    auto t4 = chrono::high_resolution_clock::now();

    auto seq_ms = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
    auto par_ms = chrono::duration_cast<chrono::microseconds>(t4 - t3).count();

    cout << "Последовательно: min=" << seq.first << " max=" << seq.second
         << " | time=" << seq_ms << " us\n";
    cout << "Параллельно(OpenMP): min=" << par.first << " max=" << par.second
         << " | time=" << par_ms << " us\n\n";
}

// ======================= Часть 2: Структуры данных =======================

// ---------- Односвязный список ----------
struct Node {
    int value;
    Node* next;
    Node(int v) : value(v), next(nullptr) {}
};

class SinglyLinkedList {
private:
    Node* head;
public:
    SinglyLinkedList() : head(nullptr) {}

    // Добавление в начало (O(1))
    void push_front(int v) {
        Node* n = new Node(v);
        n->next = head;
        head = n;
    }

    // Поиск (O(n))
    bool find(int v) const {
        Node* cur = head;
        while (cur) {
            if (cur->value == v) return true;
            cur = cur->next;
        }
        return false;
    }

    // Удаление первого вхождения (O(n))
    bool remove(int v) {
        Node* cur = head;
        Node* prev = nullptr;
        while (cur) {
            if (cur->value == v) {
                if (prev) prev->next = cur->next;
                else head = cur->next;
                delete cur;
                return true;
            }
            prev = cur;
            cur = cur->next;
        }
        return false;
    }

    // Печать (ограничим вывод, чтобы не спамить)
    void print_limited(int limit = 30) const {
        Node* cur = head;
        int cnt = 0;
        while (cur && cnt < limit) {
            cout << cur->value << ' ';
            cur = cur->next;
            cnt++;
        }
        if (cur) cout << "...";
        cout << "\n";
    }

    // Очистка памяти
    ~SinglyLinkedList() {
        Node* cur = head;
        while (cur) {
            Node* nxt = cur->next;
            delete cur;
            cur = nxt;
        }
    }
};

// ---------- Стек (на динамическом массиве) ----------
class Stack {
private:
    int* data;
    int cap;
    int topIndex;
public:
    Stack(int capacity = 1000) : cap(capacity), topIndex(-1) {
        data = new int[cap];
    }

    void push(int v) {
        if (topIndex + 1 >= cap) {
            cout << "Stack overflow!\n";
            return;
        }
        data[++topIndex] = v;
    }

    int pop() {
        if (isEmpty()) {
            cout << "Stack underflow!\n";
            return -1;
        }
        return data[topIndex--];
    }

    bool isEmpty() const { return topIndex < 0; }

    ~Stack() { delete[] data; }
};

// ---------- Очередь (кольцевой буфер) ----------
class Queue {
private:
    int* data;
    int cap;
    int head;
    int tail;
    int count;
public:
    Queue(int capacity = 10000) : cap(capacity), head(0), tail(0), count(0) {
        data = new int[cap];
    }

    bool isEmpty() const { return count == 0; }

    // Добавление в конец
    void enqueue(int v) {
        if (count == cap) {
            cout << "Queue overflow!\n";
            return;
        }
        data[tail] = v;
        tail = (tail + 1) % cap;
        count++;
    }

    // Удаление из начала
    int dequeue() {
        if (isEmpty()) {
            cout << "Queue underflow!\n";
            return -1;
        }
        int v = data[head];
        head = (head + 1) % cap;
        count--;
        return v;
    }

    ~Queue() { delete[] data; }
};

// Параллельное добавление: корректно, но с синхронизацией (иначе гонки данных!)
void parallel_push_list(SinglyLinkedList& lst, int M) {
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        int x = rand_int(1, 100);
        #pragma omp critical
        lst.push_front(x);
    }
}

void parallel_enqueue_queue(Queue& q, int M) {
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        int x = rand_int(1, 100);
        #pragma omp critical
        q.enqueue(x);
    }
}

void part2_structures(int M) {
    cout << "===== ЧАСТЬ 2: СТРУКТУРЫ ДАННЫХ =====\n";

    // Список: операции
    SinglyLinkedList lst;
    lst.push_front(10);
    lst.push_front(20);
    lst.push_front(30);
    cout << "Список после добавлений: ";
    lst.print_limited();

    cout << "find(20) = " << (lst.find(20) ? "true" : "false") << "\n";
    cout << "remove(20) = " << (lst.remove(20) ? "true" : "false") << "\n";
    cout << "Список после удаления 20: ";
    lst.print_limited();

    // Стек
    Stack st(10);
    st.push(1); st.push(2); st.push(3);
    cout << "Stack pop: " << st.pop() << "\n";
    cout << "Stack pop: " << st.pop() << "\n";
    cout << "Stack isEmpty: " << (st.isEmpty() ? "true" : "false") << "\n";

    // Очередь
    Queue q(50);
    q.enqueue(5); q.enqueue(6); q.enqueue(7);
    cout << "Queue dequeue: " << q.dequeue() << "\n";
    cout << "Queue isEmpty: " << (q.isEmpty() ? "true" : "false") << "\n\n";

    // Производительность: последовательное vs параллельное добавление
    cout << "== Сравнение добавления " << M << " элементов ==\n";

    // Список: последовательное
    SinglyLinkedList lst_seq;
    auto t1 = chrono::high_resolution_clock::now();
    for (int i = 0; i < M; ++i) lst_seq.push_front(rand_int(1, 100));
    auto t2 = chrono::high_resolution_clock::now();

    // Список: параллельное (с critical)
    SinglyLinkedList lst_par;
    auto t3 = chrono::high_resolution_clock::now();
    parallel_push_list(lst_par, M);
    auto t4 = chrono::high_resolution_clock::now();

    auto seq_us = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
    auto par_us = chrono::duration_cast<chrono::microseconds>(t4 - t3).count();
    cout << "Список: seq=" << seq_us << " us | par=" << par_us << " us\n";

    // Очередь: последовательное
    Queue q_seq(M + 10);
    auto t5 = chrono::high_resolution_clock::now();
    for (int i = 0; i < M; ++i) q_seq.enqueue(rand_int(1, 100));
    auto t6 = chrono::high_resolution_clock::now();

    // Очередь: параллельное (с critical)
    Queue q_par(M + 10);
    auto t7 = chrono::high_resolution_clock::now();
    parallel_enqueue_queue(q_par, M);
    auto t8 = chrono::high_resolution_clock::now();

    auto seq2_us = chrono::duration_cast<chrono::microseconds>(t6 - t5).count();
    auto par2_us = chrono::duration_cast<chrono::microseconds>(t8 - t7).count();
    cout << "Очередь: seq=" << seq2_us << " us | par=" << par2_us << " us\n\n";

    cout << "Примечание: из-за #pragma omp critical параллельное добавление\n"
         << "может не дать ускорения — это нормально, т.к. есть блокировки.\n\n";
}

// ======================= Часть 3: Динамическая память и указатели =======================

double average_sequential(const int* arr, int N) {
    long long sum = 0;
    for (int i = 0; i < N; ++i) sum += arr[i];
    return (double)sum / N;
}

double average_parallel(const int* arr, int N) {
    long long sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; ++i) {
        sum += arr[i];
    }
    return (double)sum / N;
}

void part3_dynamic(int N) {
    cout << "===== ЧАСТЬ 3: ДИНАМИЧЕСКАЯ ПАМЯТЬ И УКАЗАТЕЛИ =====\n";

    // 1) динамический массив
    int* arr = new int[N];
    for (int i = 0; i < N; ++i) arr[i] = rand_int(1, 100);

    // 2) среднее последовательно
    auto t1 = chrono::high_resolution_clock::now();
    double avg_seq = average_sequential(arr, N);
    auto t2 = chrono::high_resolution_clock::now();

    // 3) среднее параллельно reduction
    auto t3 = chrono::high_resolution_clock::now();
    double avg_par = average_parallel(arr, N);
    auto t4 = chrono::high_resolution_clock::now();

    auto seq_us = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
    auto par_us = chrono::duration_cast<chrono::microseconds>(t4 - t3).count();

    cout << "Average seq = " << avg_seq << " | time=" << seq_us << " us\n";
    cout << "Average par = " << avg_par << " | time=" << par_us << " us\n";

    // 4) освобождение памяти
    delete[] arr;
    cout << "Память освобождена через delete[].\n\n";
}

// ======================= main =======================
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cout << "OpenMP threads available: " << omp_get_max_threads() << "\n\n";

    int N = 30;      // размер массива для вывода
    int M = 200000;  // сколько элементов добавлять в структуры (для сравнения)
    int D = 5000000; // размер динамического массива для среднего (больше = заметнее ускорение)

    part1_arrays(N);
    part2_structures(M);
    part3_dynamic(D);

    return 0;
}
