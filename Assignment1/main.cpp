// Подключаем стандартную библиотеку для ввода/вывода (cout, endl)
#include <iostream>

// Подключаем библиотеку для генерации случайных чисел (rand, srand)
#include <cstdlib>

// Подключаем библиотеку для работы со временем (time)
#include <ctime>

// Подключаем библиотеку для замера времени выполнения
#include <chrono>

// Подключаем библиотеку для работы с предельными значениями int
#include <limits>

// Если компилятор поддерживает OpenMP
#ifdef _OPENMP
// Подключаем библиотеку OpenMP
#include <omp.h>
#endif

// Используем стандартное пространство имён
using namespace std;

// Используем пространство имён chrono для замера времени
using namespace std::chrono;

// ======================================================
// ФУНКЦИЯ: заполнение массива случайными числами [1..100]
// ======================================================
void fill_random(int* arr, size_t n) {
    // Цикл по всем элементам массива
    for (size_t i = 0; i < n; i++) {
        // Генерируем случайное число от 1 до 100
        arr[i] = rand() % 100 + 1;
    }
}

// ======================================================
// ЗАДАНИЕ 1
// Создание динамического массива из 50 000 элементов
// Нахождение среднего значения
// ======================================================
void task1() {
    // Размер массива
    const size_t N = 50000;

    // Динамическое выделение памяти под массив
    int* arr = new int[N];

    // Заполняем массив случайными числами
    fill_random(arr, N);

    // Переменная для хранения суммы элементов
    long long sum = 0;

    // Считаем сумму элементов массива
    for (size_t i = 0; i < N; i++) {
        sum += arr[i];
    }

    // Вычисляем среднее значение
    double avg = static_cast<double>(sum) / static_cast<double>(N);

    // Выводим результат
    cout << "Task 1 - Average value (N=50000): " << avg << endl;

    // Освобождаем динамически выделенную память
    delete[] arr;
}

// ======================================================
// ЗАДАНИЕ 2
// Последовательный поиск min и max в массиве
// ======================================================
void task2_sequential_minmax(const int* arr, size_t n, int &mn, int &mx) {
    // Инициализируем минимальное значение максимально возможным int
    mn = numeric_limits<int>::max();

    // Инициализируем максимальное значение минимально возможным int
    mx = numeric_limits<int>::min();

    // Проходим по массиву
    for (size_t i = 0; i < n; i++) {
        // Проверка на минимум
        if (arr[i] < mn) mn = arr[i];

        // Проверка на максимум
        if (arr[i] > mx) mx = arr[i];
    }
}

// ======================================================
// Запуск задания 2 с замером времени
// ======================================================
void task2() {
    // Размер массива
    const size_t N = 1000000;

    // Выделяем память под массив
    int* arr = new int[N];

    // Заполняем массив случайными числами
    fill_random(arr, N);

    // Переменные для min и max
    int mn, mx;

    // Запоминаем время начала
    auto t1 = high_resolution_clock::now();

    // Последовательный поиск min/max
    task2_sequential_minmax(arr, N, mn, mx);

    // Запоминаем время окончания
    auto t2 = high_resolution_clock::now();

    // Вычисляем длительность выполнения в миллисекундах
    auto ms = duration_cast<milliseconds>(t2 - t1).count();

    // Выводим результат
    cout << "Task 2 - Sequential min/max: min=" << mn
         << ", max=" << mx
         << ", time=" << ms << " ms" << endl;

    // Освобождаем память
    delete[] arr;
}

// ======================================================
// ЗАДАНИЕ 3
// Параллельный поиск min/max с OpenMP
// ======================================================
void task3_parallel_minmax(const int* arr, size_t n, int &mn, int &mx) {
    // Инициализация глобальных min/max
    mn = numeric_limits<int>::max();
    mx = numeric_limits<int>::min();

#ifdef _OPENMP
    // Параллельная область
    #pragma omp parallel
    {
        // Локальный минимум для потока
        int local_min = numeric_limits<int>::max();

        // Локальный максимум для потока
        int local_max = numeric_limits<int>::min();

        // Распределяем цикл между потоками
        #pragma omp for nowait
        for (long long i = 0; i < (long long)n; i++) {
            if (arr[i] < local_min) local_min = arr[i];
            if (arr[i] > local_max) local_max = arr[i];
        }

        // Критическая секция — безопасное обновление общих данных
        #pragma omp critical
        {
            if (local_min < mn) mn = local_min;
            if (local_max > mx) mx = local_max;
        }
    }
#else
    // Если OpenMP не поддерживается — обычный последовательный вариант
    task2_sequential_minmax(arr, n, mn, mx);
#endif
}

// ======================================================
// Сравнение sequential и parallel
// ======================================================
void task3() {
    const size_t N = 1000000;
    int* arr = new int[N];

    fill_random(arr, N);

    int mn1, mx1;
    auto s1 = high_resolution_clock::now();
    task2_sequential_minmax(arr, N, mn1, mx1);
    auto s2 = high_resolution_clock::now();

    int mn2, mx2;
    auto p1 = high_resolution_clock::now();
    task3_parallel_minmax(arr, N, mn2, mx2);
    auto p2 = high_resolution_clock::now();

#ifdef _OPENMP
    cout << "OpenMP threads: " << omp_get_max_threads() << endl;
#endif

    cout << "Task 3 - Sequential time: "
         << duration_cast<milliseconds>(s2 - s1).count() << " ms" << endl;

    cout << "Task 3 - Parallel time: "
         << duration_cast<milliseconds>(p2 - p1).count() << " ms" << endl;

    delete[] arr;
}

// ======================================================
// ЗАДАНИЕ 4
// Среднее значение: sequential vs OpenMP reduction
// ======================================================
double avg_sequential(const int* arr, size_t n) {
    long long sum = 0;
    for (size_t i = 0; i < n; i++) {
        sum += arr[i];
    }
    return (double)sum / n;
}

double avg_parallel(const int* arr, size_t n) {
#ifdef _OPENMP
    long long sum = 0;

    // reduction — безопасное суммирование между потоками
    #pragma omp parallel for reduction(+:sum)
    for (long long i = 0; i < (long long)n; i++) {
        sum += arr[i];
    }

    return (double)sum / n;
#else
    return avg_sequential(arr, n);
#endif
}

// ======================================================
// Главная функция
// ======================================================
int main() {
    // Инициализация генератора случайных чисел
    srand(time(nullptr));

    // Запуск всех заданий
    task1();
    task2();
    task3();

    // Размер массива 5 000 000
    const size_t N = 5000000;
    int* arr = new int[N];
    fill_random(arr, N);

    auto s1 = high_resolution_clock::now();
    double a1 = avg_sequential(arr, N);
    auto s2 = high_resolution_clock::now();

    auto p1 = high_resolution_clock::now();
    double a2 = avg_parallel(arr, N);
    auto p2 = high_resolution_clock::now();

    cout << "Task 4 - Sequential avg: " << a1
         << ", time=" << duration_cast<milliseconds>(s2 - s1).count() << " ms" << endl;

    cout << "Task 4 - Parallel avg: " << a2
         << ", time=" << duration_cast<milliseconds>(p2 - p1).count() << " ms" << endl;

    delete[] arr;

    return 0;
}
