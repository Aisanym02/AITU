// Подключаем библиотеку для ввода и вывода (cout, endl)
#include <iostream>

// Подключаем функции rand(), srand()
#include <cstdlib>

// Подключаем функцию time()
#include <ctime>

// Подключаем средства для измерения времени выполнения
#include <chrono>

// Подключаем OpenMP для параллельных вычислений
#include <omp.h>

// Используем стандартное пространство имён
using namespace std;

// Используем пространство имён chrono для работы со временем
using namespace chrono;

int main() {

    // Инициализация генератора случайных чисел
    // Используем текущее время, чтобы числа были разными при каждом запуске
    srand(time(nullptr));

    // ===============================
    // ЗАДАНИЕ 1
    // ===============================

    // Размер массива
    const int N1 = 50000;

    // Динамическое выделение памяти под массив из 50 000 целых чисел
    int* arr1 = new int[N1];

    // Заполняем массив случайными числами от 1 до 100
    for (int i = 0; i < N1; i++) {
        arr1[i] = rand() % 100 + 1;
    }

    // Переменная для хранения суммы элементов массива
    long long sum1 = 0;

    // Последовательно суммируем все элементы массива
    for (int i = 0; i < N1; i++) {
        sum1 += arr1[i];
    }

    // Вычисляем среднее значение
    double avg1 = static_cast<double>(sum1) / N1;

    // Выводим результат
    cout << "Task 1 - Average value: " << avg1 << endl;

    // Освобождаем динамически выделенную память
    delete[] arr1;

    // ===============================
    // ЗАДАНИЕ 2
    // ===============================

    // Размер массива
    const int N2 = 1000000;

    // Динамическое выделение массива из 1 000 000 элементов
    int* arr2 = new int[N2];

    // Заполняем массив случайными числами
    for (int i = 0; i < N2; i++) {
        arr2[i] = rand();
    }

    // Запоминаем время начала последовательного алгоритма
    auto start_seq = high_resolution_clock::now();

    // Инициализируем минимальное и максимальное значения
    int min_val = arr2[0];
    int max_val = arr2[0];

    // Последовательный поиск минимума и максимума
    for (int i = 1; i < N2; i++) {
        if (arr2[i] < min_val) min_val = arr2[i];
        if (arr2[i] > max_val) max_val = arr2[i];
    }

    // Фиксируем время окончания алгоритма
    auto end_seq = high_resolution_clock::now();

    // Выводим найденные значения
    cout << "Task 2 - Sequential Min: " << min_val
         << ", Max: " << max_val << endl;

    // Выводим время выполнения последовательного алгоритма
    cout << "Sequential time: "
         << duration_cast<milliseconds>(end_seq - start_seq).count()
         << " ms" << endl;

    // ===============================
    // ЗАДАНИЕ 3 (OpenMP)
    // ===============================

    // Инициализируем переменные для параллельного поиска
    int min_par = arr2[0];
    int max_par = arr2[0];

    // Запоминаем время начала параллельного алгоритма
    auto start_par = high_resolution_clock::now();

    // Параллельный цикл OpenMP
    // reduction(min:min_par) — корректно объединяет минимальные значения
    // reduction(max:max_par) — корректно объединяет максимальные значения
#pragma omp parallel for reduction(min:min_par) reduction(max:max_par)
    for (int i = 0; i < N2; i++) {
        if (arr2[i] < min_par) min_par = arr2[i];
        if (arr2[i] > max_par) max_par = arr2[i];
    }

    // Фиксируем время окончания параллельного алгоритма
    auto end_par = high_resolution_clock::now();

    // Выводим результаты параллельного поиска
    cout << "Task 3 - Parallel Min: " << min_par
         << ", Max: " << max_par << endl;

    // Выводим время выполнения параллельного алгоритма
    cout << "Parallel time: "
         << duration_cast<milliseconds>(end_par - start_par).count()
         << " ms" << endl;

    // Освобождаем память
    delete[] arr2;

    // ===============================
    // ЗАДАНИЕ 4
    // ===============================

    // Размер массива
    const int N3 = 5000000;

    // Динамическое выделение массива
    int* arr3 = new int[N3];

    // Заполняем массив случайными числами от 0 до 99
    for (int i = 0; i < N3; i++) {
        arr3[i] = rand() % 100;
    }

    // ---- Последовательное вычисление среднего ----

    // Запоминаем время начала
    auto start_avg_seq = high_resolution_clock::now();

    // Переменная для суммы
    long long sum_seq = 0;

    // Последовательное суммирование
    for (int i = 0; i < N3; i++) {
        sum_seq += arr3[i];
    }

    // Вычисляем среднее значение
    double avg_seq = static_cast<double>(sum_seq) / N3;

    // Время окончания
    auto end_avg_seq = high_resolution_clock::now();

    // ---- Параллельное вычисление среднего ----

    // Запоминаем время начала
    auto start_avg_par = high_resolution_clock::now();

    // Переменная для параллельной суммы
    long long sum_par = 0;

    // Параллельное суммирование с использованием reduction
#pragma omp parallel for reduction(+:sum_par)
    for (int i = 0; i < N3; i++) {
        sum_par += arr3[i];
    }

    // Вычисляем среднее значение
    double avg_par = static_cast<double>(sum_par) / N3;

    // Время окончания
    auto end_avg_par = high_resolution_clock::now();

    // Вывод последовательного результата
    cout << "Task 4 - Sequential avg: " << avg_seq
         << ", time: "
         << duration_cast<milliseconds>(end_avg_seq - start_avg_seq).count()
         << " ms" << endl;

    // Вывод параллельного результата
    cout << "Task 4 - Parallel avg: " << avg_par
         << ", time: "
         << duration_cast<milliseconds>(end_avg_par - start_avg_par).count()
         << " ms" << endl;

    // Освобождаем память
    delete[] arr3;

    // Завершаем программу
    return 0;
}
