#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

static inline int owner_of_row(int global_row, int rows_per_proc) {
    return global_row / rows_per_proc;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 512; // размер матрицы
    if (argc >= 2) N = std::atoi(argv[1]);
    if (N <= 0) N = 512;

    if (N % size != 0) {
        if (rank == 0) {
            std::cout << "For MPI_Scatter in this task, N must be divisible by number of processes.\n";
            std::cout << "Given N=" << N << ", processes=" << size << ".\n";
        }
        MPI_Finalize();
        return 0;
    }

    int rows_per_proc = N / size;
    int local_rows = rows_per_proc;

    // Храним расширенную матрицу [A | b] размером N x (N+1)
    std::vector<double> full_aug; // rank 0
    std::vector<double> local_aug(local_rows * (N + 1));

    if (rank == 0) {
        full_aug.resize((size_t)N * (N + 1));

        // Генерация СЛАУ с диагональным преобладанием для устойчивости
        std::mt19937_64 rng(42);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        for (int i = 0; i < N; i++) {
            double rowsum = 0.0;
            for (int j = 0; j < N; j++) {
                double v = dist(rng);
                full_aug[(size_t)i*(N+1) + j] = v;
                rowsum += std::abs(v);
            }
            // усиливаем диагональ
            full_aug[(size_t)i*(N+1) + i] += rowsum;

            // b
            full_aug[(size_t)i*(N+1) + N] = dist(rng);
        }
    }

    double t0 = MPI_Wtime();

    // Scatter строк расширенной матрицы
    MPI_Scatter(
        rank == 0 ? full_aug.data() : nullptr,
        local_rows * (N + 1),
        MPI_DOUBLE,
        local_aug.data(),
        local_rows * (N + 1),
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    // Буфер для pivot-строки (расширенная строка)
    std::vector<double> pivot(N + 1);

    // Прямой ход
    for (int k = 0; k < N; k++) {
        int pivot_owner = owner_of_row(k, rows_per_proc);

        // Если текущая pivot-строка принадлежит этому процессу — копируем в pivot
        if (rank == pivot_owner) {
            int local_k = k - rank * rows_per_proc;
            for (int j = 0; j < N + 1; j++) {
                pivot[j] = local_aug[(size_t)local_k*(N+1) + j];
            }

            // Нормализация pivot-строки
            double diag = pivot[k];
            if (std::abs(diag) < 1e-12) diag = (diag >= 0 ? 1e-12 : -1e-12);
            for (int j = k; j < N + 1; j++) pivot[j] /= diag;
            pivot[k] = 1.0;

            // записываем обратно
            for (int j = 0; j < N + 1; j++) {
                local_aug[(size_t)local_k*(N+1) + j] = pivot[j];
            }
        }

        // Рассылаем pivot-строку всем
        MPI_Bcast(pivot.data(), N + 1, MPI_DOUBLE, pivot_owner, MPI_COMM_WORLD);

        // Обнуляем элементы ниже диагонали в колонке k для локальных строк i > k
        int global_row_start = rank * rows_per_proc;
        for (int li = 0; li < local_rows; li++) {
            int i = global_row_start + li;
            if (i <= k) continue;

            double factor = local_aug[(size_t)li*(N+1) + k];
            if (std::abs(factor) < 1e-18) continue;

            for (int j = k; j < N + 1; j++) {
                local_aug[(size_t)li*(N+1) + j] -= factor * pivot[j];
            }
            local_aug[(size_t)li*(N+1) + k] = 0.0;
        }
    }

    // Собираем матрицу обратно на rank 0 для обратного хода
    MPI_Gather(
        local_aug.data(),
        local_rows * (N + 1),
        MPI_DOUBLE,
        rank == 0 ? full_aug.data() : nullptr,
        local_rows * (N + 1),
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    std::vector<double> x(N, 0.0);

    if (rank == 0) {
        // Обратный ход (последовательно на rank 0)
        for (int i = N - 1; i >= 0; i--) {
            double sum = full_aug[(size_t)i*(N+1) + N];
            for (int j = i + 1; j < N; j++) {
                sum -= full_aug[(size_t)i*(N+1) + j] * x[j];
            }
            // диагональ должна быть 1 после нормализации
            x[i] = sum;
        }
    }

    double t1 = MPI_Wtime();

    if (rank == 0) {
        std::cout << "N=" << N << ", processes=" << size << "\n";
        std::cout << "Execution time: " << (t1 - t0) << " seconds.\n";
        std::cout << "x[0..4]: ";
        for (int i = 0; i < std::min(5, N); i++) std::cout << x[i] << (i==4?'\n':' ');
        std::cout << "\n";
    }

    MPI_Finalize();
    return 0;
}
