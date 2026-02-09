#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

static const int INF = 1000000000;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 300; // граф NxN (для Флойда лучше не делать слишком большим)
    if (argc >= 2) N = std::atoi(argv[1]);
    if (N <= 0) N = 300;

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

    std::vector<int> full; // rank 0
    std::vector<int> local(local_rows * N);
    std::vector<int> gathered(N * N); // для Allgather (полная матрица у всех)

    if (rank == 0) {
        full.resize(N * N);
        std::mt19937 rng(7);
        std::uniform_int_distribution<int> wdist(1, 20);
        std::uniform_real_distribution<double> p(0.0, 1.0);

        // Генерация взвешенного графа
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i == j) full[i*N + j] = 0;
                else {
                    // вероятность ребра
                    if (p(rng) < 0.15) full[i*N + j] = wdist(rng);
                    else full[i*N + j] = INF;
                }
            }
        }
    }

    double t0 = MPI_Wtime();

    MPI_Scatter(
        rank == 0 ? full.data() : nullptr,
        local_rows * N,
        MPI_INT,
        local.data(),
        local_rows * N,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    // Сразу соберем стартовую матрицу у всех
    MPI_Allgather(
        local.data(), local_rows * N, MPI_INT,
        gathered.data(), local_rows * N, MPI_INT,
        MPI_COMM_WORLD
    );

    // Флойд–Уоршелл
    // local содержит строки [rank*rows_per_proc .. (rank+1)*rows_per_proc-1]
    int global_row_start = rank * rows_per_proc;

    for (int k = 0; k < N; k++) {
        // k-я строка лежит в gathered[k*N .. k*N+N-1]
        const int* row_k = &gathered[k * N];

        for (int li = 0; li < local_rows; li++) {
            int i = global_row_start + li;
            int* row_i = &local[li * N];

            int dik = row_i[k];
            if (dik >= INF) continue;

            for (int j = 0; j < N; j++) {
                int alt = dik + row_k[j];
                if (alt < row_i[j]) row_i[j] = alt;
            }
        }

        // Обновили local -> делаем Allgather для следующей итерации
        MPI_Allgather(
            local.data(), local_rows * N, MPI_INT,
            gathered.data(), local_rows * N, MPI_INT,
            MPI_COMM_WORLD
        );
    }

    // Собрать финальную матрицу на rank 0
    MPI_Gather(
        local.data(), local_rows * N, MPI_INT,
        rank == 0 ? full.data() : nullptr, local_rows * N, MPI_INT,
        0, MPI_COMM_WORLD
    );

    double t1 = MPI_Wtime();

    if (rank == 0) {
        std::cout << "N=" << N << ", processes=" << size << "\n";
        std::cout << "Execution time: " << (t1 - t0) << " seconds.\n";
        std::cout << "dist[0][0..9]: ";
        for (int j = 0; j < std::min(10, N); j++) {
            int v = full[0*N + j];
            if (v >= INF/2) std::cout << "INF ";
            else std::cout << v << " ";
        }
        std::cout << "\n";
    }

    MPI_Finalize();
    return 0;
}
