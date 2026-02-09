#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <numeric>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // N можно передать параметром: mpirun -np 4 ./task1 1000000
    long long N = 1000000;
    if (argc >= 2) N = std::atoll(argv[1]);
    if (N <= 0) N = 1000000;

    std::vector<double> full;           // только у rank 0
    std::vector<int> counts(size), displs(size);

    // Разбиение N на size с учетом остатка (Scatterv)
    long long base = N / size;
    long long rem  = N % size;
    for (int i = 0; i < size; i++) {
        counts[i] = static_cast<int>(base + (i < rem ? 1 : 0));
    }
    displs[0] = 0;
    for (int i = 1; i < size; i++) displs[i] = displs[i - 1] + counts[i - 1];

    // Локальный буфер
    std::vector<double> local(counts[rank]);

    if (rank == 0) {
        full.resize(N);
        std::mt19937_64 rng(12345);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (long long i = 0; i < N; i++) full[i] = dist(rng);
    }

    double t0 = MPI_Wtime();

    // Scatterv: делим массив даже если N не делится нацело
    MPI_Scatterv(
        rank == 0 ? full.data() : nullptr,
        counts.data(),
        displs.data(),
        MPI_DOUBLE,
        local.data(),
        counts[rank],
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    // Локальные суммы и суммы квадратов
    long double local_sum = 0.0L;
    long double local_sumsq = 0.0L;
    for (double x : local) {
        local_sum += x;
        local_sumsq += (long double)x * (long double)x;
    }

    long double global_sum = 0.0L;
    long double global_sumsq = 0.0L;

    MPI_Reduce(&local_sum,   &global_sum,   1, MPI_LONG_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sumsq, &global_sumsq, 1, MPI_LONG_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double t1 = MPI_Wtime();

    if (rank == 0) {
        long double mean = global_sum / (long double)N;
        // std = sqrt( (sum(x^2)/N) - mean^2 )
        long double ex2 = global_sumsq / (long double)N;
        long double var = ex2 - mean * mean;
        if (var < 0) var = 0; // защита от -0 из-за числ. ошибок
        long double stdev = std::sqrt((double)var);

        std::cout << "N=" << N << ", processes=" << size << "\n";
        std::cout << "Mean: " << (double)mean << "\n";
        std::cout << "StdDev: " << (double)stdev << "\n";
        std::cout << "Execution time: " << (t1 - t0) << " seconds.\n";
    }

    MPI_Finalize();
    return 0;
}
