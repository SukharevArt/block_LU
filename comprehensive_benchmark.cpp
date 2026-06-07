#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cmath>
#include "recursive_block_lu.h"
#include "utils.h"

using namespace std;

double runSingleExperiment(int matrix_size, int block_size, int num_threads, double* A, double* L, double* U) {
    omp_set_num_threads(num_threads);
    
    double start_time = omp_get_wtime();
    LU_Decomposition(A, L, U, matrix_size, block_size);
    double end_time = omp_get_wtime();
    
    return (end_time - start_time) * 1000;
}

vector<double> runMultipleExperiments(int matrix_size, int block_size, int num_threads, int num_runs) {
    vector<double> times;
    
    for (int run = 0; run < num_runs; run++) {
        double* A = new double[matrix_size * matrix_size];
        double* L = new double[matrix_size * matrix_size];
        double* U = new double[matrix_size * matrix_size];
        
        srand(time(NULL) + run);
        for (int i = 0; i < matrix_size; i++) {
            for (int j = 0; j < matrix_size; j++) {
                A[i * matrix_size + j] = (double)(rand() % 1000) / 10.0 + 1.0;
            }
        }
        
        double time = runSingleExperiment(matrix_size, block_size, num_threads, A, L, U);
        times.push_back(time);
        
        delete[] A;
        delete[] L;
        delete[] U;
    }
    
    return times;
}

// Вычисление статистики
struct Statistics {
    double mean;
    double std_dev;
    double min;
    double max;
    double median;
};

Statistics computeStatistics(const vector<double>& times) {
    Statistics stats;
    int n = times.size();
    
    vector<double> sorted_times = times;
    sort(sorted_times.begin(), sorted_times.end());
    
    double sum = 0.0;
    for (double time : times) {
        sum += time;
    }
    stats.mean = sum / n;
    
    double variance = 0.0;
    for (double time : times) {
        variance += (time - stats.mean) * (time - stats.mean);
    }
    stats.std_dev = sqrt(variance / n);
    
    stats.min = sorted_times[0];
    stats.max = sorted_times[n - 1];
    
    if (n % 2 == 0) {
        stats.median = (sorted_times[n/2 - 1] + sorted_times[n/2]) / 2.0;
    } else {
        stats.median = sorted_times[n/2];
    }
    
    return stats;
}

int main() {
    cout << "Comprehensive Benchmark for Recursive Block LU Decomposition" << endl;
    cout << "=============================================================" << endl;
    cout << endl;
    
    vector<int> matrix_sizes = {100, 200, 500, 1000, 2000, 3000, 4000, 5000};
    vector<int> block_sizes = {8, 16, 32, 64, 128, 256};
    vector<int> thread_counts = {1, 2, 4, 6, 8};
    int num_runs = 10; // Количество случайных матриц для каждого теста
    
    ofstream csv_file("comprehensive_benchmark_results.csv");
    csv_file << "MatrixSize,BlockSize,Threads,MeanTime,StdDev,MinTime,MaxTime,MedianTime" << endl;
    
    for (int matrix_size : matrix_sizes) {
        cout << "Testing matrix size: " << matrix_size << "x" << matrix_size << endl;
        
        for (int block_size : block_sizes) {
            if (block_size > matrix_size) continue;
            
            for (int num_threads : thread_counts) {
                cout << "  Block size: " << block_size << ", Threads: " << num_threads << "..." << flush;
                
                vector<double> times = runMultipleExperiments(matrix_size, block_size, num_threads, num_runs);
                
                Statistics stats = computeStatistics(times);
                
                csv_file << matrix_size << "," << block_size << "," << num_threads << ","
                        << stats.mean << "," << stats.std_dev << "," << stats.min << ","
                        << stats.max << "," << stats.median << endl;
                
                cout << " Done (Mean: " << fixed << setprecision(2) << stats.mean << " ms)" << endl;
            }
        }
        cout << endl;
    }
    
    csv_file.close();
    
    cout << "Benchmark completed successfully!" << endl;
    cout << "Results saved to: comprehensive_benchmark_results.csv" << endl;
    cout << endl;
    
    // Вывод сводной статистики
    cout << "Summary Statistics:" << endl;
    cout << "==================" << endl;
    cout << "Total tests performed: " << matrix_sizes.size() * block_sizes.size() * thread_counts.size() * num_runs << endl;
    
    return 0;
}