#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include "recursive_block_lu.h"
#include "utils.h"

using namespace std;

int tryCalculateOptimalBlockSize(int n, int num_threads) {
    if (n <= 16) return n;
    if (n <= 32) return 8;
    if (n <= 64) return 16;
    if (n <= 128) return 16;
    if (n <= 256) return 32;
    if (n <= 512) return 64;
    return std::min(128, n / (num_threads * 2));
}

int main(int argc, char** argv) {
    int n = 8;
    int block_size = 0;
    
    if (argc > 1) {
        n = atoi(argv[1]);
        if (n <= 0) {
            cout << "Invalid matrix size. Using default size 8." << endl;
            n = 8;
        }
    }
    
    if (argc > 2) {
        block_size = atoi(argv[2]);
        if (block_size <= 0) {
            block_size = 0;
        } else if (block_size > n) {
            cout << "Block size too large. Using automatic selection." << endl;
            block_size = 0;
        }
    }

    int num_threads = omp_get_max_threads();
    if (block_size == 0) {
        block_size = tryCalculateOptimalBlockSize(n, num_threads);
        cout << "Block size automatically selected: " << block_size << endl;
    }
    
    cout << "Recursive Block LU Decomposition with OpenMP" << endl;
    cout << "============================================" << endl;
    cout << "Matrix size: " << n << "x" << n << endl;
    cout << "Block size: " << block_size << endl;
    cout << "Number of threads: " << num_threads << endl;
    cout << endl;

    double* A = new double[n * n];
    double* L = new double[n * n];
    double* U = new double[n * n];

    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = (double)(rand() % 100) / 10.0 + 1.0;
        }
    }

    double* A_original = new double[n * n];
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A_original[i * n + j] = A[i * n + j];
        }
    }

    if (n <= 10) {
        printMatrix(A, n, "Original Matrix A");
    }

    double start_time = omp_get_wtime();
    LU_Decomposition(A, L, U, n);
    double end_time = omp_get_wtime();

    if (n <= 10) {
        printMatrix(L, n, "Lower Triangular Matrix L");
        printMatrix(U, n, "Upper Triangular Matrix U");
    }

    bool correct = verifyLU(A_original, L, U, n);
    
    delete[] A_original;
    
    cout << "Verification: " << (correct ? "PASSED" : "FAILED") << endl;
    cout << "Execution time: " << (end_time - start_time) * 1000 << " ms" << endl;

    delete[] A;
    delete[] L;
    delete[] U;

    return 0;
}
