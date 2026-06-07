#include <iostream>
#include <cmath>

using namespace std;

void printMatrix(double* matrix, int n, const char* name) {
    cout << name << ":" << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << matrix[i * n + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

bool verifyLU(double* A, double* L, double* U, int n) {
    double* result = new double[n * n];
    double* diff = new double[n * n];
    
    // Multiply L and U
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i * n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                result[i * n + j] += L[i * n + k] * U[k * n + j];
            }
        }
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            diff[i * n + j] = result[i * n + j] - A[i * n + j];
        }
    }

    double norm_A = 0.0;
    double norm_diff = 0.0;
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            norm_A += A[i * n + j] * A[i * n + j];
            norm_diff += diff[i * n + j] * diff[i * n + j];
        }
    }
    
    norm_A = sqrt(norm_A);
    norm_diff = sqrt(norm_diff);
    
    bool correct = (norm_diff / norm_A) < 0.01;

    delete[] result;
    delete[] diff;
    return correct;
}
