#include <omp.h>
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace std;

// decomposition for a block
void blockDecomposition(double* A, double* L, double* U, int n) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                L[i * n + j] = 1.0;
            } else {
                L[i * n + j] = 0.0;
            }
            U[i * n + j] = 0.0;
        }
    }

    for (int k = 0; k < n; k++) {
        #pragma omp parallel for
        for (int j = k; j < n; j++) {
            double sum = 0.0;
            for (int p = 0; p < k; p++) {
                sum += L[k * n + p] * U[p * n + j];
            }
            U[k * n + j] = A[k * n + j] - sum;
        }
        #pragma omp parallel for
        for (int i = k + 1; i < n; i++) {
            double sum = 0.0;
            for (int p = 0; p < k; p++) {
                sum += L[i * n + p] * U[p * n + k];
            }
            L[i * n + k] = (A[i * n + k] - sum) / U[k * n + k];
        }
    }
}

// block LU Decomposition
void LU_Decomposition(double* A, double* L, double* U, int n, int block_size_param) {
    int block_size = (block_size_param > 0) ? block_size_param : max(n/10, 1);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                L[i * n + j] = 1.0;
            } else {
                L[i * n + j] = 0.0;
            }
            U[i * n + j] = 0.0;
        }
    }

    int max_block_size = min(block_size, n);
    double* A_kk = new double[max_block_size * max_block_size];
    double* L_kk = new double[max_block_size * max_block_size];
    double* U_kk = new double[max_block_size * max_block_size];

    for (int k = 0; k < n; k += block_size) {
        int current_block_size = min(block_size, n - k);
        int remaining_size = n - (k + current_block_size);

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < current_block_size; i++) {
            for (int j = 0; j < current_block_size; j++) {
                A_kk[i * current_block_size + j] = A[(k + i) * n + (k + j)];
            }
        }

        blockDecomposition(A_kk, L_kk, U_kk, current_block_size);

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < current_block_size; i++) {
            for (int j = 0; j < current_block_size; j++) {
                L[(k + i) * n + (k + j)] = L_kk[i * current_block_size + j];
                U[(k + i) * n + (k + j)] = U_kk[i * current_block_size + j];
            }
        }

        if (remaining_size > 0) {
            double* A_k_rest = new double[current_block_size * remaining_size];
            double* A_rest_k = new double[remaining_size * current_block_size];
            double* A_rest_rest = new double[remaining_size * remaining_size];
            double* L_rest_k = new double[remaining_size * current_block_size];
            double* U_k_rest = new double[current_block_size * remaining_size];
            
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < current_block_size; i++) {
                for (int j = 0; j < remaining_size; j++) {
                    A_k_rest[i * remaining_size + j] = A[(k + i) * n + (k + current_block_size + j)];
                }
            }

            #pragma omp parallel for collapse(2)
            for (int i = 0; i < remaining_size; i++) {
                for (int j = 0; j < current_block_size; j++) {
                    A_rest_k[i * current_block_size + j] = A[(k + current_block_size + i) * n + (k + j)];
                }
            }

            #pragma omp parallel for collapse(2)
            for (int i = 0; i < remaining_size; i++) {
                for (int j = 0; j < remaining_size; j++) {
                    A_rest_rest[i * remaining_size + j] = A[(k + current_block_size + i) * n + (k + current_block_size + j)];
                }
            }
            #pragma omp parallel for
            for (int j = 0; j < remaining_size; j++) {
                for (int i = 0; i < current_block_size; i++) {
                    double sum = 0.0;
                    for (int p = 0; p < i; p++) {
                        sum += L_kk[i * current_block_size + p] * U_k_rest[p * remaining_size + j];
                    }
                    U_k_rest[i * remaining_size + j] = A_k_rest[i * remaining_size + j] - sum;
                }
            }

            #pragma omp parallel for collapse(2)
            for (int i = 0; i < current_block_size; i++) {
                for (int j = 0; j < remaining_size; j++) {
                    U[(k + i) * n + (k + current_block_size + j)] = U_k_rest[i * remaining_size + j];
                }
            }

            #pragma omp parallel for
            for (int i = 0; i < remaining_size; i++) {
                for (int j = 0; j < current_block_size; j++) {
                    double sum = 0.0;
                    for (int p = 0; p < j; p++) {
                        sum += L_rest_k[i * current_block_size + p] * U_kk[p * current_block_size + j];
                    }
                    L_rest_k[i * current_block_size + j] = (A_rest_k[i * current_block_size + j] - sum) / U_kk[j * current_block_size + j];
                }
            }

            #pragma omp parallel for collapse(2)
            for (int i = 0; i < remaining_size; i++) {
                for (int j = 0; j < current_block_size; j++) {
                    L[(k + current_block_size + i) * n + (k + j)] = L_rest_k[i * current_block_size + j];
                }
            }

            #pragma omp parallel for collapse(2)
            for (int i = 0; i < remaining_size; i++) {
                for (int j = 0; j < remaining_size; j++) {
                    double sum = 0.0;
                    for (int p = 0; p < current_block_size; p++) {
                        sum += L_rest_k[i * current_block_size + p] * U_k_rest[p * remaining_size + j];
                    }
                    A_rest_rest[i * remaining_size + j] -= sum;
                }
            }

            #pragma omp parallel for collapse(2)
            for (int i = 0; i < remaining_size; i++) {
                for (int j = 0; j < remaining_size; j++) {
                    A[(k + current_block_size + i) * n + (k + current_block_size + j)] = A_rest_rest[i * remaining_size + j];
                }
            }
            
            delete[] A_k_rest;
            delete[] A_rest_k;
            delete[] A_rest_rest;
            delete[] L_rest_k;
            delete[] U_k_rest;
        }
    }
    
    delete[] A_kk;
    delete[] L_kk;
    delete[] U_kk;
}
