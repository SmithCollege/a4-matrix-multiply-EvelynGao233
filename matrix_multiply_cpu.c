#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int sizes[] = {128, 256, 512, 1024};

void matrixMultiCPU(float *M, float *N, float *P, int Width) {
    for (int i = 0; i < Width; ++i) {
        for (int j = 0; j < Width; ++j) {
            float sum = 0;
            for (int k = 0; k < Width; ++k) {
                sum += M[i * Width + k] * N[k * Width + j];
            }
            P[i * Width + j] = sum;
        }
    }
}

int main() {
    for (int s = 0; s < 4; s++) {
        int Width = sizes[s];
        float *M = (float *)malloc(Width * Width * sizeof(float));
        float *N = (float *)malloc(Width * Width * sizeof(float));
        float *P = (float *)malloc(Width * Width * sizeof(float));

        // Initialize matrices M and N
        for (int i = 0; i < Width * Width; i++) {
            M[i] = 1.0f;
            N[i] = 2.0f;
        }

        // Timing start
        clock_t start = clock();

        // Perform matrix multiplication on CPU
        matrixMultiCPU(M, N, P, Width);

        // Timing end
        clock_t end = clock();
        double total_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;

        // Print the time taken for each size
        printf("Width: %d, Time taken (CPU): %f ms\n", Width, total_time);

        printf("Matrix multiplication on CPU completed.\n");

        free(M);
        free(N);
        free(P);
    }
    return 0;
}
