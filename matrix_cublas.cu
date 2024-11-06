#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void matrixMultiCuBLAS(float *M, float *N, float *P, int Width) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t handle;

    cublasCreate(&handle);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Width, Width, Width, &alpha, M, Width, N, Width, &beta, P, Width);

    cublasDestroy(handle);
}

int main() {
    int sizes[] = {128, 256, 512, 1024};

    for (int s = 0; s < 4; s++) {
        int Width = sizes[s];
        int size = Width * Width * sizeof(float);
        float *M, *N, *P;

        cudaMallocManaged(&M, size);
        cudaMallocManaged(&N, size);
        cudaMallocManaged(&P, size);

        // Initialize M and N
        for (int i = 0; i < Width * Width; i++) {
            M[i] = 1.0f;
            N[i] = 2.0f;
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        matrixMultiCuBLAS(M, N, P, Width);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // Print time 
        printf("Matrix size: %d x %d, Time taken (cuBLAS): %f ms\n", Width, Width, milliseconds);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaFree(M);
        cudaFree(N);
        cudaFree(P);

        printf("Matrix multiplication on cuBLAS for size %d completed.\n\n", Width);
    }

    return 0;
}
