#include <stdio.h>
#include <cuda.h>

__global__ void matrixMultKernel(float *M, float *N, float *P, int Width) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (Row < Width && Col < Width) {
        float Pvalue = 0;
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[Row * Width + k] * N[k * Width + Col];
        }
        P[Row * Width + Col] = Pvalue;
    }
}

int main() {
    int sizes[] = {128, 256, 512, 1024};

    for (int s = 0; s < 4; s++) {
        int Width = sizes[s];
        int size = Width * Width * sizeof(float);
        float *M, *N, *P;

        // Allocate memory
        cudaMallocManaged(&M, size);
        cudaMallocManaged(&N, size);
        cudaMallocManaged(&P, size);

        // Initialize M and N
        for (int i = 0; i < Width * Width; i++) {
            M[i] = 1.0f;
            N[i] = 2.0f;
        }

        // Adjust block size
        int blockSize = (Width >= 512) ? 32 : 16;
        dim3 dimBlock(blockSize, blockSize);
        dim3 dimGrid((Width + dimBlock.x - 1) / dimBlock.x, (Width + dimBlock.y - 1) / dimBlock.y);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // start
        cudaEventRecord(start);

        matrixMultKernel<<<dimGrid, dimBlock>>>(M, N, P, Width);
        cudaDeviceSynchronize();

        // stop
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // calculate time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // Print the time
        printf("Matrix size: %d x %d, Block size: %d x %d, Time taken (GPU): %f ms\n", 
               Width, Width, dimBlock.x, dimBlock.y, milliseconds);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaFree(M);
        cudaFree(N);
        cudaFree(P);

        printf("Matrix multiplication on GPU for size %d completed.\n\n", Width);
    }

    return 0;
}
