#include <stdio.h>
#include <cuda.h>

__global__ void matrixMultiTiledKernel(float *d_M, float *d_N, float *d_P, int Width, int TILE_WIDTH) {
    extern __shared__ float shared_mem[];
    float* Mds = shared_mem;
    float* Nds = &Mds[TILE_WIDTH * TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    float Pvalue = 0;

    for (int m = 0; m < (Width + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        if (Row < Width && m * TILE_WIDTH + tx < Width)
            Mds[ty * TILE_WIDTH + tx] = d_M[Row * Width + m * TILE_WIDTH + tx];
        else
            Mds[ty * TILE_WIDTH + tx] = 0.0;

        if (Col < Width && m * TILE_WIDTH + ty < Width)
            Nds[ty * TILE_WIDTH + tx] = d_N[(m * TILE_WIDTH + ty) * Width + Col];
        else
            Nds[ty * TILE_WIDTH + tx] = 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty * TILE_WIDTH + k] * Nds[k * TILE_WIDTH + tx];
        }
        __syncthreads();
    }
    if (Row < Width && Col < Width) {
        d_P[Row * Width + Col] = Pvalue;
    }
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

        for (int i = 0; i < Width * Width; i++) {
            M[i] = 1.0f;
            N[i] = 2.0f;
        }

        int TILE_WIDTH = (Width >= 512) ? 32 : 16;
        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
        dim3 dimGrid((Width + TILE_WIDTH - 1) / TILE_WIDTH, (Width + TILE_WIDTH - 1) / TILE_WIDTH);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        size_t sharedMemSize = 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float);
        matrixMultiTiledKernel<<<dimGrid, dimBlock, sharedMemSize>>>(M, N, P, Width, TILE_WIDTH);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Matrix size: %d x %d, Time taken (GPU): %f ms\n", Width, Width, milliseconds);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaFree(M);
        cudaFree(N);
        cudaFree(P);

        printf("Matrix multiplication on Tiled GPU for size %d completed.\n\n", Width);
    }

    return 0;
}
