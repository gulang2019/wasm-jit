
__global__
void matmul_shared(
    double* A,
    double* B,
    double* C,
    int M, int N, int K
) {
    __shared__ double As[32][32];
    __shared__ double Bs[32][32];

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // M dimension
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // N dimension

    if (idx < M && idy < N) {
        double sum = 0;
        for (int k = 0; k < K; k += 32) {
            // Load tiles into shared memory with boundary checks
            if (k + threadIdx.x < K)
                As[threadIdx.y][threadIdx.x] = A[idx * K + k + threadIdx.x];
            else
                As[threadIdx.y][threadIdx.x] = 0;

            if (k + threadIdx.y < K)
                Bs[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * N + idy];
            else
                Bs[threadIdx.y][threadIdx.x] = 0;

            __syncthreads();

            // Compute partial results for the current tile
            for (int kk = 0; kk < 32; kk++) {
                sum += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];
            }
            __syncthreads();
        }

        // Store result in C
        C[idx * N + idy] = sum;
    }
}