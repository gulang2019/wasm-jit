
__global__ void row_sums(const double* A, double* rowSums, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        double sumExp = 0.0;
        for (int col = 0; col < N; ++col) {
            sumExp += exp(A[row * N + col]);
        }
        rowSums[row] = sumExp;
    }
}