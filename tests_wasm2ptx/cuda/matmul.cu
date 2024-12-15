

__global__ 
void matmul(
    double* A,
    double* B,
    double* C,
    int M, int N, int K 
) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= M * N) {
        int i = idx / N;
        int j = idx % N;
        double sum = 0;
        for (int k = 0; k < K; k++) {
            sum += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}