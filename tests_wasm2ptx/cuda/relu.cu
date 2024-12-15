
__global__
void ReLU(
    double* A,
    double* B,
    int M, int N
) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        B[idx] = A[idx] > 0 ? A[idx] : 0;
    }
}