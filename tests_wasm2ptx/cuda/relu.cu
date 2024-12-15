extern "C"{
__global__
void relu(
    double* A,
    double* B,
    int M
) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M) {
        B[idx] = A[idx] > 0 ? A[idx] : 0;
    }
}
}