extern "C"{
// Kernel 2: Compute B[i,j] = exp(A[i,j]) / sumExp[i].
__global__ void softmax(const double* A, const double* rowSums, double* B, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx < total) {
        int row = idx / N;
        double val = exp(A[idx]) / rowSums[row];
        B[idx] = val;
    }
}
}