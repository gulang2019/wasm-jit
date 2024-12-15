#include "test.h"

void compute_ground_truth(
    double* hA, 
    double* hB,
    double* hOut_gt, 
    int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += hA[i * K + k] * hB[k * N + j];
            }
            hOut_gt[i * N + j] = sum;
        }
    }
}

bool Matmul::_run(
    CUfunction kernel
){
    CUdeviceptr dA, dB, dOut, dOut_gt;
    int M = 128, N = 128, K = 128;
    size_t i;
    double * hA, * hB, * hOut, * hOut_gt;
    
    hA = (double*)malloc(M * K * sizeof(double));
    hB = (double*)malloc(K * N * sizeof(double));
    hOut = (double*)malloc(M * N * sizeof(double));
    hOut_gt = (double*)malloc(M * N * sizeof(double));

    void* args[6];

    // Generate input for execution, and create output buffers.
    for (i = 0; i < M * K; ++i) {
        hA[i] = (double)i;
    }
    for (i = 0; i < K * N; ++i) {
        hB[i] = (double)i;
    }
    CUDA_SAFE_CALL(cuMemAlloc(&dA, M * K * sizeof(double)));
    CUDA_SAFE_CALL(cuMemAlloc(&dB, K * N * sizeof(double)));
    CUDA_SAFE_CALL(cuMemAlloc(&dOut, M * N * sizeof(double)));    
    
    CUDA_SAFE_CALL(cuMemcpyHtoD(dA, hA, M * K * sizeof(double)));
    CUDA_SAFE_CALL(cuMemcpyHtoD(dB, hB, K * N * sizeof(double)));

    args[0] = &dA;
    args[1] = &dB;
    args[2] = &dOut;
    args[3] = &M;
    args[4] = &N;
    args[5] = &K;

    CUDA_SAFE_CALL( cuLaunchKernel(kernel,
                                (M + 31) / 32,  (N + 31) / 32, 1, // grid dim
                                32, 32, 1, // block dim
                                0, NULL, // shared mem and stream
                                args, 0)); // arguments
    CUDA_SAFE_CALL(cuCtxSynchronize()); // Retrieve and print output.
    CUDA_SAFE_CALL(cuMemcpyDtoH(hOut, dOut, M * N * sizeof(double)));

    compute_ground_truth(hA, hB, hOut_gt, M, N, K);
    bool is_all_close = all_close(hOut, hOut_gt, M * N);
    
    // Release resources.
    CUDA_SAFE_CALL(cuMemFree(dA));
    CUDA_SAFE_CALL(cuMemFree(dB));
    CUDA_SAFE_CALL(cuMemFree(dOut));
    free(hA);
    free(hB);
    free(hOut);
    free(hOut_gt);
    return is_all_close;

}
