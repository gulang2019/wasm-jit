#include "test.h"

void compute_ground_truth(
    double* hA, 
    double* rowSums,
    double* hOut_gt, 
    int M, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            hOut_gt[i*N+j] = exp(hA[i*N+j]) / rowSums[i];
        }
    }
}

bool Softmax::_run(
    CUfunction kernel
){
    CUdeviceptr dA, dB, dOut, dOut_gt;
    int M = 128, N = 128;
    size_t i;
    double * hA, * hB, * hOut, * hOut_gt;
    
    hA = (double*)malloc(M * N * sizeof(double));
    hB = (double*)malloc(M * sizeof(double));
    hOut = (double*)malloc(M * N * sizeof(double));
    hOut_gt = (double*)malloc(M * N * sizeof(double));

    void* args[5];

    // Generate input for execution, and create output buffers.
    for (i = 0; i < M * N; ++i) {
        hA[i] = (double)(1 - 2 * ((i & 1) << 1));
    }
    for (i = 0; i < M; ++i) {
        hB[i] = (double)i + 1;
    }
    CUDA_SAFE_CALL(cuMemAlloc(&dA, M * N * sizeof(double)));
    CUDA_SAFE_CALL(cuMemAlloc(&dB, M * sizeof(double)));
    CUDA_SAFE_CALL(cuMemAlloc(&dOut, M * N * sizeof(double)));    
    
    CUDA_SAFE_CALL(cuMemcpyHtoD(dA, hA, M * N * sizeof(double)));
    CUDA_SAFE_CALL(cuMemcpyHtoD(dB, hB, M * sizeof(double)));

    args[0] = &dA;
    args[1] = &dB;
    args[2] = &dOut;
    args[3] = &M;
    args[4] = &N;

    CUDA_SAFE_CALL( cuLaunchKernel(kernel,
                                (M*N + 31) / 32,  1, 1, // grid dim
                                32, 1, 1, // block dim
                                0, NULL, // shared mem and stream
                                args, 0)); // arguments
    CUDA_SAFE_CALL(cuCtxSynchronize()); // Retrieve and print output.
    CUDA_SAFE_CALL(cuMemcpyDtoH(hOut, dOut, M * N * sizeof(double)));

    compute_ground_truth(hA, hB, hOut_gt, M, N);
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
