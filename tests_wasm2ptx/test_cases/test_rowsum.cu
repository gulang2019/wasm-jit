#include "test.h"

void compute_ground_truth(double* hX, double* hOut_gt, int M, int N) {
    for (size_t i = 0; i < M; ++i) {
        double sum = 0;
        for (size_t j = 0; j < N; ++j) {
            sum += hX[i * N + j];
        }
        hOut_gt[i] = sum;
    }
}

bool RowSum::_run(
    CUfunction kernel
){
    CUdeviceptr dX, dOut, dOut_gt;
    size_t i;
    int M = 128, N = 64;
    double * hX, * hOut, * hOut_gt;
    hX = (double*)malloc(M * N * sizeof(double));
    hOut = (double*)malloc(M * sizeof(double));
    hOut_gt = (double*)malloc(M * sizeof(double));
    void* args[4];

    // Generate input for execution, and create output buffers.
    for (i = 0; i < M * N; ++i) {
        hX[i] = (double)i * (1 - ((i &1) << 1));
    }
    CUDA_SAFE_CALL(cuMemAlloc(&dX,   M * N * sizeof(double)));
    CUDA_SAFE_CALL(cuMemAlloc(&dOut, M * sizeof(double)));
    CUDA_SAFE_CALL(cuMemAlloc(&dOut_gt, M * sizeof(double))); 

    CUDA_SAFE_CALL(cuMemcpyHtoD(dX, hX, M * N * sizeof(double)));

    args[0] = &dX;
    args[1] = &dOut;
    args[2] = &M;
    args[3] = &N;

    CUDA_SAFE_CALL( cuLaunchKernel(kernel,
                                (M + 31) / 32,  1, 1, // grid dim
                                32, 1, 1, // block dim
                                0, NULL, // shared mem and stream
                                args, 0)); // arguments
    CUDA_SAFE_CALL(cuCtxSynchronize()); // Retrieve and print output.
    CUDA_SAFE_CALL(cuMemcpyDtoH(hOut, dOut, M*sizeof(double)));

    compute_ground_truth(hX, hOut_gt, M, N);
    bool is_all_close = all_close(hOut, hOut_gt, M);
    
    // Release resources.
    CUDA_SAFE_CALL(cuMemFree(dX));
    CUDA_SAFE_CALL(cuMemFree(dOut));
    free(hX);
    free(hOut);
    free(hOut_gt);
    return is_all_close;
}