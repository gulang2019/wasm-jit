#include "test.h"

void compute_ground_truth(double* hX, double* hOut_gt, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        hOut_gt[i] = std::max(hX[i], 0.);
    }
}

bool ReLU::_run(
    CUfunction kernel
){
    CUdeviceptr dX, dOut, dOut_gt;
    size_t i;
    int size = 128;
    size_t bufferSize = size * sizeof(double);
    double * hX, * hOut, * hOut_gt;
    hX = (double*)malloc(bufferSize);
    hOut = (double*)malloc(bufferSize);
    hOut_gt = (double*)malloc(bufferSize);
    void* args[3];

    // Generate input for execution, and create output buffers.
    for (i = 0; i < size; ++i) {
        hX[i] = (double)i * (1 - ((i &1) << 1));
    }
    CUDA_SAFE_CALL(cuMemAlloc(&dX,   bufferSize));
    CUDA_SAFE_CALL(cuMemAlloc(&dOut, bufferSize));
    CUDA_SAFE_CALL(cuMemAlloc(&dOut_gt, bufferSize)); 

    CUDA_SAFE_CALL(cuMemcpyHtoD(dX, hX, bufferSize));

    args[0] = &dX;
    args[1] = &dOut;
    args[2] = &size;

    CUDA_SAFE_CALL( cuLaunchKernel(kernel,
                                (size + 31) / 32,  1, 1, // grid dim
                                32, 1, 1, // block dim
                                0, NULL, // shared mem and stream
                                args, 0)); // arguments
    CUDA_SAFE_CALL(cuCtxSynchronize()); // Retrieve and print output.
    CUDA_SAFE_CALL(cuMemcpyDtoH(hOut, dOut, bufferSize));

    compute_ground_truth(hX, hOut_gt, size);
    bool is_all_close = all_close(hOut, hOut_gt, size);
    
    // Release resources.
    CUDA_SAFE_CALL(cuMemFree(dX));
    CUDA_SAFE_CALL(cuMemFree(dOut));
    free(hX);
    free(hOut);
    free(hOut_gt);
    return is_all_close;
}