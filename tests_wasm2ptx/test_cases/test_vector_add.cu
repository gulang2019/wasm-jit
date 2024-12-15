#include "test.h"

void compute_ground_truth(double* hX, double* hY, double* hOut_gt, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        hOut_gt[i] = hX[i] + hY[i];
    }
}

bool VectorAdd::_run(
    CUfunction kernel
){
    CUdeviceptr dX, dY, dOut, dOut_gt;
    size_t i;
    size_t bufferSize = size * sizeof(double);
    double * hX, * hY, * hOut, * hOut_gt;
    hX = (double*)malloc(bufferSize);
    hY = (double*)malloc(bufferSize);
    hOut = (double*)malloc(bufferSize);
    hOut_gt = (double*)malloc(bufferSize);
    void* args[3];

    // Generate input for execution, and create output buffers.
    for (i = 0; i < size; ++i) {
        hX[i] = (double)i;
        hY[i] = (double)i * 2;
    }
    CUDA_SAFE_CALL(cuMemAlloc(&dX,   bufferSize));
    CUDA_SAFE_CALL(cuMemAlloc(&dY,   bufferSize));
    CUDA_SAFE_CALL(cuMemAlloc(&dOut, bufferSize));
    CUDA_SAFE_CALL(cuMemAlloc(&dOut_gt, bufferSize)); 

    CUDA_SAFE_CALL(cuMemcpyHtoD(dX, hX, bufferSize));
    CUDA_SAFE_CALL(cuMemcpyHtoD(dY, hY, bufferSize));

    args[0] = &dX;
    args[1] = &dY;
    args[2] = &dOut;

    CUDA_SAFE_CALL( cuLaunchKernel(kernel,
                                (size + 31) / 32,  1, 1, // grid dim
                                32, 1, 1, // block dim
                                0, NULL, // shared mem and stream
                                args, 0)); // arguments
    CUDA_SAFE_CALL(cuCtxSynchronize()); // Retrieve and print output.
    CUDA_SAFE_CALL(cuMemcpyDtoH(hOut, dOut, bufferSize));

    compute_ground_truth(hX, hY, hOut_gt, size);
    bool is_all_close = all_close(hOut, hOut_gt, size);
    
    // Release resources.
    CUDA_SAFE_CALL(cuMemFree(dX));
    CUDA_SAFE_CALL(cuMemFree(dY));
    CUDA_SAFE_CALL(cuMemFree(dOut));
    free(hX);
    free(hY);
    free(hOut);
    free(hOut_gt);
    return is_all_close;

}