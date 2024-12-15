#pragma once
#include <string>
#include <cuda_runtime.h>
#include "cuda.h"
#include "nvPTXCompiler.h"


#define CUDA_SAFE_CALL(x)                                               \
    do {                                                                \
        CUresult result = x;                                            \
        if (result != CUDA_SUCCESS) {                                   \
            const char *msg;                                            \
            cuGetErrorName(result, &msg);                               \
            printf("error: %s failed with error %s\n", #x, msg);        \
            exit(1);                                                    \
        }                                                               \
    } while(0)

#define NVPTXCOMPILER_SAFE_CALL(x)                                       \
    do {                                                                 \
        nvPTXCompileResult result = x;                                   \
        if (result != NVPTXCOMPILE_SUCCESS) {                            \
            printf("error: %s failed with error code %d\n", #x, result); \
            exit(1);                                                     \
        }                                                                \
    } while(0)

// extern 
// __global__
// void ReLU(
//     double* A,
//     double* B,
//     int M, int N
// );

// __global__
// void matmul(
//     double* A,
//     double* B,
//     double* C,
//     int M, int N, int K 
// );

// __global__
// void row_sums(const double* A, double* rowSums, int M, int N);

// __global__
// void softmax(const double* A, const double* rowSums, double* B, int M, int N);

// __global__
// void vector_add(const double* A, const double* B, double* C, int N);


struct TestCase {
    std::string name;
    bool verbose;
    char* elf = nullptr;
    void run();
    TestCase(std::string name, 
    bool verbose): name(name), verbose(verbose) {}
    virtual bool _run(CUfunction kernel) = 0;
    void _compile();
    virtual ~TestCase() {
        if (elf != nullptr) {
            free(elf);
        }
    }
};

struct VectorAdd: public TestCase{
    const int size = 1024;
    VectorAdd(std::string name, bool verbose): TestCase(name, verbose) {}
    bool _run(
        CUfunction kernel = nullptr
    ) override;
};

struct Matmul: public TestCase{
    const int size = 1024;
    Matmul(std::string name, bool verbose): TestCase(name, verbose) {}
    bool _run(
        CUfunction kernel = nullptr
    ) override;
};

bool all_close(const double* A, const double* B, int N);