#include <stdio.h>
#include <string.h>
#include "cuda.h"
#include "nvPTXCompiler.h"

#define NUM_THREADS 128
#define NUM_BLOCKS 32
#define SIZE NUM_THREADS * NUM_BLOCKS

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


const char *ptxCode = "                                      \
   .version 8.4                                           \n \
   .target sm_89                                          \n \
   .address_size 64                                       \n \
   .visible .entry simpleVectorAdd(                       \n \
        .param .u64 simpleVectorAdd_param_0,              \n \
        .param .u64 simpleVectorAdd_param_1,              \n \
        .param .u64 simpleVectorAdd_param_2               \n \
   ) {                                                    \n \
        .reg .f32   %f<4>;                                \n \
        .reg .b32   %r<5>;                                \n \
        .reg .b64   %rd<11>;                              \n \
        ld.param.u64    %rd1, [simpleVectorAdd_param_0];  \n \
        ld.param.u64    %rd2, [simpleVectorAdd_param_1];  \n \
        ld.param.u64    %rd3, [simpleVectorAdd_param_2];  \n \
        cvta.to.global.u64      %rd4, %rd3;               \n \
        cvta.to.global.u64      %rd5, %rd2;               \n \
        cvta.to.global.u64      %rd6, %rd1;               \n \
        mov.u32         %r1, %ctaid.x;                    \n \
        mov.u32         %r2, %ntid.x;                     \n \
        mov.u32         %r3, %tid.x;                      \n \
        mad.lo.s32      %r4, %r2, %r1, %r3;               \n \
        mul.wide.u32    %rd7, %r4, 4;                     \n \
        add.s64         %rd8, %rd6, %rd7;                 \n \
        ld.global.f32   %f1, [%rd8];                      \n \
        add.s64         %rd9, %rd5, %rd7;                 \n \
        ld.global.f32   %f2, [%rd9];                      \n \
        add.f32         %f3, %f1, %f2;                    \n \
        add.s64         %rd10, %rd4, %rd7;                \n \
        st.global.f32   [%rd10], %f3;                     \n \
        ret;                                              \n \
   } ";


int elfLoadAndKernelLaunch(void* elf, size_t elfSize)
{
    CUdevice cuDevice;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;
    CUdeviceptr dX, dY, dOut;
    size_t i;
    size_t bufferSize = SIZE * sizeof(float);
    float a;
    float hX[SIZE], hY[SIZE], hOut[SIZE];
    void* args[3];

    CUDA_SAFE_CALL(cuInit(0));
    CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));

    CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
    CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, elf, 0, 0, 0));
    CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "simpleVectorAdd"));

    // Generate input for execution, and create output buffers.
    for (i = 0; i < SIZE; ++i) {
        hX[i] = (float)i;
        hY[i] = (float)i * 2;
    }
    CUDA_SAFE_CALL(cuMemAlloc(&dX,   bufferSize));
    CUDA_SAFE_CALL(cuMemAlloc(&dY,   bufferSize));
    CUDA_SAFE_CALL(cuMemAlloc(&dOut, bufferSize));

    CUDA_SAFE_CALL(cuMemcpyHtoD(dX, hX, bufferSize));
    CUDA_SAFE_CALL(cuMemcpyHtoD(dY, hY, bufferSize));

    args[0] = &dX;
    args[1] = &dY;
    args[2] = &dOut;

    CUDA_SAFE_CALL( cuLaunchKernel(kernel,
                                   NUM_BLOCKS,  1, 1, // grid dim
                                   NUM_THREADS, 1, 1, // block dim
                                   0, NULL, // shared mem and stream
                                   args, 0)); // arguments
    CUDA_SAFE_CALL(cuCtxSynchronize()); // Retrieve and print output.

    CUDA_SAFE_CALL(cuMemcpyDtoH(hOut, dOut, bufferSize));
    for (i = 0; i < SIZE; ++i) {
        printf("Result:[%ld]:%f\n", i, hOut[i]);
    }

    // Release resources.
    CUDA_SAFE_CALL(cuMemFree(dX));
    CUDA_SAFE_CALL(cuMemFree(dY));
    CUDA_SAFE_CALL(cuMemFree(dOut));
    CUDA_SAFE_CALL(cuModuleUnload(module));
    CUDA_SAFE_CALL(cuCtxDestroy(context));
    return 0;
}

int main(int _argc, char *_argv[])
{

    nvPTXCompilerHandle compiler = NULL;
    nvPTXCompileResult status;

    size_t elfSize, infoSize, errorSize;
    char *elf, *infoLog, *errorLog;
    unsigned int minorVer, majorVer;

    const char* compile_options[] = { "--gpu-name=sm_89",
                                      "--verbose"
                                    };

    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetVersion(&majorVer, &minorVer));
    printf("Current PTX Compiler API Version : %d.%d\n", majorVer, minorVer);
    
    printf("src code : %s\n", ptxCode);

    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerCreate(&compiler,
                                                (size_t)strlen(ptxCode),  /* ptxCodeLen */
                                                ptxCode)                  /* ptxCode */
                            );

    status = nvPTXCompilerCompile(compiler,
                                  2,                 /* numCompileOptions */
                                  compile_options);  /* compileOptions */

    if (status != NVPTXCOMPILE_SUCCESS) {
        NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetErrorLogSize(compiler, &errorSize));

        if (errorSize != 0) {
            errorLog = (char*)malloc(errorSize+1);
            NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetErrorLog(compiler, errorLog));
            printf("Error log: %s\n", errorLog);
            free(errorLog);
        }
        exit(1);
    }

    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetCompiledProgramSize(compiler, &elfSize));

    elf = (char*) malloc(elfSize);
    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetCompiledProgram(compiler, (void*)elf));

    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetInfoLogSize(compiler, &infoSize));

    if (infoSize != 0) {
        infoLog = (char*)malloc(infoSize+1);
        NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetInfoLog(compiler, infoLog));
        printf("Info log: %s\n", infoLog);
        free(infoLog);
    }

    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerDestroy(&compiler));

    // Load the compiled GPU assembly code 'elf'
    elfLoadAndKernelLaunch(elf, elfSize);

    free(elf);
    return 0;
}