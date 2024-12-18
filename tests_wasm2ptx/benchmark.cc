#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream> 
#include <cassert>
#include <chrono>

#include "test.h"

// #define NUM_THREADS 128
// #define NUM_BLOCKS 32
// #define SIZE NUM_THREADS * NUM_BLOCKS


void TestCase::_compile() {
    nvPTXCompilerHandle compiler = NULL;
    nvPTXCompileResult status;

    size_t elfSize, infoSize, errorSize;
    char *infoLog, *errorLog;
    unsigned int minorVer, majorVer;

    const char* compile_options[] = { "--gpu-name=sm_80",
                                      "--verbose"
                                    };

    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetVersion(&majorVer, &minorVer));
    // printf("Current PTX Compiler API Version : %d.%d\n", majorVer, minorVer);

    // Load ptx code from file test_wasm2ptx/ptx/{name}.ptx
    std::string ptxCode;
    std::string ptxPath = "tests_wasm2ptx/ptx_ground_truth/" + name + ".ptx";
    FILE* ptxFile = fopen(ptxPath.c_str(), "r");
    if (ptxFile == NULL) {
        printf("Error: Cannot open file %s\n", ptxPath.c_str());
        return;
    }
    char line[1024];
    while (fgets(line, 1024, ptxFile)) {
        ptxCode += line;
    }
    fclose(ptxFile);

    if (verbose) {
        printf("PTX code:\n%s\n", ptxCode.c_str());
    }

    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerCreate(&compiler,
                                                ptxCode.size(),  /* ptxCodeLen */
                                                ptxCode.c_str())                  /* ptxCode */
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

    if (infoSize != 0 and verbose) {
        infoLog = (char*)malloc(infoSize+1);
        NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetInfoLog(compiler, infoLog));
        printf("Info log: %s\n", infoLog);
        free(infoLog);
    }

    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerDestroy(&compiler));
}

void TestCase::run() {
    // Load the compiled GPU assembly code 'elf'
    auto start = std::chrono::high_resolution_clock::now();
    _compile();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("Compilation time: %f seconds\n", elapsed.count());
    if(elf == nullptr) {
        printf("Error %s: No compiled code\n", name.c_str());
        return;
    }
    CUdevice cuDevice;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;

    CUDA_SAFE_CALL(cuInit(0));
    CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));

    CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
    CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, elf, 0, 0, 0));
    CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, name.c_str()));

    bool passed = _run(kernel);
    printf("%s: %s\n", name.c_str(), passed ? "PASSED" : "FAILED");

    CUDA_SAFE_CALL(cuModuleUnload(module));
    CUDA_SAFE_CALL(cuCtxDestroy(context));

}

bool all_close(const double* A, const double* B, int N)
 {
    for (int i = 0; i < N; i++) {
        if (std::abs(A[i] - B[i]) > 1e-6) {
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    std::vector<std::string> testNames;
    bool verbose = false;
    for (int i = 1; i < argc; i++){
        if (strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        }
        else testNames.push_back(argv[i]);
    }
    if (testNames.size() == 0) {
        testNames.push_back("vector_add");
        testNames.push_back("matmul");
        testNames.push_back("row_sum");
        testNames.push_back("softmax");
        testNames.push_back("relu");
    }
    for (auto name : testNames) {
        if (verbose) {
            printf("Running test %s\n", name);
        }
        if (name == "vector_add") {
            VectorAdd test("vector_add", verbose);
            test.run();
        }
        else if (name == "matmul") {
            Matmul test("matmul", verbose);
            test.run();
        }
        else if (name == "row_sum") {
            RowSum test("row_sum", verbose);
            test.run();
        }
        else if (name == "softmax") {
            Softmax test("softmax", verbose);
            test.run();
        }
        else if (name == "relu") {
            ReLU test("relu", verbose);
            test.run();
        }
        else {
            printf("Unknown test name: %s\n", name);
            exit(1);
        }
    }
    return 0;
}