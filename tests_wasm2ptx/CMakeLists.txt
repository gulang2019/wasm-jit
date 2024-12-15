cmake_minimum_required(VERSION 3.18)
project(wasm2ptx LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 70)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# file(GLOB CUDA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cu" "${CMAKE_CURRENT_SOURCE_DIR}/test_cases/*.cu")
file(GLOB CUDA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test_cases/*.cu")
# file(GLOB CPP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test_cases/*.cu")

# include_directories("/usr/local/cuda-11.8/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

find_library(NVPTXCOMPILER_LIBRARY
    nvptxcompiler_static
    PATHS /usr/local/cuda-11.8/lib64
    NO_DEFAULT_PATH
)

if(NOT NVPTXCOMPILER_LIBRARY)
    message(FATAL_ERROR "libnvptxcompiler_static.a not found.")
endif()

find_library(CUDA_DRIVER_LIBRARY
    cuda
    PATHS /usr/local/cuda-11.8/lib64/stubs
    NO_DEFAULT_PATH
)

if(NOT CUDA_DRIVER_LIBRARY)
    message(FATAL_ERROR "CUDA Driver API library (libcuda.so) not found.")
endif()


set(SOURCES
    benchmark.cc
    ${CUDA_SOURCES}
)

add_executable(wasm2ptx ${SOURCES})

target_link_libraries(wasm2ptx PRIVATE ${CUDA_DRIVER_LIBRARY})

target_link_libraries(wasm2ptx PRIVATE ${NVPTXCOMPILER_LIBRARY})
# target_compile_options(wasm2ptx PRIVATE -Wall -Wextra -O2)
# target_compile_options(wasm2ptx PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda -Xcompiler -O2>)

# set_property(TARGET wasm2ptx PROPERTY CUDA_SEPARABLE_COMPILATION ON)
