NVCC=/usr/local/cuda/bin/nvcc
CXX=g++
CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

simpleVectorAddition: simpleVectorAddition.c
	gcc simpleVectorAddition.c -o simpleVectorAddition \
             -I $(CUDA_PATH)/include \
             -L $(CUDA_PATH)/lib64 \
             /usr/local/cuda/lib64/libnvptxcompiler_static.a -lcuda -lm -lpthread