#CUDAPATH=/usr/local/cuda
export HIP_PATH=/home/timmy/Documents/ROC0401/HIP/HIP
#this will change from system to system
#include ./hip.prologue.make
# Have this point to an old enough gcc (for nvcc)
GCCPATH=/usr

#NVCC=${CUDAPATH}/bin/nvcc
HIPCC=${HIP_PATH}/bin/hipcc


     ${HIPCC} -c transpose_kernel.cpp
     ${HIPCC} -c transpose_test.cpp
     ${HIPCC} -lm -o transpose_test transpose_test.o transpose_kernel.o
