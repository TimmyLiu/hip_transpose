#ifndef __TRANSPOSE_KERNEL__
#define __TRANSPOSE_KERNEL__
#include "hip_runtime.h"

extern "C" __global__ void transpose_kernel(hipLaunchParm lp, float *input_matrix, float *output_matrix, size_t input_row_size, size_t input_col_size);

#endif
