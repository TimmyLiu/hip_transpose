#include "transpose_kernel.h"

// defination of template specialization
template<>
__global__ void transpose_kernel<float, 1, 1>(hipLaunchParm lp, float *input_matrix, float *output_matrix, size_t input_row_size, size_t input_col_size,
                                        size_t batch_size)
{

}
