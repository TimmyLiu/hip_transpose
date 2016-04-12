#ifndef __TRANSPOSE_KERNEL__
#define __TRANSPOSE_KERNEL__
#include "hip_runtime.h"


template<typename T, int micro_tile_col_size, int micro_tile_row_ize >
inline __global__ void transpose_kernel(hipLaunchParm lp, T *input_matrix, T *output_matrix, size_t input_row_size, size_t input_col_size,
                                        size_t batch_size)
{
//KERNELBEGIN;
    size_t local_idx_0 = hipThreadIdx_x;// 0-15
    size_t local_idx_1 = hipThreadIdx_y;// 0-15
    size_t block_idx_0 = hipBlockIdx_x;// index of work groups
    size_t block_idx_1 = hipBlockIdx_y;
    size_t block_dim_0 = hipBlockDim_x;// size of work groups 16
    size_t block_dim_1 = hipBlockDim_y;// size of work groups 16
    size_t grid_dim_0 = hipGridDim_x;// number of blocks

    const size_t reshape_factor = 4;
    const size_t unroll_factor = 16;

    __shared__ T lds[micro_tile_row_ize][micro_tile_col_size];

    size_t batch_idx = 0;
    size_t blocks_per_batch = grid_dim_0 / batch_size;
    batch_idx += (block_idx_0) / blocks_per_batch;

    input_matrix += batch_idx * input_col_size * input_row_size;

    size_t input_offset = 0;
    input_offset += input_col_size * block_idx_1 * block_dim_1 * 4;// each WG works on 64 by 64 block
    input_offset += (block_idx_0 % blocks_per_batch) * block_dim_0 * 4;

    input_matrix += input_offset;
    for(int i = 0; i < unroll_factor; i++)
    {
        //each iteration 256 work items will read from a 4 x 64 subblock
        //there are 16 iterations
        size_t subblock_idx_0 = local_idx_0 + (local_idx_1 % 4) * 16;
        size_t subblock_idx_1 = local_idx_1 / 4 + i * 4;
        //transpose happened here
        lds[subblock_idx_0][subblock_idx_1] = input_matrix[subblock_idx_1 * input_col_size + subblock_idx_0];
    }

    __syncthreads();

    output_matrix += batch_idx * input_col_size * input_row_size;
    size_t output_offset = 0;
    output_offset += input_row_size * (block_idx_0 % blocks_per_batch) * block_dim_0 * 4;//input_row_size == ouput_col_size
    output_offset += block_idx_1 * block_dim_1 * 4;

    output_matrix += output_offset;

    for(int i = 0; i < unroll_factor; i++)
    {
        size_t subblock_idx_0 = local_idx_0 + (local_idx_1 % 4) * 16;// 0-63
        size_t subblock_idx_1 = local_idx_1 / 4 + i * 4;// 0-3, 4-7 ... 60-63
        T  temp = lds[subblock_idx_1][subblock_idx_0];
        output_matrix[subblock_idx_1 * input_row_size + subblock_idx_0] = temp;//lds[subblock_idx_1][subblock_idx_0];
    }

//KERNELEND;
}

#endif
