#include <iostream>
#include <stdlib.h>
#include <vector>
#include <hip_runtime.h>
#include "transpose_kernel.h"

int main(int argc, char **argv)
{
    std::cout << "hip transpose test" << std::endl;
    if(argc < 4)
    {
       std::cout << "please identify the input matrix size and batch size" << std::endl;
       return 1;
    }
    int input_row_size = atoi(argv[1]);
    int input_col_size = atoi(argv[2]);
    int batch_size = atoi(argv[3]);

    std::cout << "input_row_size = " << input_row_size << ", input_col_size = " << input_col_size << std::endl;
    std::cout << "batch_size = " << batch_size << std::endl;

    int output_row_size = input_col_size;
    int output_col_size = input_row_size;
    std::vector<float> input_matrix(input_row_size * input_col_size * batch_size);
    std::vector<float> output_matrix(output_row_size * output_col_size * batch_size, 0);

    //fill the input matrix with values
    for(int b = 0; b < batch_size; b++)
    {
       for(int i = 0; i < input_row_size; i++)
        {
            for(int j = 0; j < input_col_size; j++)
            {
                input_matrix[b * input_row_size * input_col_size + i * input_col_size + j] = (float)(b * input_row_size * input_col_size + i * input_col_size + j);
            }
        }
    }

    //print some input matrix
    for(int i = 0; i < 16; i++)
    {
        for(int j = 0; j < 16; j++)
        {
            std::cout << input_matrix[i * input_col_size + j + 0*input_col_size*input_row_size] << " ";
        }
        std::cout << std::endl;
    }

    hipError_t err;
    //create device memory
    float *input_matrix_device, *output_matrix_device;
    err = hipMalloc(&input_matrix_device, batch_size * input_row_size * input_col_size * sizeof(float));
    if(err == hipSuccess)
       std::cout << "input_matrix_device allocation was successful" << std::endl;
    else
       std::cout << "input_matrix_device allocation was unsuccessful" << std::endl;
    err = hipMalloc(&output_matrix_device, batch_size * output_row_size * output_col_size * sizeof(float));
    if(err == hipSuccess)
       std::cout << "output_matrix_device allocation was successful" << std::endl;
    else
       std::cout << "output_matrix_device allocation was unsuccessful" << std::endl;

    //copy data to device
    err = hipMemcpy(input_matrix_device, &input_matrix[0], batch_size * input_row_size * input_col_size * sizeof(float), hipMemcpyHostToDevice);
    if(err == hipSuccess)
       std::cout << "input_matrix_device copy host to device was successful" << std::endl;
    else
       std::cout << "input_matrix_device copy host to device was unsuccessful" << std::endl;

    //launch kernel
    int block_size = 16;//dim3(input_row_size/16/4, input_col_size/16/4, 1)
    const int micro_tile_size = 64;
    //this is hip bug that I have to reverse the dim3 values
    hipLaunchKernel(HIP_KERNEL_NAME(transpose_kernel<float,micro_tile_size,micro_tile_size>), dim3(input_col_size/micro_tile_size * batch_size, input_row_size/micro_tile_size, 1),
                          dim3(16, 16, 1), 0, 0, input_matrix_device, output_matrix_device,
                          input_row_size, input_col_size, batch_size );
    hipDeviceSynchronize();
    //copy data to back to host
    err = hipMemcpy(&output_matrix[0], output_matrix_device, batch_size * output_row_size * output_col_size * sizeof(float), hipMemcpyDeviceToHost);

    if(err == hipSuccess)
       std::cout << "output_matrix_device copy device to host was successful" << std::endl;
    else
       std::cout << "output_matrix_device copy device to host was unsuccessful" << std::endl;

    //print output matrix

    for(int i = 0; i < 16; i++)
    {
        for(int j = 0; j < 16; j++)
        {
            std::cout << output_matrix[i * output_col_size + j + 0*input_col_size*input_row_size] << " ";
        }
        std::cout << std::endl;
    }


    //check result
    bool passed = true;
    for(int b = 0; b < batch_size; b++)
    {
        for(int i = 0; i < input_row_size; i++)
        {
            for(int j = 0; j < input_col_size; j++)
            {
                if(input_matrix[b * input_col_size*input_row_size + i * input_col_size + j] != output_matrix[b * input_col_size*input_row_size + j * input_row_size + i])
                {
                    passed = false;
                    break;
                }
            }
        }
    }
    if(passed)
       std::cout << "correctness PASSED" << std::endl;
    else
       std::cout << "correctness FAILED" << std::endl;
    //free device memory
    err = hipFree(input_matrix_device);
    if(err == hipSuccess)
       std::cout << "input_matrix_device free was successful" << std::endl;
    else
       std::cout << "input_matrix_device free was unsuccessful" << std::endl;
    err = hipFree(output_matrix_device);
    if(err == hipSuccess)
       std::cout << "output_matrix_device free was successful" << std::endl;
    else
       std::cout << "output_matrix_device free was unsuccessful" << std::endl;
}
