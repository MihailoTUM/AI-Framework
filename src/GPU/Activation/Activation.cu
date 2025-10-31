#include "Activation.h"
#include "../GPUTensor.h"
#include <string>
#include <cuda_runtime.h>

Activation::Activation(std::string _func) {
    _function = _func;
};

GPUTensor Activation::forward(const GPUTensor& input) {
    if(_function == "RELU") {
        return relu(input);
    }
    return input;
};

GPUTensor Activation::relu(const GPUTensor& input) {
    GPUTensor result(input.getRows(), input.getCols());
    result.setParent1(&input);

    int threads = 256;
    int blocks = (input.getRows() * input.getCols() + threads - 1)/ threads;

    reluGPU<<<blocks, threads>>>(input.getDMatrix(), result.getDMatrix(), input.getRows(), input.getCols());
    cudaDeviceSynchronize();

    return result;
};

__global__ void reluGPU(float* A, float* C, int rows, int cols) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int size = rows * cols;

    if(idx < size) {
        if(A[idx] > 0) {
            C[idx] = A[idx];
        }
        else {
            C[idx] = 0.0f;
        }
    }
}