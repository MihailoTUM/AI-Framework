#include "../GPUTensor.h"
#include <string>
#include <cuda_runtime.h>

#pragma once

class Activation {
    private:
        std::string _function;

    public:
    Activation(std::string _func);
    GPUTensor forward(const GPUTensor& input);

    // operations
    GPUTensor relu(const GPUTensor& input);
};

__global__ void reluGPU(float* A, float* C, int rows, int cols);