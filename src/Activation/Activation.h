#pragma once

#include "../Tensor/Tensor.h"
#include <cuda_runtime.h>


class Activation {
    private:
        char function;

    public:
    Activation(char nFunction = 'R');

    // getters
    char getFunction() const;

    // activation functions
    void reluCPU(float* A, float* C, int rows, int cols);

    void sigmoidCPU(float *A, float *C, int rows, int cols);
    float sigmoid(float input);

    void tanhCPU(float *A, float *C, int rows, int cols);
    float tanh(float input);

    Tensor forward(const Tensor& input);
    Tensor relu(const Tensor& input);
    Tensor sigmoid(const Tensor& input);
    Tensor tanh(const Tensor& input);
};

__global__ void reluGPU(float *A, float *C, int rows, int cols);
__global__ void sigmoidGPU(float *A, float *C, int rows, int cols);
__global__ void tanhGPU(float *A, float *C, int rows, int cols);