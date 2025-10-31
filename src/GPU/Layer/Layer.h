#include "../GPUTensor.h"
#include "../Activation/Activation.h"
#include <string>

#pragma once 

class Layer {
    private:
        int n_input;
        int n_output;
        std::string _function;

        GPUTensor weights;
        GPUTensor bias;
        Activation act;

    public:
        Layer(int nI, int nO, std::string _func);
        GPUTensor forward(const GPUTensor& input);
};