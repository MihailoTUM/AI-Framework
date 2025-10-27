#pragma once

#include "../Tensor/Tensor.h"
#include "../Activation/Activation.h"

class DenseLayer {
    private:
        Tensor weights;
        Tensor bias;
        Activation activation;

    public:
    DenseLayer(int n_input, int n_output, char device, char activation);

    Tensor forward(const Tensor& input);

};

