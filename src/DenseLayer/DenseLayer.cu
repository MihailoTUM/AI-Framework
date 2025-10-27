#include "DenseLayer.h"
#include <iostream>


DenseLayer::DenseLayer(int n_input, int n_output, char device, char activation): 
        weights(n_input, n_output, device, true), 
        bias(1, n_output, device, true),
        activation(activation)
    {};

Tensor DenseLayer::forward(const Tensor& input) {
        Tensor output = activation.forward(input * weights + bias);
        return output;
    };
