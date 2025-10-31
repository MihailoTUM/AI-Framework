#include "Layer.h"
#include "../GPUTensor.h"
#include "../Activation/Activation.h"
#include <string>

Layer::Layer(int nI, int nO, std::string _func): 
    n_input(nI), n_output(nO), _function(_func), weights(nI, nO), bias(1, nO), act(_func) {};

GPUTensor Layer::forward(const GPUTensor& input) {
    return act.forward(input * weights + bias);
}
