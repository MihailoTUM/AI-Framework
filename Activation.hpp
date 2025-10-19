#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include "Tensor.hpp"
#include <string>

template <typename T>
class Activation {
    private: 
        std::string type; 

    public:

    Activation(std::string nType = "RELU") {
        type = nType;
    }

    Tensor<T> forward(const Tensor<T>& output) const {
        if(type == "RELU") {
            return relu(output);
        }
        return output;
    };

    Tensor<T> relu(const Tensor<T>& output) const {
        Tensor<T> local(output.getRows(), output.getCols(), "CPU", output.getMatrix());
        for(int i = 0; i < output.getRows(); i++) {
            for(int j = 0; j < output.getCols(); j++) {
                if(output.get(i, j) < 0) {
                    local.fill(i, j, 0);
                }
            }
        }
        return local;
    }

    // Tensor sigmoid(Tensor output&) const {
    // }

    // Tensor tanh(Tensor output&) const {

    // };
};

#endif