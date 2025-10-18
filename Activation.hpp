#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include "Tensor.hpp"
#include <string>

template <typename T>
class Activation {
    private: 
        std::string type; 

    Activation(std::string nType = "RELU") {
        type = nType;
    }

    Tensor forward(const Tensor output&) const {
        if(type == "RELU") {
            return relu(output);
        }
        return output;
    };

    Tensor relu(Tensor output&) const {
        for(int i = 0; i < output.rows; i++) {
            for(int j = 0; j < output.cols; j++) {
                if(output.get(i, j) < 0) {
                    output.fill(i, j, 0);
                }
            }
        }
        return output;
    }

    Tensor sigmoid(Tensor output&) const {
    }

    Tensor tanh(Tensor output&) const {

    };
};

#endif