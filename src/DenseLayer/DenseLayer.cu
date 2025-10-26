#include "Tensor.cu"
#include "Activation.cu"
#include <iostream>

class DenseLayer {
    private:
        Tensor weights;
        Tensor bias;
        Activation activation;

    public:
    DenseLayer(int n_input, int n_output, char device, char activation): 
        weights(n_input, n_output, device, true), 
        bias(1, n_output, device, true),
        activation(activation)
    {};

    Tensor forward(const Tensor& input) {
        Tensor output = activation.forward(input * weights + bias);
        return output;
    };
};


int main() {
    Tensor input(1, 8, 'G', true);
    DenseLayer layer(8, 4, 'G', 'R');

    Tensor output = layer.forward(input);
    output.print();


    return 0;
}