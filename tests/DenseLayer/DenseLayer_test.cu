#include "../../src/Tensor/Tensor.h"
#include "../../src/Activation/Activation.h"
#include "../../src/DenseLayer/DenseLayer.h"

int initDenseLayer() {
    DenseLayer layer(8, 6, 'G', 'R');
    return 1;
}

int forward() {
    DenseLayer layer(6, 2, 'G', 'R');
    Tensor input(1, 6, 'G');

    Tensor output = layer.forward(input);
    output.print();

    return 1;
}


int main() {
    // initDenseLayer();
    forward();

    return 0;
};
