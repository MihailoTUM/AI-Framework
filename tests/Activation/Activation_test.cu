#include "../../src/Activation/Activation.h";
#include "../../src/Tensor/Tensor.h";
#include <cassert>
#include <iostream>

int relu() {
    Tensor input(2, 2, 'G', true);
    Activation act('R');

    Tensor output = act.forward(input);

    assert(output.getValue(0, 0) >= 0);
    assert(output.getValue(1, 0) >= 0);
    assert(output.getValue(0, 1) >= 0);
    assert(output.getValue(1, 1) >= 0);

    std::cout << "relu() - successful";
    return 1;
};

int sigmoid() {
    Tensor input(2, 2, 'G', true);
    Activation act('S');

    Tensor output = act.forward(input);

    assert(1 > output.getValue(0, 0) > 0);
    assert(1 > output.getValue(0, 1) > 0);
    assert(1 > output.getValue(1, 0) > 0);
    assert(1 > output.getValue(1, 1) > 0);

    std::cout << "sigmoid() - successful";
    return 1;
};

int tanh() {
    Tensor input(2, 2, 'G', true);
    Activation act('T');

    Tensor output = act.forward(input);

    assert(1 > output.getValue(0, 0) > -1);
    assert(1 > output.getValue(0, 1) > -1);
    assert(1 > output.getValue(1, 0) > -1);
    assert(1 > output.getValue(1, 1) > -1);

    std::cout << "tanh() - successful";
    return 1;
};

int linear() {
    Tensor input(2, 2, 'G', true);
    Activation act('N');

    Tensor output = act.forward(input);

    assert(output.getValue(0, 0) == input.getValue(0, 0));
    assert(output.getValue(0, 1) == input.getValue(0, 1));
    assert(output.getValue(1, 0) == input.getValue(1, 0));
    assert(output.getValue(1, 1) == input.getValue(1, 1));

    std::cout << "linear() - successful";
    return 1;
};

int main() {

    // relu();
    // sigmoid();
    // tanh();
    // linear();

    return 0;
}