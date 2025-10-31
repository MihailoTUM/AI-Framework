#include "../../src/GPU/GPUTensor.h"
#include "../../src/GPU/Activation/Activation.h"
#include <iostream>

void init() {
    GPUTensor t(3, 3);
    t.print();
}

void add() {};

void matmul() {
    GPUTensor t1(2, 3);
    t1.print();

    GPUTensor t2(3, 1);
    t2.print();

    GPUTensor result = t1 * t2;
    result.copyToHost();
    result.print();
};

void set() {
    GPUTensor t(3, 3);
    t.toOnes();
    t.print();

    GPUTensor t2(3, 3);
    t.toZeros();
    t.print();
}

void transpose() {
    GPUTensor t(2, 4);
    t.print();

    t.transpose();
    t.print();
}

void model() {
    GPUTensor input(1, 8);
    input.print();

    GPUTensor weights(8, 6);
    weights.print();

    GPUTensor bias(1, 6);
    bias.print();

    GPUTensor output = input * weights + bias;

    Activation act("RELU");
    GPUTensor out = act.forward(output);
    out.copyToHost();
    out.print();

    std::cout << output.traverse() << "\n";
}


int main() {
    model();

    return 0;
}