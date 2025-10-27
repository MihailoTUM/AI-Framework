#include "../../src/Tensor/Tensor.h"
#include <cassert>
#include <iostream>

int initTensor() {
    Tensor tensor(3, 3, 'G', true);
    tensor.print();

    assert(true);
    std::cout << "initTensor() - successful";

    return 1;
}

int addTensors() {
    int correct = 0;

    Tensor t1(3, 3, 'G', true);
    Tensor t2(3, 3, 'G', true);
    Tensor result = t1 + t2;

    assert(result.getValue(0, 0) == (t1.getValue(0, 0) + t2.getValue(0, 0)));
    assert(result.getValue(1, 2) == (t1.getValue(1, 2) + t2.getValue(1, 2)));
    correct++;

    std::cout << "addTensors() - successful";
    return correct;
};

int addBroadcastTensors() {
    Tensor t1(3, 3, 'G', true);
    // t1.print();
    // std::cout << "\n";

    Tensor t2(1, 3, 'G', true);
    // t2.print();
    // std::cout << "\n";

    Tensor result = t1 + t2;
    // result.print();

    assert(result.getValue(0, 0) == (t1.getValue(0, 0) + t2.getValue(0, 0)));
    assert(result.getValue(1, 2) == (t1.getValue(1, 2) + t2.getValue(0, 2)));

    std::cout << "addBroadcastTensors - successful";
    return 1;
};

int matmulTensors() {
    Tensor t1(2, 6, 'G', true);
    Tensor t2(6, 3, 'G', true);

    Tensor result = t1 * t2;
    
    //
    float sum1 = 0;
    for(int i = 0; i < t1.getCols(); i++) {
        sum1 += t1.getValue(0, i) * t2.getValue(i, 0);
    }

    float sum2 = 0;
    for(int j = 0; j < t1.getCols(); j++) {
        sum2 += t1.getValue(1, j) * t2.getValue(j, 0);
    }

    assert(result.getValue(0, 0) == sum1);
    assert(result.getValue(1, 0) == sum2);

    std::cout << "matmulTensors - successful";
    return 1;
}

int scalarTensor() {
    Tensor t(3, 3, 'G', true);
    float scalar = 1.234;

    Tensor output = scalar * t;

    assert(output.getValue(0, 0) == (t.getValue(0, 0) * scalar));
    assert(output.getValue(1, 2) == (t.getValue(1, 2) * scalar));

    std::cout << "scalarTensor - succesful";

    return 1;
};

int sumAlongX() {
    Tensor t(3, 6, 'G', true);
    Tensor s = t.sum(0);

    float sum = 0;
    for(int i = 0; i < t.getRows(); i++) {
         sum += t.getValue(i, 1);
    };

    assert(s.getValue(0, 1) == sum);

    std::cout << "sumAlongX() - successful";
    return 1;
};

int sumAlongY() {
    Tensor t(3, 6, 'G', true);
    Tensor s = t.sum(1);

    float sum = 0;
    for(int i = 0; i < t.getCols(); i++) {
        sum += t.getValue(1, i);
    }

    assert(s.getValue(0, 1) == sum);

    std::cout << "sumAlongY() - successful";
    return 1;
}

int transpose() {
    Tensor t(2, 3, 'G', true);

    // transposed (3, 2)
    Tensor tranpose = t.transpose();

    assert(t.getValue(1, 2) == tranpose.getValue(2, 1));
    std::cout << "transpose() - successful";

    return 1;
}

int main() {
    // initTensor();
    // addTensors();
    // addBroadcastTensors();
    // matmulTensors();
    // scalarTensor();
    // sumAlongX();
    // sumAlongY();
    transpose();
    
    return 0;
}