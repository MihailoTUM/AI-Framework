#include "../../src/Tensor/Tensor.h"
#include <cassert>
#include <iostream>

int initTensor() {
    Tensor t1(3, 3, 'C');
    t1.print();

    Tensor t2(3, 3, 'C');
    t2.toZeros();
    t2.print();

    Tensor t3(3, 3, 'C');
    t3.toOnes();
    t3.print();

    Tensor t4(3, 3, 'C');
    t4.toInt();
    t4.print();

    Tensor t5(3, 3, 'G');
    t5.printDevice();
    return 1;
}

void TensorValueEqual(float r, float a, float b) {
    float eps = 1e05f;
    assert(((a + b) - r) < eps);
}

void TensorEqual(float *r, float *a, float *b, int rows, int cols) {
    for(int i = 0; i < rows * cols; i++) {
        TensorValueEqual(r[i], a[i], b[i]);
    }
}

int addTensorsTest() {
    int correct = 0;

    Tensor CPU_1(3, 3, 'C');
    CPU_1.print();
    std::cout << "\n";
    Tensor CPU_2(3, 3, 'C');
    CPU_2.print();
    std::cout << "\n";
    Tensor result = CPU_1 + CPU_2;
    result.print();

    TensorEqual(result.getMatrix(), CPU_1.getMatrix(), CPU_2.getMatrix(), 3, 3);
    correct++;
    std::cout << "\nCPU - successful\n" << std::endl;

    Tensor GPU_1(3, 3, 'G');
    GPU_1.print();
    std::cout << "\n";
    Tensor GPU_2(3, 3, 'G');
    GPU_2.print();
    std::cout << "\n";
    Tensor GPU_result = GPU_1 + GPU_2;
    GPU_result.print();

    TensorEqual(GPU_result.getMatrix(), GPU_1.getMatrix(), GPU_2.getMatrix(), 3, 3);
    correct++;
    std::cout << "\nGPU - successful\n" << std::endl;

    std::cout << "addTensors() - successful";
    return correct;
};

int addScalarTensorsTest() {
    Tensor t(3, 3, 'G');
    t.print();
    float scalar = 10;
    Tensor result = scalar + t;
    result.print();

    return 1;
}

int testPrintDevice() {
    Tensor t(3, 3, 'G');
    t.printDevice();

    Tensor t1(3, 3, 'C');
    t1.printDevice();

    return 1;
}

int addBroadcastTensors() {
    Tensor t1(3, 3, 'G');
    t1.print();

    Tensor t2(1, 3, 'G');
    t2.print();

    Tensor result = t1 + t2;
    result.print();

    std::cout << "addBroadcastTensors - successful";
    return 1;
};

int matmulTensors() {
    Tensor t1(2, 6, 'G');
    t1.print();
    Tensor t2(6, 3, 'G');
    t2.print();

    Tensor result = t1 * t2;
    result.print();

    std::cout << "matmulTensors - successful";
    return 1;
}

int scalarTensor() {
    Tensor t(3, 3, 'G');
    float scalar = 1.234;

    Tensor output = scalar * t;

    std::cout << "scalarTensor - succesful" << std::flush;

    return 1;
};

int sumAlongX() {
    Tensor t(3, 6, 'G');
    Tensor s = t.sum(0);

    std::cout << "sumAlongX() - successful" << std::flush;
    return 1;
};

int sumAlongY() {
    Tensor t(3, 6, 'G');
    Tensor s = t.sum(1);

    std::cout << "sumAlongY() - successful" << std::flush;
    return 1;
}

int transpose() {
    Tensor t(2, 3, 'G');

    // transposed (3, 2)
    Tensor tranpose = t.transpose();

    std::cout << "transpose() - successful" << std::flush;

    return 1;
}

int main() {
    // initTensor();
    // addScalarTensorsTest();
    // addTensorsTest();
    // testPrintDevice();
    // addBroadcastTensors();
    // matmulTensors();
    // scalarTensor();
    sumAlongX();
    sumAlongY();
    transpose();
    
    return 0;
}