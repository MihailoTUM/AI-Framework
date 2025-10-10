#include "Tensor.hpp"
#include <iostream>
#include <cassert>

void test_random_init() {
    std::cout << "Random init" << std::endl;

    std::cout << "INT Tensor" << std::endl;
    Tensor<int> tInt(3, 3);
    tInt.print();

    std::cout << "" << std::endl;

    std::cout << "FLOAT Tensor" << std::endl;
    Tensor<float> tFloat(3, 3);
    tFloat.print();
}

void test_custom_init() {
    // std::cout << "Custom init" << std::endl;

    int i_array[9] = { 1, 2, 4, 0, -1, 2, 4, 1, -2 };
    Tensor<int> tInt(3, 3, "CPU", i_array);
    // tInt.print();

    assert(tInt.get(0, 0) == 1);
    assert(tInt.get(2, 2) == -2);
    std::cout << "Successful custom INT init" << std::endl;

    float f_array[9] = { -5, -0.25, 0.234, -0.032, 3.2, 1.4, -0.8, -0.001, 0.76};
    Tensor<float> tFloat(3, 3, "CPU", f_array);
    // tFloat.print();

    assert(tFloat.get(0, 0) == -5);
    assert(tFloat.get(2, 2) == 0.76f);
    std::cout << "Successful custom FLOAT init" << std::endl;
};

void test_addition() {

};

void test_subtraction() {

};

void test_matrixmultiplication() {

};

int main() {
    // test_random_init();
    test_custom_init();
    
    return 0;
};