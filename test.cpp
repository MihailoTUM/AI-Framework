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
    int array_1[9] = { 1, 2, 4, 0, -1, 2, 4, 1, -2 };
    Tensor<int> tInt_1(3, 3, "CPU", array_1);
    tInt_1.print();

    std::cout << "" << std::endl;

    int array_2[9] = { -2, 0, 4, 5, 1, 0, 0, 2, 1 };
    Tensor<int> tInt_2(3, 3, "CPU", array_2);
    tInt_2.print();

    std::cout << "" << std::endl; 

    Tensor<int> result = tInt_1 + tInt_2;
    result.print();

    assert(result.get(0, 0) == -1);
    assert(result.get(1, 1) == 0);

    std::cout << "" << std::endl;

    std::cout << "Successful INT addition" <<std::endl; 

};

void test_subtraction() {
    int array_1[9] = { 1, 2, 4, 0, -1, 2, 4, 1, -2 };
    Tensor<int> tInt_1(3, 3, "CPU", array_1);
    tInt_1.print();

    std::cout << "" << std::endl;

    int array_2[9] = { -2, 0, 4, 5, 1, 0, 0, 2, 1 };
    Tensor<int> tInt_2(3, 3, "CPU", array_2);
    tInt_2.print();

    std::cout << "" << std::endl; 

    Tensor<int> result = tInt_1 - tInt_2;
    result.print();

    assert(result.get(0, 0) == 3);
    assert(result.get(1, 1) == -2);

    std::cout << "" << std::endl;

    std::cout << "Successful INT subtraction" <<std::endl; 

};

void test_scalarmultiplication() {
    int array_1[9] = { 1, 2, 4, 0, -1, 2, 4, 1, -2 };
    Tensor<int> tInt_1(3, 3, "CPU", array_1);
    tInt_1.print();

    std::cout << "" << std::endl;

    Tensor<int> result = tInt_1 * 10;
    result.print();

    assert(result.get(0, 0) == 10);
    assert(result.get(2, 1) == 10);

    std::cout << "Successful scalar multiplication" << std::endl;
}

void test_matrixmultiplication() {
    int array_1[6] = { 1, 2, 4, 0, -1, 2 };
    Tensor<int> tInt_1(2, 3, "CPU", array_1);
    tInt_1.print();

    std::cout << "" << std::endl;

    int array_2[3] = { -2, 0, 4 };
    Tensor<int> tInt_2(3, 1, "CPU", array_2);
    tInt_2.print();

    std::cout << "" << std::endl; 

    Tensor<int> result = tInt_1 * tInt_2;
    result.print();

    assert(result.get(0, 0) == 14);
    assert(result.get(1, 0) == 8);

    std::cout << "" << std::endl;

    std::cout << "Successful matrix multiplication" <<std::endl; 

};

void test_negatematrix() {
    int array_1[6] = { 1, 2, 4, 0, -1, 2 };
    Tensor<int> tInt_1(2, 3, "CPU", array_1);
    tInt_1.print();

    std::cout << "" << std::endl;

    Tensor<int> tInt_2 = -tInt_1;
    tInt_2.print();

    assert(tInt_2.get(0, 0) == -1);
    assert(tInt_2.get(1, 2) == -2);

    std::cout << "Success" << std::endl;
}

void test_sum() {
    // axis = 0;
    int i_array[6] = { 1, 2, 4, 0, -1, 2 };
    Tensor<int> tInt(2, 3, "CPU", i_array);
    tInt.print();

    std::cout << "" << std::endl;

    Tensor<int> sum = tInt.sum(0);
    sum.print();

    assert(sum.get(0, 0) == 1);
    assert(sum.get(0, 1) == 1);
    assert(sum.get(0, 2) == 6);

    std::cout << "Success for sum axis = 0" << std::endl;

    Tensor<int> b(2, 3, "CPU", i_array);
    b.print();

    std::cout << "" << std::endl;

    Tensor<int> a = b.sum(1);
    a.print();

    assert(a.get(0, 0) == 7);
    assert(a.get(0, 1) == 1);

    std::cout << "Success for sum axis = 1" << std::endl;
}

void test_mean() {
    // axis = 0;
    int i_array[6] = { 1, 2, 4, 0, -1, 2 };
    Tensor<int> tInt(2, 3, "CPU", i_array);
    tInt.print();

    std::cout << "" << std::endl;

    Tensor<int> sum = tInt.mean(0);
    sum.print();

    assert(sum.get(0, 0) == 0);
    assert(sum.get(0, 1) == 0);
    assert(sum.get(0, 2) == 3);

    std::cout << "Success for sum axis = 0" << std::endl;

    Tensor<int> b(2, 3, "CPU", i_array);
    b.print();

    std::cout << "" << std::endl;

    Tensor<int> a = b.mean(1);
    a.print();

    assert(a.get(0, 0) == 2);
    assert(a.get(0, 1) == 0);

    std::cout << "Success for sum axis = 1" << std::endl;
};

int main() {
    // test_random_init();
    // test_custom_init();
    // test_addition();
    // test_subtraction();
    // test_scalarmultiplication();
    // test_matrixmultiplication();
    // test_negatematrix();
    // test_sum(); 
    test_mean();

    return 0;
};