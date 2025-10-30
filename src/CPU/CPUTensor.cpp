#include "CPUTensor.h"
#include <iostream>

CPUTensor::CPUTensor(int nRows, int nCols) {
    rows = nRows;
    cols = nCols;
    matrix =  new float[rows * cols];

    parent1 = nullptr;
    parent2 = nullptr;
};  

int CPUTensor::getRows() const {
    return rows;
}

int CPUTensor::getCols() const {
    return cols;
}

float* CPUTensor::getMatrix() {
    return matrix;
}

void CPUTensor::print() {
    for(int i = 0; i < getRows(); i++) {
        for(int j = 0; i < getCols(); j++) {
            std::cout << getMatrix()[i * getCols() + j] << " ";
        }
        std::cout << "\n";
    }
}

CPUTensor CPUTensor::operator+(const CPUTensor& input) {
    if(getRows() == input.getRows() && getCols() == input.getCols()) {
        CPUTensor result(getRows(), getCols());

        for(int i = 0; i < rows * cols; i++) {
            result.getMatrix()[i] = getMatrix()[i] + input.getMatrix()[i];
        };

        return result;
    }
    else {
        throw std::invalid_argument("rows or cols differ");
    }
};

int main() {
    CPUTensor t1(3, 3);
    t1.print();

    return 0;
}