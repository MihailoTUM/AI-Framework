#include <iostream>
#include <random>
#include <string>
#include "Tensor.hpp"


    // // Tensor Tensor::sum(int axis = 0) {

    // // };

    // // Tensor Tensor::mean(int axis = 0) {

    // // };

    // // returns newly negated Tensor
    // Tensor Tensor::operator -() const {
    //     Tensor g(rows, cols);
    //     for(int i = 0; i < rows; i++) {
    //         for(int j = 0; j < cols; j++) {
    //             g.fill(i, j, -matrix[i * cols + j]);
    //         }
    //     }
    //     return g;
    // }

    // // matrix-addition and matrix-subtraction: returns new Tensor
    // Tensor Tensor::operator +(const Tensor& other) const {
    //     if(rows == other.rows && cols == other.cols) {
    //         Tensor g(rows, cols);
    //         for(int i = 0; i < rows; i++) {
    //             for(int j = 0; j < cols; j++) {
    //                 g.fill(i, j, matrix[i * cols + j] + other.matrix[i * cols + j]);
    //             }
    //         }
    //         return g;
    //     } else {
    //         return Tensor(0, 0);
    //     };
    // };
    // Tensor Tensor::operator -(const Tensor& other) const {
    //     if(rows == other.rows && cols == other.cols) {
    //         Tensor g(rows, cols);
    //         for(int i = 0; i < rows; i++) {
    //             for(int j = 0; j < cols; j++) {
    //                 g.fill(i, j, matrix[i * cols + j] - other.matrix[i * cols + j]);
    //             }
    //         }
    //         return g;
    //     } else {
    //         return Tensor(0, 0);
    //     }; 
    // }

    // // scalar-multiplication: returns new Tensor
    // Tensor Tensor::operator *(float scalar) const {
    //     Tensor g(rows, cols);
    //     for(int i = 0; i < rows; i++) {
    //         for(int j = 0; j < cols; j++) {
    //             g.fill(i, j, scalar * matrix[i * cols + j]);
    //         }
    //     }
    //     return g;
    // }

    // // matrix-multiplication: returns new Tensor
    // Tensor Tensor::operator *(const Tensor& other) const {
    //     if(cols == other.rows) {
    //         Tensor g(rows, other.cols);
    //         for(int x = 0; x < rows; x++) {
    //             for(int y = 0; y < other.cols; y++) {
    //                 float value = 0;
    //                 for(int k = 0; k < cols; k++) {
    //                     value += matrix[cols * x + k] * other.matrix[other.cols * k + y];
    //                 }
    //                 g.fill(x, y, value);
    //             }
    //         }
    //         return g;
    //     }
    //     else {
    //         return Tensor(0, 0);
    //     }
    // }


