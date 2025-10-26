#pragma once

#include <cuda_runtime.h>

class Tensor {
    private:
        float* matrix;
        int rows;
        int cols;
        char device;

    public: 
        Tensor(int nRows, int nCols, char nDevice, bool random = true);
        ~Tensor();

        // getters for matrix
        int getRows() const;
        int getCols() const;
        float getValue(int row, int col) const;
        
        // getters for tensors
        float* getMatrix() const;
        char getDevice() const;

        // setters
        void setValue(int row, int cols, float value);

        // init
        void initMatrixToZeros();
        void initMatrixRandom();

        // print
        void print();

        // matrix operations CPU-based
        void addMatrixCPU(float* A, float *B, float *C, int rows, int cols) const;
        void addBroadcastCPU(float* A, float *B, float *C, int rows, int cols) const;

        void scalarCPU(float *A, float scalar, float *C, int rows, int cols) const;
        void matmulCPU(float* A, float *B, float *C, int rows, int mix, int cols) const;

        // operators
        Tensor operator+(const Tensor& other) const;
        Tensor operator*(const Tensor& other) const;
        Tensor operator*(float scalar) const;
        Tensor operator-() const;
        Tensor sum(int axis = 0) const;
        Tensor mean(int axis = 0) const;
        Tensor transpose();
};

        // matrix operations GPU-based
__global__ void addMatrixGPU(float* A, float *B, float *C, int rows, int cols);
__global__ void addBroadcastGPU(float *A, float *B, float *C, int rows, int cols);
__global__ void scalarGPU(float *A, float scalar, float *C, int rows, int cols);
__global__ void matmulGPU(float *A, float *B, float *C, int nA, int nB, int nC);
__global__ void sumGPU(float *A, int axis, float *C, int rows, int cols);
__global__ void meanGPU(float *A, int axis, float *C, int rows, int cols);
