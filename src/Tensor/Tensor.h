#pragma once

#include <cuda_runtime.h>

class Tensor {
    private:
        float* matrix;
        int rows;
        int cols;
        char device;
        mutable const Tensor *parent1;
        mutable const Tensor *parent2;

    public: 
        Tensor(int nRows, int nCols, char nDevice);
        ~Tensor();

        // getters for matrix
        int getRows() const;
        int getCols() const;
        float getValue(int row, int col) const;
        
        // getters for tensors
        float* getMatrix() const;
        char getDevice() const;
        const Tensor*getParent1() const;
        const Tensor*getParent2() const;

        // setters
        void setValue(int row, int cols, float value);
        void setParent1(const Tensor *c);
        void setParent2(const Tensor *c);

        // init
        void initMatrixToZeros();
        void initMatrixToOnes();

        void intiMatrixRandomInt();
        void initMatrixRandom();

        void toZeros();
        void toOnes();
        void toInt();

        // print
        void print() const;
        void printDevice() const;

        bool checkGPU() const;

        // matrix operations CPU-based
        void addMatrixCPU(float* A, float *B, float *C, int rows, int cols) const;
        void addMatrixScalarCPU(float *A, float scalar, float *C, int row, int cols) const;
        void addBroadcastCPU(float* A, float *B, float *C, int rows, int cols) const;

        void scalarCPU(float *A, float scalar, float *C, int rows, int cols) const;
        void matmulCPU(float* A, float *B, float *C, int rows, int mix, int cols) const;

        // operators
        Tensor operator+(const Tensor& other) const;
        Tensor operator+(float scalar) const;

        Tensor operator*(const Tensor& other) const;
        Tensor operator*(float scalar) const;
        Tensor operator-() const;
        Tensor sum(int axis = 0) const;
        Tensor mean(int axis = 0) const;
        Tensor transpose();
};

    Tensor operator*(float scalar, const Tensor& t);
    Tensor operator+(float scalar, const Tensor& t);

        // matrix operations GPU-based
__global__ void addMatrixGPU(float* A, float *B, float *C, int rows, int cols);
__global__ void addMatrixScalarGPU(float *A, float scalar, float *C, int rows, int cols);

__global__ void addBroadcastGPU(float *A, float *B, float *C, int rows, int cols);
__global__ void scalarGPU(float *A, float scalar, float *C, int rows, int cols);
__global__ void matmulGPU(float *A, float *B, float *C, int nA, int nB, int nC);
__global__ void sumGPU(float *A, int axis, float *C, int rows, int cols);
__global__ void meanGPU(float *A, int axis, float *C, int rows, int cols);
