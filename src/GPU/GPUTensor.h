#include <cuda_runtime.h>

#pragma once

class GPUTensor {
    private:
        int id;
        float *matrix;
        float *d_matrix;

        int rows; 
        int cols;
        const GPUTensor *parent1;
        const GPUTensor *parent2;

    public:
        // constructor
        GPUTensor(int nRows, int nCols);
        GPUTensor(const GPUTensor& input);
        ~GPUTensor();

        // helpers
        void init();
        int countGPU();
        float* uploadToGPU();
        void copyToHost();
        void toZeros();
        void toOnes();

        // graph-related

        int traverse();
        void subTraverse(const GPUTensor* pointer, int& count);

        // operations
        GPUTensor operator+(const GPUTensor& input) const;
        GPUTensor operator*(const GPUTensor& input) const;    
        GPUTensor operator*(float scalar) const;
        GPUTensor operator+(float scalar) const;
        GPUTensor operator-();
        GPUTensor sum(int axis);
        GPUTensor mean(int axis);
        void transpose();

        // getters
        int getId() const;
        int getRows() const;
        int getCols() const;
        float* getMatrix() const;
        float* getDMatrix() const;

        const GPUTensor* getParent1() const;
        const GPUTensor* getParent2() const;

        // setters
        void setParent1(const GPUTensor *p1);
        void setParent2(const GPUTensor *p2);


        // print
        void print();
        void printGPUSpecs();
};

    GPUTensor operator*(float scalar, const GPUTensor& input);
    GPUTensor operator+(float scalar, const GPUTensor& input);

__global__ void addMatrixGPU(float* A, float* B, float* C, int rows, int cols);
__global__ void addMatrixBroadCastGPU(float *A, float *B, float *C, int rows, int cols);
__global__ void mulMatrixGPU(float* A, float* B, float* C, int nA, int nB, int nC);
__global__ void scalarMulMatrixGPU(float* A, float scalar, float *C, int rows, int cols);
__global__ void scalarAddMatrixGPU(float* A, float scalar, float* C, int rows, int cols);
__global__ void negateMatrixGPU(float* A, float* B, int rows, int cols);
__global__ void sumMatrixGPU(float* A, int axis, float* C, int rows, int cols);
__global__ void meanMatrixGPU(float* A, int axis, float* C, int rows, int cols);