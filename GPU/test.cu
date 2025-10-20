#include <iostream>
#include <random>
#include <cuda_runtime.h>

__global__ void tensorAdd(float* A, float* B, float* C, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = rows * cols;
    if(idx < size) {
        C[idx] = A[idx] + B[idx];
    };
}

__global__ void tensorMultiply(float* A, float* B, float*C, int rows, int cols) {

}

class Tensor {
    private:
        float* matrix;
        int rows;
        int cols;

    public:
    Tensor(int nRows, int nCols, bool random = true) {
        this->rows = nRows;
        this->cols = nCols;
        this->matrix = new float[this->rows * this->cols];
        if(random) {
            init();
        }
        else {
            initZeros();
        }
    };

    void init() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> distFloat(-1.0f, 1.0f);

        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                matrix[i * cols + j] = distFloat(gen);
            }
        }
    }

    void initZeros() {
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                matrix[i * cols + j] = 0.0f;
            };
        };
    };

    void print() {
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                std::cout << matrix[i * cols + j] << " ";
            }
            std::cout << "\n";
        };
    };

    int getRows() { return rows; };

    int getCols() { return cols; };

    float* getMatrix() {
        return matrix;
    }

    Tensor operator+(Tensor& other) const {
        if(rows == other.rows && cols == other.cols) {
            Tensor r(rows, cols);
            size_t size = rows * cols * sizeof(float);

            float *d_A, *d_B, *d_C;
            cudaMalloc(&d_A, size);
            cudaMalloc(&d_B, size);
            cudaMalloc(&d_C, size);

            cudaMemcpy(d_A, matrix, size, cudaMemcpyHostToDevice);

        }
    };
};

int main() {
    Tensor tensorA(3, 3);
    Tensor tensorB(3, 3);
    Tensor tensorC(3, 3, false);

    size_t size = tensorA.getRows() * tensorA.getCols() * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, tensorA.getMatrix(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, tensorB.getMatrix(), size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (tensorA.getRows() * tensorA.getCols() + threads - 1)/threads;
    tensorAdd<<<blocks, threads>>>(d_A, d_B, d_C, tensorA.getRows(), tensorA.getCols());

    cudaMemcpy(tensorC.getMatrix(), d_C, size, cudaMemcpyDeviceToHost);

    for(int i = 0; i < tensorA.getRows(); i++) {
        for(int j = 0; j < tensorA.getCols(); j++) {
            std::cout << tensorC.getMatrix()[i * tensorA.getCols() + j] << " ";
        }
        std::cout << "\n";
    };

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}