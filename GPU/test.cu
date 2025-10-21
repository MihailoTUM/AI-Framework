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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = rows * cols;
    if(idx < size) {
        C[idx] = A[idx] * B[idx];
    };
}

__global__ void tensorMatmul(float *A, float *B, float * C, int nA, int nB, int nC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx / nC;
    int j = idx % nC;

    int size = nA * nC;
    if (idx < size) {
        float sum = 0.0f;
        for(int run = 0; run < nB; run++) {
            sum += A[run + i * nB] * B[run * nC + j];
        }
        C[i * nC + j] = sum;
    };
};

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
        std::uniform_real_distribution<float> distFloat(0.0f, 1.0f);

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

    int getRows() const { return rows; };

    int getCols() const { return cols; };

    float* getMatrix() const { return matrix; };

    Tensor operator+(Tensor& other) const {
        if(rows == other.rows && cols == other.cols) {
            Tensor r(rows, cols);
            size_t size = rows * cols * sizeof(float);

            float *d_A, *d_B, *d_C;
            cudaMalloc(&d_A, size);
            cudaMalloc(&d_B, size);
            cudaMalloc(&d_C, size);

            cudaMemcpy(d_A, matrix, size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, other.matrix, size, cudaMemcpyHostToDevice);
            
            int threads = 256;
            int blocks = (other.getRows() * other.getCols() + threads - 1)/threads;
            tensorAdd<<<blocks, threads>>>(d_A, d_B, d_C, other.getRows(), other.getCols());

            cudaMemcpy(r.getMatrix(), d_C, size, cudaMemcpyDeviceToHost);

            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            return r;
        }
        else {
            throw std::invalid_argument("Wrong dimensions!");
        }
    }

    Tensor operator*(Tensor& other) const {
        if(rows == other.rows && cols == other.cols) {
            Tensor r(rows, cols);

            int nSize = getRows() * getCols();
            size_t size = nSize * sizeof(float);

            float *d_A, *d_B, *d_C;

            cudaMalloc(&d_A, size);
            cudaMalloc(&d_B, size);
            cudaMalloc(&d_C, size);
            cudaMemcpy(d_A, getMatrix(), size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, other.getMatrix(), size, cudaMemcpyHostToDevice);

            int threads = 256; //threads per block
            int blocks = (rows * cols + threads - 1)/threads;

            tensorMultiply<<<blocks, threads>>>(d_A, d_B, d_C, other.getRows(), other.getCols());

            cudaMemcpy(r.getMatrix(), d_C, size, cudaMemcpyDeviceToHost);

            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            return r;
        }
        else if(cols == other.rows) {
            Tensor r(rows, other.cols);

            int nSize = getRows() * other.getCols();
            size_t size = nSize * sizeof(float);

            float *d_A, *d_B, *d_C;

            cudaMalloc(&d_A, size);
            cudaMalloc(&d_B, size);
            cudaMalloc(&d_C, size);
            cudaMemcpy(d_A, getMatrix(), size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, other.getMatrix(), size, cudaMemcpyHostToDevice);

            int threads = 256;
            int blocks = (rows * cols + threads - 1)/threads;

            tensorMatmul<<<blocks, threads>>>(d_A, d_B, d_C, getRows(), getCols(), other.getCols());

            cudaDeviceSynchronize();
            cudaMemcpy(r.getMatrix(), d_C, size, cudaMemcpyDeviceToHost);

            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            return r;
        }
        else {
            throw std::invalid_argument("Wrong dimensions!");
        }
    };

};

int main() {
    Tensor tensorA(3, 2);
    tensorA.print();
    std::cout << "\n";

    Tensor tensorB(2, 2);
    tensorB.print();
    std::cout << "\n";

    Tensor tensorC = tensorA * tensorB;
    tensorC.print();

    return 0;
}