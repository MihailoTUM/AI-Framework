#include "Tensor.h"
#include <iostream>
#include <random>
#include <cuda_runtime.h>

__global__ void addMatrixGPU(float* A, float* B, float* C, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = rows * cols;

    if(idx < size) {
        C[idx] = A[idx] + B[idx];
    };
}

__global__ void matmulGPU(float *A, float *B, float * C, int nA, int nB, int nC) {
    int size = nA * nC;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx / nC;
    int j = idx % nC;


    if (idx < size) {
        float sum = 0.0f;
        for(int run = 0; run < nB; run++) {
            sum += A[run + i * nB] * B[run * nC + j];
        }
        C[i * nC + j] = sum;
    };
};

__global__ void scalarGPU(float *A, float scalar, float* C, int rows, int cols) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int size = rows * cols;
    if(idx < size) {
        C[idx] = scalar * A[idx];
    }
}

__global__ void addBroadcastGPU(float *A, float *B, float *C, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = rows * cols;
    int i = idx / cols;
    int j = idx % cols;

    if(idx < size) {
        C[i * cols + j] = A[i * cols + j] + B[j];
    }
}

__global__ void sumGPU(float *A, int axis, float *C, int rows, int cols) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int size = rows * cols;

    int i = idx / cols; 
    int j = idx % cols;

    if(idx >= size) return;

        if(axis == 0){
            // (5, 3) -> (1, 3);
            atomicAdd(&C[j], A[i * cols + j]);
        }
        else {
            //axis == 1;
            atomicAdd(&C[i], A[i * cols + j]);
        }
    
}; 

__global__ void meanGPU(float* A, int axis, float *C, int rows, int cols) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int size = rows * cols;

    int i = idx / cols;
    int j = idx % cols;

    if(idx >= size) return;

    if(axis == 0) {
        atomicAdd(&C[i], A[i * cols + j]/rows);
    }
    else {
        atomicAdd(&C[i], A[i * cols + j]/cols);
    };
};

Tensor::Tensor(int nRows, int nCols, char nDevice, bool random) {
    rows = nRows;
    cols = nCols;
    device = nDevice;
    matrix = new float[rows * cols];

    if(random) {
        initMatrixRandom();
    }
    else {
        initMatrixToZeros();
    }
};

Tensor::~Tensor() {
    delete[] matrix;
}

int Tensor::getRows() const {
    return rows;
}

int Tensor::getCols() const {
    return cols;
}

float Tensor::getValue(int row, int col) const {
    return matrix[row * cols + col];
};

float* Tensor::getMatrix() const {
    return matrix;
}

char Tensor::getDevice() const {
    return device;
}

void Tensor::setValue(int row, int col, float value) {
    matrix[row * cols + col] = value;
};

void Tensor::initMatrixToZeros() {
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                setValue(i, j, 0.0f);
            };
        };
    };

void Tensor::initMatrixRandom() {

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                setValue(i, j, dist(gen));
            };
        };
    };

void Tensor::print() const {
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                std::cout << getValue(i, j) << " ";
            };
            std::cout << "\n";
        };
    }

void Tensor::addMatrixCPU(float *A, float *B, float *C, int rows, int cols) const {
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                C[i * cols + j] = A[i * cols + j] + B[i * cols + j];
            }
        }
    }

void Tensor::matmulCPU(float *A, float* B, float *C, int rows, int mix, int cols) const {
        for(int x = 0; x < rows; x++) {
            for(int y = 0; y < cols; y++) {
                float value = 0;
                    for(int k = 0; k < cols; k++) {
                        value += A[cols * x + k] * B[cols * k + y];
                    }
                C[x * cols + y] = value;
            }     
        }
    }

void Tensor::scalarCPU(float* A, float scalar, float *C, int rows, int cols) const{
        for(int i = 0; rows; i++) {
            for(int j = 0; j < cols; j++) {
                A[i * cols + j] = scalar * C[i * cols + j];
            }
        }
    }

void Tensor::addBroadcastCPU(float *A, float *B, float* C, int rows, int cols) const {
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                C[i * cols + j] = A[i * cols + j] + B[j];
            }
        } 
    };

Tensor Tensor::operator+(const Tensor& other) const {
        // allow broadcasting
        if(rows == other.rows && cols == other.cols) {
            if(device == other.device) {
                Tensor result(rows, cols, device, false);
                if(device == 'C') {
                    std::cout << "HAPPENS ON CPU \n";
                    addMatrixCPU(getMatrix(), other.getMatrix(), result.getMatrix(), rows, cols);
                }
                else if(device == 'G') {
                    std::cout << "HAPPENS ON GPU \n";
                    size_t size = rows * cols * sizeof(float);

                    float *d_A, *d_B, *d_C;
                    cudaMalloc(&d_A, size);
                    cudaMalloc(&d_B, size);
                    cudaMalloc(&d_C, size);

                    cudaMemcpy(d_A, getMatrix(), size, cudaMemcpyHostToDevice);
                    cudaMemcpy(d_B, other.getMatrix(), size, cudaMemcpyHostToDevice);
                
                    int threads = 256;
                    int blocks = (rows * cols + threads - 1)/threads;
                    addMatrixGPU<<<blocks, threads>>>(d_A, d_B, d_C, other.getRows(), other.getCols());

                    cudaMemcpy(result.getMatrix(), d_C, size, cudaMemcpyDeviceToHost);

                    cudaFree(d_A);
                    cudaFree(d_B);
                    cudaFree(d_C);
                }
                else {
                    throw std::invalid_argument("Invalid");
                };
                return result;
            }
            else {
                throw std::invalid_argument("Not on the same device!");
            }
        }
        else if(cols == other.cols && other.rows == 1) {
            if(getDevice() == other.getDevice()) {
                Tensor result(rows, cols, getDevice(), false);
                if(device == 'C') {
                    std::cout << "HAPPENS ON CPU";
                    addBroadcastCPU(getMatrix(), other.getMatrix(), result.getMatrix(), rows, cols);
                    return result;
                }
                else if(device == 'G') {
                    std::cout << "HAPPENS ON GPU \n";
                    size_t sizeA = rows * cols * sizeof(float);
                    size_t sizeB = other.cols * sizeof(float);

                    float *d_A, *d_B, *d_C;
                    cudaMalloc(&d_A, sizeA);
                    cudaMalloc(&d_B, sizeB);
                    cudaMalloc(&d_C, sizeA);

                    cudaMemcpy(d_A, getMatrix(), sizeA, cudaMemcpyHostToDevice);
                    cudaMemcpy(d_B, other.getMatrix(), sizeB, cudaMemcpyHostToDevice);
                
                    int threads = 256;
                    int blocks = (rows * cols + threads - 1)/threads;
                    addBroadcastGPU<<<blocks, threads>>>(d_A, d_B, d_C, getRows(), getCols());

                    cudaMemcpy(result.getMatrix(), d_C, sizeA, cudaMemcpyDeviceToHost);

                    cudaFree(d_A);
                    cudaFree(d_B);
                    cudaFree(d_C);

                    return result;
                }   
                else {
                    throw std::invalid_argument("Invalid arguments passed!");
                }
            }
            else {
                throw std::invalid_argument("Invalid argument!");
            }
        }
        else {
            throw std::invalid_argument("Invalid dimensions!");
        }
    };

Tensor Tensor::operator*(const Tensor& other) const {
        if(cols == other.rows) {
            if(device == other.device) {
                Tensor result(rows, other.cols, device, false);
                if(device == 'C') {
                    std::cout << "HAPPENS ON CPU \n";
                    matmulCPU(getMatrix(), other.getMatrix(), result.getMatrix(), rows, cols, other.cols);
                }
                else if(device == 'G') {
                    std::cout << "HAPPENS ON GPU \n";
                    size_t size_A = rows * cols * sizeof(float);
                    size_t size_B = other.getRows() * other.getCols() * sizeof(float);
                    size_t size_C = rows * getCols() * sizeof(float);

                    float *d_A, *d_B, *d_C;
                    cudaMalloc(&d_A, size_A);
                    cudaMalloc(&d_B, size_B);
                    cudaMalloc(&d_C, size_C);

                    cudaMemcpy(d_A, getMatrix(), size_A, cudaMemcpyHostToDevice);
                    cudaMemcpy(d_B, other.getMatrix(), size_B, cudaMemcpyHostToDevice);
                
                    int threads = 256;
                    int blocks = (rows * cols + threads - 1)/threads;
                    matmulGPU<<<blocks, threads>>>(d_A, d_B, d_C, getRows(), getCols(), other.getCols());

                    cudaMemcpy(result.getMatrix(), d_C, size_C, cudaMemcpyDeviceToHost);

                    cudaFree(d_A);
                    cudaFree(d_B);
                    cudaFree(d_C);
                }
                else {
                    throw std::invalid_argument("Invalid");
                };
                return result;
            }
            else {
                throw std::invalid_argument("Not on the same device!");
            }
        }
        else {
            throw std::invalid_argument("Invalid dimensions!");
        }
    }

Tensor Tensor::operator* (float scalar) const {
        Tensor result(getRows(), getCols(), getDevice(), false);
        if(getDevice() == 'C') {
            std::cout << "HAPPENS ON CPU";
            scalarCPU(getMatrix(), scalar, result.getMatrix(), getRows(), getCols());
        }
        else if(getDevice() == 'G'){
                std::cout << "HAPPENS ON GPU \n";
                size_t size = rows * cols * sizeof(float);

                float *d_A, *d_C;
                cudaMalloc(&d_A, size);
                cudaMalloc(&d_C, size);

                cudaMemcpy(d_A, getMatrix(), size, cudaMemcpyHostToDevice);
                
                int threads = 256;
                int blocks = (rows * cols + threads - 1)/threads;
                scalarGPU<<<blocks, threads>>>(d_A, scalar, d_C, getRows(), getCols());

                cudaMemcpy(result.getMatrix(), d_C, size, cudaMemcpyDeviceToHost);

                cudaFree(d_A);
                cudaFree(d_C);
        }
        else {
            throw std::invalid_argument("Invalid arguments!");
        };
        return result;
    };

Tensor Tensor::operator-() const {
        Tensor result(rows, cols, device, false);
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                result.getMatrix()[i * cols + j] = getValue(i, j);
            }
        }
        return result;
    }
Tensor Tensor::sum(int axis) const {
        if(axis == 0) {
            if(device == 'C') {
            Tensor result(1, getCols(), getDevice(), false);
             for(int k = 0; k < this->cols; k++) {
                float sum = 0;
                for(int i = 0; i < this->rows; i++) {
                    sum += this->matrix[i * this->cols + k];
                }
                result.setValue(0, k, sum);
            }
            return result;
            } else if(device == 'G') {
                Tensor result(1, getCols(), getDevice(), false);
                std::cout << "HAPPENS ON GPU \n";
                size_t sizeA = getRows() * getCols() * sizeof(float);
                size_t sizeC = getCols() * sizeof(float);

                float *d_A, *d_C;
                cudaMalloc(&d_A, sizeA);
                cudaMalloc(&d_C, sizeC);

                cudaMemcpy(d_A, getMatrix(), sizeA, cudaMemcpyHostToDevice);
                
                int threads = 256;
                int blocks = (rows * cols + threads - 1)/threads;
                sumGPU<<<blocks, threads>>>(d_A, axis, d_C, getRows(), getCols());

                cudaMemcpy(result.getMatrix(), d_C, sizeC, cudaMemcpyDeviceToHost);

                cudaFree(d_A);
                cudaFree(d_C);

                return result;
            }
            else {
                throw std::invalid_argument("Invalid device argument!");
            }
        }   
        else if (axis == 1) {
            if(device == 'C') {
                Tensor result(getRows(), 1, getDevice(), false);
                for(int k = 0; k < this->rows; k++) {
                    float sum = 0;
                    for(int i = 0; i < this->cols; i++) {
                        sum += this->matrix[k * this->cols + i];
                    }
                    result.setValue(k, 0, sum);
                }
                return result;
            } 
            else if(device == 'G') {
                Tensor result(getRows(), 1, getDevice(), false);
                std::cout << "HAPPENS ON GPU \n";
                size_t sizeA = getRows() * getCols() * sizeof(float);
                size_t sizeC = getRows() * sizeof(float);

                float *d_A, *d_C;
                cudaMalloc(&d_A, sizeA);
                cudaMalloc(&d_C, sizeC);

                cudaMemcpy(d_A, getMatrix(), sizeA, cudaMemcpyHostToDevice);
                
                int threads = 256;
                int blocks = (rows * cols + threads - 1)/threads;
                sumGPU<<<blocks, threads>>>(d_A, axis, d_C, getRows(), getCols());

                cudaMemcpy(result.getMatrix(), d_C, sizeC, cudaMemcpyDeviceToHost);

                cudaFree(d_A);
                cudaFree(d_C);

                return result;
            }
            else {
                throw std::invalid_argument("Invalid device argument!");
            }
        }
        else {
            throw std::invalid_argument("Invalid axis > 1");
        }
    };
Tensor Tensor::mean(int axis) const {
        if(axis == 0) {
            Tensor result(1, getCols(), getDevice(), false);
             for(int k = 0; k < this->cols; k++) {
                float sum = 0;
                for(int i = 0; i < this->rows; i++) {
                    sum += this->matrix[i * this->cols + k];
                }
                result.setValue(0, k, sum/rows);
            }
            return result;
        }   
        else if (axis == 1) {
            Tensor result(getRows(), 1, getDevice(), false);
            for(int k = 0; k < this->rows; k++) {
                float sum = 0;
                for(int i = 0; i < this->cols; i++) {
                    sum += this->matrix[k * this->cols + i];
                }
                result.setValue(k, 0, sum/cols);
            }
            return result;
        }
        else {
            throw std::invalid_argument("Invalid axis > 1");
        }
    };
Tensor Tensor::transpose() {
    Tensor result(getCols(), getRows(), device, false);
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                result.getMatrix()[j * rows + i] = matrix[i * cols + j];
            }
        }
        return result;
};
Tensor operator*(float scalar, const Tensor& t) {
    return t * scalar;
}