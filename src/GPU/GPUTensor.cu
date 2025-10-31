#include "GPUTensor.h"
#include <iostream>
#include <random>
#include <cuda_runtime.h>

GPUTensor::GPUTensor(int nRows, int nCols) {
    id = 0;
    rows = nRows;
    cols = nCols;
    matrix = new float[rows * cols];

    parent1 = nullptr;
    parent2 = nullptr;

    init();
    d_matrix = nullptr;
    uploadToGPU();
}

GPUTensor::GPUTensor(const GPUTensor& input) {
    rows = input.rows;
    cols = input.cols;
    id = input.id;

    matrix = new float[rows * cols];
    for(int i = 0; i < rows * cols; i++){
        matrix[i] = input.getMatrix()[i];
    };
    
    size_t size = rows * cols * sizeof(float);
    cudaMalloc(&d_matrix, size);
    cudaMemcpy(d_matrix, input.d_matrix, size, cudaMemcpyDeviceToDevice);

    parent1 = input.parent1;
    parent2 = input.parent2;
}

GPUTensor::~GPUTensor() {
    delete[] matrix;
    if(d_matrix) {
        cudaFree(d_matrix);
    }
}


// helpers
void GPUTensor::init() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for(int i = 0; i < rows * cols; i++) {
        matrix[i] = dist(gen);
    };
}

int GPUTensor::countGPU() {
    int countDevice = 0;
    cudaError_t err = cudaGetDeviceCount(&countDevice);
    if(err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 0;
    };
    return countDevice;
}

float* GPUTensor::uploadToGPU() {
    size_t size = rows * cols * sizeof(float);
    cudaMalloc(&d_matrix, size);
    cudaMemcpy(d_matrix, matrix, size, cudaMemcpyHostToDevice);

    // optional returing GPU pointer
    return d_matrix;
}

void GPUTensor::copyToHost() {
    size_t size = rows * cols * sizeof(float);
    cudaMemcpy(matrix, getDMatrix(), size, cudaMemcpyDeviceToHost);
}

void GPUTensor::toOnes() {
    for(int i = 0; i < rows * cols; i++) {
        matrix[i] = 1.0f;
    }
    uploadToGPU();
};

void GPUTensor::toZeros() {
    for(int i = 0; i < rows * cols; i++) {
        matrix[i] = 0.0f;
    }
    uploadToGPU();
}

// graph-related
int GPUTensor::traverse() {
    int count = 0;
    subTraverse(this, count);
    return count;
}

void GPUTensor::subTraverse(const GPUTensor* pointer, int& count) {
    if(pointer->getParent1()) {
        subTraverse(pointer->getParent1(), count);
    }
    if(pointer->getParent2()) {
        subTraverse(pointer->getParent2(), count);
    }
    count++;
}

// getters
int GPUTensor::getId() const {
    return id;
}

int GPUTensor::getRows() const {
    return rows;
}

int GPUTensor::getCols() const {
    return cols;
}

float* GPUTensor::getMatrix() const {
    return matrix;
}

float* GPUTensor::getDMatrix() const {
    return d_matrix;
}

const GPUTensor* GPUTensor::getParent1() const {
    return parent1;
};

const GPUTensor* GPUTensor::getParent2() const {
    return parent2;
};

// setters
void GPUTensor::setParent1(const GPUTensor* p1) {
    parent1 = p1;
}

void GPUTensor::setParent2(const GPUTensor* p2) {
    parent2 = p2;
}

// print
void GPUTensor::print() {
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if(err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    };

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << prop.name << "\n";

    std::cout << "(" << rows << ", " << cols << ")\n";

    for(int i = 0; i < getRows(); i++) {
        for(int j = 0; j < getCols(); j++) {
            std::cout << getMatrix()[i * cols + j] << " ";
        }
        std::cout << "\n";
    };

    std::cout << "\n";
};

void GPUTensor::printGPUSpecs() {
    int deviceCount = countGPU();
    if(deviceCount == 0) return;

    std::cout << "\n";

    for(int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "Device " << i << ": " << prop.name << "\n";
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  Total global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
        std::cout << "  Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB\n";
        std::cout << "  Registers per block: " << prop.regsPerBlock << "\n";
        std::cout << "  Warp size: " << prop.warpSize << "\n";
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "  Max threads per dimension: ["
                  << prop.maxThreadsDim[0] << ", "
                  << prop.maxThreadsDim[1] << ", "
                  << prop.maxThreadsDim[2] << "]\n";
        std::cout << "  Max grid size: ["
                  << prop.maxGridSize[0] << ", "
                  << prop.maxGridSize[1] << ", "
                  << prop.maxGridSize[2] << "]\n";
        std::cout << "  Clock rate: " << prop.clockRate / 1000.0f << " MHz\n";
        std::cout << "  Memory pitch: " << prop.memPitch << "\n";
        std::cout << "  Device overlap: " << prop.deviceOverlap << "\n";
        std::cout << "  MultiProcessor count: " << prop.multiProcessorCount << "\n\n";  
    };

}


// operations
GPUTensor GPUTensor::operator+(const GPUTensor& input) const {
    if(rows == input.rows && cols == input.cols) {
        GPUTensor result(rows, cols);
        result.setParent1(this);
        result.setParent2(&input);

        int threads = 256;
        int blocks = (rows * cols + threads - 1)/ threads;
        addMatrixGPU<<<blocks, threads>>>(this->d_matrix, input.getDMatrix(), result.getDMatrix(), rows, cols);
        cudaDeviceSynchronize();

        return result;
    }
    else if(input.rows == 1 && cols == input.cols) {
        GPUTensor result(rows, cols);
        result.setParent1(this);
        result.setParent2(&input);

        int threads = 256;
        int blocks = (rows * cols + threads - 1)/ threads;
        addMatrixBroadCastGPU<<<blocks, threads>>>(this->d_matrix, input.getDMatrix(), result.getDMatrix(), rows, cols);
        cudaDeviceSynchronize();

        return result;
    }
    else {
        throw std::invalid_argument("Invalid shape of Tensors!");
    }
};

GPUTensor GPUTensor::operator*(const GPUTensor& input) const {
    if(cols == input.rows) {
        GPUTensor result(rows, input.cols);
        result.setParent1(this);
        result.setParent2(&input);

        int threads = 256;
        int blocks = (rows * input.cols + threads - 1)/ threads;
        mulMatrixGPU<<<blocks, threads>>>(this->d_matrix, input.getDMatrix(), result.getDMatrix(), rows, cols, input.cols);
        cudaDeviceSynchronize();

        return result;
    }
    // (x, y) * (x, y) = element-wise multiplication
    else {
        throw std::invalid_argument("Invalid shape of Tensors!");
    }
};

GPUTensor GPUTensor::operator*(float scalar) const {
    GPUTensor result(rows, cols);
    result.setParent1(this);

    int threads = 256;
    int blocks = (rows * cols + threads - 1)/ threads;
    scalarMulMatrixGPU<<<blocks, threads>>>(this->d_matrix, scalar, result.getDMatrix(), rows, cols);
    cudaDeviceSynchronize();

    return result;
};

GPUTensor GPUTensor::operator+(float scalar) const {
    GPUTensor result(rows, cols);
    result.setParent1(this);

    int threads = 256;
    int blocks = (rows * cols + threads - 1)/ threads;
    scalarAddMatrixGPU<<<blocks, threads>>>(this->d_matrix, scalar, result.getDMatrix(), rows, cols);
    cudaDeviceSynchronize();

    return result;
};

GPUTensor GPUTensor::operator-() {
    GPUTensor result(rows, cols);
    result.setParent1(this);

    int threads = 256;
    int blocks = (rows * cols + threads - 1)/ threads;
    negateMatrixGPU<<<blocks, threads>>>(this->d_matrix, result.getDMatrix(), rows, cols);
    cudaDeviceSynchronize();

    return result;
};

GPUTensor GPUTensor::sum(int axis) {
    if(axis == 0){
        GPUTensor result(1, cols);
        result.setParent1(this);

        int threads = 256;
        int blocks = (rows * cols + threads - 1)/ threads;
        sumMatrixGPU<<<blocks, threads>>>(this->d_matrix, axis, result.getDMatrix(), rows, cols);
        cudaDeviceSynchronize();

        return result;
    }
    else if(axis == 1) {
        GPUTensor result(rows, 1);
        result.setParent1(this);

        int threads = 256;
        int blocks = (rows * cols + threads - 1)/ threads;

        sumMatrixGPU<<<blocks, threads>>>(this->d_matrix, axis, result.getDMatrix(), rows, cols);
        cudaDeviceSynchronize();

        return result;
    }
    else {
        throw std::invalid_argument("Invalid axis!");
    };
};

GPUTensor GPUTensor::mean(int axis) {
    if(axis == 0){
        GPUTensor result(1, cols);
        result.setParent1(this);

        int threads = 256;
        int blocks = (rows * cols + threads - 1)/ threads;

        meanMatrixGPU<<<blocks, threads>>>(this->d_matrix, axis, result.getDMatrix(), rows, cols);
        cudaDeviceSynchronize();

        return result;
    }
    else if(axis == 1) {
        GPUTensor result(rows, 1);
        result.setParent1(this);

        int threads = 256;
        int blocks = (rows * cols + threads - 1)/ threads;

        meanMatrixGPU<<<blocks, threads>>>(this->d_matrix, axis, result.getDMatrix(), rows, cols);
        cudaDeviceSynchronize();

        return result;
    }
    else {
        throw std::invalid_argument("Invalid axis!");
    }
};

void GPUTensor::transpose() {
    int s = rows;
    rows = cols;
    cols = s;  
}

__global__ void addMatrixGPU(float *A, float* B, float* C, int rows, int cols) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int size = rows * cols;

    if(idx < size) {
        C[idx] = A[idx] + B[idx];
    }
};

__global__ void mulMatrixGPU(float *A, float *B, float * C, int nA, int nB, int nC) {
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

__global__ void scalarMulMatrixGPU(float* A, float scalar, float* C, int rows, int cols) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int size = rows * cols;

    if(idx < size) {
        C[idx] = scalar * A[idx];
    }
};

__global__ void scalarAddMatrixGPU(float* A, float scalar, float* C, int rows, int cols) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int size = rows * cols;

    if(idx < size) {
        C[idx] = scalar + A[idx];
    };
};

__global__ void negateMatrixGPU(float* A, float* B, int rows, int cols) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int size = rows * cols;

    if(idx < size) {
        B[idx] = -A[idx];
    }
}

__global__ void sumMatrixGPU(float* A, int axis, float* C, int rows, int cols) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int size = rows * cols;

    int i = idx / cols; 
    int j = idx % cols;

    if(idx >= size) return;

    if(axis == 0){
        atomicAdd(&C[j], A[i * cols + j]);
    }
    else {
        atomicAdd(&C[i], A[i * cols + j]);
    };
};   

__global__ void meanMatrixGPU(float* A, int axis, float* C, int rows, int cols) {
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
}

__global__ void addMatrixBroadCastGPU(float *A, float *B, float *C, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = rows * cols;
    int i = idx / cols;
    int j = idx % cols;

    if(idx < size) {
        C[i * cols + j] = A[i * cols + j] + B[j];
    }
};

GPUTensor operator*(float scalar, const GPUTensor& input) {
    return input * scalar;
};

GPUTensor operator+(float scalar, const GPUTensor& input) {
    return input +  scalar;
}