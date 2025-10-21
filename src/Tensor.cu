#include <iostream>
#include <random>

__global__ void addMatrixGPU(float* A, float* B, float* C, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = rows * cols;

    if(idx < size) {
        C[idx] = A[idx] + B[idx];
    };
}

__global__ void matmulGPU(float *A, float *B, float * C, int nA, int nB, int nC) {
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

class Tensor {
    private:
        float* matrix;
        int rows;
        int cols;
        char device;

    public:
    // standard constructor
    Tensor(int nRows, int nCols, char nDevice, bool random = true) {
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

    }

    ~Tensor() {
        delete[] matrix;
    }

    // getters
    int getRows() const { return rows; };
    int getCols() const { return cols; };
    float* getMatrix() const { return matrix; };
    float getValue(int row, int col) const { return matrix[row * cols + col]; };
    char getDevice() const { return device; };

    //setters
    void setValue(int row, int col, float value) {
        matrix[row * cols + col] = value;
    }

    void initMatrixToZeros() {
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                setValue(i, j, 0.0f);
            };
        };
    };

    void initMatrixRandom() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                setValue(i, j, dist(gen));
            };
        };
    };

    void print() {
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                std::cout << getValue(i, j) << " ";
            };
            std::cout << "\n";
        };
    }

    void addMatrixCPU(float *A, float *B, float *C, int rows, int cols) const {
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                C[i * cols + j] = A[i * cols + j] + B[i * cols + j];
            }
        }
    }

    void matmulCPU(float *A, float* B, float *C, int rows, int mix, int cols) const {
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

    void scalarCPU(float* A, float scalar, float *C, int rows, int cols) const{
        for(int i = 0; rows; i++) {
            for(int j = 0; j < cols; j++) {
                A[i * cols + j] = scalar * C[i * cols + j];
            }
        }
    }

    void addBroadcastCPU(float *A, float *B, float* C, int rows, int cols) const {
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                C[i * cols + j] = A[i * cols + j] + B[j];
            }
        } 
    };

    Tensor operator+(const Tensor& other) const {
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
        else if(cols == other.cols && rows == 1) {
            if(getDevice() == other.getDevice()) {
                Tensor result(rows, cols, getDevice(), false);
                if(device == 'C') {
                    std::cout << "HAPPENS ON CPU";
                    addBroadcastCPU(getMatrix(), other.getMatrix(), result.getMatrix(), rows, cols);
                }
                else if(device == 'G') {

                } 
                else {

                }
            }
        }
        else {
            throw std::invalid_argument("Invalid dimensions!");
        }
    };

    Tensor operator*(const Tensor& other) const {
        if(cols == other.rows) {
            if(device == other.device) {
                Tensor result(rows, other.cols, device, false);
                if(device == 'C') {
                    std::cout << "HAPPENS ON CPU \n";
                    matmulCPU(getMatrix(), other.getMatrix(), result.getMatrix(), rows, cols, other.cols);
                }
                else if(device == 'G') {
                    std::cout << "HAPPENS ON GPU \n";
                    size_t size = rows * other.getCols() * sizeof(float);

                    float *d_A, *d_B, *d_C;
                    cudaMalloc(&d_A, size);
                    cudaMalloc(&d_B, size);
                    cudaMalloc(&d_C, size);

                    cudaMemcpy(d_A, getMatrix(), size, cudaMemcpyHostToDevice);
                    cudaMemcpy(d_B, other.getMatrix(), size, cudaMemcpyHostToDevice);
                
                    int threads = 256;
                    int blocks = (rows * cols + threads - 1)/threads;
                    matmulGPU<<<blocks, threads>>>(d_A, d_B, d_C, getRows(), getCols(), other.getCols());

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
        else {
            throw std::invalid_argument("Invalid dimensions!");
        }
    }

    Tensor operator* (float scalar) const {
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

    //neg
    Tensor operator-() const {
        Tensor result(rows, cols, device, false);
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                result.getMatrix()[i * cols + j] = getValue(i, j);
            }
        }
        return result;
    }
    
    //sum
    Tensor sum(int axis = 0) const {
        if(axis == 0) {
            Tensor result(1, getCols(), getDevice(), false);
             for(int k = 0; k < this->cols; k++) {
                float sum = 0;
                for(int i = 0; i < this->rows; i++) {
                    sum += this->matrix[i * this->cols + k];
                }
                result.setValue(0, k, sum);
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
                result.setValue(k, 0, sum);
            }
            return result;
        }
        else {
            throw std::invalid_argument("Invalid axis > 1");
        }
    };
    //mean

    Tensor mean(int axis = 0) const {
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
};

    Tensor operator*(float scalar, const Tensor& t) {
        return t * scalar;
    }

int main() {

    Tensor a (3, 2, 'G', true);
    a.print();
    std::cout << "\n";

    Tensor b(2, 6, 'G', true);
    b.print();
    std::cout << "\n";

    float scalar = 10.0f;

    Tensor c = scalar * a;
    c.print();

    std::cout << "\n";

    Tensor sum = b.sum(0);
    sum.print();

    std::cout << "\n";

    Tensor mean = b.mean(0);
    mean.print();

    return 0;
}