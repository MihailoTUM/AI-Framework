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
    float getValue(int row, int col) { return matrix[row * cols + col]; };

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


    Tensor operator+(const Tensor& other) const {
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
        else {
            throw std::invalid_argument("Invalid dimensions!");
        }
    };
};


int main() {

    Tensor a (3, 3, 'C', true);
    a.print();
    std::cout << "\n";

    Tensor b(3, 3, 'G', true);
    b.print();
    std::cout << "\n";

    Tensor c = a + b;
    c.print();

    return 0;
}