#include <iostream>
#include "Tensor.cu"

__global__ void reluGPU(float *A, float *C, int rows, int cols) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int size = rows * cols;

    if(idx < size) {
        if(A[idx] < 0) {
            C[idx] = 0.0f;
        } else {
            C[idx] = A[idx];
        }
    };
};


class Activation {
    private:
        char function;
    
    public:
    Activation(char func = 'R') {
        function = func;
    }

    char getFunction() { return function; };

    void reluCPU (float *A, int rows, int cols) {
        for(int i = 0; i < rows* cols; i++) {
            if(A[i] < 0) {
                A[i] = 0;
            };
        };
    };

    Tensor forward(const Tensor& other) {
        if(function == 'R') {
            return relu(other);
        }
        return other;
    };

    Tensor relu(const Tensor& input) {
        int rows = input.getRows();
        int cols = input.getCols();
        Tensor result(rows, cols, input.getDevice(), false);

        if(input.getDevice() == 'C') {
            std::cout << "HAPPENS ON CPU" << std::endl;
            reluCPU(input.getMatrix(), rows, cols);
            return result;
        }   
        else {
            std::cout << "HAPPENS ON GPU" << std::endl;
            size_t size = rows * cols * sizeof(float);
            
            float *d_A, *d_C;
            cudaMalloc(&d_A, size);
            cudaMalloc(&d_C, size);

            cudaMemcpy(d_A, input.getMatrix(), size, cudaMemcpyHostToDevice);

            int threads = 256;
            int blocks = (rows * cols + threads - 1)/threads;

            reluGPU<<<blocks, threads>>>(d_A, d_C, rows, cols);

            cudaMemcpy(result.getMatrix(), d_C, size, cudaMemcpyDeviceToHost);

            cudaFree(d_A);
            cudaFree(d_C);
            
            return result;
        }   
    };

    Tensor sigmoid();

    Tensor tanh();
};

int main() {

    Tensor cuda(3, 3, 'G', true);
    cuda.print();

    std::cout << "\n";

    Activation ac = ('R');
    Tensor relu = ac.forward(cuda);
    relu.print();

    return 0;
}