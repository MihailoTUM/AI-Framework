#include "Tensor.hpp"

template <typename T>
class DenseLayer {
    private:
    Tensor<T>* weights;
    Tensor<T>* bias;

    int batch;
    int n_input;
    int n_output;

    public:
    DenseLayer(int batchN, int n_inputN, int n_outputN) {
        batch = batchN;
        n_input = n_inputN;
        n_output = n_outputN;
        
        weights = new Tensor(batch, n_input);
        bias = new Tensor(n_output, 1);
    };

    Tensor forward(const Tensor input&) {
        Tensor matmul = weights * input;
        Tensor output = matmul + bias;
        return output;
    };
};

int main() {

    return 0;
}