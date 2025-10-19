// #include "Tensor.hpp"

// template <typename T>
// class DenseLayer {
//     private:
//     Tensor<T> weights;
//     Tensor<T> bias;

//     int n_input;
//     int n_output;

//     public:
//     DenseLayer(int n_inputN, int n_outputN): n_input(n_inputN), n_output(n_outputN), weights(n_inputN, n_outputN, "CPU"), bias(1, n_outputN) {};

//     Tensor<T> forward(const Tensor<T>& input) {
//         Tensor matmul = input * weights;
//         Tensor output = matmul + bias;
//         return output;
//     };

//     Tensor<T> get_weights() const {
//         return weights;
//     }

//     Tensor<T> get_bias () const {
//         return bias;
//     }
// };
