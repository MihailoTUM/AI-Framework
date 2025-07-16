#include <iostream>
#include <string>
#include <random>

class Tensor {
    public:
        float* pointer;
        int* shape;
        int rows;
        int cols;
        int size;

        Tensor(int n_rows, int n_cols=1, bool random=false) {
            if(n_rows == 0 || n_cols == 0){
                throw std::invalid_argument("Invalid rows or cols!");
            }

            shape = new int[2];
            rows = n_rows;
            shape[0] = rows;
            cols = n_cols;
            shape[1] = cols;

            if(n_cols == 0) {
                size = n_rows;
            }
            else {
                size = n_rows * n_cols;
            }

            pointer = new float[size];

            if(random) {
                std::random_device rd;
                std::mt19937 gen(rd());

                std::uniform_real_distribution<float> dist_float(-1.0f, 1.0f);

                for(int i = 0; i < size; i++) {
                    pointer[i] = std::round(dist_float(gen) * 100000) / 100000.0f;
                };
            }
            else {
                for(int i = 0; i < size; i++) {
                    pointer[i] = 0.0f;
                };
            }

        }

        void printShape() {
            std::cout << "(" << shape[0] << ", " << shape[1] << ")" << std::endl;
        }

        int* getShape() {
            return shape;
        }

        void print() {
            std::cout << "[";
            for(int i = 0; i < rows; i++) {
                if(i == 0) {
                    std::cout << "[ ";
                }
                else {
                    std::cout << " [ ";
                }
                for(int j = 0; j < cols; j++) {
                    std::cout << pointer[i * cols + j] << " ";
                }
                std::cout << "]";
                if(i < rows - 1) {
                    std::cout << "\n";
                }
            }
            std::cout << "]\n\n";
        }

        void set(int row, int col, float value) {
            pointer[col + row * col] = value;
        }

        static Tensor add(Tensor matrix1, Tensor matrix2) {
            if(matrix1.rows != matrix2.rows || matrix1.cols != matrix2.cols) {
                throw std::invalid_argument("Invalid shapes!");
            }

            Tensor additionTensor(matrix1.rows, matrix2.cols);

            for(int i = 0; i < matrix1.size; i++) {
                    additionTensor.pointer[i] = matrix1.pointer[i] + matrix2.pointer[i];
            }

            return additionTensor;
        }

        static Tensor multiply(Tensor matrix1, Tensor matrix2) {
            if(matrix1.rows != matrix2.rows || matrix1.cols != matrix2.cols) {
                throw std::invalid_argument("Invalid shapes!");
            }

            Tensor multiplyTensor(matrix1.rows, matrix2.cols);
            
            for(int i = 0; i < matrix1.size; i++) {
                multiplyTensor.pointer[i] = matrix1.pointer[i] * matrix2.pointer[i];
            }

            return multiplyTensor;

        }
        
        static float dotProduct(Tensor vector1, Tensor vector2) {
            if(vector1.shape[0] != vector2.shape[0]) {
                throw std::invalid_argument("Invalid shapes!");
            }
            int length = vector1.shape[0];
            float outcome = 0.0;

            for(int i = 0; i < length; i++) {
                outcome += vector1.pointer[i] * vector2.pointer[i];
            }

            return outcome;
        }

        static Tensor matmul(Tensor matrix1, Tensor matrix2) {
            if(matrix1.cols != matrix2.rows) {
                throw std::invalid_argument("Invalid shapes!");
            }

            int newSize = matrix1.rows * matrix2.cols;
            Tensor matmulTensor(matrix1.rows, matrix2.cols);

            for (int j = 0; j < matmulTensor.cols; j++) {
                for(int i = 0; i < matmulTensor.rows; i++) {
                    matmulTensor.pointer[j + matmulTensor.cols * i] = Tensor::dotProduct(matrix1.getRowVector(i), matrix2.getColumVector(j));
                }
            }

            return matmulTensor;

        };

        Tensor getRowVector(int row) {
            if(row > rows) {
                throw std::invalid_argument("Invalid row!");
            }

            Tensor rowVector (cols, 1, false);
            for(int i = 0; i < cols; i++) {
                rowVector.pointer[i] = pointer[i + row * cols];
            }

            return rowVector;
        };

        Tensor getColumVector(int col) {
            if(col > cols) {
                throw std::invalid_argument("Invalid col!");
            }

            Tensor colVector(rows, 1, false);
            for(int i = 0; i < rows; i++) {
                colVector.pointer[i] = pointer[col + cols * i];
            }
            return colVector;
        };
};


class Gradient {
    public:
    Tensor tensor;
    Gradient(int rows, int cols, bool random): tensor(rows, cols, random) {

    }
    
};


int main() {
    Tensor input(1, 5, true);
    input.printShape();
    input.print();
    Tensor weights(5, 6, true);
    weights.printShape();
    weights.print();

    Tensor output = Tensor::matmul(input, weights);
    output.printShape();
    output.print();

    return 0;
}