#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <iostream>
#include <string>
#include <random>

template <typename T>
class Tensor {
    private:
        T* matrix;
        int rows;
        int cols;
        std::string device;

        template <typename U, typename F, typename G>
        static void runDependingOnType(F f, G g) {
            if constexpr (std::is_floating_point<U>::value) f();
            else if constexpr (std::is_integral<U>::value) g();
        }

    public:


        Tensor(int nRows, int nCols, std::string nDevice = "CPU") {
            this->rows = nRows;
            this->cols = nCols;
            this->device = nDevice;
            this->matrix = new T[this->rows * this->cols];

            runDependingOnType<T>(
                [this]() { initFloat(); },
                [this]() { initInt(); }
            );

            runDependingOnType<T>(
                [this] () { randomFloat(); },
                [this] () { randomInt(); }
            );

        };

        Tensor(int nRows): Tensor(nRows, 1, "CPU") {
            this->matrix = new T[this->rows];
            runDependingOnType<T>(
                [this]() { initFloat(); },
                [this]() { initInt(); }
            );
        }

        Tensor(int nRows, int nCols, std::string nDevice, T*array): Tensor(nRows, nCols, nDevice) {
            this->matrix = array;
        };

        int getRows() const {
            return rows;
        }

        int getCols() const {
            return cols;
        }

        T* getMatrix() const {
            return matrix;
        }

        void initInt() {
            for(int i = 0; i < rows; i++) {
                for(int j = 0; j < cols; j++) {
                    this->matrix[i * cols + j] = 0;
                }
            }
        }
        void initFloat() {
            for(int i = 0; i < rows; i++) {
                for(int j = 0; j < cols; j++) {
                    matrix[cols * i + j] = 0.0f;
                }
            }
        }

        void randomInt() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> dist(1, 5);

            for(int i = 0; i < rows; i++) {
                for(int j = 0; j < cols; j++) {
                    fill(i, j, dist(gen));
                }
            }
        };
        void randomFloat() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> distFloat(-1.0f, 1.0f);

            for(int i = 0; i < rows; i++) {
                for(int j = 0; j < cols; j++) {
                    fill(i, j, distFloat(gen));
                }
            }
        };

        template <typename P>
        void fill(int row, int col, P value) {
            if(row >= this->rows || col >= this->cols) {
                throw std::invalid_argument("Out of bounds for either row or col");
            }
            matrix[row * cols + col] = value;
        }

        T get(int row, int col) const {
            if(row >= this->rows || col >= this->cols) {
                throw std::invalid_argument("Out of bounds for either row or col"); 
            }
            return this->matrix[row * this->cols + col];
        };

        void print() const {
            for(int i = 0; i < rows; i++) {
                for(int j = 0; j < cols; j++) {
                    std::cout << matrix[cols * i + j] << " ";
                }
                std::cout << "" << std::endl;
            }
        };

        // addition & subtraction of Tensors -> returns new Tensor
        Tensor operator +(const Tensor& other) const {
            if(rows == other.rows && cols == other.cols) {
                Tensor g(rows, cols);
                for(int i = 0; i < rows; i++) {
                    for(int j = 0; j < cols; j++) {
                        g.fill(i, j, matrix[i * cols + j] + other.matrix[i * cols + j]);
                    }
                }
                return g;
            } 
            else if (this->cols == other.cols && other.rows == 1) {
                Tensor g(rows, cols);
                for(int i = 0; i < rows; i++) {
                    for(int j = 0; j < cols; j++) {
                        g.fill(i, j, matrix[i * cols + j] + other.matrix[j]);
                    }
                }
                return g;
            }
            else {
                throw std::invalid_argument("Invalid dimensions for addition");
            };
        }
        Tensor operator -(const Tensor& other) const {
            if(rows == other.rows && cols == other.cols) {
                Tensor g(rows, cols);
                for(int i = 0; i < rows; i++) {
                    for(int j = 0; j < cols; j++) {
                        g.fill(i, j, matrix[i * cols + j] - other.matrix[i * cols + j]);
                    }
                }
                return g;
            } else {
                throw std::invalid_argument("Invalid dimensions for subtraction");
            }; 
        }

        // scalar multiplication
        Tensor operator *(T scalar) const {
            Tensor g(rows, cols);
            for(int i = 0; i < rows; i++) {
                for(int j = 0; j < cols; j++) {
                    g.fill(i, j, scalar * matrix[i * cols + j]);
                }
            }
            return g;
        }

        // matrix multiplication 
        Tensor operator *(const Tensor& other) const {
            if(cols == other.rows) {
            Tensor g(rows, other.cols);
                for(int x = 0; x < rows; x++) {
                    for(int y = 0; y < other.cols; y++) {
                        float value = 0;
                        for(int k = 0; k < cols; k++) {
                            value += matrix[cols * x + k] * other.matrix[other.cols * k + y];
                        }
                        g.fill(x, y, value);
                    }
                }
                return g;
            }
            else {
                throw std::invalid_argument("Invalid dimensions for matrix multiplication");
            }
        }

        // negates the existing Tensor
        Tensor operator -() const {
            Tensor g(rows, cols);
            for(int i = 0; i < rows; i++) {
                for(int j = 0; j < cols; j++) {
                    g.fill(i, j, -matrix[i * cols + j]);
                }
            }
            return g;
        }
        
        Tensor sum(int axis = 0) {
            if(axis == 0) {
                // (2, 3) -> (1, 3) (column-wise)
                Tensor g(1, this->cols);
                for(int k = 0; k < this->cols; k++) {
                    T sum = 0;
                    for(int i = 0; i < this->rows; i++) {
                        sum += this->matrix[i * this->cols + k];
                    }
                    g.fill(0, k, sum);
                }

                return g;

            }   
            else if (axis == 1) {
                // (2, 3) -> (2, 1) (row-wise)
                Tensor g(this->rows, 1);
                for(int k = 0; k < this->rows; k++) {
                    T sum = 0;
                    for(int i = 0; i < this->cols; i++) {
                        sum += this->matrix[k * this->cols + i];
                    }
                    g.fill(k, 0, sum);
                }
                return g;
            }
            else {
                throw std::invalid_argument("Invalid axis provided. Axis can only be 0 or 1");
            }
        }
        
        Tensor mean(int axis = 0) {
                // (2, 3) -> (1, 3) (column-wise)
            if(axis == 0){ 
                Tensor g(1, this->cols);
                for(int k = 0; k < this->cols; k++) {
                    T sum = 0;
                    for(int i = 0; i < this->rows; i++) {
                        sum += this->matrix[i * this->cols + k];
                    }
                    sum = static_cast<T>(sum / this->rows);
                    g.fill(0, k, sum);
                }

                return g;
            }
            else if (axis == 1) {
                // (2, 3) -> (2, 1) (row-wise)
                Tensor g(this->rows, 1);
                for(int k = 0; k < this->rows; k++) {
                    T sum = 0;
                    for(int i = 0; i < this->cols; i++) {
                        sum += this->matrix[k * this->cols + i];
                    }
                    sum = static_cast<T>(sum / this->cols);
                    g.fill(k, 0, sum);
                }
                return g;
            }
            else {
                throw std::invalid_argument("Invalid axis provided. Axis can only be 0 or 1");
            }
        }

        Tensor transpose() const {
            // (2, 3) -> (3, 2)
            Tensor g(this->cols, this->rows);

            for(int i = 0; i < this->rows; i++) {
                for(int k = 0; k < this->cols; k++) {
                    g.fill(k, i, this->matrix[i * cols + k]);
                }
            }

            return g;
        };

};


#endif