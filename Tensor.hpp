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

        Tensor(int nRows, int nCols, std::string nDevice, T*array): Tensor(nRows, nCols, nDevice) {
            this->matrix = array;
        };

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
            matrix[row * cols + col] = value;
        }

        T get(int row, int col) {
            if(row > this->rows || col > this->cols) {
                throw std::invalid_argument("Row > rows || Col > cols"); 
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

        //
        // Tensor sum(int axis = 0);
        // Tensor mean(int axis = 0);

        // // negates the existing Tensor
        // Tensor operator -() const;

        // // addition & subtraction of Tensors -> returns new Tensor
        // Tensor operator +(const Tensor& other) const;
        // Tensor operator -(const Tensor& other) const;

        // // matrix multiplication 
        // Tensor operator *(const Tensor& other) const;

        // // scalar multiplication
        // Tensor operator *(float scalar) const;

};


#endif