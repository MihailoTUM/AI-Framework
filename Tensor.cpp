#include <iostream>

class Tensor {
    private:
    float* matrix;
    int rows;
    int cols;

    public:
    Tensor(int Mrows, int Mcols) {
        rows = Mrows;
        cols = Mcols;
        matrix = new float[rows * cols];
        if(!matrix) {
            std::cerr << "Memomry allocaiton failed" << std::endl;
        }
    }
    ~Tensor() {
        delete[] matrix;
    }

    void init() {
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                this->matrix[cols * i + j] = 0.0f;
            }
        }
    }

    void fill(int row, int col, float value) {
        this->matrix[row * cols + col] = value;
    }

    void print() {
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                std::cout << this->matrix[cols * i + j] << " "; 
            }
            std::cout << " " << std::endl;
        }
    }

    // Tensor operator+(const Tensor& other) const {
    //     if(other.rows == this->rows && other.cols == this->cols) {
    //         for(int i = 0; i < rows; i++) {
    //             for(int j = 0; j < cols; j++) {

    //             }
    //         }
    //     }
    // }
};

int main() {

    Tensor t(2, 3);
    t.init();
    t.print();

    return 0;
}