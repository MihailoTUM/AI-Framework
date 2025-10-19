#include <iostream>
#include <random>

class Tensor {
    private:
        float* matrix;
        int rows;
        int cols;

    public:
    Tensor(int nRows, int nCols) {
        this->rows = nRows;
        this->cols = nCols;
        this->matrix = new float[this->rows * this->cols];
        init();
    };

    void init() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> distFloat(-1.0f, 1.0f);

        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                matrix[i * cols + j] = distFloat(gen);
            }
        }
    }

    void print() {
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                std::cout << matrix[i * cols + j] << " ";
            }
            std::cout << "\n";
        }
    }
};

int main() {

    Tensor t(3, 3);
    t.print();

    return 0;
}