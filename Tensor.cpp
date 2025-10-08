#include <iostream>
#include <random>

class Tensor {
    float* matrix;
    int rows;
    int cols;

    public:
    Tensor(int nRows, int nCols) {
        rows = nRows;
        cols = nCols;
        matrix = new float[rows * cols];
        init();
        random();
    }

    void init() {
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                matrix[cols * i + j] = 0;
            }
        }
    }

    void random() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> distFloat(-1.0f, 1.0f);

        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                fill(i, j, distFloat(gen));
            }
        }

    }

    void fill(int row, int col, float value) {
        matrix[row * cols + col] = value;
    }

    void print() {
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                std::cout << matrix[cols * i + j] << " ";
            }
            std::cout << "" << std::endl;
        }
    }

    Tensor sum(int axis = 0) {

    };

    Tensor mean(int axis = 0) {

    };

    Tensor operator -() const {
        Tensor g(rows, cols);
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                g.fill(i, j, -matrix[i * cols + j]);
            }
        }
        return g;
    }

    Tensor operator +(const Tensor& other) const {
        if(rows == other.rows && cols == other.cols) {
            Tensor g(rows, cols);
            for(int i = 0; i < rows; i++) {
                for(int j = 0; j < cols; j++) {
                    g.fill(i, j, matrix[i * cols + j] + other.matrix[i * cols + j]);
                }
            }
            return g;
        } else {
            return Tensor(0, 0);
        };
    };

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
            return Tensor(0, 0);
        }; 
    }

    Tensor operator *(float scalar) const {
        Tensor g(rows, cols);
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                g.fill(i, j, scalar * matrix[i * cols + j]);
            }
        }
        return g;
    }

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
            return Tensor(0, 0);
        }
    }

};

int main() {
    Tensor t1(3, 3);
    t1.print();

    std::cout << "\n";

    Tensor t2(3, 2);
    t2.print();

    std::cout << "\n";

    Tensor t3 = t1 * t2;
    t3.print();

    return 0;
}