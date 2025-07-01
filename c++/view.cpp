#include <iostream>

int main() {

    int rows = 5;
    int cols = 4;

    float* matrix = new float[rows * cols];

    struct MatrixView {
        float* data;
        int rows;
        int cols;
        int row_stride;
        int col_stride;


        float& operator()(int i, int j) {
            return data[i*row_stride + j*col_stride];
        }
    };

    MatrixView matrixview = { matrix, rows, cols, cols, 1 };
    std::cout << matrixview(0, 0) << std::endl;


    return 0;
}