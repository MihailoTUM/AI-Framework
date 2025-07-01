#include <iostream>
#include <string>
#include <vector>
#include <random>

using namespace std;

float randomNumber() {
    std::default_random_engine generator(std::random_device{}());
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    return distribution(generator);
}

float* createVector(int size) {
    float* arr = new float[size];
    return arr;
};

float* randomVector(int size) {
    float* vector = createVector(size);

    for (int i = 0; i < size; i++) {
        vector[i] = randomNumber();
    }

    return vector;
}

float dotProduct(float* vector1, float* vector2, int size) {
    float output = 0.0;

    for(int i = 0; i < size; i++) {
        output = output + vector1[i] * vector2[i];
    }

    return output;
};

void printVector(float* vector, int size) {
    for (int i = 0; i < size; i++) {
        cout << vector[i] << endl;
    }
    cout << "" << endl;
}

float* createMatrix(int rows, int cols) {
    float* matrix = new float[rows * cols];
    return matrix;
};

float* randomMatrix(int rows, int cols) {
    float* matrix = createMatrix(rows, cols);

    std::default_random_engine generator(std::random_device{}());
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            matrix[i * cols + j] = distribution(generator);
        }
    }

    return matrix;
}

void printMatrix(float* matrix, int rows, int cols) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            cout <<"  " << matrix[i * cols + j] << " ";
        }
        cout << " " << endl;
    }
}

float* rowOfMatrix() {
    return nullptr;
}

float* colOfMatrx(float* matrix, int rows, int cols, int col) {
    // rows and cols [n, m]
    // col is in the intervall [1, n] e.g. 2 == 2nd column

    float* column = createVector(rows);

    for(int i = 0; i < rows; i++) {
        column[i] = matrix[col + i * cols];
    }

    return column;
};

float* vectorMatrixMultiplication(float* vector, int dim, float* matrix, int rows, int cols) {
    if(dim != rows) {
        return nullptr;
    }

    float* outputVector = createVector(cols);

    for(int i = 0; i < cols; i++) {
        outputVector[i] = dotProduct(vector, colOfMatrx(matrix, rows, cols, i), rows);
    }

    return outputVector;
}

int main() {

    int rows = 5;
    int cols = 7;
    int dim = 5;

    float* matrix = randomMatrix(rows, cols);
    printMatrix(matrix, rows, cols);

    float* vector = randomVector(dim);
    printVector(vector, dim);

    float* output = vectorMatrixMultiplication(vector, dim, matrix, rows, cols);
    printVector(output, cols);

    return 0;
}
