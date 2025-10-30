

class CPUTensor {
    private:
        float *matrix;
        int rows;
        int cols;
        CPUTensor* parent1;
        CPUTensor* parent2;

    public:
        // operations
        CPUTensor(int nRows, int nCols);
        CPUTensor operator+(const CPUTensor& input);
        CPUTensor operator*(const CPUTensor& input);
        CPUTensor operator*(float scalar);
        CPUTensor operator+(float scalar);

        // 
        void print();

        // getters
        int getRows() const;
        int getCols() const;
        float *getMatrix();
};