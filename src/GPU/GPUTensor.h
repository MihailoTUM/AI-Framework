
class GPUTensor {
    private:
        float *matrix;
        int rows; 
        int cols;
        GPUTensor *parent1;
        GPUTensor *parent2;

    public:
        GPUTensor();
        GPUTensor operator+(const GPUTensor& input);
        GPUTensor operator*(const GPUTensor& input);    
        GPUTensor operator*(float scalar);
        GPUTensor operator+(float scalar);
};