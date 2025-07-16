#include <iostream>

__global__ void VectorAddition(float* a, float*b, float* c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main() {
    float* a;
    float* b;
    float* c;

    VectorAddition<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}