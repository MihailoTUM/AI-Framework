#include "../../../src/GPU/GPUTensor.h"
#include "../../../src/GPU/Layer/Layer.h"

int main() {
    GPUTensor input(3, 8);
    input.print();

    Layer l(8, 2, "RELU");

    GPUTensor output = l.forward(input);
    output.copyToHost();
    output.print();

    return 0;
}