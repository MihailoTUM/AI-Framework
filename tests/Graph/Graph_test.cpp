#include "../../src/Graph/Graph.h"
#include "../../src/Tensor/Tensor.h"


int main() {
    Graph g;
    Tensor input(2, 3, 'G');
    Tensor weights_1(3, 3, 'G');
    Tensor bias_1(1, 3, 'G');

    Tensor output1 = input * weights_1;
    Tensor output = output1 + bias_1;
    output.print();
    g.add(output);
    g.traverse();

    return 0;
}