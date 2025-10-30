#include "./Graph.h"
#include "../Tensor/Tensor.h"
#include <iostream>

Graph::Graph() {
    last = nullptr;
};

void Graph::add(const Tensor& input) {
    if(!last) {
        last = &input;
    }
};

void Graph::traverse() {
    if(!last) return;
    const Tensor*pointer = last;
    int count = 0;

    recursiveTraverse(pointer, count);

    std::cout << "#Tensors: " << count << "\n" << std::endl;
};

void recursiveTraverse(const Tensor* pointer, int& count) {
    if (pointer->getParent1())
        recursiveTraverse(pointer->getParent1(), count);
    if (pointer->getParent2())
        recursiveTraverse(pointer->getParent2(), count);
    count++;
};