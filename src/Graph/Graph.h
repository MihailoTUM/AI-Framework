#include "../Tensor/Tensor.h"

class Graph {
    private:
        mutable const Tensor *last;

    public:
        Graph();
        void add(const Tensor& input);
        void traverse();
};

    void recursiveTraverse(const Tensor* pointer, int& count);


