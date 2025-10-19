#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Tensor.hpp"
#include "Activation.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mikideeplib, m) {
    m.doc() = "Miki's deep learning framework";

    py::class_<Tensor<float>>(m, "TensorFloat")
        .def(py::init<int, int, std::string>())
        .def("print", &Tensor<float>::print);
}