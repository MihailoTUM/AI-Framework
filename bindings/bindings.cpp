#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../src/Tensor/Tensor.h"
#include "../src/Activation/Activation.h"
#include "../src/DenseLayer/DenseLayer.h"

namespace py = pybind11;

PYBIND11_MODULE(TUM, m) {
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<int, int, char, bool>(), py::arg("rows"), py::arg("cols"), py::arg("device")='C', py::arg("random")=true)
        .def("print", &Tensor::print)
        .def("getRows", &Tensor::getRows)
        .def("getCols", &Tensor::getCols)
        .def("getValue", &Tensor::getValue)
        .def("setValue", &Tensor:setValue)
        .def("__add__", &Tensor::operator+)
        .def("__matmul__", py::overload_cast<const Tensor&>(&Tensor::operator*, py::const_))
        .def("__mul__", py::overload_cast<float>(&Tensor::operator*, py::const_));
}