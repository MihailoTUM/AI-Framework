#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "Tensor.hpp"
#include "Activation.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mikideeplib, m) {
    m.doc() = "Miki's deep learning framework";

    py::class_<Tensor<float>>(m, "TensorFloat")
        .def(py::init<int, int, std::string>())
        .def(py::init<int, int, std::string, py::array_t<float>>())
        .def("print", &Tensor<float>::print)
        .def("getRows", &Tensor<float>::getRows)
        .def("getCols", &Tensor<float>::getCols)
        .def("fill", (void (Tensor<float>::*)(int, int, float)) &Tensor<float>::fill)
        .def("get", &Tensor<float>::get)
        .def("__getitem__", [](const Tensor<float>& t, std::pair<int, int> idx) {
            return t.get(idx.first, idx.second);
        })
        .def("__add__", (Tensor<float> (Tensor<float>::*)(const Tensor<float>&) const) &Tensor<float>::operator+)
        .def("__sub__", (Tensor<float> (Tensor<float>::*)(const Tensor<float>&) const) &Tensor<float>::operator-)
        .def("__mul__", py::overload_cast<float>(&Tensor<float>::operator*, py::const_))
        .def("__matmul__", py::overload_cast<const Tensor<float>&>(&Tensor<float>::operator*, py::const_))
        .def("__neg__", (Tensor<float>(Tensor<float>::*)() const)&Tensor<float>::operator-)
        .def("sum", &Tensor<float>::sum)
        .def("mean", &Tensor<float>::mean)
        .def("transpose", &Tensor<float>::transpose)
        .def("getMatrix", [](const Tensor<float>& t) {
            return pybind11::array_t<float>(
                {t.getRows(), t.getCols()},
                {sizeof(float) * t.getCols(), sizeof(float)},
                t.getMatrix()
            );
        });

    py::class_<Activation<float>>(m, "Activation")
        .def(py::init<std::string>())
        .def("forward", &Activation<float>::forward)
        .def("relu", (Tensor<float> (Activation<float>::*)(const Tensor<float>&) const) &Activation<float>::relu);
}