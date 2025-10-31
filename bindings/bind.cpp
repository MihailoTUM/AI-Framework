#include <pybind11/pybind11.h>
#include "../src/GPU/GPUTensor.h"
#include "../src/GPU/Activation/Activation.h"
#include <string>

namespace py = pybind11;

PYBIND11_MODULE(AIKI, m) {
    m.doc() = "AIKI = (AI + MIKI) Framework for AI";

    py::class_<GPUTensor>(m, "GPUTensor")
        .def(py::init<int, int>(), py::arg("rows"), py::arg("cols"))
        .def(py::init<const GPUTensor &>(), py::arg("input"))

        .def("init", &GPUTensor::init)
        .def("countGPU", &GPUTensor::countGPU)
        .def("uploadToGPU", &GPUTensor::uploadToGPU)
        .def("copyToHost", &GPUTensor::copyToHost)
        .def("toZeros", &GPUTensor::toZeros)
        .def("toOnes", &GPUTensor::toOnes)
        .def("traverse", &GPUTensor::traverse)
        .def("__add__", [](const GPUTensor &self, const GPUTensor &other) { return self + other; })
        .def("__add__", [](const GPUTensor &self, float scalar) { return self + scalar; })
        .def("__radd__", [](const GPUTensor &self, float scalar) { return scalar + self; })
        .def("__matmul__", [](const GPUTensor &self, const GPUTensor &other) { return self * other;})
        .def("__mul__", [](const GPUTensor &self, float scalar) { return self * scalar; })
        .def("__rmul__", [](const GPUTensor &self, float scalar) { return scalar * self; })
        .def("__neg__", &GPUTensor::operator-)

        .def("sum", &GPUTensor::sum, py::arg("axis") = 0)
        .def("mean", &GPUTensor::mean, py::arg("axis") = 0)
        .def("transpose", &GPUTensor::transpose)

        .def("print", &GPUTensor::print)
        .def("printGPUSpecs", &GPUTensor::printGPUSpecs);

    
    py::class_<Activation>(m, "Activation")
        .def(py::init<std::string>(), py::arg("_func"))
        .def("forward", &Activation::forward, py::arg("input"))
        .def("relu", &Activation::relu, py::arg("input"));
}