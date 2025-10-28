cmake_minimum_required(VERSION 3.20)
project(MyCUDAProject LANGUAGES CXX CUDA)

# Pybind11
find_package(pybind11 REQUIRED)

# CUDA architecture (adjust to your GPU)
set(CMAKE_CUDA_ARCHITECTURES 75)  # 75 for Turing, 86 for Ampere, etc.

# Your source files
set(SOURCES
    bindings/bindings.cpp
    src/Tensor/Tensor.cu
)

# Create Python module
pybind11_add_module(my_module ${SOURCES})
set_target_properties(my_module PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
)
