#  SPDX-License-Identifier: LGPL-3.0-only
#
#  Examples for TensorToolkit

cmake_minimum_required(VERSION 3.12)

if (QLTEN_USE_GPU)
    message(FATAL_ERROR "Examples are CPU-only for now. Build without QLTEN_USE_GPU.")
endif ()

add_executable(z2_ising_vertex z2_ising_tensor.cpp)
target_link_libraries(z2_ising_vertex
    PRIVATE TensorToolkit::TensorToolkit
)

add_executable(z2_ising_trg z2_ising_trg.cpp)
target_link_libraries(z2_ising_trg
    PRIVATE TensorToolkit::TensorToolkit
)
