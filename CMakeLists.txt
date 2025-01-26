# SPDX-License-Identifier: LGPL-3.0-only
#
# Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
# Creation Date: 2019-04-24 16:28
#
# Description: QuantumLiquids/TensorToolkit project. Main CMake file.
#


# CMake basic setting
cmake_minimum_required(VERSION 3.12)

project("TensorToolkit"
        DESCRIPTION "a high-performance tensor basic operation library written in C++"
        LANGUAGES CXX
)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/Modules/")

# Build type.
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

if (${CMAKE_BUILD_TYPE} MATCHES Debug)
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif ()

# Compilation and linking control.
set(CMAKE_CXX_STANDARD 17)
add_definitions(-Wall -g)

# Advanced options.
option(QLTEN_BUILD_UNITTEST "Build unittests for TensorKit." OFF)

option(QLTEN_TIMING_MODE "Timing mode." ON)
if (QLTEN_TIMING_MODE)
    add_definitions(-DQLTEN_TIMING_MODE)
endif ()

option(QLTEN_MPI_TIMING_MODE "Timing mode." ON)
if (QLTEN_MPI_TIMING_MODE)
    add_definitions(-DQLTEN_MPI_TIMING_MODE)
endif ()

option(QLTEN_USE_GPU "Use GPU" OFF)
if (QLTEN_USE_GPU)
    add_definitions(-DUSE_GPU=1)

    # Enable CUDA language support
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)  # Modern CUDA Toolkit detection

    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    message(STATUS "CUDA_INCLUDE_DIRS: '${CUDAToolkit_INCLUDE_DIRS}'")

    # Set CUDA architectures
    set(CMAKE_CUDA_ARCHITECTURES 70 80) # For V100 & A100, adjust if needed

    # Compiler flags
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo -Xcompiler=-fPIC")
else ()
    option(QLTEN_USE_EXTERNAL_HPTT_LIB "Use user pre-installed external hptt library" OFF)
endif ()

# Build dependencies.
if (NOT QLTEN_USE_EXTERNAL_HPTT_LIB)
    add_subdirectory(external/hptt)
endif ()

# Installation.
set(QLTEN_HEADER_PATH "${PROJECT_SOURCE_DIR}/include")
set(QLTEN_HEADER_DIR ${QLTEN_HEADER_PATH}/qlten)
set(QLTEN_INCLUDE_DEST "${CMAKE_INSTALL_PREFIX}/include")
install(DIRECTORY ${QLTEN_HEADER_DIR}
        DESTINATION ${QLTEN_INCLUDE_DEST})

# Unittests
if (QLTEN_BUILD_UNITTEST)
    enable_testing()
    find_package(GTest REQUIRED)
    add_subdirectory(tests tests)
endif ()
