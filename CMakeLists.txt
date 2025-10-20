# SPDX-License-Identifier: LGPL-3.0-only
#
# Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
# Creation Date: 2019-04-24 16:28
#
# Description: QuantumLiquids/TensorToolkit project. Main CMake file.
#


# CMake basic setting
cmake_minimum_required(VERSION 3.27) # Required for modern FindBLAS support used with AOCL.

project("TensorToolkit"
        DESCRIPTION "a high-performance tensor basic operation library written in C++"
        LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)


# options.
option(QLTEN_USE_GPU "Use GPU" OFF)
option(QLTEN_COMPILE_HPTT_LIB "Compile hptt library; Invalid if Use GPU" ON)
option(QLTEN_BUILD_UNITTEST "Build unittests for TensorKit." OFF)
option(QLTEN_BUILD_EXAMPLES "Build examples." OFF)

option(QLTEN_TIMING_MODE "Enable Timing mode." ON)
option(QLTEN_MPI_TIMING_MODE "Enable MPI timing mode." ON)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/Modules/")

# Build type.
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

if (${CMAKE_BUILD_TYPE} MATCHES Debug)
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif ()

# Compilation and linking control.
add_definitions(-Wall -g)

if (QLTEN_USE_GPU)
    set(QLTEN_COMPILE_HPTT_LIB OFF)
endif ()


if (QLTEN_TIMING_MODE)
    add_definitions(-DQLTEN_TIMING_MODE)
endif ()


if (QLTEN_MPI_TIMING_MODE)
    add_definitions(-DQLTEN_MPI_TIMING_MODE)
endif ()

# Build dependencies.
if (QLTEN_COMPILE_HPTT_LIB)
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

# Examples
if (QLTEN_BUILD_EXAMPLES)
    add_subdirectory(examples examples)
endif ()
