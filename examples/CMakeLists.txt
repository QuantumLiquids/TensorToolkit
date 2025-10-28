#  SPDX-License-Identifier: LGPL-3.0-only
#
#  Examples for TensorToolkit

cmake_minimum_required(VERSION 3.12)

if (QLTEN_USE_GPU)
    message(FATAL_ERROR "Examples are CPU-only for now. Build without QLTEN_USE_GPU.")
endif ()

# Set the header path if not already set
if(NOT DEFINED QLTEN_HEADER_PATH)
    set(QLTEN_HEADER_PATH "${PROJECT_SOURCE_DIR}/include")
endif()

if(QLTEN_COMPILE_HPTT_LIB)
    set(hptt_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/external/hptt/include")
    set(hptt_LIBRARY "${CMAKE_BINARY_DIR}/external/hptt/libhptt.a")
else()
    find_package(hptt REQUIRED)
endif ()


include(../cmake/Modules/MathBackend.cmake)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

find_package(MPI REQUIRED)

# Intel MKL multi-thread may conflict with other compiler's OpenMP
# Close OpenMP if using MKL with other compiler
# If one must use GCC OpenMP + MKL, one possible solution is to manually static link MKL
# Relevant discussion : https://stackoverflow.com/questions/54699309/multithreaded-mkl-openmp-compiled-with-gcc
if (NOT BLA_VENDOR STREQUAL "Intel10_64lp" OR CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    find_package(OpenMP REQUIRED)
    link_libraries(OpenMP::OpenMP_CXX)
endif ()

set(MATH_LIB_LINK_FLAGS "BLAS::BLAS;LAPACK::LAPACK")

add_executable(z2_ising_vertex z2_ising_tensor.cpp)
target_include_directories(z2_ising_vertex
    PRIVATE ${QLTEN_HEADER_PATH}
    PRIVATE ${hptt_INCLUDE_DIR}
    PRIVATE ${MPI_CXX_HEADER_DIR}
    PRIVATE ${BLAS_INCLUDE_DIRS}
)
target_link_libraries(z2_ising_vertex
    PRIVATE ${hptt_LIBRARY}
    PRIVATE MPI::MPI_CXX
    ${MATH_LIB_LINK_FLAGS}
)

add_executable(z2_ising_trg z2_ising_trg.cpp)
target_include_directories(z2_ising_trg
    PRIVATE ${QLTEN_HEADER_PATH}
    PRIVATE ${hptt_INCLUDE_DIR}
    PRIVATE ${MPI_CXX_HEADER_DIR}
    PRIVATE ${BLAS_INCLUDE_DIRS}
)
target_link_libraries(z2_ising_trg
    PRIVATE ${hptt_LIBRARY}
    PRIVATE MPI::MPI_CXX
    ${MATH_LIB_LINK_FLAGS}
)


