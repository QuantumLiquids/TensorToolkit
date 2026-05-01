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
        VERSION 0.1.1
        DESCRIPTION "a high-performance tensor basic operation library written in C++"
        LANGUAGES CXX
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# options.
option(QLTEN_USE_GPU "Use GPU" OFF)
option(QLTEN_COMPILE_HPTT_LIB "Compile hptt library; Invalid if Use GPU" ON)
option(QLTEN_BUILD_UNITTEST "Build unittests for TensorKit." OFF)
option(QLTEN_BUILD_EXAMPLES "Build examples." OFF)

option(QLTEN_TIMING_MODE "Enable timing instrumentation for in-tree targets." OFF)
option(QLTEN_MPI_TIMING_MODE "Enable MPI timing instrumentation for in-tree targets." OFF)


list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules")

function(qlten_prepare_mpi_discovery)
    if (NOT MPI_CXX_COMPILER)
        find_program(_qlten_mpi_cxx_wrapper NAMES mpicxx mpic++)
        if (_qlten_mpi_cxx_wrapper)
            set(MPI_CXX_COMPILER "${_qlten_mpi_cxx_wrapper}" CACHE FILEPATH
                    "MPI C++ wrapper compiler" FORCE)
        endif ()
    endif ()

    if (NOT MPI_CXX_HEADER_DIR)
        set(_qlten_mpi_probe "${MPI_CXX_COMPILER}")
        if (NOT _qlten_mpi_probe)
            find_program(_qlten_mpi_probe NAMES mpicxx mpic++)
        endif ()

        if (_qlten_mpi_probe)
            execute_process(
                    COMMAND "${_qlten_mpi_probe}" --showme:compile
                    OUTPUT_VARIABLE _qlten_mpi_compile_flags
                    RESULT_VARIABLE _qlten_mpi_probe_result
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET
            )
            if (_qlten_mpi_probe_result EQUAL 0
                    AND _qlten_mpi_compile_flags MATCHES "(^| )-I([^ ]+)")
                set(MPI_CXX_HEADER_DIR "${CMAKE_MATCH_2}" CACHE PATH
                        "MPI header directory" FORCE)
            endif ()
        endif ()
    endif ()
endfunction()

# Build type.
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

if (${CMAKE_BUILD_TYPE} MATCHES Debug)
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif ()

if (QLTEN_USE_GPU)
    set(QLTEN_COMPILE_HPTT_LIB OFF)
endif ()

set(QLTEN_HEADER_PATH "${PROJECT_SOURCE_DIR}/include")
set(QLTEN_HEADER_DIR ${QLTEN_HEADER_PATH}/qlten)
set(QLTEN_PACKAGE_CONFIG_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/TensorToolkit")
set(QLTEN_BUNDLES_HPTT OFF)
set(QLTEN_NEEDS_OPENMP OFF)
set(QLTEN_NVHPC_PGMATH_LIBRARY "")

add_library(TensorToolkit INTERFACE)
add_library(TensorToolkit::TensorToolkit ALIAS TensorToolkit)
target_compile_features(TensorToolkit INTERFACE cxx_std_17)
target_include_directories(TensorToolkit
        INTERFACE
        $<BUILD_INTERFACE:${QLTEN_HEADER_PATH}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_definitions(TensorToolkit
        INTERFACE
        $<$<BOOL:${QLTEN_USE_GPU}>:USE_GPU=1>
)

add_library(qlten_build_options INTERFACE)
target_compile_definitions(qlten_build_options
        INTERFACE
        $<$<BOOL:${QLTEN_TIMING_MODE}>:QLTEN_TIMING_MODE>
        $<$<BOOL:${QLTEN_MPI_TIMING_MODE}>:QLTEN_MPI_TIMING_MODE>
)

qlten_prepare_mpi_discovery()
find_package(MPI REQUIRED)
target_link_libraries(TensorToolkit
        INTERFACE
        MPI::MPI_CXX
)

include(cmake/Modules/MathBackend.cmake)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
qlten_resolve_lapacke()

if (NOT HP_NUMERIC_USE_MKL AND NOT QLTEN_HP_NUMERIC_LAPACKE_INCLUDE_DIR)
    message(FATAL_ERROR "Unable to locate lapacke.h for the selected HP numeric backend.")
endif ()
if (HP_NUMERIC_USE_OPENBLAS AND NOT QLTEN_HP_NUMERIC_LAPACKE_LIBRARY)
    message(FATAL_ERROR "OpenBLAS builds require a LAPACKE library, but no lapacke library was found.")
endif ()

if (NOT TARGET BLAS::BLAS)
    add_library(BLAS::BLAS INTERFACE IMPORTED)
    set_target_properties(BLAS::BLAS PROPERTIES
            INTERFACE_LINK_LIBRARIES "${BLAS_LIBRARIES}")
endif ()

if (NOT TARGET LAPACK::LAPACK)
    add_library(LAPACK::LAPACK INTERFACE IMPORTED)
    set_target_properties(LAPACK::LAPACK PROPERTIES
            INTERFACE_LINK_LIBRARIES "${LAPACK_LIBRARIES}")
endif ()

if (HP_NUMERIC_USE_MKL)
    set(_qlten_mkl_threaded_runtime OFF)
    foreach(_qlten_blas_lib IN LISTS BLAS_LIBRARIES LAPACK_LIBRARIES)
        if (_qlten_blas_lib MATCHES "mkl_(intel|gnu)_thread")
            set(_qlten_mkl_threaded_runtime ON)
            break()
        endif ()
    endforeach ()
    if (_qlten_mkl_threaded_runtime)
        set(QLTEN_HP_NUMERIC_REQUIRES_OPENMP ON)
    endif ()
endif ()

set(QLTEN_NEEDS_OPENMP ${QLTEN_HP_NUMERIC_REQUIRES_OPENMP})
if (NOT QLTEN_USE_GPU AND QLTEN_COMPILE_HPTT_LIB)
    set(QLTEN_NEEDS_OPENMP ON)
endif ()
if (QLTEN_NEEDS_OPENMP)
    find_package(OpenMP REQUIRED)
endif ()

target_compile_definitions(TensorToolkit
        INTERFACE
        ${QLTEN_HP_NUMERIC_BACKEND_DEFINE}
)

if (QLTEN_HP_NUMERIC_PUBLIC_INCLUDE_DIRS)
    foreach(_qlten_hp_numeric_include_dir IN LISTS QLTEN_HP_NUMERIC_PUBLIC_INCLUDE_DIRS)
        target_include_directories(TensorToolkit
                INTERFACE
                $<BUILD_INTERFACE:${_qlten_hp_numeric_include_dir}>
        )
    endforeach ()
endif ()

target_link_libraries(TensorToolkit
        INTERFACE
        BLAS::BLAS
        LAPACK::LAPACK
)

if (QLTEN_HP_NUMERIC_LAPACKE_LIBRARY)
    if (NOT TARGET qlten_build_lapacke)
        add_library(qlten_build_lapacke UNKNOWN IMPORTED)
        set_target_properties(qlten_build_lapacke PROPERTIES
                IMPORTED_LOCATION "${QLTEN_HP_NUMERIC_LAPACKE_LIBRARY}")
    endif ()

    target_link_libraries(TensorToolkit
            INTERFACE
            $<BUILD_INTERFACE:qlten_build_lapacke>
    )
endif ()

if (QLTEN_NEEDS_OPENMP)
    target_link_libraries(TensorToolkit
            INTERFACE
            OpenMP::OpenMP_CXX
    )
endif ()

if (QLTEN_USE_GPU)
    find_package(CUDAToolkit REQUIRED)
    find_package(CUTENSOR REQUIRED)

    target_link_libraries(TensorToolkit
            INTERFACE
            CUDA::cudart
            CUDA::cublas
            CUDA::cusolver
            CUTENSOR::CUTENSOR
    )
else ()
    if (QLTEN_COMPILE_HPTT_LIB)
        add_subdirectory(external/hptt)
        set(QLTEN_BUNDLES_HPTT ON)

        if (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
            get_filename_component(_qlten_nvhpc_bin_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
            get_filename_component(_qlten_nvhpc_lib_dir "${_qlten_nvhpc_bin_dir}/../lib" ABSOLUTE)

            find_file(_qlten_pgmath_static
                    NAMES libpgmath.a
                    HINTS "${_qlten_nvhpc_lib_dir}"
                    PATH_SUFFIXES "" compilers/lib)
            if (_qlten_pgmath_static)
                set(QLTEN_NVHPC_PGMATH_LIBRARY "${_qlten_pgmath_static}")
            else ()
                find_library(QLTEN_NVHPC_PGMATH_LIBRARY
                        NAMES pgmath
                        HINTS
                        "${_qlten_nvhpc_lib_dir}"
                        "$ENV{NVCOMPILERS}"
                        "$ENV{NVHPCSDK}"
                        "/opt/nvidia/hpc_sdk"
                        PATH_SUFFIXES compilers/lib)
            endif ()

            if (NOT QLTEN_NVHPC_PGMATH_LIBRARY)
                message(FATAL_ERROR
                        "TensorToolkit requires the NVHPC pgmath runtime to export the bundled HPTT archive.")
            endif ()
        endif ()

        target_include_directories(TensorToolkit
                INTERFACE
                $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/hptt/include>
        )
        target_link_libraries(TensorToolkit
                INTERFACE
                hptt
        )
    else ()
        find_package(hptt REQUIRED)

        set(_qlten_hptt_include_dirs "${hptt_INCLUDE_DIRS}")
        if (NOT _qlten_hptt_include_dirs)
            set(_qlten_hptt_include_dirs "${hptt_INCLUDE_DIR}")
        endif ()

        if (NOT TARGET qlten_build_hptt)
            add_library(qlten_build_hptt UNKNOWN IMPORTED)
            set_target_properties(qlten_build_hptt PROPERTIES
                    IMPORTED_LOCATION "${hptt_LIBRARY}"
                    INTERFACE_INCLUDE_DIRECTORIES "${_qlten_hptt_include_dirs}")
        endif ()

        target_link_libraries(TensorToolkit
                INTERFACE
                $<BUILD_INTERFACE:qlten_build_hptt>
        )
    endif ()
endif ()

install(DIRECTORY ${QLTEN_HEADER_DIR}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

configure_package_config_file(
        cmake/TensorToolkitConfig.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/TensorToolkitConfig.cmake"
        INSTALL_DESTINATION "${QLTEN_PACKAGE_CONFIG_DIR}"
        PATH_VARS CMAKE_INSTALL_INCLUDEDIR
)

configure_file(
        cmake/TensorToolkitDependencies.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/TensorToolkitDependencies.cmake"
        @ONLY
)

write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/TensorToolkitConfigVersion.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY SameMajorVersion
)

install(TARGETS TensorToolkit
        EXPORT TensorToolkitTargets
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT TensorToolkitTargets
        FILE TensorToolkitTargets.cmake
        NAMESPACE TensorToolkit::
        DESTINATION "${QLTEN_PACKAGE_CONFIG_DIR}")

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/TensorToolkitConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/TensorToolkitConfigVersion.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/TensorToolkitDependencies.cmake"
        DESTINATION "${QLTEN_PACKAGE_CONFIG_DIR}")

install(FILES
        cmake/Modules/MathBackend.cmake
        cmake/Modules/FindCUTENSOR.cmake
        cmake/Modules/Findhptt.cmake
        DESTINATION "${QLTEN_PACKAGE_CONFIG_DIR}/modules")

set(QLTEN_VERIFY_MPI_CXX_WRAPPER "")
if (DEFINED MPI_CXX_COMPILER AND NOT MPI_CXX_COMPILER STREQUAL "")
    set(QLTEN_VERIFY_MPI_CXX_WRAPPER "${MPI_CXX_COMPILER}")
else ()
    find_program(QLTEN_VERIFY_MPI_CXX_WRAPPER NAMES mpicxx mpic++)
endif ()
set(QLTEN_VERIFY_CONSUMER_CXX_COMPILER "${CMAKE_CXX_COMPILER}")
set(QLTEN_VERIFY_CPU_BACKEND_ARG "")
set(QLTEN_VERIFY_HPTT_INCLUDE_DIR "")
set(QLTEN_VERIFY_HPTT_LIBRARY "")
if (HP_NUMERIC_USE_MKL)
    set(QLTEN_VERIFY_CPU_BACKEND_ARG "-DHP_NUMERIC_USE_MKL=ON")
elseif (HP_NUMERIC_USE_AOCL)
    set(QLTEN_VERIFY_CPU_BACKEND_ARG "-DHP_NUMERIC_USE_AOCL=ON")
elseif (HP_NUMERIC_USE_OPENBLAS)
    set(QLTEN_VERIFY_CPU_BACKEND_ARG "-DHP_NUMERIC_USE_OPENBLAS=ON")
endif ()

if (NOT QLTEN_USE_GPU AND NOT QLTEN_COMPILE_HPTT_LIB)
    if (DEFINED hptt_INCLUDE_DIR AND NOT hptt_INCLUDE_DIR STREQUAL "")
        set(QLTEN_VERIFY_HPTT_INCLUDE_DIR "${hptt_INCLUDE_DIR}")
    elseif (DEFINED hptt_INCLUDE_DIRS AND NOT hptt_INCLUDE_DIRS STREQUAL "")
        list(GET hptt_INCLUDE_DIRS 0 QLTEN_VERIFY_HPTT_INCLUDE_DIR)
    endif ()

    if (DEFINED hptt_LIBRARY AND NOT hptt_LIBRARY STREQUAL "")
        set(QLTEN_VERIFY_HPTT_LIBRARY "${hptt_LIBRARY}")
    endif ()
endif ()

add_custom_target(verify-package-cpu
        COMMAND ${CMAKE_COMMAND}
        -DQLTEN_VERIFY_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DQLTEN_VERIFY_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
        -DQLTEN_VERIFY_CPU_BACKEND_ARG:STRING=${QLTEN_VERIFY_CPU_BACKEND_ARG}
        -DQLTEN_VERIFY_COMPILE_HPTT_LIB:BOOL=${QLTEN_COMPILE_HPTT_LIB}
        -DQLTEN_VERIFY_HPTT_INCLUDE_DIR:PATH=${QLTEN_VERIFY_HPTT_INCLUDE_DIR}
        -DQLTEN_VERIFY_HPTT_LIBRARY:FILEPATH=${QLTEN_VERIFY_HPTT_LIBRARY}
        -DQLTEN_VERIFY_TIMING_MODE:BOOL=ON
        -DQLTEN_VERIFY_MPI_TIMING_MODE:BOOL=ON
        -DQLTEN_VERIFY_MPI_CXX_COMPILER:FILEPATH=${QLTEN_VERIFY_MPI_CXX_WRAPPER}
        -DQLTEN_VERIFY_CONSUMER_CXX_COMPILER:FILEPATH=${QLTEN_VERIFY_CONSUMER_CXX_COMPILER}
        -P ${PROJECT_SOURCE_DIR}/tests/package_consumer/verify_install_cpu.cmake
        USES_TERMINAL
)

add_custom_target(verify-package-gpu
        COMMAND ${CMAKE_COMMAND}
        -DQLTEN_VERIFY_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DQLTEN_VERIFY_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
        -DQLTEN_VERIFY_CPU_BACKEND_ARG:STRING=${QLTEN_VERIFY_CPU_BACKEND_ARG}
        -DQLTEN_VERIFY_CUDA_COMPILER:FILEPATH=${CMAKE_CUDA_COMPILER}
        -DQLTEN_VERIFY_TIMING_MODE:BOOL=ON
        -DQLTEN_VERIFY_MPI_TIMING_MODE:BOOL=ON
        -DQLTEN_VERIFY_CUDA_ARCHITECTURES:STRING=${QLTEN_CUDA_ARCHITECTURES}
        -DQLTEN_VERIFY_CUTENSOR_INCLUDE_DIR:PATH=${CUTENSOR_INCLUDE_DIR}
        -DQLTEN_VERIFY_CUTENSOR_LIBRARY:FILEPATH=${CUTENSOR_LIBRARY}
        -DQLTEN_VERIFY_MPI_CXX_COMPILER:FILEPATH=${QLTEN_VERIFY_MPI_CXX_WRAPPER}
        -DQLTEN_VERIFY_CONSUMER_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
        -P ${PROJECT_SOURCE_DIR}/tests/package_consumer/verify_install_gpu.cmake
        USES_TERMINAL
)

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
