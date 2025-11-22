#  SPDX-License-Identifier: LGPL-3.0-only
#
#  Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
#  Creation Date: 2024-Jan-28
#
#  Description: QuantumLiquids/tensor project. CMake file to control unittest.
#

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

    find_package(CUTENSOR REQUIRED)
    # Find all .cc files in tests/ directory
    file(GLOB_RECURSE TEST_SOURCES */*.cc)
    message(STATUS "TEST_SOURCES" ${TEST_SOURCES})
    # Treat all .cc files in tests/ as CUDA files
    set_source_files_properties(${TEST_SOURCES} PROPERTIES LANGUAGE CUDA)
else ()
    if(QLTEN_COMPILE_HPTT_LIB)
        set(hptt_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/external/hptt/include")
        set(hptt_LIBRARY "${CMAKE_BINARY_DIR}/external/hptt/libhptt.a")
    else()
        find_package(hptt REQUIRED)
    endif ()
endif ()


include(../cmake/Modules/MathBackend.cmake)
FIND_PACKAGE(BLAS REQUIRED)
FIND_PACKAGE(LAPACK REQUIRED)

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


option(TEST_ACTUAL_COMBAT "test the actual examples of mps tensors" OFF)

if (TEST_ACTUAL_COMBAT)
    add_definitions(-DACTUALCOMBAT)
endif ()

option(TEST_LARGE_MATRIX "test large matrixs(only for omp parallel svd upto now)." OFF)
if (TEST_LARGE_MATRIX)
    add_definitions(-DTEST_LARGE_MATRIX)
endif ()


if (QLTEN_USE_GPU)
    macro(add_unittest
            TEST_NAME TEST_SRC MATH_INCLUDE_DIR LINK_LIBS LINK_LIB_FLAGS)
        add_executable(${TEST_NAME}
                ${TEST_SRC})

        target_include_directories(${TEST_NAME}
                PRIVATE ${MATH_INCLUDE_DIR}
                PRIVATE ${QLTEN_HEADER_PATH}
                PRIVATE ${CUTENSOR_INCLUDE_DIR}
                PRIVATE ${MPI_CXX_HEADER_DIR}
        )
        target_link_libraries(${TEST_NAME}
                PRIVATE CUDA::cublas_static
                PRIVATE CUDA::cusolver_static
                PRIVATE ${CUTENSOR_LIBRARY}
                PRIVATE GTest::GTest GTest::Main
                "${LINK_LIBS}"
                PRIVATE "${LINK_LIB_FLAGS}"
        ) #remove MPI dependence since GPU code not support MPI upto now

        add_test(
                NAME ${TEST_NAME}
                COMMAND "${TEST_NAME}")

        set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests)
    endmacro()
else ()
    macro(add_unittest
            TEST_NAME TEST_SRC MATH_INCLUDE_DIR LINK_LIBS LINK_LIB_FLAGS)
        add_executable(${TEST_NAME}
                ${TEST_SRC})

        target_include_directories(${TEST_NAME}
                PRIVATE ${MATH_INCLUDE_DIR}
                PRIVATE ${QLTEN_HEADER_PATH}
                PRIVATE ${hptt_INCLUDE_DIR}
                PRIVATE ${MPI_CXX_HEADER_DIR})
        target_link_libraries(${TEST_NAME}
                PRIVATE ${hptt_LIBRARY}
                PRIVATE MPI::MPI_CXX
                PRIVATE GTest::GTest GTest::Main
                "${LINK_LIBS}"
                PRIVATE "${LINK_LIB_FLAGS}"
        )

        add_test(
                NAME ${TEST_NAME}
                COMMAND "${TEST_NAME}")

        set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests)
    endmacro()
endif ()


macro(add_mpi_unittest
        TEST_NAME TEST_SRC MATH_INCLUDE_DIR LINK_LIBS LINK_LIB_FLAGS PROCESSOR_NUM)
    add_executable(${TEST_NAME}
            ${TEST_SRC})

    target_include_directories(${TEST_NAME}
            PRIVATE ${MATH_INCLUDE_DIR}
            PRIVATE ${QLTEN_HEADER_PATH}
            PRIVATE ${hptt_INCLUDE_DIR}
            PRIVATE ${MPI_CXX_HEADER_DIR})
    target_link_libraries(${TEST_NAME}
            PRIVATE GTest::GTest GTest::Main
            PRIVATE ${hptt_LIBRARY}
            PRIVATE MPI::MPI_CXX
            "${LINK_LIBS}"
            PRIVATE "${LINK_LIB_FLAGS}"
    )

    if (NOT (MPIEXEC_MAX_NUMPROCS MATCHES "1"))
        add_test(
                NAME ${TEST_NAME}
                COMMAND ${MPIEXEC_EXECUTABLE}
                ${MPIEXEC_NUMPROC_FLAG} ${PROCESSOR_NUM}
                $<TARGET_FILE:${TEST_NAME}>)
    endif ()
    set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests)
endmacro()

add_unittest(test_fu1qn "test_qltensor/test_special_qn/test_fu1qn.cc" "" "" "")
add_unittest(test_fz2u1qn "test_qltensor/test_special_qn/test_fz2u1qn.cc" "" "" "")


add_unittest(test_fermion_parity_exchange "test_utility/test_fermion_parity_exchange.cc" "" "" "")

## Test quantum number related objects
# Test quantum number vaules
add_unittest(test_qnval_u1 "test_qltensor/test_qnval/test_qnval_u1.cc" "" "" "")
# Test quantum number
add_unittest(test_qn "test_qltensor/test_qn.cc" "" "" "")

## Test quantum number sector object.
add_unittest(test_qnsct "test_qltensor/test_qnsct.cc" "" "" "")

## Test index object.
add_unittest(test_index "test_qltensor/test_index.cc" "" "" "")

## Test symmetry-blocked sparse tensor, QLTensor.
add_unittest(test_qltensor
        "test_qltensor/test_qltensor.cc"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
)

add_unittest(test_qltensor_float
        "test_qltensor/test_qltensor_float.cc"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
)

add_unittest(test_fermion_qltensor
        "test_qltensor/test_fermion_qltensor.cc"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
)

## Test tensor manipulation functions.
# Test basic tensor operations.
add_unittest(test_basic_operations
        "test_tensor_manipulation/test_basic_operations.cc"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
)
# Test tensor linear combination.
add_unittest(test_ten_linear_combine
        "test_tensor_manipulation/test_ten_linear_combine.cc"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
)
# Test tensor contraction.
add_unittest(test_ten_ctrct
        "test_tensor_manipulation/test_ten_ctrct.cc"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
)
add_unittest(test_fermion_ten_ctrct
        "test_tensor_manipulation/test_fermion_ten_ctrct.cc"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
)
#Test tensor restriction contract
add_unittest(test_ten_ctrct_1sct
        "test_tensor_manipulation/test_ten_ctrct_1sct.cc"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
)
# Test tensor SVD.
add_unittest(test_ten_svd
        "test_tensor_manipulation/test_ten_svd.cc"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
)
# EVD
add_unittest(test_sym_mat_evd
        "test_tensor_manipulation/test_sym_mat_evd.cc"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
)
# Test tensor QR.
add_unittest(test_ten_qr
        "test_tensor_manipulation/test_ten_qr.cc"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
)
# Test tensor index combination.
add_unittest(test_index_combine
        "test_tensor_manipulation/test_index_combine.cc"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
)
# Test tensor expansion
add_unittest(test_ten_expand
        "test_tensor_manipulation/test_ten_expand.cc"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
)

# Test tensor fuse index
add_unittest(test_ten_fuse_index
        "test_tensor_manipulation/test_ten_fuse_index.cc"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
)

# Test tensor block expansion
add_unittest(test_ten_blk_expand
        "test_tensor_manipulation/test_ten_block_expand.cc"
        "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}"
)
if (NOT QLTEN_USE_GPU)
    # Test tensor communication for mpi
    add_mpi_unittest(test_ten_mpi_basic
            "test_tensor_mpi/test_ten_mpi_basic.cc"
            "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}" "2"
    )


    # Test mpi svd performance
    add_mpi_unittest(test_mpi_svd
            "test_tensor_mpi/test_mpi_svd.cc"
            "${BLAS_INCLUDE_DIRS}" "" "${MATH_LIB_LINK_FLAGS}" "3"
    )
endif ()
