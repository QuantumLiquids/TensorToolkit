#  SPDX-License-Identifier: LGPL-3.0-only
# 
#  Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
#  Creation Date: 2024-Jan-28
#  
#  Description: QuantumLiquids/tensor project. CMake file to control unittest.
#

if (QLTEN_USE_EXTERNAL_HPTT_LIB)
    find_package(hptt REQUIRED)
else ()
    set(hptt_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/external/hptt/include")
    set(hptt_LIBRARY "${CMAKE_BINARY_DIR}/external/hptt/libhptt.a")
endif ()

find_package(Boost
        1.74.0
        REQUIRED
        COMPONENTS serialization mpi)

find_package(MPI REQUIRED)

option(TEST_ACTUAL_COMBAT "test the actual examples of mps tensors" OFF)

if (TEST_ACTUAL_COMBAT)
    add_definitions(-DACTUALCOMBAT)
endif ()

option(TEST_LARGE_MATRIX "test large matrixs(only for omp parallel svd upto now)." OFF)
if (TEST_LARGE_MATRIX)
    add_definitions(-DTEST_LARGE_MATRIX)
endif ()

#set omp flag
if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(OMP_FLAGS -qopenmp)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(OMP_FLAGS -fopenmp)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(OMP_FLAGS -fopenmp)
endif ()

macro(add_unittest
        TEST_NAME TEST_SRC CFLAGS LINK_LIBS LINK_LIB_FLAGS)
    add_executable(${TEST_NAME}
            ${TEST_SRC})

    target_compile_options(${TEST_NAME}
            PRIVATE ${CFLAGS}
            PRIVATE ${OMP_FLAGS}
    )
    target_include_directories(${TEST_NAME}
            PRIVATE ${QLTEN_HEADER_PATH}
            PRIVATE ${hptt_INCLUDE_DIR}
            PRIVATE ${Boost_INCLUDE_DIR}
            PRIVATE ${MPI_CXX_HEADER_DIR})
    target_link_libraries(${TEST_NAME}
            GTest::GTest GTest::Main
            ${hptt_LIBRARY}
            Boost::serialization
            Boost::mpi
            ${MPI_CXX_LINK_FLAGS}
            ${MPI_mpi_LIBRARY}
            "${LINK_LIBS}" "${LINK_LIB_FLAGS}"
    )

    add_test(
            NAME ${TEST_NAME}
            COMMAND "${TEST_NAME}")

    set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests)
endmacro()


macro(add_mpi_unittest
        TEST_NAME TEST_SRC CFLAGS LINK_LIBS LINK_LIB_FLAGS PROCESSOR_NUM)
    add_executable(${TEST_NAME}
            ${TEST_SRC})

    target_compile_options(${TEST_NAME}
            PRIVATE ${CFLAGS}
            PRIVATE ${OMP_FLAGS}
    )
    target_include_directories(${TEST_NAME}
            PRIVATE ${QLTEN_HEADER_PATH}
            PRIVATE ${hptt_INCLUDE_DIR}
            PRIVATE ${Boost_INCLUDE_DIR}
            PRIVATE ${MPI_CXX_HEADER_DIR})
    target_link_libraries(${TEST_NAME}
            GTest::GTest GTest::Main
            ${hptt_LIBRARY}
            Boost::serialization
            Boost::mpi
            ${MPI_CXX_LINK_FLAGS}
            ${MPI_mpi_LIBRARY}
            "${LINK_LIBS}" "${LINK_LIB_FLAGS}"
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


macro(compile_unittest
        TEST_NAME TEST_SRC CFLAGS LINK_LIBS LINK_LIB_FLAGS)
    add_executable(${TEST_NAME}
            ${TEST_SRC})

    target_compile_options(${TEST_NAME}
            PRIVATE ${CFLAGS}
            PRIVATE ${OMP_FLAGS}
    )
    target_include_directories(${TEST_NAME}
            PRIVATE ${hptt_INCLUDE_DIR}
            PRIVATE ${Boost_INCLUDE_DIR}
            PRIVATE ${MPI_CXX_HEADER_DIR})
    target_link_libraries(${TEST_NAME}
            GTest::GTest GTest::Main
            ${hptt_LIBRARY}
            Boost::serialization
            Boost::mpi
            ${MPI_CXX_LINK_FLAGS}
            ${MPI_mpi_LIBRARY}
            "${LINK_LIBS}" "${LINK_LIB_FLAGS}"
    )
    set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests)
endmacro()

add_unittest(test_fu1qn "test_qltensor/test_special_qn/test_fu1qn.cc" "" "" "")

add_unittest(test_fermion_parity_exchange "test_utility/test_fermion_parity_exchange.cpp" "" "" "")

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
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
)

add_unittest(test_fermion_qltensor
        "test_qltensor/test_fermion_qltensor.cc"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
)

## Test tensor manipulation functions.
# Test basic tensor operations.
add_unittest(test_basic_operations
        "test_tensor_manipulation/test_basic_operations.cc"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
)
# Test tensor linear combination.
add_unittest(test_ten_linear_combine
        "test_tensor_manipulation/test_ten_linear_combine.cc"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
)
# Test tensor contraction.
add_unittest(test_ten_ctrct
        "test_tensor_manipulation/test_ten_ctrct.cc"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
)
add_unittest(test_fermion_ten_ctrct
        "test_tensor_manipulation/test_fermion_ten_ctrct.cc"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
)
#Test tensor restriction contract
add_unittest(test_ten_ctrct_1sct
        "test_tensor_manipulation/test_ten_ctrct_1sct.cc"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
)
# Test tensor SVD.
add_unittest(test_ten_svd
        "test_tensor_manipulation/test_ten_svd.cc"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
)
# EVD
add_unittest(test_sym_mat_evd
        "test_tensor_manipulation/test_sym_mat_evd.cpp"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
)
# Test tensor QR.
add_unittest(test_ten_qr
        "test_tensor_manipulation/test_ten_qr.cc"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
)
# Test tensor index combination.
add_unittest(test_index_combine
        "test_tensor_manipulation/test_index_combine.cc"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
)
# Test tensor expansion
add_unittest(test_ten_expand
        "test_tensor_manipulation/test_ten_expand.cc"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
)

# Test tensor fuse index
add_unittest(test_ten_fuse_index
        "test_tensor_manipulation/test_ten_fuse_index.cc"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
)

# Test tensor block expansion
add_unittest(test_ten_blk_expand
        "test_tensor_manipulation/test_ten_block_expand.cc"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
)
# Test tensor communication for serialization
add_unittest(test_boost_serialization
        "test_tensor_mpi/test_ten_boost_serialization.cc"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
)

# Test boost mpi
add_mpi_unittest(test_boost_mpi
        "test_tensor_mpi/test_boost_mpi.cc"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" "2"
)

# Test tensor communication for mpi
add_mpi_unittest(test_ten_mpi_comm
        "test_tensor_mpi/test_ten_mpi_comm.cc"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" "2"
)


# Test mpi svd performance
add_mpi_unittest(test_mpi_svd
        "test_tensor_mpi/test_mpi_svd.cc"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" "3"
)
