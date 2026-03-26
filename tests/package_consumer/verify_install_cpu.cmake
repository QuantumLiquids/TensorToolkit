if (NOT DEFINED QLTEN_VERIFY_BUILD_TYPE OR QLTEN_VERIFY_BUILD_TYPE STREQUAL "")
  set(QLTEN_VERIFY_BUILD_TYPE Debug)
endif ()
if (NOT DEFINED QLTEN_VERIFY_COMPILE_HPTT_LIB)
  set(QLTEN_VERIFY_COMPILE_HPTT_LIB ON)
endif ()
if (NOT DEFINED QLTEN_VERIFY_TIMING_MODE)
  set(QLTEN_VERIFY_TIMING_MODE ON)
endif ()
if (NOT DEFINED QLTEN_VERIFY_MPI_TIMING_MODE)
  set(QLTEN_VERIFY_MPI_TIMING_MODE ON)
endif ()

string(RANDOM LENGTH 8 ALPHABET 0123456789abcdef _qlten_verify_suffix)
set(_qlten_verify_root "/tmp/qlten-verify-cpu-${_qlten_verify_suffix}")
set(package_build_dir "${_qlten_verify_root}/package-build")
set(package_prefix "${_qlten_verify_root}/prefix")
set(consumer_build_dir "${_qlten_verify_root}/consumer-build")
get_filename_component(_qlten_source_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

message(STATUS "TensorToolkit CPU package verification root: ${_qlten_verify_root}")

set(_qlten_configure_args
  -S "${CMAKE_CURRENT_LIST_DIR}/../.."
  -B "${package_build_dir}"
  -DCMAKE_BUILD_TYPE=${QLTEN_VERIFY_BUILD_TYPE}
  -DCMAKE_INSTALL_PREFIX=${package_prefix}
  -DQLTEN_USE_GPU=OFF
  -DQLTEN_COMPILE_HPTT_LIB=${QLTEN_VERIFY_COMPILE_HPTT_LIB}
  -DQLTEN_BUILD_UNITTEST=OFF
  -DQLTEN_BUILD_EXAMPLES=OFF
  -DQLTEN_TIMING_MODE=${QLTEN_VERIFY_TIMING_MODE}
  -DQLTEN_MPI_TIMING_MODE=${QLTEN_VERIFY_MPI_TIMING_MODE})

if (DEFINED QLTEN_VERIFY_CXX_COMPILER AND NOT QLTEN_VERIFY_CXX_COMPILER STREQUAL "")
  list(APPEND _qlten_configure_args -DCMAKE_CXX_COMPILER=${QLTEN_VERIFY_CXX_COMPILER})
endif ()

if (DEFINED QLTEN_VERIFY_MPI_CXX_COMPILER AND NOT QLTEN_VERIFY_MPI_CXX_COMPILER STREQUAL "")
  list(APPEND _qlten_configure_args -DMPI_CXX_COMPILER=${QLTEN_VERIFY_MPI_CXX_COMPILER})
endif ()
if (DEFINED QLTEN_VERIFY_HPTT_INCLUDE_DIR AND NOT QLTEN_VERIFY_HPTT_INCLUDE_DIR STREQUAL "")
  list(APPEND _qlten_configure_args -Dhptt_INCLUDE_DIR=${QLTEN_VERIFY_HPTT_INCLUDE_DIR})
endif ()
if (DEFINED QLTEN_VERIFY_HPTT_LIBRARY AND NOT QLTEN_VERIFY_HPTT_LIBRARY STREQUAL "")
  list(APPEND _qlten_configure_args -Dhptt_LIBRARY=${QLTEN_VERIFY_HPTT_LIBRARY})
endif ()

set(qlten_backend_arg "${QLTEN_VERIFY_CPU_BACKEND_ARG}")
if (qlten_backend_arg STREQUAL "")
  if (DEFINED ENV{MKLROOT} AND NOT "$ENV{MKLROOT}" STREQUAL "")
    set(qlten_backend_arg "-DHP_NUMERIC_USE_MKL=ON")
  elseif (DEFINED ENV{AOCL_ROOT} AND NOT "$ENV{AOCL_ROOT}" STREQUAL "")
    set(qlten_backend_arg "-DHP_NUMERIC_USE_AOCL=ON")
  else ()
    set(qlten_backend_arg "-DHP_NUMERIC_USE_OPENBLAS=ON")
  endif ()
endif ()
list(APPEND _qlten_configure_args "${qlten_backend_arg}")

execute_process(
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${_qlten_verify_root}"
  RESULT_VARIABLE cleanup_root_result
)
if (NOT cleanup_root_result EQUAL 0)
  message(FATAL_ERROR "TensorToolkit CPU verification cleanup failed: ${cleanup_root_result}")
endif ()

execute_process(
  COMMAND ${CMAKE_COMMAND} ${_qlten_configure_args}
  RESULT_VARIABLE configure_result
)
if (NOT configure_result EQUAL 0)
  message(FATAL_ERROR "TensorToolkit CPU package configure failed: ${configure_result}")
endif ()

execute_process(
  COMMAND ${CMAKE_COMMAND} --build "${package_build_dir}" -j4
  RESULT_VARIABLE build_result
)
if (NOT build_result EQUAL 0)
  message(FATAL_ERROR "TensorToolkit CPU package build failed: ${build_result}")
endif ()

execute_process(
  COMMAND ${CMAKE_COMMAND} --install "${package_build_dir}"
  RESULT_VARIABLE install_result
)
if (NOT install_result EQUAL 0)
  message(FATAL_ERROR "TensorToolkit CPU package install failed: ${install_result}")
endif ()

set(_qlten_installed_targets_file
  "${package_prefix}/lib/cmake/TensorToolkit/TensorToolkitTargets.cmake")
file(READ "${_qlten_installed_targets_file}" _qlten_installed_targets)
string(FIND "${_qlten_installed_targets}" "${_qlten_source_root}" _qlten_source_root_pos)
if (NOT _qlten_source_root_pos EQUAL -1)
  message(FATAL_ERROR
    "TensorToolkit CPU package export leaked the source tree into "
    "${_qlten_installed_targets_file}.")
endif ()

set(_qlten_consumer_configure_args
  -S "${CMAKE_CURRENT_LIST_DIR}"
  -B "${consumer_build_dir}"
  -DCMAKE_PREFIX_PATH=${package_prefix})

if (DEFINED QLTEN_VERIFY_CONSUMER_CXX_COMPILER
    AND NOT QLTEN_VERIFY_CONSUMER_CXX_COMPILER STREQUAL "")
  list(APPEND _qlten_consumer_configure_args
    -DCMAKE_CXX_COMPILER=${QLTEN_VERIFY_CONSUMER_CXX_COMPILER})
elseif (DEFINED QLTEN_VERIFY_CXX_COMPILER AND NOT QLTEN_VERIFY_CXX_COMPILER STREQUAL "")
  list(APPEND _qlten_consumer_configure_args
    -DCMAKE_CXX_COMPILER=${QLTEN_VERIFY_CXX_COMPILER})
endif ()
if (DEFINED QLTEN_VERIFY_HPTT_INCLUDE_DIR AND NOT QLTEN_VERIFY_HPTT_INCLUDE_DIR STREQUAL "")
  list(APPEND _qlten_consumer_configure_args -Dhptt_INCLUDE_DIR=${QLTEN_VERIFY_HPTT_INCLUDE_DIR})
endif ()
if (DEFINED QLTEN_VERIFY_HPTT_LIBRARY AND NOT QLTEN_VERIFY_HPTT_LIBRARY STREQUAL "")
  list(APPEND _qlten_consumer_configure_args -Dhptt_LIBRARY=${QLTEN_VERIFY_HPTT_LIBRARY})
endif ()

execute_process(
  COMMAND ${CMAKE_COMMAND} ${_qlten_consumer_configure_args}
  RESULT_VARIABLE consumer_configure_result
)
if (NOT consumer_configure_result EQUAL 0)
  message(FATAL_ERROR "TensorToolkit CPU consumer configure failed: ${consumer_configure_result}")
endif ()

execute_process(
  COMMAND ${CMAKE_COMMAND} --build "${consumer_build_dir}" -j4
  RESULT_VARIABLE consumer_build_result
)
if (NOT consumer_build_result EQUAL 0)
  message(FATAL_ERROR "TensorToolkit CPU consumer build failed: ${consumer_build_result}")
endif ()

execute_process(
  COMMAND ${CMAKE_CTEST_COMMAND} --test-dir "${consumer_build_dir}" --output-on-failure
  RESULT_VARIABLE consumer_test_result
)
if (NOT consumer_test_result EQUAL 0)
  message(FATAL_ERROR "TensorToolkit CPU consumer test failed: ${consumer_test_result}")
endif ()
