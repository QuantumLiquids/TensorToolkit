cmake_minimum_required(VERSION 3.27)
project(TensorToolkitPackageConsumer LANGUAGES CXX)

option(QLTEN_PACKAGE_CONSUMER_ENABLE_GPU "Build the GPU package smoke consumer" OFF)

if(QLTEN_PACKAGE_CONSUMER_ENABLE_GPU)
  enable_language(CUDA)
endif()

find_package(TensorToolkit CONFIG REQUIRED)

get_target_property(_qlten_tt_compile_defs
  TensorToolkit::TensorToolkit INTERFACE_COMPILE_DEFINITIONS)
if(_qlten_tt_compile_defs)
  foreach(_qlten_compile_def IN LISTS _qlten_tt_compile_defs)
    if(_qlten_compile_def MATCHES "(^|[<:])QLTEN(_MPI)?_TIMING_MODE($|[>:])")
      message(FATAL_ERROR
        "TensorToolkit::TensorToolkit must not force timing macros on consumers: "
        "${_qlten_compile_def}")
    endif()
  endforeach()
endif()

set(qlten_package_smoke_source smoke_cpu.cc)
if(QLTEN_PACKAGE_CONSUMER_ENABLE_GPU)
  set(qlten_package_smoke_source smoke_gpu.cu)
endif()

add_executable(qlten_package_smoke "${qlten_package_smoke_source}")
target_link_libraries(qlten_package_smoke PRIVATE TensorToolkit::TensorToolkit)

set(_qlten_consumer_expects_openmp OFF)
get_target_property(_qlten_tt_link_libs TensorToolkit::TensorToolkit INTERFACE_LINK_LIBRARIES)
if(_qlten_tt_link_libs)
  foreach(_qlten_link_item IN LISTS _qlten_tt_link_libs)
    if(_qlten_link_item MATCHES "OpenMP::OpenMP_CXX")
      set(_qlten_consumer_expects_openmp ON)
      break()
    endif()
  endforeach()
endif()
if(_qlten_consumer_expects_openmp)
  target_compile_definitions(qlten_package_smoke PRIVATE QLTEN_EXPECT_OPENMP=1)
endif()

enable_testing()
add_test(
  NAME qlten_package_smoke
  COMMAND "$<TARGET_FILE:qlten_package_smoke>")

if(QLTEN_PACKAGE_CONSUMER_ENABLE_GPU)
  set_tests_properties(qlten_package_smoke PROPERTIES SKIP_RETURN_CODE 77)
endif()
