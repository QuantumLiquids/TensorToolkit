cmake_minimum_required(VERSION 3.27)
project(TensorToolkitPackageConsumer LANGUAGES CXX)

option(QLTEN_PACKAGE_CONSUMER_ENABLE_GPU "Build the GPU package smoke consumer" OFF)

if(QLTEN_PACKAGE_CONSUMER_ENABLE_GPU)
  enable_language(CUDA)
endif()

find_package(TensorToolkit CONFIG REQUIRED)

set(qlten_package_smoke_source smoke_cpu.cc)
if(QLTEN_PACKAGE_CONSUMER_ENABLE_GPU)
  set(qlten_package_smoke_source smoke_gpu.cu)
endif()

add_executable(qlten_package_smoke "${qlten_package_smoke_source}")
target_link_libraries(qlten_package_smoke PRIVATE TensorToolkit::TensorToolkit)

enable_testing()
add_test(
  NAME qlten_package_smoke
  COMMAND "$<TARGET_FILE:qlten_package_smoke>")

if(QLTEN_PACKAGE_CONSUMER_ENABLE_GPU)
  set_tests_properties(qlten_package_smoke PROPERTIES SKIP_RETURN_CODE 77)
endif()
