
cmake_minimum_required(VERSION 3.7 FATAL_ERROR)
project (HPTT C CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(ENABLE_IBM OFF)
set(HPTT_REQUIRES_OPENMP OFF)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "ppc64le")
  set(ENABLE_IBM ON)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  set(HPTT_REQUIRES_OPENMP ON)
  set(HPTT_CXX_FLAGS ${HPTT_CXX_FLAGS} -xhost)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  set(HPTT_REQUIRES_OPENMP ON)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(HPTT_REQUIRES_OPENMP ON)
  if(ENABLE_IBM)
  else()
    set(HPTT_CXX_FLAGS ${HPTT_CXX_FLAGS} -march=native -mtune=native)
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(HPTT_REQUIRES_OPENMP ON)
  set(HPTT_CXX_FLAGS ${HPTT_CXX_FLAGS} -march=native)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "PGI")
  set(HPTT_CXX_FLAGS ${HPTT_CXX_FLAGS} -silent -w -Mnovect)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "XL")
  set(HPTT_REQUIRES_OPENMP ON)
endif()

if(ENABLE_AVX)
  set(HPTT_CXX_FLAGS ${HPTT_CXX_FLAGS} -mavx -DHPTT_ARCH_AVX)
elseif(ENABLE_ARM)
  set(HPTT_CXX_FLAGS ${HPTT_CXX_FLAGS} -mfpu=neon -DHPTT_ARCH_ARM)
elseif(ENABLE_IBM)
  set(HPTT_CXX_FLAGS ${HPTT_CXX_FLAGS} -mtune=native -DHPTT_ARCH_IBM -maltivec -mabi=altivec)
endif()

set(HPTT_SRCS src/hptt.cpp src/plan.cpp src/transpose.cpp src/utils.cpp)

add_library(hptt STATIC ${HPTT_SRCS})
target_compile_features(hptt PRIVATE cxx_std_11)
target_include_directories(hptt
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_options(hptt PRIVATE ${HPTT_CXX_FLAGS})

if (HPTT_REQUIRES_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(hptt PRIVATE OpenMP::OpenMP_CXX)
endif ()

install(TARGETS hptt
        EXPORT TensorToolkitTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h")
