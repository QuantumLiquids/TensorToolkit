# SPDX-License-Identifier: LGPL-3.0-only
#
# Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
# Creation Date: 2024-11-28
#
# Description: QuantumLiquids/tensor project. CMake module to find cuTENSOR.
#

include_guard(GLOBAL)

set(CUTENSOR_ROOT "${CUTENSOR_ROOT}" CACHE PATH
    "cuTENSOR installation prefix or root hint. Examples: /opt/nvidia/hpc_sdk/.../targets/x86_64-linux or $ENV{HOME}/.local/usr")

set(_cutensor_candidate_roots "")
set(_cutensor_search_roots "")

macro(_cutensor_append_root root_value)
  if (DEFINED ${root_value} AND NOT "${${root_value}}" STREQUAL "")
    list(APPEND _cutensor_candidate_roots "${${root_value}}")
  endif ()
endmacro()

macro(_cutensor_append_env_root env_name)
  if (DEFINED ENV{${env_name}} AND NOT "$ENV{${env_name}}" STREQUAL "")
    list(APPEND _cutensor_candidate_roots "$ENV{${env_name}}")
  endif ()
endmacro()

if (CUTENSOR_ROOT)
  list(APPEND _cutensor_candidate_roots "${CUTENSOR_ROOT}")
endif ()

_cutensor_append_env_root(CUTENSOR_ROOT)
_cutensor_append_env_root(CUDAToolkit_ROOT)
_cutensor_append_env_root(CUDA_ROOT)
_cutensor_append_env_root(CUDA_HOME)

if (DEFINED CUDAToolkit_ROOT AND NOT "${CUDAToolkit_ROOT}" STREQUAL "")
  list(APPEND _cutensor_candidate_roots "${CUDAToolkit_ROOT}")
endif ()

if (DEFINED CUDAToolkit_LIBRARY_ROOT AND NOT "${CUDAToolkit_LIBRARY_ROOT}" STREQUAL "")
  list(APPEND _cutensor_candidate_roots "${CUDAToolkit_LIBRARY_ROOT}")
endif ()

if (DEFINED CUDAToolkit_LIBRARY_DIR AND NOT "${CUDAToolkit_LIBRARY_DIR}" STREQUAL "")
  get_filename_component(_cutensor_cuda_lib_root "${CUDAToolkit_LIBRARY_DIR}" DIRECTORY)
  list(APPEND _cutensor_candidate_roots "${_cutensor_cuda_lib_root}")
endif ()

if (DEFINED CUDAToolkit_BIN_DIR AND NOT "${CUDAToolkit_BIN_DIR}" STREQUAL "")
  get_filename_component(_cutensor_cuda_root "${CUDAToolkit_BIN_DIR}" DIRECTORY)
  list(APPEND _cutensor_candidate_roots "${_cutensor_cuda_root}")
endif ()

if (DEFINED CMAKE_CUDA_COMPILER AND NOT "${CMAKE_CUDA_COMPILER}" STREQUAL "")
  get_filename_component(_cutensor_cuda_bin_from_compiler "${CMAKE_CUDA_COMPILER}" DIRECTORY)
  get_filename_component(_cutensor_cuda_root_from_compiler "${_cutensor_cuda_bin_from_compiler}" DIRECTORY)
  list(APPEND _cutensor_candidate_roots "${_cutensor_cuda_root_from_compiler}")
endif ()

if (DEFINED ENV{NVHPC} AND NOT "$ENV{NVHPC}" STREQUAL "")
  file(GLOB _cutensor_nvhpc_roots
    "$ENV{NVHPC}/Linux_x86_64/*/math_libs/*/targets/x86_64-linux")
  list(APPEND _cutensor_candidate_roots ${_cutensor_nvhpc_roots})
endif ()

foreach (_cutensor_root IN LISTS _cutensor_candidate_roots)
  if (_cutensor_root STREQUAL "")
    continue()
  endif ()

  cmake_path(NORMAL_PATH _cutensor_root OUTPUT_VARIABLE _cutensor_root_norm)
  list(APPEND _cutensor_search_roots "${_cutensor_root_norm}")

  if (_cutensor_root_norm MATCHES "^(.*)/cuda/([0-9]+\\.[0-9]+)(/.*)?$")
    list(APPEND _cutensor_search_roots
      "${CMAKE_MATCH_1}/math_libs/${CMAKE_MATCH_2}/targets/x86_64-linux")
  endif ()

  file(GLOB _cutensor_sdk_math_lib_roots
    "${_cutensor_root_norm}/math_libs/*/targets/x86_64-linux")
  list(APPEND _cutensor_search_roots ${_cutensor_sdk_math_lib_roots})

  list(APPEND _cutensor_search_roots
    "${_cutensor_root_norm}/usr"
    "${_cutensor_root_norm}/usr/local")
endforeach ()

list(REMOVE_DUPLICATES _cutensor_search_roots)

set(_cutensor_include_suffixes
  include
  usr/include
  targets/x86_64-linux/include)

set(_cutensor_library_suffixes
  lib
  lib64
  lib/x86_64-linux-gnu
  usr/lib
  usr/lib64
  usr/lib/x86_64-linux-gnu
  targets/x86_64-linux/lib)

set(_cutensor_versioned_library_dirs "")
foreach (_cutensor_root IN LISTS _cutensor_search_roots)
  file(GLOB _cutensor_globbed_dirs
    "${_cutensor_root}/lib/libcutensor/*"
    "${_cutensor_root}/lib64/libcutensor/*"
    "${_cutensor_root}/usr/lib/libcutensor/*"
    "${_cutensor_root}/usr/lib64/libcutensor/*"
    "${_cutensor_root}/usr/lib/x86_64-linux-gnu/libcutensor/*")
  foreach (_cutensor_dir IN LISTS _cutensor_globbed_dirs)
    if (EXISTS "${_cutensor_dir}/libcutensor.so"
        OR EXISTS "${_cutensor_dir}/libcutensor.so.2"
        OR EXISTS "${_cutensor_dir}/libcutensorMg.so")
      list(APPEND _cutensor_versioned_library_dirs "${_cutensor_dir}")
    endif ()
  endforeach ()
endforeach ()
list(REMOVE_DUPLICATES _cutensor_versioned_library_dirs)

find_path(CUTENSOR_INCLUDE_DIR
  NAMES cutensor.h
  HINTS ${_cutensor_search_roots}
  PATH_SUFFIXES ${_cutensor_include_suffixes}
  NO_DEFAULT_PATH
  DOC "Path to cuTENSOR include directory")
if (NOT CUTENSOR_INCLUDE_DIR)
  find_path(CUTENSOR_INCLUDE_DIR
    NAMES cutensor.h
    PATH_SUFFIXES ${_cutensor_include_suffixes}
    DOC "Path to cuTENSOR include directory")
endif ()

find_library(CUTENSOR_LIBRARY
  NAMES cutensor
  HINTS ${_cutensor_versioned_library_dirs} ${_cutensor_search_roots}
  PATH_SUFFIXES ${_cutensor_library_suffixes}
  NO_DEFAULT_PATH
  DOC "Path to the cuTENSOR library")
if (NOT CUTENSOR_LIBRARY)
  find_library(CUTENSOR_LIBRARY
    NAMES cutensor
    PATH_SUFFIXES ${_cutensor_library_suffixes}
    DOC "Path to the cuTENSOR library")
endif ()

if (CUTENSOR_LIBRARY)
  get_filename_component(_cutensor_library_dir "${CUTENSOR_LIBRARY}" DIRECTORY)
  set(_cutensor_mg_hints "${_cutensor_library_dir}")
else ()
  set(_cutensor_mg_hints ${_cutensor_versioned_library_dirs} ${_cutensor_search_roots})
endif ()
list(REMOVE_DUPLICATES _cutensor_mg_hints)

find_library(CUTENSOR_MG_LIBRARY
  NAMES cutensorMg
  HINTS ${_cutensor_mg_hints}
  PATH_SUFFIXES ${_cutensor_library_suffixes}
  NO_DEFAULT_PATH
  DOC "Path to the cuTENSOR Mg library")

set(CUTENSOR_INCLUDE_DIRS "${CUTENSOR_INCLUDE_DIR}")
set(CUTENSOR_LIBRARIES "${CUTENSOR_LIBRARY}")
if (CUTENSOR_MG_LIBRARY)
  list(APPEND CUTENSOR_LIBRARIES "${CUTENSOR_MG_LIBRARY}")
endif ()

set(CUTENSOR_VERSION "")
if (CUTENSOR_INCLUDE_DIR AND EXISTS "${CUTENSOR_INCLUDE_DIR}/cutensor.h")
  file(STRINGS "${CUTENSOR_INCLUDE_DIR}/cutensor.h" _cutensor_version_major_line
       REGEX "^#define CUTENSOR_MAJOR[ \t]+[0-9]+")
  file(STRINGS "${CUTENSOR_INCLUDE_DIR}/cutensor.h" _cutensor_version_minor_line
       REGEX "^#define CUTENSOR_MINOR[ \t]+[0-9]+")
  file(STRINGS "${CUTENSOR_INCLUDE_DIR}/cutensor.h" _cutensor_version_patch_line
       REGEX "^#define CUTENSOR_PATCH[ \t]+[0-9]+")
  if (_cutensor_version_major_line)
    string(REGEX REPLACE ".*CUTENSOR_MAJOR[ \t]+([0-9]+).*" "\\1"
           _cutensor_version_major "${_cutensor_version_major_line}")
    string(REGEX REPLACE ".*CUTENSOR_MINOR[ \t]+([0-9]+).*" "\\1"
           _cutensor_version_minor "${_cutensor_version_minor_line}")
    string(REGEX REPLACE ".*CUTENSOR_PATCH[ \t]+([0-9]+).*" "\\1"
           _cutensor_version_patch "${_cutensor_version_patch_line}")
    set(CUTENSOR_VERSION
      "${_cutensor_version_major}.${_cutensor_version_minor}.${_cutensor_version_patch}")
  endif ()
endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUTENSOR
  REQUIRED_VARS CUTENSOR_INCLUDE_DIR CUTENSOR_LIBRARY
  VERSION_VAR CUTENSOR_VERSION
  REASON_FAILURE_MESSAGE
    "Set CUTENSOR_ROOT to the cuTENSOR prefix, or set CUTENSOR_INCLUDE_DIR and CUTENSOR_LIBRARY explicitly, or set CUDAToolkit_ROOT for CUDA installs in non-standard locations. Supported layouts include NVHPC math_libs target trees and user-local installs such as \$HOME/.local/usr.")

if (CUTENSOR_FOUND)
  if (NOT TARGET CUTENSOR::CUTENSOR)
    add_library(CUTENSOR::CUTENSOR SHARED IMPORTED)
    set_target_properties(CUTENSOR::CUTENSOR PROPERTIES
      IMPORTED_LOCATION "${CUTENSOR_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${CUTENSOR_INCLUDE_DIR}")
  endif ()

  if (CUTENSOR_MG_LIBRARY AND NOT TARGET CUTENSOR::CUTENSOR_MG)
    add_library(CUTENSOR::CUTENSOR_MG SHARED IMPORTED)
    set_target_properties(CUTENSOR::CUTENSOR_MG PROPERTIES
      IMPORTED_LOCATION "${CUTENSOR_MG_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${CUTENSOR_INCLUDE_DIR}")
  endif ()

  message(STATUS "Found cuTENSOR ${CUTENSOR_VERSION}: ${CUTENSOR_LIBRARY}")
  message(STATUS "cuTENSOR include directory: ${CUTENSOR_INCLUDE_DIR}")
  if (CUTENSOR_MG_LIBRARY)
    message(STATUS "cuTENSOR Mg library: ${CUTENSOR_MG_LIBRARY}")
  endif ()
else ()
  string(REPLACE ";" "\n  " _cutensor_search_roots_message "${_cutensor_search_roots}")
  message(STATUS "cuTENSOR search roots considered:\n  ${_cutensor_search_roots_message}")
endif ()

mark_as_advanced(
  CUTENSOR_ROOT
  CUTENSOR_INCLUDE_DIR
  CUTENSOR_LIBRARY
  CUTENSOR_MG_LIBRARY)
