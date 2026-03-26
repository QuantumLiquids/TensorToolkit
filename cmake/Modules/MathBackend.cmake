# SPDX-License-Identifier: LGPL-3.0-only

include_guard(DIRECTORY)

option(HP_NUMERIC_USE_MKL "Use Intel MKL as BLAS backend" OFF)
option(HP_NUMERIC_USE_AOCL "Use AMD AOCL as BLAS backend" OFF)
option(HP_NUMERIC_USE_OPENBLAS "Use OpenBLAS as BLAS backend" OFF)

set(QLTEN_HP_NUMERIC_BACKEND_DEFINE "")
set(QLTEN_HP_NUMERIC_BLAS_INCLUDE_DIRS "")
set(QLTEN_HP_NUMERIC_PUBLIC_INCLUDE_DIRS "")
set(QLTEN_HP_NUMERIC_REQUIRES_OPENMP OFF)
set(QLTEN_HP_NUMERIC_BLA_VENDOR "")
set(QLTEN_HP_NUMERIC_LAPACKE_LIBRARY "")
set(QLTEN_HP_NUMERIC_LAPACKE_INCLUDE_DIR "")

function(qlten_resolve_lapacke)
    if (HP_NUMERIC_USE_MKL)
        return ()
    endif ()

    unset(QLTEN_HP_NUMERIC_LAPACKE_INCLUDE_DIR CACHE)
    unset(QLTEN_HP_NUMERIC_LAPACKE_LIBRARY CACHE)
    unset(QLTEN_HP_NUMERIC_LAPACKE_INCLUDE_DIR)
    unset(QLTEN_HP_NUMERIC_LAPACKE_LIBRARY)

    set(_qlten_hp_numeric_search_roots "")
    if (HP_NUMERIC_USE_AOCL)
        list(APPEND _qlten_hp_numeric_search_roots "$ENV{AOCL_ROOT}")
    elseif (HP_NUMERIC_USE_OPENBLAS)
        list(APPEND _qlten_hp_numeric_search_roots
                "$ENV{OpenBLAS_ROOT}"
                "$ENV{LAPACK_ROOT}"
                "/usr"
                "/usr/local"
                "/usr/lib"
                "/usr/lib64"
                "/lib"
                "/lib64"
                "/opt/homebrew/opt/openblas"
                "/opt/homebrew/opt/lapack")
    endif ()

    foreach(_qlten_lib IN LISTS BLAS_LIBRARIES LAPACK_LIBRARIES)
        get_filename_component(_qlten_lib_dir "${_qlten_lib}" DIRECTORY)
        if (_qlten_lib_dir)
            get_filename_component(_qlten_prefix1 "${_qlten_lib_dir}/.." ABSOLUTE)
            get_filename_component(_qlten_prefix2 "${_qlten_lib_dir}/../.." ABSOLUTE)
            list(APPEND _qlten_hp_numeric_search_roots
                    "${_qlten_lib_dir}"
                    "${_qlten_prefix1}"
                    "${_qlten_prefix2}")
        endif ()
    endforeach ()
    list(REMOVE_DUPLICATES _qlten_hp_numeric_search_roots)

    set(_qlten_hp_numeric_lib_suffixes lib lib64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu)
    if (DEFINED CMAKE_LIBRARY_ARCHITECTURE AND NOT CMAKE_LIBRARY_ARCHITECTURE STREQUAL "")
        list(APPEND _qlten_hp_numeric_lib_suffixes "lib/${CMAKE_LIBRARY_ARCHITECTURE}")
    endif ()

    find_path(QLTEN_HP_NUMERIC_LAPACKE_INCLUDE_DIR
            NAMES lapacke.h
            HINTS ${_qlten_hp_numeric_search_roots}
            PATH_SUFFIXES include include/openblas
    )

    find_library(QLTEN_HP_NUMERIC_LAPACKE_LIBRARY
            NAMES lapacke
            HINTS ${_qlten_hp_numeric_search_roots}
            PATH_SUFFIXES ${_qlten_hp_numeric_lib_suffixes}
    )

    set(_qlten_hp_numeric_public_include_dirs "${QLTEN_HP_NUMERIC_BLAS_INCLUDE_DIRS}")
    if (QLTEN_HP_NUMERIC_LAPACKE_INCLUDE_DIR)
        list(APPEND _qlten_hp_numeric_public_include_dirs "${QLTEN_HP_NUMERIC_LAPACKE_INCLUDE_DIR}")
    endif ()
    list(REMOVE_DUPLICATES _qlten_hp_numeric_public_include_dirs)

    set(QLTEN_HP_NUMERIC_LAPACKE_INCLUDE_DIR "${QLTEN_HP_NUMERIC_LAPACKE_INCLUDE_DIR}" PARENT_SCOPE)
    set(QLTEN_HP_NUMERIC_LAPACKE_LIBRARY "${QLTEN_HP_NUMERIC_LAPACKE_LIBRARY}" PARENT_SCOPE)
    set(QLTEN_HP_NUMERIC_PUBLIC_INCLUDE_DIRS "${_qlten_hp_numeric_public_include_dirs}" PARENT_SCOPE)
endfunction()

if (NOT HP_NUMERIC_USE_MKL AND NOT HP_NUMERIC_USE_AOCL AND NOT HP_NUMERIC_USE_OPENBLAS)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
        set(_hp_numeric_cpu_vendor "")
        if (EXISTS "/proc/cpuinfo")
            file(STRINGS "/proc/cpuinfo" _hp_numeric_cpu_vendor_lines REGEX "^vendor_id")
            list(LENGTH _hp_numeric_cpu_vendor_lines _hp_numeric_cpu_vendor_count)
            if (_hp_numeric_cpu_vendor_count GREATER 0)
                list(GET _hp_numeric_cpu_vendor_lines 0 _hp_numeric_cpu_vendor_line)
                if (_hp_numeric_cpu_vendor_line MATCHES "AuthenticAMD")
                    set(_hp_numeric_cpu_vendor "amd")
                elseif (_hp_numeric_cpu_vendor_line MATCHES "GenuineIntel")
                    set(_hp_numeric_cpu_vendor "intel")
                endif ()
            endif ()
        endif ()

        if (_hp_numeric_cpu_vendor STREQUAL "amd")
            set(HP_NUMERIC_USE_AOCL ON)
        else ()
            set(HP_NUMERIC_USE_MKL ON)
        endif ()
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        set(HP_NUMERIC_USE_OPENBLAS ON)
    else ()
        set(HP_NUMERIC_USE_OPENBLAS ON)
    endif ()
endif ()

set(_hp_numeric_backend_count 0)
if (HP_NUMERIC_USE_MKL)
    math(EXPR _hp_numeric_backend_count "${_hp_numeric_backend_count} + 1")
endif ()
if (HP_NUMERIC_USE_AOCL)
    math(EXPR _hp_numeric_backend_count "${_hp_numeric_backend_count} + 1")
endif ()
if (HP_NUMERIC_USE_OPENBLAS)
    math(EXPR _hp_numeric_backend_count "${_hp_numeric_backend_count} + 1")
endif ()

if (_hp_numeric_backend_count EQUAL 0)
    message(FATAL_ERROR "No BLAS backend selected. Enable one of HP_NUMERIC_USE_MKL, HP_NUMERIC_USE_AOCL, HP_NUMERIC_USE_OPENBLAS.")
elseif (_hp_numeric_backend_count GREATER 1)
    message(FATAL_ERROR "Multiple BLAS backends selected. Enable only one of HP_NUMERIC_USE_MKL, HP_NUMERIC_USE_AOCL, HP_NUMERIC_USE_OPENBLAS.")
endif ()

set(BLA_VENDOR "" CACHE STRING "Preferred BLAS vendor for HP numeric backend" FORCE)
set(BLAS_INCLUDE_DIRS "" CACHE STRING "Include paths for the selected BLAS backend" FORCE)

if (HP_NUMERIC_USE_MKL)
    if (NOT DEFINED ENV{MKLROOT} OR "$ENV{MKLROOT}" STREQUAL "")
        message(FATAL_ERROR "HP_NUMERIC_USE_MKL is ON but MKLROOT is not defined. Source the MKL environment or export MKLROOT before configuring.")
    endif ()

    set(QLTEN_HP_NUMERIC_BACKEND_DEFINE "HP_NUMERIC_BACKEND_MKL")
    set(QLTEN_HP_NUMERIC_BLA_VENDOR "Intel10_64lp")
    set(QLTEN_HP_NUMERIC_BLAS_INCLUDE_DIRS "$ENV{MKLROOT}/include")
    set(QLTEN_HP_NUMERIC_PUBLIC_INCLUDE_DIRS "${QLTEN_HP_NUMERIC_BLAS_INCLUDE_DIRS}")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        set(QLTEN_HP_NUMERIC_REQUIRES_OPENMP ON)
    endif ()
elseif (HP_NUMERIC_USE_AOCL)
    if (NOT DEFINED ENV{AOCL_ROOT} OR "$ENV{AOCL_ROOT}" STREQUAL "")
        message(FATAL_ERROR "HP_NUMERIC_USE_AOCL is ON but AOCL_ROOT is not defined. Export AOCL_ROOT (or source your AOCL vendor environment script) before configuring.")
    endif ()

    set(QLTEN_HP_NUMERIC_BACKEND_DEFINE "HP_NUMERIC_BACKEND_AOCL")
    set(QLTEN_HP_NUMERIC_BLA_VENDOR "AOCL_mt")
    list(APPEND CMAKE_PREFIX_PATH "$ENV{AOCL_ROOT}")
    set(QLTEN_HP_NUMERIC_BLAS_INCLUDE_DIRS "$ENV{AOCL_ROOT}/include")
    set(QLTEN_HP_NUMERIC_PUBLIC_INCLUDE_DIRS "${QLTEN_HP_NUMERIC_BLAS_INCLUDE_DIRS}")
    set(QLTEN_HP_NUMERIC_REQUIRES_OPENMP ON)
elseif (HP_NUMERIC_USE_OPENBLAS)
    set(QLTEN_HP_NUMERIC_BACKEND_DEFINE "HP_NUMERIC_BACKEND_OPENBLAS")
    set(QLTEN_HP_NUMERIC_BLA_VENDOR "OpenBLAS")
    set(QLTEN_HP_NUMERIC_REQUIRES_OPENMP ON)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        set(OpenBLAS_ROOT "/opt/homebrew/opt/openblas/")
        set(LAPACK_ROOT "/opt/homebrew/opt/lapack")
        list(APPEND CMAKE_PREFIX_PATH "${OpenBLAS_ROOT}" "${LAPACK_ROOT}")
        set(QLTEN_HP_NUMERIC_BLAS_INCLUDE_DIRS "${OpenBLAS_ROOT}/include")
    elseif (UNIX AND NOT APPLE) # use for github action
        # On Linux (e.g. Ubuntu), OpenBLAS headers are usually in /usr/include/openblas or /usr/include
        if (EXISTS "/usr/include/openblas/cblas.h")
            set(QLTEN_HP_NUMERIC_BLAS_INCLUDE_DIRS "/usr/include/openblas")
        elseif (EXISTS "/usr/include/cblas.h")
            set(QLTEN_HP_NUMERIC_BLAS_INCLUDE_DIRS "/usr/include")
        endif()
    endif ()
    set(QLTEN_HP_NUMERIC_PUBLIC_INCLUDE_DIRS "${QLTEN_HP_NUMERIC_BLAS_INCLUDE_DIRS}")
endif ()

if (NOT HP_NUMERIC_USE_MKL)
    qlten_resolve_lapacke()
endif ()

set(BLA_VENDOR "${QLTEN_HP_NUMERIC_BLA_VENDOR}" CACHE STRING "Preferred BLAS vendor for HP numeric backend" FORCE)
set(BLAS_INCLUDE_DIRS "${QLTEN_HP_NUMERIC_BLAS_INCLUDE_DIRS}" CACHE STRING "Include paths for the selected BLAS backend" FORCE)

if (QLTEN_HP_NUMERIC_BLAS_INCLUDE_DIRS)
    message(STATUS "HP numeric headers: ${QLTEN_HP_NUMERIC_BLAS_INCLUDE_DIRS}")
endif ()
