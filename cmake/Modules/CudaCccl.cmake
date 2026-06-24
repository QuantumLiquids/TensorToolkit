function(qlten_collect_cuda_cccl_include_dirs out_var)
  set(_qlten_cuda_cccl_include_dirs "")
  set(_qlten_cuda_include_roots "")

  foreach(_qlten_cuda_include_dir IN LISTS CUDAToolkit_INCLUDE_DIRS CUDAToolkit_INCLUDE_DIR)
    if (_qlten_cuda_include_dir)
      list(APPEND _qlten_cuda_include_roots "${_qlten_cuda_include_dir}")
    endif ()
  endforeach ()

  foreach(_qlten_cuda_root IN ITEMS
      "${CUDAToolkit_TARGET_DIR}"
      "${CUDAToolkit_ROOT}"
      "$ENV{CUDAToolkit_ROOT}"
      "$ENV{CUDA_HOME}"
      "$ENV{CUDA_PATH}")
    if (_qlten_cuda_root)
      list(APPEND _qlten_cuda_include_roots "${_qlten_cuda_root}/include")
    endif ()
  endforeach ()

  if (CUDAToolkit_LIBRARY_DIR)
    get_filename_component(_qlten_cuda_library_root
      "${CUDAToolkit_LIBRARY_DIR}/.." ABSOLUTE)
    list(APPEND _qlten_cuda_include_roots
      "${_qlten_cuda_library_root}/include")
  endif ()

  list(REMOVE_DUPLICATES _qlten_cuda_include_roots)
  foreach(_qlten_cuda_include_root IN LISTS _qlten_cuda_include_roots)
    if (EXISTS "${_qlten_cuda_include_root}/cccl/cuda/std/complex")
      list(APPEND _qlten_cuda_cccl_include_dirs
        "${_qlten_cuda_include_root}/cccl")
    endif ()
  endforeach ()

  list(REMOVE_DUPLICATES _qlten_cuda_cccl_include_dirs)
  set(${out_var} "${_qlten_cuda_cccl_include_dirs}" PARENT_SCOPE)
endfunction()
