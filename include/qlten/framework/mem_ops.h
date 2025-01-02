// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author:   Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2024-12-26
*
* Description: Unify the API for memory operations
 * For CUDA code, all the functions in current file only work in device.
 * API name follow the CPU C style
*/


#ifndef QLTEN_FRAMEWORK_MEM_OPS_H
#define QLTEN_FRAMEWORK_MEM_OPS_H

#ifndef USE_GPU
#include <cstdlib>      // malloc, free, calloc
#include <cstring>      // memcpy, memset
#endif

#ifdef USE_GPU
#include <cuda_runtime.h>
#endif

namespace qlten {
#ifdef USE_GPU
#include <cuda_runtime.h>
// Handles CUDA runtime errors
inline void HandleCudaError(cudaError_t error, int line) {
  if (error != cudaSuccess) {
    std::cerr << "CUDA Error at line " << line << ": "
              << cudaGetErrorString(error) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}
#define HANDLE_CUDA_ERROR(x) HandleCudaError(x, __LINE__)
#endif

inline void *QLMalloc(std::size_t size) {
#ifndef USE_GPU
  return std::malloc(size);
#else
  void *data;
  HANDLE_CUDA_ERROR(cudaMalloc((void **) &data, size));
  return data;
#endif
}

inline void *QLMemset(void *dest, int ch, std::size_t count) {
#ifndef USE_GPU
  return memset(dest, ch, count);
#else
  auto cuda_err = cudaMemset((void *) dest, ch, count);
  if (cuda_err != cudaSuccess) {
    std::cerr << "cudaMemset error : " << cuda_err << std::endl;
  }
  return dest;
#endif
}

/**
 * Initialize as 0 on a block of continuous memory with # of bytes = num * size_of_elem
 * By definition only num * size_of_elem make sense, but
 * by convention we regard size_of_elem as something like sizeof(double) or sizeof(complex<double>),
 * and regard num as the number of elements.
 */
inline void *QLCalloc(std::size_t num, std::size_t size_of_elem) {
#ifndef USE_GPU
  return calloc(num, size_of_elem);
#else
  void *data;
  auto cuda_err = cudaMalloc((void **) &data, num * size_of_elem);
  if (cuda_err != cudaSuccess) {
    std::cerr << "cudaMalloc error : " << cuda_err << std::endl;
  }
  cuda_err = cudaMemset((void *) data, 0, num * size_of_elem);
  if (cuda_err != cudaSuccess) {
    std::cerr << "cudaMemset error : " << cuda_err << std::endl;
  }
  return data;
#endif
}

inline void QLFree(void *data) {
#ifndef USE_GPU // CPU code
  std::free(data);
#else // GPU code
  auto cuda_err = cudaFree(data);
//  assert(cuda_err == cudaSuccess );
  HANDLE_CUDA_ERROR(cuda_err);

#endif
}

void *QLMemcpy(void *dest, const void *src, std::size_t count) {
#ifndef USE_GPU // CPU code
  return memcpy(dest, src, count);
#else
  auto cuda_err = cudaMemcpy(dest, src, count, cudaMemcpyDeviceToDevice);
  if (cuda_err != cudaSuccess) {
    std::cerr << "cudaMemcpy error (1): " << cuda_err << std::endl;
  }
  return dest;
#endif
}
}//qlten

#endif //QLTEN_FRAMEWORK_MEM_OPS_H
