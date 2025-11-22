// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Haoxin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2020-11-24 20:13
*
* Description: QuantumLiquids/tensor project. High performance BLAS Level 1 related
* functions based on MKL, OpenBlas or cuBlas.
*/

/**
@file blas_level1.h
@brief High performance BLAS Level 1 related functions based on MKL.
*/
#ifndef QLTEN_FRAMEWORK_HP_NUMERIC_BLAS_LEVEL1_H
#define QLTEN_FRAMEWORK_HP_NUMERIC_BLAS_LEVEL1_H

#include <cassert>                              // assert
#include "qlten/framework/value_t.h"            // QLTEN_Double, QLTEN_Complex
#include "qlten/framework/flops_count.h"        // flop
#include "qlten/framework/hp_numeric/gpu_set.h" //CublasHandleManager
#ifdef Release
#define NDEBUG
#endif

#ifndef USE_GPU     // use CPU

#include "qlten/framework/hp_numeric/backend_selector.h"
#else
#include <cublas_v2.h>
#endif
namespace qlten {
/// High performance numerical functions.
namespace hp_numeric {

#ifndef USE_GPU
inline void VectorAddTo(
    const QLTEN_Double *x,
    const size_t size,
    QLTEN_Double *y,
    const QLTEN_Double a = 1.0
) {
  cblas_daxpy(size, a, x, 1, y, 1);
#ifdef QLTEN_COUNT_FLOPS
  flop += 2 * size;
#endif
}

inline void VectorAddTo(
    const QLTEN_Float *x,
    const size_t size,
    QLTEN_Float *y,
    const QLTEN_Float a = 1.0f
) {
  cblas_saxpy(size, a, x, 1, y, 1);
#ifdef QLTEN_COUNT_FLOPS
  flop += 2 * size;
#endif
}

inline void VectorAddTo(
    const QLTEN_Complex *x,
    const size_t size,
    QLTEN_Complex *y,
    const QLTEN_Complex a = 1.0
) {
  cblas_zaxpy(size, &a, x, 1, y, 1);
#ifdef QLTEN_COUNT_FLOPS
  flop += 8 * size;
#endif
}

inline void VectorAddTo(
    const QLTEN_ComplexFloat *x,
    const size_t size,
    QLTEN_ComplexFloat *y,
    const QLTEN_ComplexFloat a = 1.0f
) {
  cblas_caxpy(size, &a, x, 1, y, 1);
#ifdef QLTEN_COUNT_FLOPS
  flop += 8 * size;
#endif
}


inline void VectorScaleCopy(
    const QLTEN_Double *x,
    const size_t size,
    QLTEN_Double *y,
    const QLTEN_Double a = 1.0
) {
  cblas_dcopy(size, x, 1, y, 1);
  cblas_dscal(size, a, y, 1);
#ifdef QLTEN_COUNT_FLOPS
  flop += size;
#endif
}

inline void VectorScaleCopy(
    const QLTEN_Float *x,
    const size_t size,
    QLTEN_Float *y,
    const QLTEN_Float a = 1.0f
) {
  cblas_scopy(size, x, 1, y, 1);
  cblas_sscal(size, a, y, 1);
#ifdef QLTEN_COUNT_FLOPS
  flop += size;
#endif
}

inline void VectorScaleCopy(
    const QLTEN_Complex *x,
    const size_t size,
    QLTEN_Complex *y,
    const QLTEN_Complex a = 1.0
) {
  cblas_zcopy(size, x, 1, y, 1);
  cblas_zscal(size, &a, y, 1);
#ifdef QLTEN_COUNT_FLOPS
  flop += 6 * size;
#endif
}

inline void VectorScaleCopy(
    const QLTEN_ComplexFloat *x,
    const size_t size,
    QLTEN_ComplexFloat *y,
    const QLTEN_ComplexFloat a = 1.0f
) {
  cblas_ccopy(size, x, 1, y, 1);
  cblas_cscal(size, &a, y, 1);
#ifdef QLTEN_COUNT_FLOPS
  flop += 6 * size;
#endif
}

inline void VectorCopy(
    const QLTEN_Double *source,
    const size_t size,
    QLTEN_Double *dest
) {
  cblas_dcopy(size, source, 1, dest, 1);
}

inline void VectorCopy(
    const QLTEN_Float *source,
    const size_t size,
    QLTEN_Float *dest
) {
  cblas_scopy(size, source, 1, dest, 1);
}

inline void VectorCopy(
    const QLTEN_Complex *source,
    const size_t size,
    QLTEN_Complex *dest
) {
  cblas_zcopy(size, source, 1, dest, 1);
}

inline void VectorCopy(
    const QLTEN_ComplexFloat *source,
    const size_t size,
    QLTEN_ComplexFloat *dest
) {
  cblas_ccopy(size, source, 1, dest, 1);
}

inline void VectorScale(
    QLTEN_Double *x,
    const size_t size,
    const QLTEN_Double a
) {
  cblas_dscal(size, a, x, 1);
#ifdef QLTEN_COUNT_FLOPS
  flop += size;
#endif
}

inline void VectorScale(
    QLTEN_Float *x,
    const size_t size,
    const QLTEN_Float a
) {
  cblas_sscal(size, a, x, 1);
#ifdef QLTEN_COUNT_FLOPS
  flop += size;
#endif
}

inline void VectorScale(
    QLTEN_Complex *x,
    const size_t size,
    const QLTEN_Complex a
) {
  cblas_zscal(size, &a, x, 1);
#ifdef QLTEN_COUNT_FLOPS
  flop += 6 * size;
#endif
}

inline void VectorScale(
    QLTEN_ComplexFloat *x,
    const size_t size,
    const QLTEN_ComplexFloat a
) {
  cblas_cscal(size, &a, x, 1);
#ifdef QLTEN_COUNT_FLOPS
  flop += 6 * size;
#endif
}

/**
 * @note return sqrt(sum(x^2)) instead of sum(x^2)
 */
inline double Vector2Norm(
    QLTEN_Double *x,
    const size_t size
) {
  return cblas_dnrm2(size, x, 1);
#ifdef QLTEN_COUNT_FLOPS
  flop += 2 * size;
#endif
}

inline float Vector2Norm(
    QLTEN_Float *x,
    const size_t size
) {
  return cblas_snrm2(size, x, 1);
#ifdef QLTEN_COUNT_FLOPS
  flop += 2 * size;
#endif
}

inline double Vector2Norm(
    QLTEN_Complex *x,
    const size_t size
) {
  return cblas_dznrm2(size, x, 1);
#ifdef QLTEN_COUNT_FLOPS
  flop += 8 * size;
#endif
}

inline float Vector2Norm(
    QLTEN_ComplexFloat *x,
    const size_t size
) {
  return cblas_scnrm2(size, x, 1);
#ifdef QLTEN_COUNT_FLOPS
  flop += 8 * size;
#endif
}

inline void VectorRealToCplx(
    const QLTEN_Double *real,
    const size_t size,
    QLTEN_Complex *cplx
) {
  for (size_t i = 0; i < size; ++i) { cplx[i] = real[i]; }
}

inline void VectorRealToCplx(
    const QLTEN_Float *real,
    const size_t size,
    QLTEN_ComplexFloat *cplx
) {
  for (size_t i = 0; i < size; ++i) { cplx[i] = real[i]; }
}

/**
 * @brief Calculate the sum of squares of the elements in the vector.
 * @param x Pointer to the vector elements.
 * @param size Number of elements in the vector.
 * @return Sum of squares of the vector elements.
 */
inline double VectorSumSquares(
    const QLTEN_Double *x,
    const size_t size
) {
  return cblas_ddot(size, x, 1, x, 1);
#ifdef QLTEN_COUNT_FLOPS
  flop += 2 * size;
#endif
}

inline float VectorSumSquares(
    const QLTEN_Float *x,
    const size_t size
) {
  return cblas_sdot(size, x, 1, x, 1);
#ifdef QLTEN_COUNT_FLOPS
  flop += 2 * size;
#endif
}

/**
 * @brief Calculate the sum of squares of the elements in the complex vector.
 * @param x Pointer to the complex vector elements.
 * @param size Number of elements in the vector.
 * @return Sum of squares of the complex vector elements.
 */
inline double VectorSumSquares(
    const QLTEN_Complex *x,
    const size_t size
) {
  double sum_squares_real =
      cblas_ddot(size, reinterpret_cast<const double *>(x), 2, reinterpret_cast<const double *>(x), 2);
  double sum_squares_imag =
      cblas_ddot(size, reinterpret_cast<const double *>(x) + 1, 2, reinterpret_cast<const double *>(x) + 1, 2);
  return sum_squares_real + sum_squares_imag;
#ifdef QLTEN_COUNT_FLOPS
  flop += 4 * size;
#endif
}

inline float VectorSumSquares(
    const QLTEN_ComplexFloat *x,
    const size_t size
) {
  float sum_squares_real =
      cblas_sdot(size, reinterpret_cast<const float *>(x), 2, reinterpret_cast<const float *>(x), 2);
  float sum_squares_imag =
      cblas_sdot(size, reinterpret_cast<const float *>(x) + 1, 2, reinterpret_cast<const float *>(x) + 1, 2);
  return sum_squares_real + sum_squares_imag;
#ifdef QLTEN_COUNT_FLOPS
  flop += 4 * size;
#endif
}

#else //USE GPU

inline void VectorAddTo(
    const QLTEN_Double *d_x,  // Device pointer
    const size_t size,
    QLTEN_Double *d_y,        // Device pointer
    const QLTEN_Double a = 1.0
) {
  // cuBLAS AXPY performs: y = a * x + y
  if (cublasDaxpy(CublasHandleManager::GetHandle(),
                  size, &a,
                  d_x, 1,
                  d_y, 1) != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("cuBLAS DAXPY operation failed.");
  }
#ifdef QLTEN_COUNT_FLOPS
  flop += 2 * size;
#endif
}

inline void VectorAddTo(
    const QLTEN_Float *d_x,  // Device pointer
    const size_t size,
    QLTEN_Float *d_y,        // Device pointer
    const QLTEN_Float a = 1.0f
) {
  // cuBLAS AXPY performs: y = a * x + y
  if (cublasSaxpy(CublasHandleManager::GetHandle(),
                  size, &a,
                  d_x, 1,
                  d_y, 1) != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("cuBLAS SAXPY operation failed.");
  }
#ifdef QLTEN_COUNT_FLOPS
  flop += 2 * size;
#endif
}

// Function to perform vector addition using cuBLAS
inline void VectorAddTo(
    const QLTEN_Complex *x,
    const size_t size,
    QLTEN_Complex *y,
    const QLTEN_Complex a = QLTEN_Complex(1.0, 0.0)
) {
  cublasHandle_t &handle = CublasHandleManager::GetHandle();

  if (cublasZaxpy(handle, size,
                  reinterpret_cast<const cuDoubleComplex *>(&a),
                  reinterpret_cast<const cuDoubleComplex *>(x), 1,
                  reinterpret_cast<cuDoubleComplex *>(y), 1) != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("cuBLAS ZAXPY operation failed.");
  }

#ifdef QLTEN_COUNT_FLOPS
  flop += 8 * size;
#endif
}

inline void VectorAddTo(
    const QLTEN_ComplexFloat *x,
    const size_t size,
    QLTEN_ComplexFloat *y,
    const QLTEN_ComplexFloat a = QLTEN_ComplexFloat(1.0f, 0.0f)
) {
  cublasHandle_t &handle = CublasHandleManager::GetHandle();

  if (cublasCaxpy(handle, size,
                  reinterpret_cast<const cuComplex *>(&a),
                  reinterpret_cast<const cuComplex *>(x), 1,
                  reinterpret_cast<cuComplex *>(y), 1) != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("cuBLAS CAXPY operation failed.");
  }

#ifdef QLTEN_COUNT_FLOPS
  flop += 8 * size;
#endif
}

inline void VectorScaleCopy(
    const QLTEN_Double *x,
    const size_t size,
    QLTEN_Double *y,
    const QLTEN_Double a = 1.0
) {
  // Get cuBLAS handle
  cublasHandle_t &handle = CublasHandleManager::GetHandle();

  // Copy and scale vector
  cublasStatus_t status;

  // Perform y = x and then y = a * y (scale)
  status = cublasDcopy(handle, size, x, 1, y, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasDcopy failed!" << std::endl;
  }

  status = cublasDscal(handle, size, &a, y, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasDscal failed!" << std::endl;
  }

#ifdef QLTEN_COUNT_FLOPS
  flop += size;
#endif
}

inline void VectorScaleCopy(
    const QLTEN_Float *x,
    const size_t size,
    QLTEN_Float *y,
    const QLTEN_Float a = 1.0f
) {
  // Get cuBLAS handle
  cublasHandle_t &handle = CublasHandleManager::GetHandle();

  // Copy and scale vector
  cublasStatus_t status;

  // Perform y = x and then y = a * y (scale)
  status = cublasScopy(handle, size, x, 1, y, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasScopy failed!" << std::endl;
  }

  status = cublasSscal(handle, size, &a, y, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasSscal failed!" << std::endl;
  }

#ifdef QLTEN_COUNT_FLOPS
  flop += size;
#endif
}

inline void VectorScaleCopy(
    const QLTEN_Complex *x,
    const size_t size,
    QLTEN_Complex *y,
    const QLTEN_Complex a = 1.0
) {
  // Get cuBLAS handle
  cublasHandle_t &handle = CublasHandleManager::GetHandle();

  // Copy and scale complex vector
  cublasStatus_t status;

  // Perform y = x and then y = a * y (scale)
  status = cublasZcopy(handle,
                       size,
                       reinterpret_cast<const cuDoubleComplex *>(x),
                       1,
                       reinterpret_cast<cuDoubleComplex *>(y),
                       1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasZcopy failed!" << std::endl;
  }

  status = cublasZscal(handle,
                       size,
                       reinterpret_cast<const cuDoubleComplex *>(&a),
                       reinterpret_cast<cuDoubleComplex *>(y),
                       1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasZscal failed!" << std::endl;
  }

#ifdef QLTEN_COUNT_FLOPS
  flop += 4 * size;
#endif
}

inline void VectorScaleCopy(
    const QLTEN_ComplexFloat *x,
    const size_t size,
    QLTEN_ComplexFloat *y,
    const QLTEN_ComplexFloat a = 1.0f
) {
  // Get cuBLAS handle
  cublasHandle_t &handle = CublasHandleManager::GetHandle();

  // Copy and scale complex vector
  cublasStatus_t status;

  // Perform y = x and then y = a * y (scale)
  status = cublasCcopy(handle,
                       size,
                       reinterpret_cast<const cuComplex *>(x),
                       1,
                       reinterpret_cast<cuComplex *>(y),
                       1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasCcopy failed!" << std::endl;
  }

  status = cublasCscal(handle,
                       size,
                       reinterpret_cast<const cuComplex *>(&a),
                       reinterpret_cast<cuComplex *>(y),
                       1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasCscal failed!" << std::endl;
  }

#ifdef QLTEN_COUNT_FLOPS
  flop += 4 * size;
#endif
}

inline void VectorCopy(
    const QLTEN_Double *source,
    const size_t size,
    QLTEN_Double *dest
) {
  // Get cuBLAS handle
  cublasHandle_t &handle = CublasHandleManager::GetHandle();
  // Copy vector
  cublasStatus_t status = cublasDcopy(handle, size, source, 1, dest, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasDcopy failed!" << std::endl;
  }
}

inline void VectorCopy(
    const QLTEN_Float *source,
    const size_t size,
    QLTEN_Float *dest
) {
  // Get cuBLAS handle
  cublasHandle_t &handle = CublasHandleManager::GetHandle();
  // Copy vector
  cublasStatus_t status = cublasScopy(handle, size, source, 1, dest, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasScopy failed!" << std::endl;
  }
}

inline void VectorCopy(
    const QLTEN_Complex *source,
    const size_t size,
    QLTEN_Complex *dest
) {
  // Get cuBLAS handle
  cublasHandle_t &handle = CublasHandleManager::GetHandle();

  // Copy complex vector
  cublasStatus_t status = cublasZcopy(handle,
                                      size,
                                      reinterpret_cast<const cuDoubleComplex *>(source),
                                      1,
                                      reinterpret_cast<cuDoubleComplex *>(dest),
                                      1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasZcopy failed!" << std::endl;
  }
}

inline void VectorCopy(
    const QLTEN_ComplexFloat *source,
    const size_t size,
    QLTEN_ComplexFloat *dest
) {
  // Get cuBLAS handle
  cublasHandle_t &handle = CublasHandleManager::GetHandle();

  // Copy complex vector
  cublasStatus_t status = cublasCcopy(handle,
                                      size,
                                      reinterpret_cast<const cuComplex *>(source),
                                      1,
                                      reinterpret_cast<cuComplex *>(dest),
                                      1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasCcopy failed!" << std::endl;
  }
}

inline void VectorScale(
    QLTEN_Double *x,
    const size_t size,
    const QLTEN_Double a
) {
  // Get cuBLAS handle
  cublasHandle_t &handle = CublasHandleManager::GetHandle();

  // Scale vector
  cublasStatus_t status = cublasDscal(handle, size, &a, x, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasDscal failed!" << std::endl;
  }

#ifdef QLTEN_COUNT_FLOPS
  flop += size;
#endif
}

inline void VectorScale(
    QLTEN_Float *x,
    const size_t size,
    const QLTEN_Float a
) {
  // Get cuBLAS handle
  cublasHandle_t &handle = CublasHandleManager::GetHandle();

  // Scale vector
  cublasStatus_t status = cublasSscal(handle, size, &a, x, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasSscal failed!" << std::endl;
  }

#ifdef QLTEN_COUNT_FLOPS
  flop += size;
#endif
}

inline void VectorScale(
    QLTEN_Complex *x,
    const size_t size,
    const QLTEN_Complex a
) {
  // Get cuBLAS handle
  cublasHandle_t &handle = CublasHandleManager::GetHandle();

  // Scale complex vector
  cublasStatus_t status = cublasZscal(handle,
                                      size,
                                      reinterpret_cast<const cuDoubleComplex *>(&a),
                                      reinterpret_cast<cuDoubleComplex *>(x),
                                      1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasZscal failed!" << std::endl;
  }

#ifdef QLTEN_COUNT_FLOPS
  flop += 4 * size;
#endif
}

inline void VectorScale(
    QLTEN_ComplexFloat *x,
    const size_t size,
    const QLTEN_ComplexFloat a
) {
  // Get cuBLAS handle
  cublasHandle_t &handle = CublasHandleManager::GetHandle();

  // Scale complex vector
  cublasStatus_t status = cublasCscal(handle,
                                      size,
                                      reinterpret_cast<const cuComplex *>(&a),
                                      reinterpret_cast<cuComplex *>(x),
                                      1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasCscal failed!" << std::endl;
  }

#ifdef QLTEN_COUNT_FLOPS
  flop += 4 * size;
#endif
}

inline double Vector2Norm(
    QLTEN_Double *x,
    const size_t size
) {
  // Get cuBLAS handle
  cublasHandle_t &handle = CublasHandleManager::GetHandle();

  // Compute 2-norm
  double norm;
  cublasStatus_t status = cublasDnrm2(handle, size, x, 1, &norm);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasDnrm2 failed!" << std::endl;
  }

#ifdef QLTEN_COUNT_FLOPS
  flop += 2 * size;
#endif

  return norm;
}

inline float Vector2Norm(
    QLTEN_Float *x,
    const size_t size
) {
  // Get cuBLAS handle
  cublasHandle_t &handle = CublasHandleManager::GetHandle();

  // Compute 2-norm
  float norm;
  cublasStatus_t status = cublasSnrm2(handle, size, x, 1, &norm);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasSnrm2 failed!" << std::endl;
  }

#ifdef QLTEN_COUNT_FLOPS
  flop += 2 * size;
#endif

  return norm;
}

inline double Vector2Norm(
    QLTEN_Complex *x,
    const size_t size
) {
  // Get cuBLAS handle
  cublasHandle_t &handle = CublasHandleManager::GetHandle();

  // Compute 2-norm of complex vector
  double norm;
  cublasStatus_t status = cublasDznrm2(handle, size, reinterpret_cast<const cuDoubleComplex *>(x), 1, &norm);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasDznrm2 failed!" << std::endl;
  }

#ifdef QLTEN_COUNT_FLOPS
  flop += 8 * size;
#endif

  return norm;
}

inline float Vector2Norm(
    QLTEN_ComplexFloat *x,
    const size_t size
) {
  // Get cuBLAS handle
  cublasHandle_t &handle = CublasHandleManager::GetHandle();

  // Compute 2-norm of complex vector
  float norm;
  cublasStatus_t status = cublasScnrm2(handle, size, reinterpret_cast<const cuComplex *>(x), 1, &norm);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasScnrm2 failed!" << std::endl;
  }

#ifdef QLTEN_COUNT_FLOPS
  flop += 8 * size;
#endif

  return norm;
}

// CUDA kernel to convert real array to complex array
__global__ 
inline void VectorRealToCplxKernel(
    const QLTEN_Double *real, QLTEN_Complex *cplx, size_t size
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    cplx[idx] = QLTEN_Complex(real[idx], 0.0); // Set imaginary part to 0
  }
}

__global__ 
inline void VectorRealToCplxKernel(
    const QLTEN_Float *real, QLTEN_ComplexFloat *cplx, size_t size
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    cplx[idx] = QLTEN_ComplexFloat(real[idx], 0.0f); // Set imaginary part to 0
  }
}

inline void VectorRealToCplx(
    const QLTEN_Double *real,
    const size_t size,
    QLTEN_Complex *cplx
) {
  int threadsPerBlock = 256; // Number of threads per block
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  VectorRealToCplxKernel<<<blocksPerGrid, threadsPerBlock>>>(real, cplx, size);

  // Synchronize to ensure kernel execution completes
  cudaDeviceSynchronize();
}

inline void VectorRealToCplx(
    const QLTEN_Float *real,
    const size_t size,
    QLTEN_ComplexFloat *cplx
) {
  int threadsPerBlock = 256; // Number of threads per block
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  VectorRealToCplxKernel<<<blocksPerGrid, threadsPerBlock>>>(real, cplx, size);

  // Synchronize to ensure kernel execution completes
  cudaDeviceSynchronize();
}

// VectorSumSquares for Double precision
inline double VectorSumSquares(
    const QLTEN_Double *x,
    const size_t size
) {
  // Get cuBLAS handle
  cublasHandle_t &handle = CublasHandleManager::GetHandle();

  // Compute sum of squares
  double sum_squares;
  cublasStatus_t status = cublasDdot(handle, size, x, 1, x, 1, &sum_squares);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasDdot failed!" << std::endl;
  }

#ifdef QLTEN_COUNT_FLOPS
  flop += 2 * size;
#endif

  return sum_squares;
}

inline float VectorSumSquares(
    const QLTEN_Float *x,
    const size_t size
) {
  // Get cuBLAS handle
  cublasHandle_t &handle = CublasHandleManager::GetHandle();

  // Compute sum of squares
  float sum_squares;
  cublasStatus_t status = cublasSdot(handle, size, x, 1, x, 1, &sum_squares);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasSdot failed!" << std::endl;
  }

#ifdef QLTEN_COUNT_FLOPS
  flop += 2 * size;
#endif

  return sum_squares;
}

// VectorSumSquares for Complex precision
inline double VectorSumSquares(
    const QLTEN_Complex *x,
    const size_t size
) {
  // Get cuBLAS handle
  cublasHandle_t &handle = CublasHandleManager::GetHandle();

  // Compute sum of squares of complex elements
  double sum_squares_real, sum_squares_imag;

  cublasStatus_t status_real = cublasDdot(handle,
                                          size,
                                          reinterpret_cast<const double *>(x),
                                          2,
                                          reinterpret_cast<const double *>(x),
                                          2,
                                          &sum_squares_real);
  cublasStatus_t status_imag = cublasDdot(handle,
                                          size,
                                          reinterpret_cast<const double *>(x) + 1,
                                          2,
                                          reinterpret_cast<const double *>(x) + 1,
                                          2,
                                          &sum_squares_imag);

  if (status_real != CUBLAS_STATUS_SUCCESS || status_imag != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasDdot failed!" << std::endl;
  }

  return sum_squares_real + sum_squares_imag;
#ifdef QLTEN_COUNT_FLOPS
  flop += 4 * size;
#endif
}

inline float VectorSumSquares(
    const QLTEN_ComplexFloat *x,
    const size_t size
) {
  // Get cuBLAS handle
  cublasHandle_t &handle = CublasHandleManager::GetHandle();

  // Compute sum of squares of complex elements
  float sum_squares_real, sum_squares_imag;

  cublasStatus_t status_real = cublasSdot(handle,
                                          size,
                                          reinterpret_cast<const float *>(x),
                                          2,
                                          reinterpret_cast<const float *>(x),
                                          2,
                                          &sum_squares_real);
  cublasStatus_t status_imag = cublasSdot(handle,
                                          size,
                                          reinterpret_cast<const float *>(x) + 1,
                                          2,
                                          reinterpret_cast<const float *>(x) + 1,
                                          2,
                                          &sum_squares_imag);

  if (status_real != CUBLAS_STATUS_SUCCESS || status_imag != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasSdot failed!" << std::endl;
  }

  return sum_squares_real + sum_squares_imag;
#ifdef QLTEN_COUNT_FLOPS
  flop += 4 * size;
#endif
}
#endif // USE GPU

} /* hp_numeric */
} /* qlten */


#endif /* ifndef QLTEN_FRAMEWORK_HP_NUMERIC_BLAS_LEVEL1_H */
