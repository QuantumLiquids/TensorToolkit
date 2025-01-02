// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Author: Haoxin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2020-11-28 19:07
*
* Description: QuantumLiquids/tensor project. High performance BLAS Level 3 related
* functions based on MKL, OpenBLAS or CUBLAS
*/

/**
@file blas_level3.h
@brief High performance BLAS Level 3 related functions based on MKL.
*/
#ifndef QLTEN_FRAMEWORK_HP_NUMERIC_BLAS_LEVEL3_H
#define QLTEN_FRAMEWORK_HP_NUMERIC_BLAS_LEVEL3_H

#include <cassert>                        // assert
#include "qlten/framework/value_t.h"      // QLTEN_Double, QLTEN_Complex
#include "qlten/framework/flops_count.h"  // flop

#ifdef Release
#define NDEBUG
#endif

#ifndef USE_GPU     // use CPU
#ifndef USE_OPENBLAS

#include "mkl.h"      // cblas_*axpy, cblas_*scal

#else

#include <cblas.h>

#endif
#else
#include <cublas_v2.h>
#endif
namespace qlten {

/// High performance numerical functions.
namespace hp_numeric {
#ifndef USE_GPU     // use CPU
///< C = alpha * A * B + beta * C
inline void MatMultiply(
    const double alpha,
    const QLTEN_Double *a,
    const QLTEN_Double *b,
    const size_t m,
    const size_t k,
    const size_t n,
    const QLTEN_Double beta,
    QLTEN_Double *c) {
  cblas_dgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      m, n, k,
      alpha,
      a, k,
      b, n,
      beta,
      c, n
  );
#ifdef QLTEN_COUNT_FLOPS
  flop += m * n * (2 * k + 2);
#endif
}

/**
 * C = alpha * A * B + beta * C
 *
 * @param alpha alpha = 1 for bosonic and = \pm 1 for fermionic
 */
inline void MatMultiply(
    const double alpha,
    const QLTEN_Complex *a,
    const QLTEN_Complex *b,
    const size_t m,
    const size_t k,
    const size_t n,
    const QLTEN_Complex beta,
    QLTEN_Complex *c) {
  QLTEN_Complex alpha_complex(alpha);
  cblas_zgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      m, n, k,
      &alpha_complex,
      a, k,
      b, n,
      &beta,
      c, n
  );
#ifdef QLTEN_COUNT_FLOPS
  flop += m * n * (8 * k + 8);
#endif
}

inline void MatMultiply(
    const double alpha,
    const QLTEN_Double *a,
    const CBLAS_TRANSPOSE cblas_transpose_a,
    const QLTEN_Double *b,
    const CBLAS_TRANSPOSE cblas_transpose_b,
    const size_t m,
    const size_t k,
    const size_t n,
    const size_t lda,
    const size_t ldb,
    const QLTEN_Double beta,
    QLTEN_Double *c) {
  cblas_dgemm(
      CblasRowMajor, cblas_transpose_a, cblas_transpose_b,
      m, n, k,
      alpha,
      a, lda,
      b, ldb,
      beta,
      c, n
  );
#ifdef QLTEN_COUNT_FLOPS
  flop += m * n * (2 * k + 2);
#endif
}

inline void MatMultiply(
    const double alpha,
    const QLTEN_Complex *a,
    const CBLAS_TRANSPOSE cblas_transpose_a,
    const QLTEN_Complex *b,
    const CBLAS_TRANSPOSE cblas_transpose_b,
    const size_t m,
    const size_t k,
    const size_t n,
    const size_t lda,
    const size_t ldb,
    const QLTEN_Complex beta,
    QLTEN_Complex *c) {
  QLTEN_Complex alpha_complex(alpha);
  cblas_zgemm(
      CblasRowMajor, cblas_transpose_a, cblas_transpose_b,
      m, n, k,
      &alpha_complex,
      a, lda,
      b, ldb,
      &beta,
      c, n
  );
#ifdef QLTEN_COUNT_FLOPS
  flop += m * n * (8 * k + 8);
#endif
}

#ifndef USE_OPENBLAS

inline void MatMultiplyBatch(
    const QLTEN_Double **a_array, const QLTEN_Double **b_array,
    const MKL_INT *m_array, const MKL_INT *k_array, const MKL_INT *n_array,
    const QLTEN_Double *beta_array,
    QLTEN_Double **c_array,
    const MKL_INT group_count) {

  const CBLAS_LAYOUT Layout = CblasRowMajor;
  const MKL_INT *lda_array = k_array;
  const MKL_INT *ldb_array = n_array;
  const MKL_INT *ldc_array = n_array;

#ifdef QLTEN_USE_MKL_GEMM_BATCH
  // NOTE: DONOT use this part code now, except contracting one index
  // because when c_array has some same elements (c_array[i]==c_array[j] with i!=j),
  //different CPU&caches will load&read the data at the same time.
  CBLAS_TRANSPOSE* transa_array = (CBLAS_TRANSPOSE *) malloc(group_count* sizeof(CBLAS_TRANSPOSE));
  CBLAS_TRANSPOSE* transb_array = (CBLAS_TRANSPOSE *) malloc(group_count* sizeof(CBLAS_TRANSPOSE));
  for (MKL_INT i = 0 ; i < group_count ; i++){
    transa_array[i] = CblasNoTrans;
  }
  for (MKL_INT i = 0 ; i < group_count ; i++){
    transb_array[i] = CblasNoTrans;
  }

  QLTEN_Double *alpha_array = (QLTEN_Double *) malloc(group_count* sizeof(QLTEN_Double));
  for (MKL_INT i = 0 ; i < group_count ; i++){
    alpha_array[i] = 1.0;
  }

  MKL_INT* group_size = (MKL_INT *) malloc(group_count* sizeof(MKL_INT));
  for (MKL_INT i = 0 ; i < group_count ; i++){
    group_size[i] = 1;
  }

  cblas_dgemm_batch (
      Layout,
      transa_array, transb_array,
      m_array, n_array, k_array,
      alpha_array,
      a_array, lda_array,
      b_array, ldb_array,
      beta_array,
      c_array, ldc_array,
      group_count,
      group_size);

  free(transa_array);
  free(transb_array);
  free(alpha_array);
  free(group_size);
#else // Use direct gemm loop.

  auto idx = 0;
  for (MKL_INT i = 0; i < group_count; ++i) {
    for (MKL_INT j = 0; j < 1; ++j) {
      cblas_dgemm(
          Layout,
          CblasNoTrans, CblasNoTrans,
          m_array[i], n_array[i], k_array[i],
          1.0,
          a_array[idx], lda_array[i],
          b_array[idx], ldb_array[i],
          beta_array[i],
          c_array[idx], ldc_array[i]);
      ++idx;
#ifdef QLTEN_COUNT_FLOPS
      flop += m_array[i] * n_array[i] * (2 * k_array[i] + 2);
#endif
    }
}

#endif
}

inline void MatMultiplyBatch(
    const QLTEN_Complex **a_array,
    const QLTEN_Complex **b_array,
    const MKL_INT *m_array, const MKL_INT *k_array, const MKL_INT *n_array,
    const QLTEN_Complex *beta_array,
    QLTEN_Complex **c_array,
    const MKL_INT group_count) {

  const CBLAS_LAYOUT Layout = CblasRowMajor;

  const MKL_INT *lda_array = k_array;
  const MKL_INT *ldb_array = n_array;
  const MKL_INT *ldc_array = n_array;

#ifdef QLTEN_USE_MKL_GEMM_BATCH
  // NOTE: DONOT use this part code now, except contracting one index
  // because when c_array has some same elements (c_array[i]==c_array[j] with i!=j),
  //different CPU&caches will load&read the data at the same time.
  CBLAS_TRANSPOSE* transa_array = (CBLAS_TRANSPOSE *) malloc(group_count* sizeof(CBLAS_TRANSPOSE));
  CBLAS_TRANSPOSE* transb_array = (CBLAS_TRANSPOSE *) malloc(group_count* sizeof(CBLAS_TRANSPOSE));
  for (MKL_INT i = 0 ; i < group_count ; i++){
    transa_array[i] = CblasNoTrans;
  }
  for (MKL_INT i = 0 ; i < group_count ; i++){
    transb_array[i] = CblasNoTrans;
  }

  QLTEN_Complex *alpha_array = (QLTEN_Complex *) malloc(group_count* sizeof(QLTEN_Complex));
  for (MKL_INT i = 0 ; i < group_count ; i++){
    alpha_array[i] = QLTEN_Complex(1.0);
  }

  MKL_INT* group_size = (MKL_INT *) malloc(group_count* sizeof(MKL_INT));
  for (MKL_INT i = 0 ; i < group_count ; i++){
    group_size[i] = 1;
  }

  const void** a_array_void_pointer = (const void** ) malloc(group_count* sizeof(void*));
  for(size_t i=0;i < group_count; i++){
    a_array_void_pointer[i] = (const void*) a_array[i];
  }
  const void** b_array_void_pointer = (const void** ) malloc(group_count* sizeof(void*));
  for(size_t i=0;i < group_count; i++){
    b_array_void_pointer[i] = (const void*) b_array[i];
  }
  void** c_array_void_pointer = (void** ) malloc(group_count* sizeof(void*));
  for(size_t i=0;i < group_count; i++){
    c_array_void_pointer[i] = (void*) c_array[i];
  }

  cblas_zgemm_batch (
      Layout,
      transa_array, transb_array,
      m_array, n_array, k_array,
      alpha_array,
      a_array_void_pointer, lda_array,
      b_array_void_pointer, ldb_array,
      beta_array,
      c_array_void_pointer, ldc_array,
      group_count,
      group_size);

  free(a_array_void_pointer);
  free(b_array_void_pointer);
  free(c_array_void_pointer);

  free(transa_array);
  free(transb_array);
  free(alpha_array);
  free(group_size);
#else // Use direct gemm loop.
  QLTEN_Complex alpha = QLTEN_Complex(1.0);
  auto idx = 0;
  for (MKL_INT i = 0; i < group_count; ++i) {
    for (MKL_INT j = 0; j < 1; ++j) {
      cblas_zgemm(
          Layout,
          CblasNoTrans, CblasNoTrans,
          m_array[i], n_array[i], k_array[i],
          &alpha,
          a_array[idx], lda_array[i],
          b_array[idx], ldb_array[i],
          &beta_array[i],
          c_array[idx], ldc_array[i]);
      ++idx;
#ifdef QLTEN_COUNT_FLOPS
      flop += m_array[i] * n_array[i] * (8 * k_array[i] + 8);
#endif
    }
  }
#endif
}

#endif
#else

const char *cublasGetErrorString(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    default: return "Unknown cuBLAS error";
  }
}

#ifndef NDEBUG
//row-major matrix A, m: rows, n: cols; lda: leading-dimension, usually be n
inline void print_matrix2(const int &m, const int &n, const double *A, const int &lda) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      std::printf("%0.10f ", A[i * lda + j]);
    }
    std::printf("\n");
  }
}
#endif

///< C = alpha * A * B + beta * C
inline void MatMultiply(
    const double alpha,
    const QLTEN_Double *a,
    const QLTEN_Double *b,
    const size_t m,
    const size_t k,
    const size_t n,
    const QLTEN_Double beta,
    QLTEN_Double *c) {

  // Use cuBLAS with column-major matrices, so we transpose the inputs
  cublasHandle_t &handle = CublasHandleManager::GetHandle();

  // reinterpret the Row-major matrix data
  // as a Column-major matrix which operated by transpose upon original Row-major matrix
  auto status = cublasDgemm(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      n, m, k,
      &alpha,
      b, n,
      a, k,
      &beta,
      c, n
  );
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasDgemm failed: " << cublasGetErrorString(status) << std::endl;
  }
#ifndef NDEBUG
//  double *h_a;
//  double *h_b;
//  double *h_c;
//
//  h_a = (QLTEN_Double *) malloc((k * m) * sizeof(QLTEN_Double));
//  h_b = (QLTEN_Double *) malloc(k * n * sizeof(QLTEN_Double));
//  h_c = (QLTEN_Double *) malloc((m * n) * sizeof(QLTEN_Double));
//
//  cudaMemcpy(h_a, a, (k * m) * sizeof(QLTEN_Double), cudaMemcpyDeviceToHost);
//  cudaMemcpy(h_b, b, (k * n) * sizeof(QLTEN_Double), cudaMemcpyDeviceToHost);
//  cudaMemcpy(h_c, c, (m * n) * sizeof(QLTEN_Double), cudaMemcpyDeviceToHost);
//  std::cout << std::endl;
//  std::cout << "A matrix:" << std::endl;
//  print_matrix2(m, k, h_a, k);
//  std::cout << std::endl;
//
//  std::cout << "B matrix:" << std::endl;
//  print_matrix2(k, n, h_b, n);
//  std::cout << std::endl;
//  std::cout << "C matrix:" << std::endl;
//  print_matrix2(m, n, h_c, n);
//  std::cout << std::endl;
//  free(h_a);
//  free(h_b);
//  free(h_c);
#endif
#ifdef QLTEN_COUNT_FLOPS
  flop += 2 * m * n * k;  // Adjust FLOP count based on operations
#endif
}

///< C = alpha * A * B + beta * C
inline void MatMultiply(
    const double alpha,
    const QLTEN_Complex *a,
    const QLTEN_Complex *b,
    const size_t m,
    const size_t k,
    const size_t n,
    const QLTEN_Complex beta,
    QLTEN_Complex *c) {

  // Use cuBLAS with column-major matrices, so we transpose the inputs
  cublasHandle_t &handle = CublasHandleManager::GetHandle();
  const QLTEN_Complex alpha_complex(alpha);
  // Transpose A and B, then multiply
  auto status = cublasZgemm(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      n, m, k,
      reinterpret_cast<const cuDoubleComplex *>(&alpha_complex),
      reinterpret_cast<const cuDoubleComplex *>(b), n,
      reinterpret_cast<const cuDoubleComplex *>(a), k,
      reinterpret_cast<const cuDoubleComplex *>(&beta),
      reinterpret_cast<cuDoubleComplex *>(c), n
  );

#ifdef QLTEN_COUNT_FLOPS
  flop += m * n * k * 8;  // Adjust FLOP count based on operations
#endif
}

inline void MatMultiply(
    const double alpha,
    const QLTEN_Double *a,
    const cublasOperation_t cblas_transpose_a,
    const QLTEN_Double *b,
    const cublasOperation_t cblas_transpose_b,
    const size_t m,
    const size_t k,
    const size_t n,
    const size_t lda,
    const size_t ldb,
    const QLTEN_Double beta,
    QLTEN_Double *c) {

  // Use cuBLAS with column-major matrices, so we transpose the inputs
  cublasHandle_t &handle = CublasHandleManager::GetHandle();

  // reinterpret the Row-major matrix data
  // as a Column-major matrix which operated by transpose upon original Row-major matrix
  auto status = cublasDgemm(
      handle,
      cblas_transpose_b, cblas_transpose_a,
      n, m, k,
      &alpha,
      b, ldb,
      a, lda,
      &beta,
      c, n
  );
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasDgemm failed!" << std::endl;
  }
#ifdef QLTEN_COUNT_FLOPS
  flop += m * n * (2 * k + 2);
#endif
}

inline void MatMultiply(
    const double alpha,
    const QLTEN_Complex *a,
    const cublasOperation_t cblas_transpose_a,
    const QLTEN_Complex *b,
    const cublasOperation_t cblas_transpose_b,
    const size_t m,
    const size_t k,
    const size_t n,
    const size_t lda,
    const size_t ldb,
    const QLTEN_Complex beta,
    QLTEN_Complex *c) {
  // Use cuBLAS with column-major matrices, so we transpose the inputs
  cublasHandle_t &handle = CublasHandleManager::GetHandle();
  const QLTEN_Complex alpha_complex(alpha);
  // Transpose A and B, then multiply
  auto status = cublasZgemm(
      handle,
      cblas_transpose_b, cblas_transpose_a,
      n, m, k,
      reinterpret_cast<const cuDoubleComplex *>(&alpha_complex),
      reinterpret_cast<const cuDoubleComplex *>(b), ldb,
      reinterpret_cast<const cuDoubleComplex *>(a), lda,
      reinterpret_cast<const cuDoubleComplex *>(&beta),
      reinterpret_cast<cuDoubleComplex *>(c), n
  );
#ifdef QLTEN_COUNT_FLOPS
  flop += m * n * (8 * k + 8);
#endif
}

#endif  // USE_GPU

} /* hp_numeric */
} /* qlten */
#endif /* ifndef QLTEN_FRAMEWORK_HP_NUMERIC_BLAS_LEVEL3_H */
