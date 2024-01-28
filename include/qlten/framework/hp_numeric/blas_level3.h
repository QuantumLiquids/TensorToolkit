// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-28 19:07
*
* Description: QuantumLiquids/tensor project. High performance BLAS Level 3 related
* functions based on MKL.
*/

/**
@file blas_level3.h
@brief High performance BLAS Level 3 related functions based on MKL.
*/
#ifndef QLTEN_FRAMEWORK_HP_NUMERIC_BLAS_LEVEL3_H
#define QLTEN_FRAMEWORK_HP_NUMERIC_BLAS_LEVEL3_H

#include "qlten/framework/value_t.h"      // QLTEN_Double, QLTEN_Complex
#include "qlten/framework/flops_count.h"  // flop

#ifdef Release
#define NDEBUG
#endif
#include <cassert>     // assert

#ifndef USE_OPENBLAS

#include "mkl.h"      // cblas_*axpy, cblas_*scal

#else

#include <cblas.h>

#endif
namespace qlten {

/// High performance numerical functions.
namespace hp_numeric {

inline void MatMultiply(
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
      1.0,
      a, k,
      b, n,
      beta,
      c, n
  );
#ifdef QLTEN_COUNT_FLOPS
  flop += m * n * (2 * k + 2);
#endif
}

inline void MatMultiply(
    const QLTEN_Complex *a,
    const QLTEN_Complex *b,
    const size_t m,
    const size_t k,
    const size_t n,
    const QLTEN_Complex beta,
    QLTEN_Complex *c) {
  QLTEN_Complex alpha(1.0);
  cblas_zgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      m, n, k,
      &alpha,
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
      1.0,
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
  QLTEN_Complex alpha(1.0);
  cblas_zgemm(
      CblasRowMajor, cblas_transpose_a, cblas_transpose_b,
      m, n, k,
      &alpha,
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

} /* hp_numeric */
} /* qlten */
#endif /* ifndef QLTEN_FRAMEWORK_HP_NUMERIC_BLAS_LEVEL3_H */
