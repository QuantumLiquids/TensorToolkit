// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-24 20:13
*
* Description: QuantumLiquids/tensor project. High performance BLAS Level 1 related
* functions based on MKL.
*/

/**
@file blas_level1.h
@brief High performance BLAS Level 1 related functions based on MKL.
*/
#ifndef QLTEN_FRAMEWORK_HP_NUMERIC_BLAS_LEVEL1_H
#define QLTEN_FRAMEWORK_HP_NUMERIC_BLAS_LEVEL1_H

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

inline void VectorCopy(
    const QLTEN_Double *source,
    const size_t size,
    QLTEN_Double *dest
) {
  cblas_dcopy(size, source, 1, dest, 1);
}

inline void VectorCopy(
    const QLTEN_Complex *source,
    const size_t size,
    QLTEN_Complex *dest
) {
  cblas_zcopy(size, source, 1, dest, 1);
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
    QLTEN_Complex *x,
    const size_t size,
    const QLTEN_Complex a
) {
  cblas_zscal(size, &a, x, 1);
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

inline double Vector2Norm(
    QLTEN_Complex *x,
    const size_t size
) {
  return cblas_dznrm2(size, x, 1);
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

} /* hp_numeric */
} /* qlten */


#endif /* ifndef QLTEN_FRAMEWORK_HP_NUMERIC_BLAS_LEVEL1_H */
