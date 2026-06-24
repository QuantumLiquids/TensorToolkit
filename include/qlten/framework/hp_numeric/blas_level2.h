// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2026-06-24
*
* Description: QuantumLiquids/tensor project. High performance BLAS Level 2
* related functions based on MKL, OpenBlas, AOCL/BLIS or KML.
*/

/**
@file blas_level2.h
@brief High performance BLAS Level 2 related functions.
*/
#ifndef QLTEN_FRAMEWORK_HP_NUMERIC_BLAS_LEVEL2_H
#define QLTEN_FRAMEWORK_HP_NUMERIC_BLAS_LEVEL2_H

#include <cstddef>                          // size_t
#include "qlten/framework/value_t.h"        // QLTEN_Double, QLTEN_Complex
#include "qlten/framework/flops_count.h"    // flop


#ifndef USE_GPU
#include "qlten/framework/hp_numeric/backend_selector.h"
#endif

namespace qlten {
/// High performance numerical functions.
namespace hp_numeric {

#ifndef USE_GPU
/**
 * General rank-1 update of a row-major matrix:
 * \f[ A \leftarrow \alpha\, x\, y^{T} + A, \f]
 * where `a` is a `rows` by `cols` matrix stored row-major with leading
 * dimension `cols`, `x` is a length-`rows` vector with stride `incx`, and
 * `y` is a length-`cols` vector with stride `incy`. Wraps the BLAS
 * `?ger`/`?geru` routines.
 */
inline void MatrixRankOneUpdate(
    const QLTEN_Double alpha,
    const QLTEN_Double *x,
    const size_t incx,
    const QLTEN_Double *y,
    const size_t incy,
    const size_t rows,
    const size_t cols,
    QLTEN_Double *a
) {
  cblas_dger(
      CblasRowMajor, rows, cols,
      alpha,
      x, incx,
      y, incy,
      a, cols
  );
#ifdef QLTEN_COUNT_FLOPS
  flop += 2 * rows * cols;
#endif
}

inline void MatrixRankOneUpdate(
    const QLTEN_Float alpha,
    const QLTEN_Float *x,
    const size_t incx,
    const QLTEN_Float *y,
    const size_t incy,
    const size_t rows,
    const size_t cols,
    QLTEN_Float *a
) {
  cblas_sger(
      CblasRowMajor, rows, cols,
      alpha,
      x, incx,
      y, incy,
      a, cols
  );
#ifdef QLTEN_COUNT_FLOPS
  flop += 2 * rows * cols;
#endif
}

inline void MatrixRankOneUpdate(
    const QLTEN_Complex alpha,
    const QLTEN_Complex *x,
    const size_t incx,
    const QLTEN_Complex *y,
    const size_t incy,
    const size_t rows,
    const size_t cols,
    QLTEN_Complex *a
) {
  cblas_zgeru(
      CblasRowMajor, rows, cols,
      &alpha,
      x, incx,
      y, incy,
      a, cols
  );
#ifdef QLTEN_COUNT_FLOPS
  flop += 8 * rows * cols;
#endif
}

inline void MatrixRankOneUpdate(
    const QLTEN_ComplexFloat alpha,
    const QLTEN_ComplexFloat *x,
    const size_t incx,
    const QLTEN_ComplexFloat *y,
    const size_t incy,
    const size_t rows,
    const size_t cols,
    QLTEN_ComplexFloat *a
) {
  cblas_cgeru(
      CblasRowMajor, rows, cols,
      &alpha,
      x, incx,
      y, incy,
      a, cols
  );
#ifdef QLTEN_COUNT_FLOPS
  flop += 8 * rows * cols;
#endif
}
#endif // USE_GPU

} /* hp_numeric */
} /* qlten */

#endif /* ifndef QLTEN_FRAMEWORK_HP_NUMERIC_BLAS_LEVEL2_H */
