// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-12-02 14:15
* 
* Description: QuantumLiquids/tensor project. High performance LAPACK related functions
* based on MKL.
*/

/**
@file lapack.h
@brief High performance LAPACK related functions based on MKL.
*/
#ifndef QLTEN_FRAMEWORK_HP_NUMERIC_LAPACK_H
#define QLTEN_FRAMEWORK_HP_NUMERIC_LAPACK_H

#include "qlten/framework/value_t.h"
#include "qlten/framework/flops_count.h"  // flop

#include <algorithm>    // min
#include <cstring>      // memcpy, memset

#ifdef Release
#define NDEBUG
#endif

#include <cassert>     // assert

#ifndef USE_OPENBLAS

#include "mkl.h"      // cblas_*axpy, cblas_*scal

#else

#include <cblas.h>
#include <lapacke.h>

#endif

namespace qlten {

namespace hp_numeric {

inline lapack_int MatSVD(
    QLTEN_Double *mat,
    const size_t m, const size_t n,
    QLTEN_Double *&u,
    QLTEN_Double *&s,
    QLTEN_Double *&vt
) {
  auto lda = n;
  size_t ldu = std::min(m, n);
  size_t ldvt = n;
  u = (QLTEN_Double *) malloc((ldu * m) * sizeof(QLTEN_Double));
  s = (QLTEN_Double *) malloc(ldu * sizeof(QLTEN_Double));
  vt = (QLTEN_Double *) malloc((ldvt * ldu) * sizeof(QLTEN_Double));
#ifdef FAST_SVD
  auto info = LAPACKE_dgesdd(
      LAPACK_ROW_MAJOR, 'S',
      m, n,
      mat, lda,
      s,
      u, ldu,
      vt, ldvt
  );
#else // More stable
  double *superb = new double[m];
  auto info = LAPACKE_dgesvd(
      LAPACK_ROW_MAJOR, 'S', 'S',
      m, n,
      mat, lda,
      s,
      u, ldu,
      vt, ldvt,
      superb
  );
  delete[] superb;
#endif
  assert(info == 0);
#ifdef QLTEN_COUNT_FLOPS
  flop += 4 * m * n * n - 4 * n * n * n / 3;
  // a rough estimation
#endif
  return info;
}

inline lapack_int MatSVD(
    QLTEN_Complex *mat,
    const size_t m, const size_t n,
    QLTEN_Complex *&u,
    QLTEN_Double *&s,
    QLTEN_Complex *&vt
) {

  auto lda = n;
  size_t ldu = std::min(m, n);
  size_t ldvt = n;
  u = (QLTEN_Complex *) malloc((ldu * m) * sizeof(QLTEN_Complex));
  s = (QLTEN_Double *) malloc(ldu * sizeof(QLTEN_Double));
  vt = (QLTEN_Complex *) malloc((ldvt * ldu) * sizeof(QLTEN_Complex));
#ifdef FAST_SVD
  auto info = LAPACKE_zgesdd(
      LAPACK_ROW_MAJOR, 'S',
      m, n,
      reinterpret_cast<lapack_complex_double *>(mat), lda,
      s,
      reinterpret_cast<lapack_complex_double *>(u), ldu,
      reinterpret_cast<lapack_complex_double *>(vt), ldvt
  );
#else // stable
  double *superb = new double[m];
  auto info = LAPACKE_zgesvd(
      LAPACK_ROW_MAJOR, 'S', 'S',
      m, n,
      reinterpret_cast<lapack_complex_double *>(mat), lda,
      s,
      reinterpret_cast<lapack_complex_double *>(u), ldu,
      reinterpret_cast<lapack_complex_double *>(vt), ldvt,
      superb
  );
  delete[] superb;

#endif
  assert(info == 0);
#ifdef QLTEN_COUNT_FLOPS
  flop += 8 * m * n * n - 8 * n * n * n / 3;
  // a rough estimation
#endif
  return info;
}

// eigen value decomposition for symmetric/hermitian matrix
// the memory of d an u matrices are allocated outside the function
// d should be initialized as zero matrix
inline lapack_int SymMatEVD(
    const QLTEN_Double *mat,
    const size_t n,
    QLTEN_Double *d, // eigen values but in matrix form
    QLTEN_Double *u  // the column is the eigen vector
) {
  QLTEN_Double *eigenvalues = (QLTEN_Double *) malloc(n * sizeof(QLTEN_Double));
  memcpy(u, mat, n * n * sizeof(QLTEN_Double));

  auto info = LAPACKE_dsyev(
      LAPACK_ROW_MAJOR, 'V', 'U', // Compute eigenvalues and eigenvectors, upper triangle
      n,
      u, n,
      eigenvalues
  );

  assert(info == 0);
  for (size_t i = 0; i < n; i++) {
    d[i * n + i] = eigenvalues[i];
  }
  free(eigenvalues);
#ifdef QLTEN_COUNT_FLOPS
  flop += 4 * n * n * n / 3;
  // a rough estimation for EVD
#endif
  return info;
}

inline lapack_int SymMatEVD(
    const QLTEN_Complex *mat,
    const size_t n,
    QLTEN_Double *d,
    QLTEN_Complex *u
) {
  QLTEN_Double *eigenvalues = (QLTEN_Double *) malloc(n * sizeof(QLTEN_Double));
  memcpy(u, mat, n * n * sizeof(QLTEN_Complex));

  auto info = LAPACKE_zheev(
      LAPACK_ROW_MAJOR, 'V', 'U', // Compute eigenvalues and eigenvectors, upper triangle
      n,
      reinterpret_cast<lapack_complex_double *>(u), n,
      eigenvalues
  );

  assert(info == 0);
  for (size_t i = 0; i < n; i++) {
    d[i * n + i] = eigenvalues[i];
  }
  free(eigenvalues);
#ifdef QLTEN_COUNT_FLOPS
  flop += 8 * n * n * n / 3;
  // a rough estimation for complex EVD
#endif
  return info;
}

inline void MatQR(
    QLTEN_Double *mat,
    const size_t m, const size_t n,
    QLTEN_Double *&q,
    QLTEN_Double *&r
) {
  auto k = std::min(m, n);
  size_t elem_type_size = sizeof(QLTEN_Double);
  auto tau = (QLTEN_Double *) malloc(k * elem_type_size);
  LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, mat, n, tau);

  // Create R matrix
  r = (QLTEN_Double *) malloc((k * n) * elem_type_size);
  for (size_t i = 0; i < k; ++i) {
    memset(r + i * n, 0, i * elem_type_size);
    memcpy(r + i * n + i, mat + i * n + i, (n - i) * elem_type_size);
  }

  // Create Q matrix
  LAPACKE_dorgqr(LAPACK_ROW_MAJOR, m, k, k, mat, n, tau);     // or: orthogonal
  free(tau);
  q = (QLTEN_Double *) malloc((m * k) * elem_type_size);
  if (m == n) {
    memcpy(q, mat, (m * n) * elem_type_size);
  } else {
    for (size_t i = 0; i < m; ++i) {
      memcpy(q + i * k, mat + i * n, k * elem_type_size);
    }
  }
#ifdef QLTEN_COUNT_FLOPS
  flop += 2 * m * n * n - 2 * n * n * n / 3;
  // the book "Numerical Linear Algebra" by Trefethen and Bau
  // assume Householder transformations
#endif
}

inline void MatQR(
    QLTEN_Complex *mat,
    const size_t m, const size_t n,
    QLTEN_Complex *&q,
    QLTEN_Complex *&r
) {
  auto k = std::min(m, n);
  size_t elem_type_size = sizeof(QLTEN_Complex);
  auto tau = (QLTEN_Complex *) malloc(k * elem_type_size);

  LAPACKE_zgeqrf(LAPACK_ROW_MAJOR, m, n,
                 reinterpret_cast<lapack_complex_double *>(mat),
                 n, reinterpret_cast<lapack_complex_double *>(tau));

  // Create R matrix
  r = (QLTEN_Complex *) malloc((k * n) * elem_type_size);
  for (size_t row = 0; row < k; ++row) {
    std::fill(r + row * n, r + row * n + row, 0);
    std::copy(mat + row * n + row, mat + row * n + n, r + row * n + row);
  }

  // Create Q matrix
  LAPACKE_zungqr(LAPACK_ROW_MAJOR, m, k, k,
                 reinterpret_cast<lapack_complex_double *>(mat),
                 n, reinterpret_cast<lapack_complex_double *>(tau));
  free(tau);
  q = (QLTEN_Complex *) malloc((m * k) * elem_type_size);
  if (m == n) {
    memcpy(q, mat, (m * n) * elem_type_size);
  } else {
    for (size_t i = 0; i < m; ++i) {
      memcpy(q + i * k, mat + i * n, k * elem_type_size);
    }
  }
#ifdef QLTEN_COUNT_FLOPS
  flop += 8 * m * n * n - 8 * n * n * n / 3;
  // the book "Numerical Linear Algebra" by Trefethen and Bau
  // assume Householder transformations
  // roughly estimate for complex number
#endif
}
} /* hp_numeric */
} /* qlten */
#endif /* ifndef QLTEN_FRAMEWORK_HP_NUMERIC_LAPACK_H */






//void qr( double* const _Q, double* const _R, double* const _A, const size_t _m, const size_t _n) {
//// Maximal rank is used by Lapacke
//const size_t rank = std::min(_m, _n);

//// Tmp Array for Lapacke
//const std::unique_ptr<double[]> tau(new double[rank]);

//// Calculate QR factorisations
//LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, (int) _m, (int) _n, _A, (int) _n, tau.get());

//// Copy the upper triangular Matrix R (rank x _n) into position
//for(size_t row =0; row < rank; ++row) {
//memset(_R+row*_n, 0, row*sizeof(double)); // Set starting zeros
//memcpy(_R+row*_n+row, _A+row*_n+row, (_n-row)*sizeof(double)); // Copy upper triangular part from Lapack result.
//}

//// Create orthogonal matrix Q (in tmpA)
//LAPACKE_dorgqr(LAPACK_ROW_MAJOR, (int) _m, (int) rank, (int) rank, _A, (int) _n, tau.get());

////Copy Q (_m x rank) into position
//if(_m == _n) {
//memcpy(_Q, _A, sizeof(double)*(_m*_n));
//} else {
//for(size_t row =0; row < _m; ++row) {
//memcpy(_Q+row*rank, _A+row*_n, sizeof(double)*(rank));
//}
//}
//}
