// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Haoxin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2020-12-02 14:15
* 
* Description: QuantumLiquids/tensor project. High performance LAPACK related functions
* based on MKL/Opanblas, or cuSolver.
*
* The matrices in the wrapped API are all assumed to be ROW-MAJOR. cuSolver and cuBlas are all use
* COLUMN-MAJOR matrix as default. Current file has carefully dealt with this incompatibility by introducing
* matrix transpose or reordering. Users please simply regard the matrices in the wrappers are ROW-MAJOR.
*/

/**
@file lapack.h
@brief High performance LAPACK related functions based on MKL.
*/
#ifndef QLTEN_FRAMEWORK_HP_NUMERIC_LAPACK_H
#define QLTEN_FRAMEWORK_HP_NUMERIC_LAPACK_H

#include <algorithm>    // min
#include <cstring>      // memcpy, memset
#include <cassert>      // assert

#ifndef  USE_GPU
#ifndef USE_OPENBLAS
#include "mkl.h"       
#else
#include <cblas.h>
#include <lapacke.h>
#endif
#else
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#endif

#include "qlten/framework/value_t.h"
#include "qlten/framework/flops_count.h"  // flop

#ifdef Release
#define NDEBUG
#endif

namespace qlten {

namespace hp_numeric {
#ifndef  USE_GPU
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
#else // define USE GPU

#ifndef  NDEBUG
//row-major matrix A, m: rows, n: cols; lda: leading-dimension, usually be n
inline void print_matrix(const int &m, const int &n, const double *A, const int &lda) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      std::printf("%0.10f ", A[i * lda + j]);
    }
    std::printf("\n");
  }
}
#endif

// QLTEN_Double *mat will be overwritten
inline cusolverStatus_t MatSVD(
    QLTEN_Double *mat,
    const size_t m, const size_t n,
    QLTEN_Double *&u,
    QLTEN_Double *&s,
    QLTEN_Double *&vt
) {
#ifndef NDEBUG
//  cudaPointerAttributes attr;
//  cudaPointerGetAttributes(&attr, mat);
//  if (attr.type != cudaMemoryTypeDevice) {
//    std::cerr << "Error: mat is not a valid device pointer." << std::endl;
//    exit(EXIT_FAILURE);
//  }
//  double *h_mat;
//  h_mat = (QLTEN_Double *) malloc((n * m) * sizeof(QLTEN_Double));
//  HANDLE_CUDA_ERROR(cudaMemcpy(h_mat, mat, (n * m) * sizeof(QLTEN_Double), cudaMemcpyDeviceToHost));
//  std::cout << "rows(m) : " << m << std::endl;
//  std::cout << "cols(n) : " << n << std::endl;
//  print_matrix(m, n, h_mat, n);
#endif
  cusolverDnHandle_t handle = CusolverHandleManager::GetHandle();

  int lwork = 0;
  QLTEN_Double *d_work = nullptr;
  int *devInfo = nullptr;
  cudaMalloc((void **) &devInfo, sizeof(int));

  size_t k = std::min(m, n);
  cudaMalloc((void **) &u, m * k * sizeof(QLTEN_Double));
  QLTEN_Double *ut;
  cudaMalloc((void **) &ut, m * k * sizeof(QLTEN_Double));
  cudaMalloc((void **) &s, k * sizeof(QLTEN_Double));
  cudaMalloc((void **) &vt, n * k * sizeof(QLTEN_Double));

  gesvdjInfo_t params = nullptr;
  cusolverDnCreateGesvdjInfo(&params);
  HANDLE_CUSOLVER_ERROR(
      cusolverDnDgesvdj_bufferSize(
          handle, CUSOLVER_EIG_MODE_VECTOR, 1, n, m, mat, n, s, vt, n, ut, m, &lwork, params));

  HANDLE_CUDA_ERROR(cudaMalloc((void **) &d_work, lwork * sizeof(QLTEN_Double)));

  auto status =
      cusolverDnDgesvdj(
          handle, CUSOLVER_EIG_MODE_VECTOR, 1, n, m, mat, n, s, vt, n, ut, m, d_work, lwork, devInfo, params);
  HANDLE_CUSOLVER_ERROR(status);

#ifndef NDEBUG
  int info;
  cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
  if (0 == info) {

  } else if (0 > info) {
    std::printf("%d-th parameter is wrong \n", -info);
    exit(1);
  } else {
    std::printf("WARNING: info = %d : gesvdj does not converge \n", info);
  }
  assert(0 == info);

//  double *h_u;
//  double *h_s;
//  double *h_v;
//
//  h_u = (QLTEN_Double *) malloc((k * m) * sizeof(QLTEN_Double));
//  h_s = (QLTEN_Double *) malloc(k * sizeof(QLTEN_Double));
//  h_v = (QLTEN_Double *) malloc((k * n) * sizeof(QLTEN_Double));
//
//  cudaMemcpy(h_v, vt, (k * n) * sizeof(QLTEN_Double), cudaMemcpyDeviceToHost);
//  cudaMemcpy(h_s, s, (k) * sizeof(QLTEN_Double), cudaMemcpyDeviceToHost);

//  std::cout << std::endl;
//  std::cout << "Vt matrix:" << std::endl;
//  print_matrix(k, n, h_v, n);
//  std::cout << std::endl;
//  std::cout << "singular values :" << std::endl;
//  for (size_t i = 0; i < k; i++) {
//    std::printf("%0.10f ", h_s[i]);
//  }

//  double residual = 0;
//  int executed_sweeps = 0;
//  cusolverDnXgesvdjGetSweeps(handle, params, &executed_sweeps);
//  cusolverDnXgesvdjGetResidual(handle, params, &residual);
//  std::printf("internal residual |E|_F = %E \n", residual);
//  std::printf("number of executed sweeps = %d \n", executed_sweeps);

#endif
  // Transpose ut to u
  cublasHandle_t cublasHandle = CublasHandleManager::GetHandle();
  const QLTEN_Double alpha = 1.0;
  const QLTEN_Double beta = 0.0;
  cublasDgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, k, m, &alpha, ut, m, &beta, nullptr, m, u, k);

#ifndef NDEBUG
//  cudaMemcpy(h_u, u, (k * m) * sizeof(QLTEN_Double), cudaMemcpyDeviceToHost);
//  std::cout << std::endl;
//  std::cout << "U matrix :" << std::endl;
//  print_matrix(m, k, h_u, k);
//  free(h_u);
//  free(h_v);
//  free(h_s);
//  free(h_mat);
#endif
  cusolverDnDestroyGesvdjInfo(params);
  cudaFree(ut);
  cudaFree(d_work);
  cudaFree(devInfo);
  return status;
}

inline cusolverStatus_t MatSVD(
    QLTEN_Complex *mat,
    const size_t m, const size_t n,
    QLTEN_Complex *&u,
    QLTEN_Double *&s,
    QLTEN_Complex *&vt
) {
  cusolverDnHandle_t handle = CusolverHandleManager::GetHandle();

  int lwork = 0;
  cuDoubleComplex *d_work = nullptr;
  int *devInfo = nullptr;
  cudaMalloc((void **) &devInfo, sizeof(int));

  size_t k = std::min(m, n);
  cudaMalloc((void **) &u, m * k * sizeof(QLTEN_Complex));
  cuDoubleComplex *ut;
  cudaMalloc((void **) &ut, m * k * sizeof(QLTEN_Complex));
  cudaMalloc((void **) &s, k * sizeof(QLTEN_Double));
  cudaMalloc((void **) &vt, n * k * sizeof(QLTEN_Complex));

  gesvdjInfo_t params = nullptr;
  cusolverDnCreateGesvdjInfo(&params);

  HANDLE_CUSOLVER_ERROR(
      cusolverDnZgesvdj_bufferSize(
          handle, CUSOLVER_EIG_MODE_VECTOR, 1, n, m,
          reinterpret_cast<cuDoubleComplex *>(mat), n,
          s,
          reinterpret_cast<cuDoubleComplex *>(vt), n,
          ut, m, &lwork, params));

  cudaMalloc((void **) &d_work, lwork * sizeof(cuDoubleComplex));

  auto status = cusolverDnZgesvdj(
      handle, CUSOLVER_EIG_MODE_VECTOR, 1, n, m,
      reinterpret_cast<cuDoubleComplex *>(mat), n,
      s,
      reinterpret_cast<cuDoubleComplex *>(vt), n,
      ut, m, d_work, lwork, devInfo, params);
  HANDLE_CUSOLVER_ERROR(status);

  // Transpose ut to u
  const cuDoubleComplex alpha = {1.0, 0.0};
  const cuDoubleComplex beta = {0.0, 0.0};
  cublasZgeam(CublasHandleManager::GetHandle(),
              CUBLAS_OP_C,
              CUBLAS_OP_N,
              k, m, &alpha, ut, m, &beta, nullptr, m,
              reinterpret_cast<cuDoubleComplex *>(u), k);

  cusolverDnDestroyGesvdjInfo(params);
  cudaFree(ut);
  cudaFree(d_work);
  cudaFree(devInfo);
  return status;
}

inline void MatQR(
    QLTEN_Double *mat,
    const size_t m, const size_t n,
    QLTEN_Double *&q,
    QLTEN_Double *&r
) {
  // cuSolver handle
  cusolverDnHandle_t handle = CusolverHandleManager::GetHandle();
  auto cublas_handle = CublasHandleManager::GetHandle();
  int lda = m; // Leading dimension for column-major layout
  auto k = std::min(m, n);
  int q_rows = m;
  int q_cols = std::min(n, m);
#ifndef NDEBUG
//  double *h_mat;
//  h_mat = (QLTEN_Double *) malloc((n * m) * sizeof(QLTEN_Double));
//  HANDLE_CUDA_ERROR(cudaMemcpy(h_mat, mat, (n * m) * sizeof(QLTEN_Double), cudaMemcpyDeviceToHost));
//  std::cout << "rows(m) : " << m << std::endl;
//  std::cout << "cols(n) : " << n << std::endl;
//  std::cout << "Input Matrix for QR : " << std::endl;
//  print_matrix(m, n, h_mat, n);
#endif
  // Copy input matrix and transpose (row-major to column-major)
  QLTEN_Double *d_A;
  HANDLE_CUDA_ERROR(cudaMalloc(&d_A, sizeof(QLTEN_Double) * m * n));

  // Transpose mat (row-major to col-major)
  const QLTEN_Double alpha = 1.0, beta = 0.0;
  HANDLE_CUBLAS_ERROR(cublasDgeam(cublas_handle,
                                  CUBLAS_OP_T,
                                  CUBLAS_OP_N,
                                  m,
                                  n,
                                  &alpha,
                                  mat,
                                  n,
                                  &beta,
                                  nullptr,
                                  m,
                                  d_A,
                                  m));

  // Allocate tau (for householder reflectors)
  QLTEN_Double *d_tau;
  HANDLE_CUDA_ERROR(cudaMalloc(&d_tau, sizeof(QLTEN_Double) * k));

  // Workspace size
  int workspace_size_geqrf(0), workspace_size_ormqr(0);
  HANDLE_CUSOLVER_ERROR(cusolverDnDgeqrf_bufferSize(handle, m, n, d_A, lda, &workspace_size_geqrf));
  HANDLE_CUSOLVER_ERROR(cusolverDnDorgqr_bufferSize(handle, m, k, k, d_A, lda, d_tau, &workspace_size_ormqr));
  int lwork = std::max(workspace_size_geqrf, workspace_size_ormqr);
  // Allocate workspace
  QLTEN_Double *workspace;
  HANDLE_CUDA_ERROR(cudaMalloc(&workspace, sizeof(QLTEN_Double) * lwork));
  // QR factorization
  int *devInfo;
  HANDLE_CUDA_ERROR(cudaMalloc(&devInfo, sizeof(int)));
  HANDLE_CUSOLVER_ERROR(cusolverDnDgeqrf(handle, m, n, d_A, lda, d_tau, workspace, lwork, devInfo));
#ifndef NDEBUG
  int info;
  cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
  if (0 == info) {

  } else if (0 > info) {
    std::printf("Dgeqrf %d-th parameter is wrong \n", -info);
    exit(1);
  } else {
    std::printf("WARNING: info = %d : Dgeqrf does not converge \n", info);
  }
  assert(0 == info);
#endif
  // Extract R
  QLTEN_Double *rt; // column-major R
  HANDLE_CUDA_ERROR(cudaMalloc(&rt, sizeof(QLTEN_Double) * k * n));
  HANDLE_CUDA_ERROR(cudaMemcpy2D(rt,
                                 k * sizeof(QLTEN_Double),
                                 d_A,
                                 m * sizeof(QLTEN_Double),
                                 k * sizeof(QLTEN_Double),
                                 n,
                                 cudaMemcpyDeviceToDevice));
  HANDLE_CUDA_ERROR(cudaMalloc(&r, sizeof(QLTEN_Double) * k * n));
  cublasDgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
              n, k, &alpha, rt, k, &beta, nullptr, n,
              r, n);
  HANDLE_CUDA_ERROR(cudaFree(rt));
  for (size_t i = 0; i < k; ++i) {
    HANDLE_CUDA_ERROR(cudaMemset(r + i * n, 0, sizeof(QLTEN_Double) * i));
  }
  // Generate Q

  HANDLE_CUSOLVER_ERROR(cusolverDnDorgqr(handle, m, k, k,
                                         d_A, lda, d_tau,
                                         workspace, lwork, devInfo));

  // Transpose Q back (col-major to row-major)
  HANDLE_CUDA_ERROR(cudaMalloc(&q, sizeof(QLTEN_Double) * q_rows * q_cols));
  cublasDgeam(cublas_handle,
              CUBLAS_OP_T,
              CUBLAS_OP_N,
              q_cols,
              q_rows,
              &alpha,
              d_A,
              m,
              &beta,
              nullptr,
              q_cols,
              q,
              q_cols);

#ifndef NDEBUG
  cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
  if (0 == info) {

  } else if (0 > info) {
    std::printf("orgqr %d-th parameter is wrong \n", -info);
    exit(1);
  } else {
    std::printf("WARNING: info = %d : orgqr does not converge \n", info);
  }
  assert(0 == info);

//  double *h_q;
//  double *h_r;
//
//  h_q = (QLTEN_Double *) malloc((q_cols * q_rows) * sizeof(QLTEN_Double));
//  h_r = (QLTEN_Double *) malloc(std::min(m, n) * n * sizeof(QLTEN_Double));
//
//  cudaMemcpy(h_q, q, (q_cols * q_rows) * sizeof(QLTEN_Double), cudaMemcpyDeviceToHost);
//  cudaMemcpy(h_r, r, std::min(m, n) * n * sizeof(QLTEN_Double), cudaMemcpyDeviceToHost);
//
//  std::cout << std::endl;
//  std::cout << "Q matrix:" << std::endl;
//  print_matrix(m, q_cols, h_q, q_cols);
//  std::cout << std::endl;
//  std::cout << "R matrix :" << std::endl;
//  print_matrix(std::min(m, n), n, h_r, n);
//  std::cout << std::endl;
//  free(h_q);
//  free(h_r);
//  free(h_mat);
#endif
  // Cleanup
  cudaFree(d_A);
  cudaFree(d_tau);
  cudaFree(workspace);
  cudaFree(devInfo);
}

inline void MatQR(
    QLTEN_Complex *mat,
    const size_t m, const size_t n,
    QLTEN_Complex *&q,
    QLTEN_Complex *&r
) {
  //assume sizeof(cuDoubleComplex) == sizeof(QLTEN_Complex)
  // cuSolver handle
  cusolverDnHandle_t handle = CusolverHandleManager::GetHandle();
  auto cublas_handle = CublasHandleManager::GetHandle();
  int lda = m; // Leading dimension for column-major layout
  auto k = std::min(m, n); //economy size
  // Allocate memory for input matrix
  cuDoubleComplex *d_A;
  HANDLE_CUDA_ERROR(cudaMalloc(&d_A, sizeof(cuDoubleComplex) * m * n));
  // Transpose mat (row-major to col-major)
  const cuDoubleComplex alpha = {1.0, 0.0}, beta = {0.0, 0.0};
  HANDLE_CUBLAS_ERROR(cublasZgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha,
                                  reinterpret_cast<const cuDoubleComplex *>(mat), n, &beta, nullptr, m,
                                  d_A, m));

  // Allocate tau
  cuDoubleComplex *d_tau;
  HANDLE_CUDA_ERROR(cudaMalloc(&d_tau, sizeof(cuDoubleComplex) * k));

  // Workspace size
  int workspace_size_geqrf, workspace_size_ormqr;
  HANDLE_CUSOLVER_ERROR(cusolverDnZgeqrf_bufferSize(handle, m, n, d_A, lda, &workspace_size_geqrf));
  HANDLE_CUSOLVER_ERROR(cusolverDnZungqr_bufferSize(handle, m, k, k, d_A, lda, d_tau, &workspace_size_ormqr));
  int lwork = std::max(workspace_size_geqrf, workspace_size_ormqr);
  // Allocate workspace
  cuDoubleComplex *workspace;
  HANDLE_CUDA_ERROR(cudaMalloc(&workspace, sizeof(cuDoubleComplex) * lwork));

  // QR factorization
  int *devInfo;
  HANDLE_CUDA_ERROR(cudaMalloc(&devInfo, sizeof(int)));
  HANDLE_CUSOLVER_ERROR(cusolverDnZgeqrf(handle, m, n, d_A, lda, d_tau, workspace, lwork, devInfo));

  // Generate R
  cuDoubleComplex *rt;
  HANDLE_CUDA_ERROR(cudaMalloc(&rt, sizeof(cuDoubleComplex) * k * n));
  HANDLE_CUDA_ERROR(cudaMemcpy2D(rt,
                                 k * sizeof(cuDoubleComplex),
                                 d_A,
                                 m * sizeof(cuDoubleComplex),
                                 k * sizeof(cuDoubleComplex),
                                 n,
                                 cudaMemcpyDeviceToDevice));
  HANDLE_CUDA_ERROR(cudaMalloc(&r, sizeof(QLTEN_Complex) * k * n));
  //transpose to row-major R
  cublasZgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
              n, k, &alpha, rt, k, &beta, nullptr, n,
              reinterpret_cast<cuDoubleComplex *>(r), n);

  HANDLE_CUDA_ERROR(cudaFree(rt));
  for (size_t i = 0; i < k; ++i) {
    HANDLE_CUDA_ERROR(cudaMemset(r + i * n, 0, sizeof(QLTEN_Complex) * i));
  }

  //Generate Q
  HANDLE_CUSOLVER_ERROR(cusolverDnZungqr(handle, m, k, k, d_A, lda,
                                         d_tau, workspace, lwork, devInfo));
  // Transpose Q back (col-major to row-major)
  cudaMalloc(&q, sizeof(QLTEN_Complex) * m * k);
  cublasZgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
              k, m, &alpha,
              d_A, m,
              &beta, nullptr, k,
              reinterpret_cast<cuDoubleComplex *>(q), k);

  // Cleanup
  cudaFree(d_A);
  cudaFree(d_tau);
  cudaFree(workspace);
  cudaFree(devInfo);
}

#endif//USE_GPU
} /* hp_numeric */
} /* qlten */
#endif /* ifndef QLTEN_FRAMEWORK_HP_NUMERIC_LAPACK_H */

