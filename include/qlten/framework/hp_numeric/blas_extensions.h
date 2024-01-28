// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2021-10-23.
*
* Description: QuantumLiquids/tensor project. High performance BLAS extensions
* functions based on MKL.
*/

/**
@file blas_extensions.h
@brief High performance BLAS extensions
*/
#ifndef QLTEN_FRAMEWORK_HP_NUMERIC_BLAS_EXTENSIONS
#define QLTEN_FRAMEWORK_HP_NUMERIC_BLAS_EXTENSIONS

#include "qlten/framework/value_t.h"      // QLTEN_Double, QLTEN_Complex

#ifdef Release
#define NDEBUG
#endif
#include <cassert>     // assert

#ifndef USE_OPENBLAS
  #include "mkl.h"
#else
  #include <cblas.h>
#endif

namespace qlten {

/// High performance numerical functions.
namespace hp_numeric {

/*
 * B = A'
 */
inline void MatrixTransposeBatch(
    const QLTEN_Double **Amat_array,
    QLTEN_Double **Bmat_array,
    const size_t *rows_array,
    const size_t *cols_array,
    const size_t group_count
) {
#if defined(INTEL_MKL_VERSION) && INTEL_MKL_VERSION >= (20210004)
  char * trans_array = new char[group_count];
  double * alpha_array = new double[group_count];
  size_t * group_size = new size_t[group_count];
  for(size_t i = 0; i < group_count; i++){
    trans_array[i] = 'T';
    alpha_array[i] = 1.0;
    group_size[i] = 1;
  }
  mkl_domatcopy_batch(
    'R', trans_array,
    rows_array, cols_array,
    alpha_array,
    Amat_array, cols_array,
    Bmat_array, rows_array,
    group_count,group_size
      );

  delete [] trans_array;
  delete [] alpha_array;
  delete [] group_size;
#elif defined(INTEL_MKL_VERSION)
  for(size_t i = 0; i < group_count; i++) {
    mkl_domatcopy(
        'R', 'T',
        rows_array[i], cols_array[i],
        1.0,
        Amat_array[i], cols_array[i],
        Bmat_array[i], rows_array[i]
        );
  }
#else
  // Use OpenBLAS style implementation here
  for (size_t i = 0; i < group_count; i++) {
    cblas_domatcopy(
        CblasRowMajor, CblasTrans,
        rows_array[i], cols_array[i],
        1.0,
        Amat_array[i], cols_array[i],
        Bmat_array[i], rows_array[i]
    );
  }
#endif
}

inline void MatrixTransposeBatch(
    const QLTEN_Complex **Amat_array,
    QLTEN_Complex **Bmat_array,
    const size_t *rows_array,
    const size_t *cols_array,
    const size_t group_count
) {
  //TODO: question: if MKL_Complex16 == QLTEN_Complex
  //?? MKL_Complex16 ** == QLTEN_Complex**?
#if defined(INTEL_MKL_VERSION) && INTEL_MKL_VERSION >= (20210004)
  char * trans_array = new char[group_count];
  MKL_Complex16 * alpha_array = new MKL_Complex16[group_count];
  size_t * group_size = new size_t[group_count];
  for(size_t i = 0; i < group_count; i++){
    trans_array[i] = 'T';
    alpha_array[i] = 1.0;
    group_size[i] = 1;
  }
  mkl_zomatcopy_batch(
    'R', trans_array,
    rows_array, cols_array,
    alpha_array,
    Amat_array, cols_array,
    Bmat_array, rows_array,
    group_count,group_size
      );

  delete [] trans_array;
  delete [] alpha_array;
  delete [] group_size;
#elif defined(INTEL_MKL_VERSION)
  for(size_t i = 0; i < group_count; i++) {
    mkl_zomatcopy(
        'R', 'T',
        rows_array[i], cols_array[i],
        MKL_Complex16(1.0),
        Amat_array[i], cols_array[i],
        Bmat_array[i], rows_array[i]
    );
  }
#else
  // Use OpenBLAS style implementation here
  double alpha[2] = {1.0, 0.0};
  for (size_t i = 0; i < group_count; i++) {
    cblas_zomatcopy(
        CblasRowMajor, CblasTrans,
        rows_array[i], cols_array[i],
        alpha,
        reinterpret_cast<const double *>(Amat_array[i]), cols_array[i] ,
        reinterpret_cast<double *>(Bmat_array[i]), rows_array[i]
    );
  }
#endif
}
} /* hp_numeric */
} /* qlten */
#endif /* ifndef QLTEN_FRAMEWORK_HP_NUMERIC_BLAS_EXTENSIONS */
