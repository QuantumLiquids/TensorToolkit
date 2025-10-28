// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Haoxin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2025-10-27.
*
* Description: QuantumLiquids/tensor project. Backend selector for high performance numerical functions.
*/

#ifndef QLTEN_FRAMEWORK_HP_NUMERIC_BACKEND_SELECTOR_H
#define QLTEN_FRAMEWORK_HP_NUMERIC_BACKEND_SELECTOR_H

/**
@file backend_selector.h
@brief Backend selector for high performance numerical functions.
*/

#if defined(HP_NUMERIC_BACKEND_MKL)
#   include "mkl.h"
#elif defined(HP_NUMERIC_BACKEND_AOCL)
#   include <cblas.h>
#   include <lapacke.h>
extern "C" {
void somatcopy_( f77_char* trans,
                 f77_int*  rows,
                 f77_int*  cols,
                 const float* alpha,
                 const float* a,
                 f77_int*  lda,
                 float*   b,
                 f77_int*  ldb );

void domatcopy_( f77_char* trans,
                 f77_int*  rows,
                 f77_int*  cols,
                 const double* alpha,
                 const double* a,
                 f77_int*  lda,
                 double*   b,
                 f77_int*  ldb );

void comatcopy_( f77_char* trans,
                 f77_int*  rows,
                 f77_int*  cols,
                 const scomplex* alpha,
                 const scomplex* a,
                 f77_int*  lda,
                 scomplex*   b,
                 f77_int*  ldb );

void zomatcopy_( f77_char* trans,
                 f77_int*  rows,
                 f77_int*  cols,
                 const dcomplex* alpha,
                 const dcomplex* a,
                 f77_int*  lda,
                 dcomplex*   b,
                 f77_int*  ldb );
}
#elif defined(HP_NUMERIC_BACKEND_OPENBLAS)
#   include <cblas.h>
#   include <lapacke.h>
#else
#   error "Unsupported hp_numeric backend"
#endif

namespace hp_numeric_backend {

constexpr const char* Vendor()
{
#if defined(HP_NUMERIC_BACKEND_MKL)
    return "intel";
#elif defined(HP_NUMERIC_BACKEND_AOCL)
    return "amd";
#elif defined(HP_NUMERIC_BACKEND_OPENBLAS)
    return "openblas";
#else
    return "unknown";
#endif
}

} // namespace hp_numeric_backend

#endif /* ifndef QLTEN_FRAMEWORK_HP_NUMERIC_BACKEND_SELECTOR_H */