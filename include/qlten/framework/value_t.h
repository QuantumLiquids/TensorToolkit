// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-09-12 20:58
* 
* Description: QuantumLiquids/tensor project. Type definitions used by this library.
* This file must be included before any MKL library header file.
*/
#ifndef QLTEN_FRAMEWORK_VALUE_T_H
#define QLTEN_FRAMEWORK_VALUE_T_H

#include <complex>    // complex
#include <vector>     // vector
#ifdef USE_GPU
#include <cuda/std/complex>
#endif

namespace qlten {

using QLTEN_Double = double;
#ifndef USE_GPU
using QLTEN_Complex = std::complex<QLTEN_Double>;
using std::abs;
using std::conj;
using std::norm;
using std::sqrt;
#else
using QLTEN_Complex = cuda::std::complex<QLTEN_Double>;
using cuda::std::abs;
using cuda::std::conj;
using cuda::std::norm;
using cuda::std::sqrt;
#endif


using CoorsT = std::vector<size_t>;
using ShapeT = std::vector<size_t>;
} /* qlten */


#define MKL_Complex16 qlten::QLTEN_Complex    // This must be defined before any MKL header file.

#endif /* ifndef QLTEN_FRAMEWORK_VALUE_T_H */
