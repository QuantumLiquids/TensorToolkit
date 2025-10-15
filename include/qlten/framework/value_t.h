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
#if __cplusplus >= 201703L
inline double arg(const QLTEN_Complex &z) { return std::arg(z); }
inline QLTEN_Complex polar(QLTEN_Double rho, QLTEN_Double theta) { return std::polar(rho, theta); }
#endif
#else
using QLTEN_Complex = cuda::std::complex<QLTEN_Double>;
using cuda::std::abs;
using cuda::std::conj;
using cuda::std::norm;
using cuda::std::sqrt;

// Provide arg/polar wrappers in qlten namespace for cuda::std::complex
inline double arg(const QLTEN_Complex &z) { return cuda::std::arg(z); }
inline QLTEN_Complex polar(QLTEN_Double rho, QLTEN_Double theta) { return cuda::std::polar(rho, theta); }

#if (__CUDACC_VER_MAJOR__ < 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ <= 5))
// Patch: Provide a custom operator<< in the same namespace as cuda::std::complex
template<typename T>
std::ostream& operator<<(std::ostream& os, const cuda::std::complex<T>& z) {
    return os << "(" << z.real() << ", " << z.imag() << ")";
}
#endif

#endif

using CoorsT = std::vector<size_t>;
using ShapeT = std::vector<size_t>;
} /* qlten */


#define MKL_Complex16 qlten::QLTEN_Complex    // This must be defined before any MKL header file.

#endif /* ifndef QLTEN_FRAMEWORK_VALUE_T_H */
