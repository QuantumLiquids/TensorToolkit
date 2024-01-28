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

namespace qlten {

using QLTEN_Double = double;
using QLTEN_Complex = std::complex<QLTEN_Double>;

using CoorsT = std::vector<size_t>;
using ShapeT = std::vector<size_t>;
} /* qlten */


#define MKL_Complex16 qlten::QLTEN_Complex    // This must be defined before any MKL header file.

#endif /* ifndef QLTEN_FRAMEWORK_VALUE_T_H */
