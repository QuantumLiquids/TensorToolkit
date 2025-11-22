// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2019-09-11 20:43
* 
* Description: QuantumLiquids/tensor project. Constants used by this library.
*/
#ifndef QLTEN_FRAMEWORK_CONSTS_H
#define QLTEN_FRAMEWORK_CONSTS_H

#include <string>

namespace qlten {

// QLTensor storage file suffix.
const std::string kQLTenFileSuffix = "qlten";

// Double numerical error.
const double kDoubleEpsilon = 1.0E-15;
// Float numerical error.
const float kFloatEpsilon = 1.0E-6f;

// Default omp threads number.
const int kTensorOperationDefaultNumThreads = 4;

} /* qlten */
#endif /* ifndef QLTEN_FRAMEWORK_CONSTS_H */
