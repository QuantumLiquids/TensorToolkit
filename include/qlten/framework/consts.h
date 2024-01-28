// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-09-11 20:43
* 
* Description: QuantumLiquids/tensor project. Constants used by this library.
*/
#ifndef QLTEN_FRAMEWORK_CONSTS_H
#define QLTEN_FRAMEWORK_CONSTS_H

#include <string>

namespace qlten {

// QLTensor storage file suffix.
const std::string kGQTenFileSuffix = "qlten";

// Double numerical error.
const double kDoubleEpsilon = 1.0E-15;

// Default omp threads number.
const int kTensorOperationDefaultNumThreads = 4;

// MPI master's rank
const size_t kMPIMasterRank = 0;
} /* qlten */
#endif /* ifndef QLTEN_FRAMEWORK_CONSTS_H */
