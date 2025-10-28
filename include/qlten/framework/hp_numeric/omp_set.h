// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author:   Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
            Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-21 15:27
*
* Description: API for setting openmp thread
*/

/**
@file omp_set.h
@brief API for setting openmp thread
*/
#ifndef QLTEN_FRAMEWORK_HP_NUMERIC_OMP_SET_H
#define QLTEN_FRAMEWORK_HP_NUMERIC_OMP_SET_H

#include <cassert>
#include <omp.h>

namespace qlten {
/// High performance numerical functions.
namespace hp_numeric {

#ifndef USE_GPU
//thread for contract, svd, qr
inline unsigned tensor_manipulation_num_threads = kTensorOperationDefaultNumThreads;

inline void SetTensorManipulationThreads(unsigned thread) {
  assert(thread > 0);
  tensor_manipulation_num_threads = thread;
  tensor_transpose_num_threads = thread;
#ifdef HP_NUMERIC_BACKEND_INTEL
  mkl_set_num_threads(thread);
#elif defined(HP_NUMERIC_BACKEND_OPENBLAS)
  openblas_set_num_threads(thread);
#elif defined(HP_NUMERIC_BACKEND_AOCL)
  // AOCL builds typically use OpenMP; control threads via OpenMP runtime
  omp_set_num_threads(thread);
#endif
}

inline unsigned GetTensorManipulationThreads() {
  return tensor_manipulation_num_threads;
}
#endif
} /* hp_numeric */
} /* qlten */



#endif /* ifndef QLTEN_FRAMEWORK_HP_NUMERIC_OMP_SET_H */
