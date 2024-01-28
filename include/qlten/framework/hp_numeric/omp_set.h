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

namespace qlten {
/// High performance numerical functions.
namespace hp_numeric {

//thread for contract, svd, qr
inline unsigned tensor_manipulation_num_threads = kTensorOperationDefaultNumThreads;

inline void SetTensorManipulationThreads(unsigned thread) {
  assert(thread > 0);
  tensor_manipulation_num_threads = thread;
  tensor_transpose_num_threads = thread;
#ifndef USE_OPENBLAS
  mkl_set_num_threads(thread);
//  mkl_set_num_threads_local(0);
//  mkl_set_dynamic(true);
#else
  openblas_set_num_threads(thread);
  //equivalent to  `export OMP_NUM_THREADS=4`, manually in script
#endif
}

inline unsigned GetTensorManipulationThreads() {
  return tensor_manipulation_num_threads;
}
} /* hp_numeric */
} /* qlten */



#endif /* ifndef QLTEN_FRAMEWORK_HP_NUMERIC_OMP_SET_H */
