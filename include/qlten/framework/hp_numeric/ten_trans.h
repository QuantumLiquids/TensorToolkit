// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-21 15:27
*
* Description: High performance tensor transpose function based on HPTT library.
*/

/**
@file ten_trans.h
@brief High performance tensor transpose function based on HPTT library.
*/
#ifndef QLTEN_FRAMEWORK_HP_NUMERIC_TEN_TRANS_H
#define QLTEN_FRAMEWORK_HP_NUMERIC_TEN_TRANS_H

#include <vector>     // vector

#ifndef PLAIN_TRANSPOSE

#include "hptt.h"

#endif

namespace qlten {

/// High performance numerical functions.
namespace hp_numeric {

inline int tensor_transpose_num_threads = kTensorOperationDefaultNumThreads;

#ifndef PLAIN_TRANSPOSE

template<typename ElemT>
void TensorTranspose(
    const std::vector<size_t> &transed_order,
    const size_t ten_rank,
    ElemT *original_data,
    const std::vector<size_t> &original_shape,
    ElemT *transed_data,
    const std::vector<size_t> &transed_shape,
    const double alpha  //scale factor, for boson tensor alpha = 1; for Fermion tensor additional minus sign may require
) {
  const int dim = ten_rank;
  int perm[dim];
  for (int i = 0; i < dim; ++i) { perm[i] = transed_order[i]; }
  int sizeA[dim];
  for (int i = 0; i < dim; ++i) { sizeA[i] = original_shape[i]; }
  int outerSizeB[dim];
  for (int i = 0; i < dim; ++i) { outerSizeB[i] = transed_shape[i]; }
  auto tentrans_plan = hptt::create_plan(perm, dim,
                                         alpha, original_data, sizeA, sizeA,
                                         0.0, transed_data, outerSizeB,
                                         hptt::ESTIMATE,
                                         tensor_transpose_num_threads, {},
                                         true
  );
  tentrans_plan->execute();
}

template<typename ElemT>
void TensorTranspose(
    const std::vector<int> &transed_order,
    const size_t ten_rank,
    ElemT *original_data,
    const std::vector<size_t> &original_shape,
    ElemT *transed_data,
    const std::vector<int> &transed_shape,
    const double alpha //scale factor, for boson tensor alpha = 1; for Fermion tensor additional minus sign may require
) {
  const int dim = ten_rank;
  int sizeA[dim];
  for (int i = 0; i < dim; ++i) { sizeA[i] = original_shape[i]; }
  auto tentrans_plan = hptt::create_plan(transed_order.data(), dim,
                                         alpha, original_data, sizeA, sizeA,
                                         0.0, transed_data, transed_shape.data(),
                                         hptt::ESTIMATE,
                                         tensor_transpose_num_threads, {},
                                         true
  );
  tentrans_plan->execute();
}

#else
template<typename ElemT, typename IntType>
void TensorTranspose(
    const std::vector<IntType> &transed_order,
    const size_t ten_rank,
    ElemT *original_data,
    const std::vector<size_t> &original_shape,
    ElemT *transed_data,
    const std::vector<IntType> &transed_shape,
    const double scale
) {
  // Calculate the strides for each dimension in the original tensor
  std::vector<size_t> original_strides(ten_rank, 1);
  size_t stride = 1;
  for (int i = ten_rank - 1; i >= 0; --i) {
    original_strides[i] = stride;
    stride *= original_shape[i];
  }

  // Calculate the total number of elements in the original tensor
  size_t total_elements = stride;

  // Calculate the strides for each dimension in the transposed tensor
  std::vector<size_t> transed_strides(ten_rank, 1);
  stride = 1;
  for (int i = ten_rank - 1; i >= 0; --i) {
    transed_strides[i] = stride;
    stride *= transed_shape[i];
  }

  // Loop over each element in the original tensor
  std::vector<size_t> original_coords(ten_rank, 0);
  std::vector<size_t> transed_coords(ten_rank);
  for (size_t index = 0; index < total_elements; ++index) {
    // Calculate the coordinates of the current element in the original tensor

    size_t remaining_index = index;

    for (int i = 0; i < ten_rank; ++i) {
      original_coords[i] = remaining_index / original_strides[i];
      remaining_index %= original_strides[i];
    }

    // Permute the coordinates according to the transposition order

    for (size_t i = 0; i < ten_rank; ++i) {
      transed_coords[i] = original_coords[transed_order[i]];
    }

    // Calculate the index of the corresponding element in the transposed tensor
    size_t transed_index = 0;
    for (size_t i = 0; i < ten_rank; ++i) {
      transed_index += transed_coords[i] * transed_strides[i];
    }

    // Copy the data to the transposed position
    transed_data[transed_index] = scale * original_data[index];
  }
}

#endif

} /* hp_numeric */
} /* qlten */
#endif /* ifndef QLTEN_FRAMEWORK_HP_NUMERIC_TEN_TRANS_H */
