// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2021-03-23 10:00
*
* Description: QuantumLiquids/tensor project. Expand two tensors.
*/

/**
@file ten_expand.h
@brief Expand two tensors.
*/
#ifndef QLTEN_TENSOR_MANIPULATION_TEN_EXPAND_H
#define QLTEN_TENSOR_MANIPULATION_TEN_EXPAND_H

#include <vector>       // vector
#include <map>          // map
#include <algorithm>    // find
#include <cassert>     // assert

#include "qlten/qltensor_all.h"     // QLTensor
#include "qlten/utility/timer.h"

#ifdef Release
#define NDEBUG
#endif

namespace qlten {

// Forward declaration
template<typename TenElemT, typename QNT>
void ExpandOneIdx_(
    QLTensor<TenElemT, QNT> *,
    QLTensor<TenElemT, QNT> *,
    const size_t,
    QLTensor<TenElemT, QNT> *
);

// Inline helpers for tensor expansion
template<typename TenElemT, typename QNT>
inline void TensorExpandPreChecker(
    const QLTensor<TenElemT, QNT> &a,
    const QLTensor<TenElemT, QNT> &b,
    const std::vector<size_t> &expand_idx_nums
) {
  assert(a.Rank() == b.Rank());     // To be expanded tensors should have the same rank
  for (size_t i = 0; i < a.Rank(); ++i) {
    if (
        find(expand_idx_nums.cbegin(), expand_idx_nums.cend(), i) ==
            expand_idx_nums.cend()
        ) {
      assert(a.GetIndexes()[i] == b.GetIndexes()[i]);
    } else {
      // Indexes of the to be expanded tensors should have the same directions
      assert(a.GetIndexes()[i].GetDir() == b.GetIndexes()[i].GetDir());
    }
  }
  // To be expanded tensors should have the same quantum number divergence or
  // be an empty tensor
  assert(a.GetQNBlkNum() == 0 || b.GetQNBlkNum() == 0 || a.Div() == b.Div());
}

template<typename TenElemT, typename QNT>
inline void ExpandedTenDivChecker(
    const QLTensor<TenElemT, QNT> &a,
    const QLTensor<TenElemT, QNT> &b,
    const QLTensor<TenElemT, QNT> &c
) {
  if (a.GetQNBlkNum() != 0) {
    assert(c.Div() == a.Div());
  } else if (b.GetQNBlkNum() != 0) {
    assert(c.Div() == b.Div());
  } else {
    assert(c.GetQNBlkNum() == 0);
  }
}

/**
Function version for tensor expansion.

@tparam TenElemT The type of tensor elements.
@tparam QNT The quantum number type of the tensors.

@param cpa Pointer to input tensor \f$ A \f$.
@param cpb Pointer to input tensor \f$ B \f$.
@param expand_idx_nums Index numbers which index to be expanded.
@param pc Pointer to result tensor \f$ C \f$.

@note cpa and cpb will actually be temporarily changed in this function. This
      defect will be fixed in the future!
*/
template<typename TenElemT, typename QNT>
void Expand(
    const QLTensor<TenElemT, QNT> *cpa,
    const QLTensor<TenElemT, QNT> *cpb,
    const std::vector<size_t> &expand_idx_nums,
    QLTensor<TenElemT, QNT> *pc
) {
#ifndef NDEBUG
  TensorExpandPreChecker(*cpa, *cpb, expand_idx_nums);
#endif /* ifndef NDEBUG */

  // TODO: Remove const_cast!!
  auto pa = const_cast<QLTensor<TenElemT, QNT> *>(cpa);
  auto pb = const_cast<QLTensor<TenElemT, QNT> *>(cpb);

  if (expand_idx_nums.size() == 1) {
    ExpandOneIdx_(pa, pb, expand_idx_nums[0], pc);
    return;
  } else {
    std::vector<size_t> expand_idx_nums_without_last_one(
        expand_idx_nums.begin(), expand_idx_nums.end() - 1
    );
    auto indexes_a = pa->GetIndexes();
    auto indexes_b = pb->GetIndexes();
    auto expand_dual_a_indexes = pa->GetIndexes();
    auto expand_dual_b_indexes = pb->GetIndexes();
    for (auto idx_num : expand_idx_nums_without_last_one) {
      expand_dual_a_indexes[idx_num] = indexes_b[idx_num];
      expand_dual_b_indexes[idx_num] = indexes_a[idx_num];
    }

    QLTensor<TenElemT, QNT> expand_tmp_a, expand_tmp_b;
    {
      QLTensor<TenElemT, QNT> expand_dual_a(expand_dual_a_indexes);
      Expand(
          pa, &expand_dual_a,
          expand_idx_nums_without_last_one,
          &expand_tmp_a
      );
    }
    {
      QLTensor<TenElemT, QNT> expand_dual_b(expand_dual_b_indexes);
      Expand(
          &expand_dual_b, pb,
          expand_idx_nums_without_last_one,
          &expand_tmp_b
      );
    }

    ExpandOneIdx_(&expand_tmp_a, &expand_tmp_b, expand_idx_nums.back(), pc);
  }

#ifndef NDEBUG
  ExpandedTenDivChecker(*pa, *pb, *pc);
#endif /* ifndef NDEBUG */
}

template<typename QNT>
inline Index<QNT> ExpandIndexAndRecordInfo(
    const Index<QNT> &idx_from_a,
    const Index<QNT> &idx_from_b,
    std::vector<bool> &is_a_idx_qnsct_expanded,
    std::map<size_t, size_t> &b_idx_qnsct_coor_expanded_idx_qnsct_coor_map
) {
  QNSectorVec<QNT> expanded_qnscts;
  auto qnscts_from_b = idx_from_b.GetQNScts();
  auto qnscts_from_a_size = idx_from_a.GetQNSctNum();
  auto qnscts_from_b_size = idx_from_b.GetQNSctNum();
  is_a_idx_qnsct_expanded = std::vector<bool>(qnscts_from_a_size);
  std::vector<size_t> idxs_of_erased_qnscts_from_b;     // record the indexes of erased qnscts from b,
  // so that we do not actually erase the qnsct from qnscts_from_b
  for (size_t sct_coor_a = 0; sct_coor_a < qnscts_from_a_size; ++sct_coor_a) {
    auto qnsct_from_a = idx_from_a.GetQNSct(sct_coor_a);
    bool has_matched_qnsct_in_qnscts_from_b = false;
    for (size_t sct_coor_b = 0; sct_coor_b < qnscts_from_b_size; ++sct_coor_b) {
      auto qnsct_from_b = qnscts_from_b[sct_coor_b];
      // Expand the dimension when two QNSectors have the same QN
      if (qnsct_from_b.GetQn() == qnsct_from_a.GetQn()) {
        auto expanded_qnsct = QNSector<QNT>(
            qnsct_from_a.GetQn(),
            qnsct_from_a.dim() + qnsct_from_b.dim()
        );
        expanded_qnscts.push_back(expanded_qnsct);
        is_a_idx_qnsct_expanded[sct_coor_a] = true;
        b_idx_qnsct_coor_expanded_idx_qnsct_coor_map[sct_coor_b] = sct_coor_a;
        idxs_of_erased_qnscts_from_b.push_back(sct_coor_b);
        has_matched_qnsct_in_qnscts_from_b = true;
        break;
      }
    }
    if (!has_matched_qnsct_in_qnscts_from_b) {
      expanded_qnscts.push_back(qnsct_from_a);
      is_a_idx_qnsct_expanded[sct_coor_a] = false;
    }
  }

  // Deal with left QNSectors from the index from tensor B
  for (size_t sct_coor_b = 0; sct_coor_b < qnscts_from_b_size; ++sct_coor_b) {
    if (
        find(
            idxs_of_erased_qnscts_from_b.cbegin(),
            idxs_of_erased_qnscts_from_b.cend(),
            sct_coor_b
        ) == idxs_of_erased_qnscts_from_b.cend()
        ) {
      expanded_qnscts.push_back(qnscts_from_b[sct_coor_b]);
      b_idx_qnsct_coor_expanded_idx_qnsct_coor_map[
          sct_coor_b
      ] = expanded_qnscts.size() - 1;
    }
  }

  return Index<QNT>(expanded_qnscts, idx_from_a.GetDir());
}

/**
Tensor expansion: special case for only expand on one index.

@tparam TenElemT The type of tensor elements.
@tparam QNT The quantum number type of the tensors.

@param pa Pointer to input tensor \f$ A \f$.
@param pb Pointer to input tensor \f$ B \f$.
@param expand_idx_num Index number which index to be expanded.
@param pc Pointer to result tensor \f$ C \f$.

@note This function will temporarily modify two input tensors! Intra-used
      function, not for normal user!
*/
template<typename TenElemT, typename QNT>
void ExpandOneIdx_(
    QLTensor<TenElemT, QNT> *pa,
    QLTensor<TenElemT, QNT> *pb,
    const size_t expand_idx_num,
    QLTensor<TenElemT, QNT> *pc
) {
#ifndef NDEBUG
  TensorExpandPreChecker(*pa, *pb, {expand_idx_num});
#endif /* ifndef NDEBUG */

#ifdef QLTEN_TIMING_MODE
  Timer expand_pre_transpose_timer("   =============> expansion_first_time_transpose");
#endif
  // Firstly we transpose the expand_idx_num-th index to the first index
  size_t ten_rank = pa->Rank();
  std::vector<size_t> transpose_order(ten_rank);
  transpose_order[0] = expand_idx_num;
  for (size_t i = 1; i <= expand_idx_num; i++) { transpose_order[i] = i - 1; }
  for (size_t i = expand_idx_num + 1; i < ten_rank; i++) { transpose_order[i] = i; }
  if (pa == pb) {
    pa->Transpose(transpose_order);
  } else {
    pa->Transpose(transpose_order);
    pb->Transpose(transpose_order);
  }
#ifdef QLTEN_TIMING_MODE
  expand_pre_transpose_timer.PrintElapsed();
#endif

#ifdef QLTEN_TIMING_MODE
  Timer expand_index_timer("   =============> expansion_expand_index_and_record_info");
#endif
  // Then we can expand the two tensor according the first indexes. For each block, the data are direct connected
  // Expand the first index
  std::vector<bool> is_a_idx_qnsct_expanded;
  std::map<size_t, size_t> b_idx_qnsct_coor_expanded_idx_qnsct_coor_map;
  auto expanded_index = ExpandIndexAndRecordInfo(
      pa->GetIndexes()[0],
      pb->GetIndexes()[0],
      is_a_idx_qnsct_expanded,
      b_idx_qnsct_coor_expanded_idx_qnsct_coor_map
  );
#ifdef QLTEN_TIMING_MODE
  expand_index_timer.PrintElapsed();
#endif
  // Expand data
  IndexVec<QNT> expanded_idxs = pa->GetIndexes();
  expanded_idxs[0] = expanded_index;
  (*pc) = QLTensor<TenElemT, QNT>(expanded_idxs);
  (pc->GetBlkSparDataTen()).ConstructExpandedDataOnFirstIndex(
      pa->GetBlkSparDataTen(),
      pb->GetBlkSparDataTen(),
      is_a_idx_qnsct_expanded,
      b_idx_qnsct_coor_expanded_idx_qnsct_coor_map
  );
#ifdef QLTEN_TIMING_MODE
  Timer expand_latter_transpose_timer("   =============> expansion_second_time_transpose");
#endif
  // transpose back
  for (size_t i = 0; i < expand_idx_num; i++) { transpose_order[i] = i + 1; }
  transpose_order[expand_idx_num] = 0;
  if (pa == pb) {
    pa->Transpose(transpose_order);
  } else {
    pa->Transpose(transpose_order);
    pb->Transpose(transpose_order);
  }
  pc->Transpose(transpose_order);
#ifdef QLTEN_TIMING_MODE
  expand_latter_transpose_timer.PrintElapsed();
#endif
#ifndef NDEBUG
  ExpandedTenDivChecker(*pa, *pb, *pc);
#endif /* ifndef NDEBUG */
}
} /* qlten */
#endif /* ifndef QLTEN_TENSOR_MANIPULATION_TEN_EXPAND_H */
