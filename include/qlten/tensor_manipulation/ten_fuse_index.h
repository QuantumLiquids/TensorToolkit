// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author:   Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
            Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2021-7-22 
*
* Description: QuantumLiquids/tensor project. Fuse two indices of tensor.
*/

/**
@file ten_fuse_index.h
@brief Fuse two indices of tensor.
*/
#ifndef QLTEN_TENSOR_MANIPULATION_TEN_FUSE_INDEX_H
#define QLTEN_TENSOR_MANIPULATION_TEN_FUSE_INDEX_H

#include <vector>       // vector
#include <map>          // map
#include <algorithm>    // find
#include <cassert>     // assert

#include "qlten/qltensor_all.h"     // QLTensor
#include "qlten/tensor_manipulation/index_combine.h"  // QNSctsOffsetInfo
#include "qlten/utility/timer.h"

#ifdef Release
#define NDEBUG
#endif

namespace qlten {

/**
 * @brief Information about a fused index, used for potential SplitIndex operation.
 *
 * This structure stores all necessary information to reverse the FuseIndex operation.
 * It records the original indices before fusion and the mapping information.
 *
 * @tparam QNT The quantum number type.
 *
 * @note Currently, this is designed for Abelian quantum numbers. For non-Abelian
 *       cases (e.g., SU(2)), additional Clebsch-Gordan coefficient information
 *       may be needed and could require storing FuseInfo inside the tensor.
 */
template<typename QNT>
struct FuseInfo {
  Index<QNT> original_idx1;   ///< The first original index before fusion.
  Index<QNT> original_idx2;   ///< The second original index before fusion.
  Index<QNT> fused_idx;       ///< The resulting fused index.

  /**
   * @brief Mapping from original QNSector pairs to the fused QNSector.
   *
   * Each tuple contains:
   * - std::get<0>: QNSector index in original_idx1
   * - std::get<1>: QNSector index in original_idx2
   * - std::get<2>: QNSector index in fused_idx
   * - std::get<3>: Offset within the fused QNSector's degeneracy space
   */
  std::vector<QNSctsOffsetInfo> qnscts_offset_info_list;

  FuseInfo() = default;

  FuseInfo(const Index<QNT> &idx1, const Index<QNT> &idx2,
           const Index<QNT> &fused, const std::vector<QNSctsOffsetInfo> &info)
      : original_idx1(idx1), original_idx2(idx2),
        fused_idx(fused), qnscts_offset_info_list(info) {}
};

// Forward declaration
template<typename QNT>
Index<QNT> FuseTwoIndexAndRecordInfo(
    const Index<QNT> &idx1,
    const Index<QNT> &idx2,
    std::vector<QNSctsOffsetInfo> &qnscts_offset_info_list
);

/**
 * @brief Fuse two indices of a tensor into one combined index.
 *
 * This function fuses two indices at positions `idx1` and `idx2` into a single
 * combined index. The fused index is placed at position 0 of the resulting tensor.
 *
 * @tparam TenElemT The element type of the tensor.
 * @tparam QNT The quantum number type of the tensor.
 *
 * @param idx1 Position of the first index to fuse (must be < idx2).
 * @param idx2 Position of the second index to fuse.
 *
 * @return FuseInfo<QNT> containing information about the fusion, which can be
 *         used for a future SplitIndex operation.
 *
 * @pre idx1 < idx2 < rank()
 * @pre Both indices must have the same direction (both IN or both OUT).
 *      Fusing indices with different directions is not supported.
 *
 * @note The indices do not need to be adjacent; the function will first
 *       transpose them to adjacent positions.
 *
 * @note The fused index will be placed at position 0. If a different position
 *       is desired, an additional Transpose() call is needed.
 *
 * ### Fermion Sign Handling
 *
 * For fermionic tensors (Z2-graded), the fermion exchange sign is handled
 * as follows:
 *
 * 1. **Transpose step**: The function first transposes idx1 and idx2 to
 *    positions 0 and 1 respectively. During this transpose, the fermion
 *    exchange sign is computed and applied based on the parity of the
 *    quantum number sectors being moved past each other. This is handled
 *    by `DataBlk::Transpose()` which uses `FermionicInplaceReorder()`.
 *
 * 2. **Fusion step**: Once the two indices are adjacent (at positions 0 and 1),
 *    the actual fusion is performed. Since the indices are now adjacent,
 *    no additional fermion exchange is needed during the fusion itself.
 *    The data is simply reorganized according to the quantum number
 *    sector mapping.
 *
 * ### Fused Index Structure
 *
 * For block-sparse tensors, the fused index structure is determined by quantum
 * number conservation. QN sectors from idx1 and idx2 that combine to the same
 * total QN are grouped together in the fused index. The detailed mapping is
 * recorded in `FuseInfo::qnscts_offset_info_list`.
 *
 * Within each combined QN sector, elements from (qnsct_i, qnsct_j) are stored
 * contiguously with row-major ordering (i.e., idx2 varies fastest).
 *
 * @warning Fusing indices with opposite directions (IN-OUT or OUT-IN) is
 *          currently not supported. Use IndexCombine with Contract for such cases.
 *
 * @todo Implement SplitIndex as the inverse operation using FuseInfo.
 *
 * @see FuseInfo, IndexCombine, SplitIndex (TODO)
 */
template<typename TenElemT, typename QNT>
FuseInfo<QNT> QLTensor<TenElemT, QNT>::FuseIndex(
    const size_t idx1,
    const size_t idx2
) {
  assert(idx1 < idx2 && idx2 < rank_);
  assert(indexes_[idx1].GetDir() == indexes_[idx2].GetDir());

  // Store original indices for FuseInfo before any modification
  Index<QNT> original_idx1 = indexes_[idx1];
  Index<QNT> original_idx2 = indexes_[idx2];

#ifdef QLTEN_TIMING_MODE
  Timer fuse_index_pre_transpose_timer("fuse_index_pre_trans");
#endif
  // Step 1: Transpose idx1 and idx2 to the leading positions.
  // It ensures that idx1 and idx2 are positioned as the first and second
  // indices, respectively, making the tensor operations more intuitive and efficient.
  // For fermionic tensors, this transpose step handles the fermion exchange sign
  // through FermionicInplaceReorder() in DataBlk::Transpose().
  std::vector<size_t> transpose_axes(rank_);
  transpose_axes[0] = idx1;
  transpose_axes[1] = idx2;
  for (size_t i = 2; i < idx1 + 2; i++) {
    transpose_axes[i] = i - 2;
  }
  for (size_t i = idx1 + 2; i < idx2 + 1; i++) {
    transpose_axes[i] = i - 1;
  }
  for (size_t i = idx2 + 1; i < rank_; i++) {
    transpose_axes[i] = i;
  }
  Transpose(transpose_axes);

#ifdef QLTEN_TIMING_MODE
  fuse_index_pre_transpose_timer.PrintElapsed();
#endif

  // Step 2: Generate new fused index and record mapping information.
  // Since the two indices are now adjacent (at positions 0 and 1),
  // no additional fermion exchange is needed during this step.
  std::vector<QNSctsOffsetInfo> qnscts_offset_info_list;
  Index<QNT> new_index = FuseTwoIndexAndRecordInfo(
      indexes_[0],
      indexes_[1],
      qnscts_offset_info_list
  );

  // Create FuseInfo before modifying the tensor
  FuseInfo<QNT> fuse_info(original_idx1, original_idx2, new_index, qnscts_offset_info_list);

  // Step 3: Update tensor metadata (index, shape, rank)
  indexes_.erase(indexes_.begin());
  indexes_[0] = new_index;
  shape_[1] = shape_[0] * shape_[1];
  shape_.erase(shape_.begin());
  rank_ = rank_ - 1;

  // Step 4: Update block sparse data tensor (data blocks and raw data)
  pblk_spar_data_ten_->FuseFirstTwoIndex(
      qnscts_offset_info_list
  );

  return fuse_info;
}

/**
 * @brief Fuse two indices and record the quantum number sector mapping.
 *
 * This helper function creates a new fused index by combining the quantum
 * number sectors of two input indices, and records the mapping information
 * needed to reorganize tensor data.
 *
 * @tparam QNT The quantum number type.
 *
 * @param idx1 The first index to fuse.
 * @param idx2 The second index to fuse.
 * @param[out] qnscts_offset_info_list Output vector containing mapping info.
 *
 * @return The new fused index.
 *
 * @pre Both indices must have defined directions (not NDIR).
 * @pre Both indices must have the same direction.
 *
 * @note For Abelian quantum numbers, the combined QN is simply qn1 + qn2.
 *       This function does not handle non-Abelian cases.
 */
template<typename QNT>
Index<QNT> FuseTwoIndexAndRecordInfo(
    const Index<QNT> &idx1,
    const Index<QNT> &idx2,
    std::vector<QNSctsOffsetInfo> &qnscts_offset_info_list
) {
  assert(idx1.GetDir() != TenIndexDirType::NDIR);
  assert(idx2.GetDir() != TenIndexDirType::NDIR);
  assert(idx1.GetDir() == idx2.GetDir());
  auto new_idx_dir = idx1.GetDir();
  auto idx1_qnsct_num = idx1.GetQNSctNum();
  auto idx2_qnsct_num = idx2.GetQNSctNum();
  qnscts_offset_info_list.reserve(idx1_qnsct_num * idx2_qnsct_num);
  std::vector<QNDgnc<QNT>> new_qn_dgnc_list;
  for (size_t i = 0; i < idx1_qnsct_num; ++i) {
    for (size_t j = 0; j < idx2_qnsct_num; ++j) {
      QNSector<QNT> qnsct_from_idx1 = idx1.GetQNSct(i);
      QNSector<QNT> qnsct_from_idx2 = idx2.GetQNSct(j);
      auto qn_from_idx1 = qnsct_from_idx1.GetQn();
      auto qn_from_idx2 = qnsct_from_idx2.GetQn();
      auto dgnc_from_idx1 = qnsct_from_idx1.GetDegeneracy();
      auto dgnc_from_idx2 = qnsct_from_idx2.GetDegeneracy();
      QNT combined_qn = qn_from_idx1 + qn_from_idx2;
      auto poss_it = std::find_if(
          new_qn_dgnc_list.begin(),
          new_qn_dgnc_list.end(),
          [&combined_qn](const QNDgnc<QNT> &qn_dgnc) -> bool {
            return qn_dgnc.first == combined_qn;
          }
      );
      if (poss_it != new_qn_dgnc_list.end()) {
        size_t offset = poss_it->second;
        poss_it->second += (dgnc_from_idx1 * dgnc_from_idx2);
        size_t qn_dgnc_idx = poss_it - new_qn_dgnc_list.begin();
        qnscts_offset_info_list.emplace_back(
            std::make_tuple(i, j, qn_dgnc_idx, offset)
        );
      } else {
        size_t qn_dgnc_idx = new_qn_dgnc_list.size();
        new_qn_dgnc_list.push_back(
            std::make_pair(combined_qn, dgnc_from_idx1 * dgnc_from_idx2)
        );
        qnscts_offset_info_list.emplace_back(
            std::make_tuple(i, j, qn_dgnc_idx, 0)
        );
      }
    }
  }
  QNSectorVec<QNT> qnscts;
  // std::vector<size_t> new_idx_qnsct_dim_offsets;
  qnscts.reserve(new_qn_dgnc_list.size());
  // new_idx_qnsct_dim_offsets.reserve(new_qn_dgnc_list.size());
  // size_t qnsct_dim_offset = 0;
  for (auto &new_qn_dgnc : new_qn_dgnc_list) {
    qnscts.push_back(QNSector<QNT>(new_qn_dgnc.first, new_qn_dgnc.second));
    // new_idx_qnsct_dim_offsets.push_back(qnsct_dim_offset);
    // qnsct_dim_offset += new_qn_dgnc.second;
  }
  return Index<QNT>(qnscts, new_idx_dir);
}
}

#endif //QLTEN_TENSOR_MANIPULATION_TEN_FUSE_INDEX_H