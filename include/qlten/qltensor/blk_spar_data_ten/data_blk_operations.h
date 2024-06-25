// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
*         Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2020-11-30 11:57
*
* Description: QuantumLiquids/tensor project. Data block level operations for block
* sparse data tensor.
*/

/**
@file data_blk_operations.h
@brief Data block level operations for block sparse data tensor.
*/
#ifndef QLTEN_QLTENSOR_BLK_SPAR_DATA_TEN_DATA_BLK_LEVEL_OPERATIONS_H
#define QLTEN_QLTENSOR_BLK_SPAR_DATA_TEN_DATA_BLK_LEVEL_OPERATIONS_H

#include "qlten/qltensor/blk_spar_data_ten/blk_spar_data_ten.h"
#include "qlten/qltensor/blk_spar_data_ten/raw_data_operations.h"
#include "qlten/qltensor/blk_spar_data_ten/raw_data_operation_tasks.h"
#include "qlten/framework/hp_numeric/lapack.h"    // MatSVD, MatQR
#include "qlten/framework/hp_numeric/omp_set.h"
#include "qlten/framework/hp_numeric/mpi_fun.h"   // MPI_Send, MPI_Recv
#include "qlten/utility/timer.h"
#include <omp.h>

#include <cstring>      // memcpy
#ifdef Release
#define NDEBUG
#endif
#include <cassert>     // assert

namespace qlten {

/**
Insert a new data block. User can decide whether allocate the memory.

@param blk_coors Block coordinates of the new data block.
@param alloc_mem Whether allocate the memory and set them to 0.

@return An iterator for this new block index <-> data block pair.

@note You can only gap the memory allocation procedure when the raw data is empty.
*/
template<typename ElemT, typename QNT>
typename BlockSparseDataTensor<ElemT, QNT>::BlkIdxDataBlkMap::iterator
BlockSparseDataTensor<ElemT, QNT>::DataBlkInsert(
    const CoorsT &blk_coors, const bool alloc_mem
) {
  assert(!blk_coors.empty());
  auto blk_idx = BlkCoorsToBlkIdx(blk_coors);
  blk_idx_data_blk_map_[blk_idx] = DataBlk<QNT>(blk_coors, *pqlten_indexes);
  size_t inserted_data_size = blk_idx_data_blk_map_[blk_idx].size;
  size_t total_data_offset = 0;

  auto iter = blk_idx_data_blk_map_.find(blk_idx);
  if (iter != blk_idx_data_blk_map_.cbegin()) {
    --iter;
    total_data_offset = iter->second.data_offset + iter->second.size;
    ++iter;
    iter->second.data_offset = total_data_offset;
  } else {
    iter->second.data_offset = 0;
  }

  for (++iter; iter != blk_idx_data_blk_map_.cend(); ++iter) {
    iter->second.data_offset += inserted_data_size;
  }
  raw_data_size_ += inserted_data_size;

  if (alloc_mem) {
    RawDataInsert_(total_data_offset, inserted_data_size, true);
  } else {
    assert(pactual_raw_data_ == nullptr);
  }

  return blk_idx_data_blk_map_.find(blk_idx);
}

template<typename ElemT, typename QNT>
typename BlockSparseDataTensor<ElemT, QNT>::BlkIdxDataBlkMap::iterator
BlockSparseDataTensor<ElemT, QNT>::DataBlkQuasiInsert(
    const CoorsT &blk_coors
) {
  assert(blk_coors.size() == ten_rank);
  auto blk_idx = BlkCoorsToBlkIdx(blk_coors);
  auto iter = blk_idx_data_blk_map_.find(blk_idx);
  if (iter != blk_idx_data_blk_map_.cend()) {
    return iter;
  } else {
    auto [iter, success] = blk_idx_data_blk_map_.insert(
        std::make_pair(blk_idx,
                       DataBlk<QNT>(blk_coors, *pqlten_indexes)
        )
    );
    return iter;
  }
}

template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::DataBlksOffsetRefresh() {
  raw_data_size_ = 0;
  for (auto &idx_data_blk: blk_idx_data_blk_map_) {
    auto &data_blk = idx_data_blk.second;
    data_blk.data_offset = raw_data_size_;
    raw_data_size_ += data_blk.size;
  }
}

/**
Insert a list of data blocks. The BlockSparseDataTensor must be empty before
performing this insertion.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::DataBlksInsert(
    const std::vector<size_t> &blk_idxs,
    const std::vector<CoorsT> &blk_coors_s,
    const bool alloc_mem,
    const bool init
) {
  assert(blk_idx_data_blk_map_.empty());
  //it's better that every CoorsT is unique.
  //if not unique, it will also work
  auto iter = blk_idxs.begin();
  for (auto &blk_coors: blk_coors_s) {
    size_t blk_idx = *iter;
    blk_idx_data_blk_map_[blk_idx] = DataBlk<QNT>(blk_coors, *pqlten_indexes);
    iter++;
  }
  raw_data_size_ = 0;
  for (auto &idx_data_blk: blk_idx_data_blk_map_) {
    auto &data_blk = idx_data_blk.second;
    data_blk.data_offset = raw_data_size_;
    raw_data_size_ += data_blk.size;
  }
  if (alloc_mem) {
    Allocate(init);
  } else {
    assert(pactual_raw_data_ == nullptr);
  }
}

/**
Insert a list of data blocks. The BlockSparseDataTensor must be empty before
performing this insertion.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::DataBlksInsert(
    const std::vector<CoorsT> &blk_coors_s,
    const bool alloc_mem,
    const bool init
) {
  assert(blk_idx_data_blk_map_.empty());
  //it's better that every CoorsT is unique.
  //if not unique, it will also work
  std::vector<size_t> blk_idxs;
  blk_idxs.reserve(blk_coors_s.size());
  for (auto &blk_coors: blk_coors_s) {
    blk_idxs.push_back(std::move(BlkCoorsToBlkIdx(blk_coors)));
  }
  DataBlksInsert(blk_idxs, blk_coors_s, alloc_mem, init);
}

template<typename ElemT, typename QNT>
DataBlk<QNT> BlockSparseDataTensor<ElemT, QNT>::GetMaxSizeDataBlk(void) const {
  if (actual_raw_data_size_ == 0) {
    return DataBlk<QNT>();
  }

  size_t max_size = 0;
  size_t max_idx;
  for (auto &idx_data_blk: blk_idx_data_blk_map_) {
    size_t this_blk_size = idx_data_blk.second.size();
    if (this_blk_size > max_size) {
      max_size = this_blk_size;
      max_idx = idx_data_blk.first;
    }
  }
  return blk_idx_data_blk_map_.at(max_idx);
}

template<typename ElemT, typename QNT>
size_t BlockSparseDataTensor<ElemT, QNT>::GetMaxDataBlkSize(void) const {
  if (actual_raw_data_size_ == 0) {
    return 0;
  }
  size_t max_size = 0;
  for (auto &idx_data_blk: blk_idx_data_blk_map_) {
    size_t this_blk_size = idx_data_blk.second.size();
    max_size = std::max(max_size, this_blk_size);
  }
  return max_size;
}
/**
Copy and rescale raw data from another tensor.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::DataBlkCopyAndScale(
    const RawDataCopyAndScaleTask<ElemT> &task,
    const ElemT *pten_raw_data
) {
  RawDataCopyAndScale_(task, pten_raw_data);
}

template<typename ElemT, typename QNT>
std::vector<RawDataFermionNormTask> BlockSparseDataTensor<ElemT, QNT>::GenFermionNormTask() const {
  std::vector<size_t> in_indices;
  for (size_t i = 0; i < ten_rank; ++i) {
    if (this->pqlten_indexes[i].GetDir() == TenIndexDirType::IN) {
      in_indices.push_back(i);
    }
  }

  std::vector<RawDataFermionNormTask> tasks;
  tasks.reserve(blk_idx_data_blk_map_.size());

  for (const auto &[idx, data_blk]: blk_idx_data_blk_map_) {
    const std::vector<bool> &parities = data_blk.GetBlkFermionParities();
    int in_count = std::count_if(in_indices.begin(), in_indices.end(),
                                 [&parities](size_t index) { return parities[index]; });

    int ex_sign = (in_count % 2 == 0) ? 1 : -1;
    tasks.emplace_back(data_blk.data_offset, data_blk.size, ex_sign);
  }

  return tasks;
}

// Some helpers for tensor contraction
inline std::vector<std::vector<size_t>> TenCtrctGenSavedAxesSet(
    const size_t a_rank,
    const size_t b_rank,
    const std::vector<std::vector<size_t>> &ctrct_axes_set
) {
  auto a_ctrct_axes = ctrct_axes_set[0];
  auto b_ctrct_axes = ctrct_axes_set[1];
  std::vector<std::vector<size_t>> saved_axes_set;
  saved_axes_set.reserve(2);
  std::vector<size_t> a_saved_axes;
  for (size_t i = 0; i < a_rank; ++i) {
    if (std::find(a_ctrct_axes.begin(), a_ctrct_axes.end(), i) ==
        a_ctrct_axes.end()
        ) {
      a_saved_axes.push_back(i);
    }
  }
  saved_axes_set.emplace_back(a_saved_axes);
  std::vector<size_t> b_saved_axes;
  for (size_t i = 0; i < b_rank; ++i) {
    if (std::find(b_ctrct_axes.begin(), b_ctrct_axes.end(), i) ==
        b_ctrct_axes.end()
        ) {
      b_saved_axes.push_back(i);
    }
  }
  saved_axes_set.emplace_back(b_saved_axes);
  return saved_axes_set;
}

inline std::pair<bool, bool> TenCtrctNeedTransCheck(
    const std::vector<std::vector<size_t>> &ctrct_axes_set,
    const std::vector<std::vector<size_t>> &saved_axes_set,
    std::vector<size_t> &a_trans_orders,
    std::vector<size_t> &b_trans_orders
) {
  a_trans_orders = saved_axes_set[0];
  a_trans_orders.insert(
      a_trans_orders.end(),
      ctrct_axes_set[0].begin(),
      ctrct_axes_set[0].end()
  );
  bool a_need_trans;
  if (std::is_sorted(a_trans_orders.begin(), a_trans_orders.end())) {
    a_need_trans = false;
  } else {
    a_need_trans = true;
  }

  b_trans_orders = ctrct_axes_set[1];
  b_trans_orders.insert(
      b_trans_orders.end(),
      saved_axes_set[1].begin(),
      saved_axes_set[1].end()
  );
  bool b_need_trans;
  if (std::is_sorted(b_trans_orders.begin(), b_trans_orders.end())) {
    b_need_trans = false;
  } else {
    b_need_trans = true;
  }

  return std::make_pair(a_need_trans, b_need_trans);
}

inline CoorsT GenTenCtrctDataBlkCoors(
    const CoorsT &a_blk_coors,
    const CoorsT &b_blk_coors,
    const std::vector<std::vector<size_t>> &saved_axes_set
) {
  CoorsT c_blk_coors;
  c_blk_coors.reserve(saved_axes_set[0].size() + saved_axes_set[1].size());
  for (auto axis: saved_axes_set[0]) {
    c_blk_coors.push_back(a_blk_coors[axis]);
  }
  for (auto axis: saved_axes_set[1]) {
    c_blk_coors.push_back(b_blk_coors[axis]);
  }
  return c_blk_coors;
}

/// Contract(OUT--->IN) give 1
/// Contract(IN--->OUT) gives potential -1
inline int FermionExchangeSignForCtrct(
    const std::vector<bool> &a_ten_blk_qn_parities,
    const std::vector<bool> &b_ten_blk_qn_parities,
    const std::vector<size_t> &a_ctrct_axes,
    const std::vector<size_t> &b_ctrct_axes,
    const std::vector<TenIndexDirType> &a_ctrct_idx_dir
) {
  int exchange_num = 0;
  std::vector<bool> concatenated_parities = a_ten_blk_qn_parities;
  concatenated_parities.insert(concatenated_parities.end(), b_ten_blk_qn_parities.begin(), b_ten_blk_qn_parities.end());

  size_t rank_a = a_ten_blk_qn_parities.size();
  std::vector<size_t> concatenated_axes(2 * a_ctrct_axes.size());

  for (size_t i = 0; i < a_ctrct_axes.size(); i++) {
    concatenated_axes[2 * i] = a_ctrct_axes[i];
    concatenated_axes[2 * i + 1] = b_ctrct_axes[i] + rank_a;
  }

  for (size_t i = 0; i < a_ctrct_axes.size(); i++) {
    // move concatenated_axes[2 * i + 1] so that it equals concatenated_axes[2 * i] + 1
    // at the same time, count the fermion exchange times and move elements in concatenated_parities
    // also, other elements in concatenated_axes may need to be redefined
    size_t axes1 = concatenated_axes[2 * i];
    bool exist_particle1 = concatenated_parities[axes1];
    size_t axes2 = concatenated_axes[2 * i + 1];
    bool exist_particle2 = concatenated_parities[axes2];
    assert(exist_particle2 == exist_particle1);

    if (exist_particle2) {
      int fermion_num_between_ctrct_axes = std::count(
          concatenated_parities.begin() + axes1 + 1, concatenated_parities.begin() + axes2, true);
      exchange_num += fermion_num_between_ctrct_axes;
    }

    // Use std::rotate to shift elements left
    std::rotate(concatenated_parities.begin() + axes1 + 1,
                concatenated_parities.begin() + axes2,
                concatenated_parities.begin() + axes2 + 1);

    for (auto &axes: concatenated_axes) {
      if (axes > axes1 && axes < axes2) {
        axes++;
      }
    }
    concatenated_axes[2 * i + 1] = axes1 + 1;

    exchange_num += exist_particle1 && (a_ctrct_idx_dir[i] == IN);
  }
  return (exchange_num & 1) ? -1 : 1;
}

/**
Generate data blocks for two tensor contraction.

@param bsdt_a Block sparse data tensor A.
@param bsdt_b Block sparse data tensor B.
@param ctrct_axes_set To-be contracted tensor axes indexes.
       For example, {{0, 1}, {3, 2}}.
*/
template<typename ElemT, typename QNT>
std::vector<RawDataCtrctTask>
BlockSparseDataTensor<ElemT, QNT>::DataBlkGenForTenCtrct(
    const BlockSparseDataTensor &bsdt_a,
    const BlockSparseDataTensor &bsdt_b,
    const std::vector<std::vector<size_t>> &ctrct_axes_set,
    const std::vector<std::vector<size_t>> &saved_axes_set
) {
  assert(!(bsdt_a.IsScalar() || bsdt_b.IsScalar()));
  //TODO: parallel
//  const int ompth = hp_numeric::tensor_manipulation_num_threads;

  const auto &a_blk_idx_data_blk_map = bsdt_a.GetBlkIdxDataBlkMap();
  const auto &b_blk_idx_data_blk_map = bsdt_b.GetBlkIdxDataBlkMap();
  auto a_blk_idx_coor_part_hash_map = GenBlkIdxQNBlkCoorPartHashMap(
      a_blk_idx_data_blk_map,
      ctrct_axes_set[0]
  );
  auto b_blk_idx_coor_part_hash_map = GenBlkIdxQNBlkCoorPartHashMap(
      b_blk_idx_data_blk_map,
      ctrct_axes_set[1]
  );
  std::vector<RawDataCtrctTask> raw_data_ctrct_tasks;
  std::unordered_map<size_t, size_t> b_blk_idx_n_map;

  std::vector<TenIndexDirType> a_ctrct_idx_dir; // use for fermion sign
  if constexpr (Fermionicable<QNT>::IsFermionic()) {
    a_ctrct_idx_dir.resize(ctrct_axes_set[0].size());
    for (size_t i = 0; i < ctrct_axes_set[0].size(); i++) {
      const Index<QNT> &index = bsdt_a->pqlten_indexes[ctrct_axes_set[0][i]];
      a_ctrct_idx_dir[i] = index.GetDir();
    }
  }

  bool c_is_scalar = IsScalar();
#ifndef NDEBUG
  if (c_is_scalar) {
    assert(saved_axes_set[0].empty() && saved_axes_set[1].empty());
  }
#endif /* ifndef NDEBUG */
  if (c_is_scalar) {
    raw_data_ctrct_tasks.reserve(a_blk_idx_coor_part_hash_map.size() / 2);
    const size_t m(1), n(1);
    QLTEN_Double beta(1.0), f_sign(1.0);  //fermion sign is set as 1.0 by default for boson cases.
    for (size_t i = 0; i < a_blk_idx_coor_part_hash_map.size(); i += 2) {
      for (size_t j = 0; j < b_blk_idx_coor_part_hash_map.size(); j += 2) {
        if (a_blk_idx_coor_part_hash_map[i + 1] == b_blk_idx_coor_part_hash_map[j + 1]) {
          auto a_blk_idx = a_blk_idx_coor_part_hash_map[i];
          auto b_blk_idx = b_blk_idx_coor_part_hash_map[j];
          const auto &a_data_blk = a_blk_idx_data_blk_map.at(a_blk_idx);
          const auto &b_data_blk = b_blk_idx_data_blk_map.at(b_blk_idx);
          size_t k = VecMultiSelectElemts(a_data_blk.shape, ctrct_axes_set[0]);
          if constexpr (Fermionicable<QNT>::IsFermionic()) {
            f_sign = FermionExchangeSignForCtrct(a_data_blk.GetBlkFermionParities(),
                                                 b_data_blk.GetBlkFermionParities(),
                                                 ctrct_axes_set[0], ctrct_axes_set[1],
                                                 a_ctrct_idx_dir);
          }
          raw_data_ctrct_tasks.push_back(
              RawDataCtrctTask(
                  a_blk_idx,
                  a_data_blk.data_offset,
                  b_blk_idx,
                  b_data_blk.data_offset,
                  m, k, n,
                  f_sign,
                  beta
              )
          );
          break;
        }
      }
    }
    if (!raw_data_ctrct_tasks.empty()) {
      raw_data_size_ = 1;
      raw_data_ctrct_tasks[0].beta = 0.0;
    }
//#pragma omp parallel for default(shared) num_threads(ompth) schedule(static)
    for (auto &task: raw_data_ctrct_tasks) {
      task.c_data_offset = 0;
    }
  } else { //if c is not scalar
    raw_data_ctrct_tasks.reserve(a_blk_idx_coor_part_hash_map.size() * b_blk_idx_coor_part_hash_map.size() / 4);
    for (size_t i = 0; i < a_blk_idx_coor_part_hash_map.size(); i += 2) {
      size_t m(0), k(0);
      for (size_t j = 0; j < b_blk_idx_coor_part_hash_map.size(); j += 2) {
        if (a_blk_idx_coor_part_hash_map[i + 1] == b_blk_idx_coor_part_hash_map[j + 1]) {
          auto a_blk_idx = a_blk_idx_coor_part_hash_map[i];
          auto b_blk_idx = b_blk_idx_coor_part_hash_map[j];
          const auto &a_data_blk = a_blk_idx_data_blk_map.at(a_blk_idx);
          const auto &b_data_blk = b_blk_idx_data_blk_map.at(b_blk_idx);
          // Calculate m, k, n
          size_t n;
          if (m == 0) {
            m = VecMultiSelectElemts(a_data_blk.shape, saved_axes_set[0]);
            k = VecMultiSelectElemts(a_data_blk.shape, ctrct_axes_set[0]);
          }
          if (b_blk_idx_n_map.find(b_blk_idx) != b_blk_idx_n_map.end()) {
            n = b_blk_idx_n_map.at(b_blk_idx);
          } else {
            n = VecMultiSelectElemts(b_data_blk.shape, saved_axes_set[1]);
            b_blk_idx_n_map.emplace(b_blk_idx, n);
          }


          // Create raw data contraction task
          auto c_blk_coors = GenTenCtrctDataBlkCoors(
              a_data_blk.blk_coors,
              b_data_blk.blk_coors,
              saved_axes_set
          );
          auto c_blk_idx = BlkCoorsToBlkIdx(c_blk_coors);
          QLTEN_Double beta;
          if (blk_idx_data_blk_map_.find(c_blk_idx) !=
              blk_idx_data_blk_map_.end()
              ) {
            beta = 1.0;
          } else {
            auto c_blk_shape = GenTenCtrctDataBlkCoors(
                a_data_blk.shape,
                b_data_blk.shape,
                saved_axes_set
            );
            blk_idx_data_blk_map_[c_blk_idx] =
                DataBlk<QNT>(std::move(c_blk_coors), std::move(c_blk_shape));
            beta = 0.0;
          }

          double f_sign = 1.0;
          if constexpr (Fermionicable<QNT>::IsFermionic()) {
            f_sign = FermionExchangeSignForCtrct(a_data_blk.GetBlkFermionParities(),
                                                 b_data_blk.GetBlkFermionParities(),
                                                 ctrct_axes_set[0], ctrct_axes_set[1],
                                                 a_ctrct_idx_dir);
          }

          raw_data_ctrct_tasks.push_back(
              RawDataCtrctTask(
                  a_blk_idx,
                  a_data_blk.data_offset,
                  b_blk_idx,
                  b_data_blk.data_offset,
                  c_blk_idx,
                  m, k, n,
                  f_sign,
                  beta
              )
          );
        }
      }
    }
    DataBlksOffsetRefresh();

//#pragma omp parallel for default(shared) num_threads(ompth) schedule(static)
    for (size_t i = 0; i < raw_data_ctrct_tasks.size(); i++) {
      auto &task = raw_data_ctrct_tasks[i];
      task.c_data_offset = blk_idx_data_blk_map_[task.c_blk_idx].data_offset;
    }
  }

  return raw_data_ctrct_tasks;
}

template<typename ElemT, typename QNT>
std::map<size_t, int>
BlockSparseDataTensor<ElemT, QNT>::CountResidueFermionSignForMatBasedCtrct(
    const std::set<size_t> &selected_blk_idx,
    const std::vector<size_t> &saved_axes_set,
    const size_t trans_critical_axe
) {
  std::map<size_t, int> idx_fermion_sign_map;
  std::vector<size_t> saved_axes_first, saved_axes_second; //also should be continuous numbers
  for (auto axe: saved_axes_set) {
    if (axe < trans_critical_axe) {
      saved_axes_first.push_back(axe);
      assert(saved_axes_first[0] == 0);
      assert(saved_axes_first.back() == saved_axes_first.size() - 1);
    } else {
      saved_axes_second.push_back(axe);
    }
  }
  for (auto &idx: selected_blk_idx) {
    const DataBlk<QNT> &data_blk = blk_idx_data_blk_map_[idx];
    const std::vector<double> fermion_parities = data_blk.GetBlkFermionParities();
    size_t fermion_num1 = std::count(
        fermion_parities.begin(), fermion_parities.begin() + saved_axes_first.size(), true);
    size_t fermion_num2 = std::count(
        fermion_parities.begin() + saved_axes_second[0], fermion_parities.end(), true);
    idx_fermion_sign_map[idx] = ((fermion_num1 & 1) && (fermion_num2 & 1)) ? -1 : 1;
  }
  return idx_fermion_sign_map;
}

/**D
SVD decomposition.
*/
template<typename ElemT, typename QNT>
std::map<size_t, DataBlkMatSvdRes<ElemT>>
BlockSparseDataTensor<ElemT, QNT>::DataBlkDecompSVD(
    const IdxDataBlkMatMap<QNT> &idx_data_blk_mat_map
) const {
  std::map<size_t, DataBlkMatSvdRes<ElemT>> idx_svd_res_map;

#ifdef QLTEN_TIMING_MODE
  Timer svd_mkl_timer("matrix svd");
  svd_mkl_timer.Suspend();
#endif
  for (auto &[idx, data_blk_mat]: idx_data_blk_mat_map) {
    ElemT *mat = RawDataGenDenseDataBlkMat_(data_blk_mat);
    ElemT *u = nullptr;
    ElemT *vt = nullptr;
    QLTEN_Double *s = nullptr;
    size_t m = data_blk_mat.rows;
    size_t n = data_blk_mat.cols;
    size_t k = m > n ? n : m;
#ifdef QLTEN_TIMING_MODE
    svd_mkl_timer.Restart();
#endif
    hp_numeric::MatSVD(mat, m, n, u, s, vt);
#ifdef QLTEN_TIMING_MODE
    svd_mkl_timer.Suspend();
#endif
    free(mat);
    idx_svd_res_map[idx] = DataBlkMatSvdRes<ElemT>(m, n, k, u, s, vt);
  }
#ifdef QLTEN_TIMING_MODE
  svd_mkl_timer.PrintElapsed();
#endif
  return idx_svd_res_map;
}

/**
SVD decomposition. Multi-thread version required MPI initial as
 MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, provided);
 Nov 9 2023, I decide to remove the multi-thread version,
 and replace it by the standard Producer-consumer scheme
*/
/*
template<typename ElemT, typename QNT>
std::map<size_t, DataBlkMatSvdRes<ElemT>>
BlockSparseDataTensor<ElemT, QNT>::DataBlkDecompSVDMaster(
    const IdxDataBlkMatMap<QNT> &idx_data_blk_mat_map,
    boost::mpi::communicator &world
) const {
#ifdef QLTEN_MPI_TIMING_MODE
  Timer data_blk_decomp_svd_master_timer("data_blk_decomp_svd_master_func");
#endif
  using namespace std;
  /// This setting give Slave
  std::map<size_t, DataBlkMatSvdRes<ElemT>> idx_svd_res_map;

  auto iter = idx_data_blk_mat_map.begin();

  const size_t slave_num = world.size() - 1;
  const size_t task_size = idx_data_blk_mat_map.size();
  // if task_size < slave num the code should also work
  // suppose procs num >= slave num
  // for parallel do not need in order
#pragma omp parallel for default(none) \
                shared(task_size, idx_svd_res_map, iter, world, cout)\
                num_threads(slave_num)\
                schedule(dynamic)
  for (size_t i = 0; i < task_size; i++) {
    size_t controlling_slave = omp_get_thread_num() + 1;
    // size_t threads =  omp_get_num_threads();
    TenDecompDataBlkMat<QNT> data_blk_mat;
    size_t idx;
#pragma omp critical
    {
      idx = iter->first;
      data_blk_mat = iter->second;
      iter++;
    }
    ElemT *mat = RawDataGenDenseDataBlkMat_(data_blk_mat);
    const size_t m = data_blk_mat.rows;
    const size_t n = data_blk_mat.cols;
    hp_numeric::MPI_Send(m, controlling_slave, 2 * controlling_slave, MPI_Comm(world));
    hp_numeric::MPI_Send(n, controlling_slave, 3 * controlling_slave, MPI_Comm(world));
    hp_numeric::MPI_Send(mat, m * n, controlling_slave, 4 * controlling_slave, MPI_Comm(world));

    const size_t ld = std::min(m, n);
    ElemT *u = (ElemT *) malloc((ld * m) * sizeof(ElemT));
    ElemT *vt = (ElemT *) malloc((ld * n) * sizeof(ElemT));
    QLTEN_Double *s = (QLTEN_Double *) malloc(ld * sizeof(QLTEN_Double));
    const size_t k = m > n ? n : m;

    //TODO change to non-block
    hp_numeric::MPI_Recv(u, ld * m, controlling_slave, 5 * controlling_slave, MPI_Comm(world));
    hp_numeric::MPI_Recv(vt, ld * n, controlling_slave, 6 * controlling_slave, MPI_Comm(world));
    hp_numeric::MPI_Recv(s, ld, controlling_slave, 7 * controlling_slave, MPI_Comm(world));
    free(mat);
#pragma omp critical
    {
      idx_svd_res_map[idx] = DataBlkMatSvdRes<ElemT>(m, n, k, u, s, vt);
    }
  }
  //make sure thread is safe
  assert(iter == idx_data_blk_mat_map.end());

  for (int slave = 1; slave < world.size(); slave++) {
    //send finish signal
    const size_t m = 0;
    const size_t n = 0;
    hp_numeric::MPI_Send(m, slave, 2 * slave, MPI_Comm(world));
    hp_numeric::MPI_Send(n, slave, 3 * slave, MPI_Comm(world));
  }
#ifdef QLTEN_MPI_TIMING_MODE
  data_blk_decomp_svd_master_timer.PrintElapsed();
#endif
  return idx_svd_res_map;
}
*/


template<typename ElemT, typename QNT>
std::map<size_t, DataBlkMatSvdRes<ElemT>>
BlockSparseDataTensor<ElemT, QNT>::DataBlkDecompSVDMaster(
    const IdxDataBlkMatMap<QNT> &idx_data_blk_mat_map,
    const boost::mpi::communicator &world
) const {
#ifdef QLTEN_MPI_TIMING_MODE
  Timer data_blk_decomp_svd_master_timer("data_blk_decomp_svd_master_func");
#endif
  std::map<size_t, DataBlkMatSvdRes<ElemT>> idx_svd_res_map;

  const size_t worker_num = world.size() - 1;
  /**
   * Send package: (idx, m, n, mat)
   * Recv package: (idx, m, n, u, s, v)
   */
  size_t m, n; //m : rows; n : cols;
  int idx; // data blk idx
  for (auto &[idx_next, data_blk_mat_next]: idx_data_blk_mat_map) {
    // Wait for a worker to become available
    MPI_Status status;
    MPI_Recv(&idx, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_Comm(world), &status);
    size_t worker_id = status.MPI_SOURCE;
    if (status.MPI_TAG > 0) { //valid results
      hp_numeric::MPI_Recv(m, worker_id, idx + 14, MPI_Comm(world));
      hp_numeric::MPI_Recv(n, worker_id, idx + 15, MPI_Comm(world));
      const size_t ld = std::min(m, n);
      QLTEN_Double *s = (QLTEN_Double *) malloc(ld * sizeof(QLTEN_Double));
      ElemT *u = (ElemT *) malloc((ld * m) * sizeof(ElemT));
      ElemT *vt = (ElemT *) malloc((ld * n) * sizeof(ElemT));
      const size_t k = m > n ? n : m;
      hp_numeric::MPI_Recv(u, ld * m, worker_id, idx + 16, MPI_Comm(world));
      hp_numeric::MPI_Recv(s, ld, worker_id, idx + 17, MPI_Comm(world));
      hp_numeric::MPI_Recv(vt, ld * n, worker_id, idx + 18, MPI_Comm(world));
      idx_svd_res_map[idx] = DataBlkMatSvdRes<ElemT>(m, n, k, u, s, vt);
    }
    // Distribute data to available worker
    ElemT *mat = RawDataGenDenseDataBlkMat_(data_blk_mat_next);
    size_t data_length = data_blk_mat_next.rows * data_blk_mat_next.cols;
    MPI_Send(&idx_next, 1, MPI_INT, worker_id, idx_next + 6, MPI_Comm(world)); // The tag should be positive
    hp_numeric::MPI_Send(data_blk_mat_next.rows, worker_id, idx_next + 7, MPI_Comm(world));
    hp_numeric::MPI_Send(data_blk_mat_next.cols, worker_id, idx_next + 8, MPI_Comm(world));
    hp_numeric::MPI_Send(mat, data_length, worker_id, idx_next + 9, MPI_Comm(world));
    free(mat);
  }

  // Send termination signal to each rank when they submit their last job
  for (int num_terminated = 0; num_terminated < (int) worker_num; num_terminated++) {
    // Wait for a worker to become available
    MPI_Status status;
    MPI_Recv(&idx, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_Comm(world), &status);
    size_t worker_id = status.MPI_SOURCE;
    // If a matrix was decomposed
    if (status.MPI_TAG > 0) {
      hp_numeric::MPI_Recv(m, worker_id, idx + 14, MPI_Comm(world));
      hp_numeric::MPI_Recv(n, worker_id, idx + 15, MPI_Comm(world));
      const size_t ld = std::min(m, n);
      QLTEN_Double *s = (QLTEN_Double *) malloc(ld * sizeof(QLTEN_Double));
      ElemT *u = (ElemT *) malloc((ld * m) * sizeof(ElemT));
      ElemT *vt = (ElemT *) malloc((ld * n) * sizeof(ElemT));
      const size_t k = m > n ? n : m;
      hp_numeric::MPI_Recv(u, ld * m, worker_id, idx + 16, MPI_Comm(world));
      hp_numeric::MPI_Recv(s, ld, worker_id, idx + 17, MPI_Comm(world));
      hp_numeric::MPI_Recv(vt, ld * n, worker_id, idx + 18, MPI_Comm(world));
      idx_svd_res_map[idx] = DataBlkMatSvdRes<ElemT>(m, n, k, u, s, vt);
    }

    // Send termination signal (tag = 0); workers receive the variable as idx
    MPI_Send(&num_terminated, 1, MPI_INT, worker_id, 0, MPI_Comm(world));
  }
  MPI_Barrier(MPI_Comm(world));
#ifdef QLTEN_MPI_TIMING_MODE
  data_blk_decomp_svd_master_timer.PrintElapsed();
#endif
  return idx_svd_res_map;
}
// The MPI SVD version corresponding to the multi-thread version.
// Nov 9 2023, I decide to remove the multi-thread version,
// and replace it by the standard Producer-consumer scheme
/*
template<typename ElemT>
void DataBlkDecompSVDSlave(boost::mpi::communicator &world) {
  size_t task_done = 0;
  size_t m, n;//check if IdxDataBlkMatMap must have >0 row and  column
  size_t slave_identifier = world.rank();
  hp_numeric::MPI_Recv(m, kMPIMasterRank, 2 * slave_identifier, MPI_Comm(world));
  hp_numeric::MPI_Recv(n, kMPIMasterRank, 3 * slave_identifier, MPI_Comm(world));
#ifdef QLTEN_MPI_TIMING_MODE
  Timer slave_total_work_timer("slave " + std::to_string(slave_identifier) + " total work");
  Timer slave_commu_timer("slave " + std::to_string(slave_identifier) + " communication and wait");
#endif
  while (m > 0 && n > 0) {
    size_t data_size = m * n;
    ElemT *mat = (ElemT *) malloc(data_size * sizeof(ElemT));
    hp_numeric::MPI_Recv(mat, data_size, kMPIMasterRank, 4 * slave_identifier, MPI_Comm(world));
#ifdef QLTEN_MPI_TIMING_MODE
    slave_commu_timer.Suspend();
#endif
    ElemT *u = nullptr;
    ElemT *vt = nullptr;
    QLTEN_Double *s = nullptr;
    hp_numeric::MatSVD(mat, m, n, u, s, vt);

    size_t ld = std::min(m, n);
#ifdef QLTEN_MPI_TIMING_MODE
    slave_commu_timer.Restart();
#endif
    hp_numeric::MPI_Send(u, ld * m, kMPIMasterRank, 5 * slave_identifier, MPI_Comm(world));
    hp_numeric::MPI_Send(vt, ld * n, kMPIMasterRank, 6 * slave_identifier, MPI_Comm(world));
    hp_numeric::MPI_Send(s, ld, kMPIMasterRank, 7 * slave_identifier, MPI_Comm(world));

    free(mat);
    free(u);
    free(vt);
    free(s);

    task_done++;

    hp_numeric::MPI_Recv(m, kMPIMasterRank, 2 * slave_identifier, MPI_Comm(world));
    hp_numeric::MPI_Recv(n, kMPIMasterRank, 3 * slave_identifier, MPI_Comm(world));
  }
#ifdef QLTEN_MPI_TIMING_MODE
  std::cout << "slave " << slave_identifier << " has done " << task_done << " tasks." << std::endl;
  slave_total_work_timer.PrintElapsed();
  slave_commu_timer.PrintElapsed();
#endif
}
 */

template<typename ElemT>
void DataBlkDecompSVDSlave(const boost::mpi::communicator &world) {
  size_t task_done = 0;
  size_t idx(0), m(0), n(0);
  size_t consumer_rank = world.rank();

  /**
   * Recv package: (idx, m, n, mat)
   * Send package: (idx, m, n, u, s, v)
   */
  MPI_Send(&idx, 1, MPI_INT, kMPIMasterRank, 0, MPI_Comm(world)); // The signal I'm available

#ifdef QLTEN_MPI_TIMING_MODE
  Timer slave_total_work_timer("worker " + std::to_string(consumer_rank) + " total work");
  Timer slave_commu_timer("worker " + std::to_string(consumer_rank) + " communication and wait");
  slave_commu_timer.Suspend();
#endif
  bool terminated(false);
  do {
    // Wait for a job
    MPI_Status status;
    MPI_Recv(&idx, 1, MPI_INT, kMPIMasterRank, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    if (status.MPI_TAG == 0) {
      terminated = true;
    } else {
#ifdef QLTEN_MPI_TIMING_MODE
      slave_commu_timer.Restart();
#endif
      hp_numeric::MPI_Recv(m, kMPIMasterRank, idx + 7, MPI_Comm(world));
      hp_numeric::MPI_Recv(n, kMPIMasterRank, idx + 8, MPI_Comm(world));
      size_t data_size = m * n;
      ElemT *mat = (ElemT *) malloc(data_size * sizeof(ElemT));
      hp_numeric::MPI_Recv(mat, data_size, kMPIMasterRank, idx + 9, MPI_Comm(world));
#ifdef QLTEN_MPI_TIMING_MODE
      slave_commu_timer.Suspend();
#endif

      ElemT *u = nullptr;
      ElemT *vt = nullptr;
      QLTEN_Double *s = nullptr;
      hp_numeric::MatSVD(mat, m, n, u, s, vt);
      free(mat);
      size_t ld = std::min(m, n);
#ifdef QLTEN_MPI_TIMING_MODE
      slave_commu_timer.Restart();
#endif
      MPI_Send(&idx, 1, MPI_INT, kMPIMasterRank, idx + 13, MPI_Comm(world)); // The tag should be positive
      hp_numeric::MPI_Send(m, kMPIMasterRank, idx + 14, MPI_Comm(world));
      hp_numeric::MPI_Send(n, kMPIMasterRank, idx + 15, MPI_Comm(world));
      hp_numeric::MPI_Send(u, ld * m, kMPIMasterRank, idx + 16, MPI_Comm(world));
      hp_numeric::MPI_Send(s, ld, kMPIMasterRank, idx + 17, MPI_Comm(world));
      hp_numeric::MPI_Send(vt, ld * n, kMPIMasterRank, idx + 18, MPI_Comm(world));
#ifdef QLTEN_MPI_TIMING_MODE
      slave_commu_timer.Suspend();
#endif
      free(u);
      free(vt);
      free(s);
      task_done++;
    }
  } while (!terminated);
  MPI_Barrier(MPI_Comm(world));

#ifdef QLTEN_MPI_TIMING_MODE
  std::cout << "[[worker " << consumer_rank << " has done " << task_done << " tasks.]]" << std::endl;
  slave_total_work_timer.PrintElapsed();
  slave_commu_timer.PrintElapsed();
#endif
  return;
}

template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::DataBlkCopySVDUdata(
    const CoorsT &blk_coors, const size_t mat_m, const size_t mat_n,
    const size_t row_offset,
    const ElemT *u, const size_t u_m, const size_t u_n,
    const std::vector<size_t> &kept_cols
) {
  assert(kept_cols.size() == mat_n);
  auto blk_idx = BlkCoorsToBlkIdx(blk_coors);
  // TODO: Remove direct touch the raw data in DataBlk* member!
  auto data = pactual_raw_data_ + blk_idx_data_blk_map_[blk_idx].data_offset;
  size_t data_idx = 0;
  for (size_t i = 0; i < mat_m; ++i) {
    for (size_t j = 0; j < mat_n; ++j) {
      data[data_idx] = u[(row_offset + i) * u_n + kept_cols[j]];
      data_idx++;
    }
  }
}

template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::DataBlkCopySVDVtData(
    const CoorsT &blk_coors, const size_t mat_m, const size_t mat_n,
    const size_t col_offset,
    const ElemT *vt, const size_t vt_m, const size_t vt_n,
    const std::vector<size_t> &kept_rows
) {
  assert(kept_rows.size() == mat_m);
  auto blk_idx = BlkCoorsToBlkIdx(blk_coors);
  // TODO: Remove direct touch the raw data in DataBlk* member!
  auto data = pactual_raw_data_ + blk_idx_data_blk_map_[blk_idx].data_offset;
  for (size_t i = 0; i < mat_m; ++i) {
    memcpy(
        data + (i * mat_n),
        vt + (kept_rows[i] * vt_n + col_offset),
        mat_n * sizeof(ElemT)
    );
  }
}

/**
QR decomposition.
*/
template<typename ElemT, typename QNT>
std::map<size_t, DataBlkMatQrRes<ElemT>>
BlockSparseDataTensor<ElemT, QNT>::DataBlkDecompQR(
    const IdxDataBlkMatMap<QNT> &idx_data_blk_mat_map
) const {
  std::map<size_t, DataBlkMatQrRes<ElemT>> idx_qr_res_map;
  for (auto &idx_data_blk_mat: idx_data_blk_mat_map) {
    auto idx = idx_data_blk_mat.first;
    auto data_blk_mat = idx_data_blk_mat.second;
    ElemT *mat = RawDataGenDenseDataBlkMat_(data_blk_mat);
    ElemT *q = nullptr;
    ElemT *r = nullptr;
    size_t m = data_blk_mat.rows;
    size_t n = data_blk_mat.cols;
    size_t k = m > n ? n : m;
    hp_numeric::MatQR(mat, m, n, q, r);
    free(mat);
    idx_qr_res_map[idx] = DataBlkMatQrRes<ElemT>(m, n, k, q, r);
  }
  return idx_qr_res_map;
}

template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::DataBlkCopyQRQdata(
    const CoorsT &blk_coors, const size_t mat_m, const size_t mat_n,
    const size_t row_offset,
    const ElemT *q, const size_t q_m, const size_t q_n
) {
  assert(mat_n == q_n);
  auto blk_idx = BlkCoorsToBlkIdx(blk_coors);
  auto &data_blk = blk_idx_data_blk_map_.at(blk_idx);
  assert(data_blk.size == (mat_m * mat_n));
  auto data = pactual_raw_data_ + data_blk.data_offset;
  memcpy(data, q + (row_offset * q_n), data_blk.size * sizeof(ElemT));
}

template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::DataBlkCopyQRRdata(
    const CoorsT &blk_coors, const size_t mat_m, const size_t mat_n,
    const size_t col_offset,
    const ElemT *r, const size_t r_m, const size_t r_n
) {
  assert(mat_m == r_m);
  auto blk_idx = BlkCoorsToBlkIdx(blk_coors);
  auto data = pactual_raw_data_ + blk_idx_data_blk_map_.at(blk_idx).data_offset;
  for (size_t i = 0; i < mat_m; ++i) {
    memcpy(
        data + (i * mat_n),
        r + (i * r_n + col_offset),
        mat_n * sizeof(ElemT)
    );
  }
}

/**
Clear data blocks and reset raw_data_size_.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::DataBlkClear_(void) {
  blk_idx_data_blk_map_.clear();
  raw_data_size_ = 0;
}

} /* qlten */
#endif /* ifndef QLTEN_QLTENSOR_BLK_SPAR_DATA_TEN_DATA_BLK_LEVEL_OPERATIONS_H */
