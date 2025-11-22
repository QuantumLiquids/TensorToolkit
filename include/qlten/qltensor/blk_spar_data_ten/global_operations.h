// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-26 21:30
*
* Description: QuantumLiquids/tensor project. Global level operations in BlockSparseDataTensor.
*/

/**
@file global_operations.h
@brief Global level operations in BlockSparseDataTensor.
*/
#ifndef QLTEN_QLTENSOR_BLK_SPAR_DATA_TEN_GLOBAL_OPERATIONS_H
#define QLTEN_QLTENSOR_BLK_SPAR_DATA_TEN_GLOBAL_OPERATIONS_H

#include "qlten/qltensor/blk_spar_data_ten/blk_spar_data_ten.h"
#include "qlten/qltensor/blk_spar_data_ten/data_blk.h"                    // DataBlk
#include "qlten/qltensor/blk_spar_data_ten/data_blk_operations.h"
#include "qlten/qltensor/blk_spar_data_ten/raw_data_operations.h"
#include "qlten/framework/value_t.h"                                      // QLTEN_Double, QLTEN_Complex
#include "qlten/framework/hp_numeric/ten_trans.h"                         // TensorTranspose
#include "qlten/framework/hp_numeric/blas_extensions.h"
#include "qlten/framework/hp_numeric/lapack.h"
#include "qlten/utility/utils_inl.h"                                      // CalcMultiDimDataOffsets, Reorder
#include "qlten/utility/timer.h"

#include <map>              // map
#include <unordered_set>    // unordered_set
#include <set>              // set

#ifdef Release
#define NDEBUG
#endif
#include <cassert>     // assert

namespace qlten {

/**
Clear all contents of this block sparse data tensor.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::Clear(void) {
  DataBlkClear_();
  RawDataFree_();
}

/**
Allocate the memory based on the size of raw_data_size_;

@param init Whether initialize the memory to 0.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::Allocate(const bool init) {
  RawDataAlloc_(raw_data_size_, init);
}

/**
Random set all elements in [0, 1].
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::Random(void) {
  if (IsScalar()) { raw_data_size_ = 1; }
  if (raw_data_size_ > actual_raw_data_size_) {
    RawDataAlloc_(raw_data_size_);
  }
  RawDataRand_();
}

/**
Fill all elements with given value.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::Fill(const ElemT &value) {
  if (IsScalar()) { raw_data_size_ = 1; }
  if (raw_data_size_ > actual_raw_data_size_) {
    RawDataAlloc_(raw_data_size_);
  }
  RawDataFill_(value);
}

/**
Element-wise multiplication with another BlockSparseDataTensor.
The two tensors may have different blocks.

@param rhs Another BlockSparseDataTensor to multiply with.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::ElementWiseMultiply(const BlockSparseDataTensor &rhs) {
  // For each block in the current tensor, perform element-wise multiplication
  for (auto &[blk_idx, data_blk] : blk_idx_data_blk_map_) {
    auto rhs_it = rhs.blk_idx_data_blk_map_.find(blk_idx);
    if (rhs_it == rhs.blk_idx_data_blk_map_.end()) {
      // If the block doesn't exist in rhs, treat it as all zeros
      // Zero out the current block
      RawDataSetZeros_(data_blk.data_offset, data_blk.size);
    } else {
      const auto &rhs_data_blk = rhs_it->second;

      // Perform element-wise multiplication for this block
      RawDataElementWiseMultiply_(
          data_blk.data_offset,
          rhs.pactual_raw_data_ + rhs_data_blk.data_offset,
          data_blk.size
      );
    }
  }
}

/**
Transpose the block sparse data tensor.

@param transed_idxes_order Transposed order of indexes.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::Transpose(
    const std::vector<size_t> &transed_idxes_order
) {
  assert(transed_idxes_order.size() == blk_shape.size());
  // Give a shorted order, do nothing
  if (std::is_sorted(transed_idxes_order.begin(), transed_idxes_order.end())) {
    return;
  }

  InplaceReorder(blk_shape, transed_idxes_order);
  blk_multi_dim_offsets_ = CalcMultiDimDataOffsets(blk_shape);

  std::vector<RawDataTransposeTask> raw_data_trans_tasks;
  BlkIdxDataBlkMap transed_blk_idx_data_blk_map;
  for (auto &blk_idx_data_blk : blk_idx_data_blk_map_) {
    DataBlk<QNT> transed_data_blk(blk_idx_data_blk.second);
    int sign = transed_data_blk.Transpose(transed_idxes_order);
    auto transed_data_blk_idx = BlkCoorsToBlkIdx(transed_data_blk.blk_coors);
    transed_blk_idx_data_blk_map[transed_data_blk_idx] = transed_data_blk;
    raw_data_trans_tasks.push_back(
        RawDataTransposeTask(
            ten_rank,
            transed_idxes_order,
            blk_idx_data_blk.first,
            blk_idx_data_blk.second.shape,
            blk_idx_data_blk.second.data_offset,
            transed_data_blk_idx,
            transed_data_blk.shape,
            sign
        )
    );
  }

  // Calculate and set data offset of each transposed data block.
  ResetDataOffset(transed_blk_idx_data_blk_map);
  RawDataTransposeTask::SortTasksByTranspoedBlkIdx(raw_data_trans_tasks);
  size_t trans_task_idx = 0;
  for (auto &blk_idx_data_blk : transed_blk_idx_data_blk_map) {
    raw_data_trans_tasks[trans_task_idx].transed_data_offset =
        blk_idx_data_blk.second.data_offset;
    trans_task_idx++;
  }
  // Update block index <-> data block map.
  blk_idx_data_blk_map_ = transed_blk_idx_data_blk_map;
  // Transpose the raw data.
  RawDataTransposeTask::SortTasksByOriginalBlkIdx(raw_data_trans_tasks);
  RawDataTranspose_(raw_data_trans_tasks);
}

template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::FuseFirstTwoIndex(
    const std::vector<std::tuple<size_t, size_t, size_t, size_t>> &qnscts_offset_info_list
) {
  if (actual_raw_data_size_ != 0) {
#ifdef QLTEN_TIMING_MODE
    Timer fuse_index_bsdt_pre_timer("use_index_bsdt_prepare");
#endif
    using QNSctsOffsetInfo = std::tuple<size_t, size_t, size_t, size_t>;
    std::map < std::pair<size_t, size_t>, size_t > map_from_old_blk_first_two_coors_to_new_blk_first_coor;
    std::map < std::pair<size_t, size_t>, size_t > map_from_old_blk_first_two_coors_to_new_blk_data_off_set;
    for (const QNSctsOffsetInfo &qnscts_offset_info : qnscts_offset_info_list) {
      std::pair<size_t, size_t> old_blk_first_two_coors = std::make_pair(
          std::get<0>(qnscts_offset_info),
          std::get<1>(qnscts_offset_info)
      );
      map_from_old_blk_first_two_coors_to_new_blk_first_coor[old_blk_first_two_coors] =
          std::get<2>(qnscts_offset_info);
      map_from_old_blk_first_two_coors_to_new_blk_data_off_set[old_blk_first_two_coors] =
          std::get<3>(qnscts_offset_info);
    }

    //we generate a new bsdt to convenient use the constructor of BSDT
    //note here pqlten_indexes has become pointing to the new indices
    BlockSparseDataTensor<ElemT, QNT> new_bsdt = BlockSparseDataTensor<ElemT, QNT>(pqlten_indexes);
    std::map < size_t, size_t > old_blk_idx_mapto_new_blk_idx;
    std::vector<CoorsT> new_blk_coors_vector;
    std::vector<size_t> new_blk_idx_vector;
    new_blk_coors_vector.reserve(blk_idx_data_blk_map_.size());
    new_blk_idx_vector.reserve(blk_idx_data_blk_map_.size());
    for (auto &[old_idx, data_blk] : blk_idx_data_blk_map_) {
      CoorsT &blk_coors = data_blk.blk_coors;
      std::pair<size_t, size_t> old_blk_first_two_coors = std::make_pair(
          blk_coors[0],
          blk_coors[1]
      );
      size_t new_blk_first_coor = map_from_old_blk_first_two_coors_to_new_blk_first_coor[old_blk_first_two_coors];
      std::vector<size_t> new_blk_coors = std::vector<size_t>(blk_coors.begin() + 1, blk_coors.end());
      new_blk_coors[0] = new_blk_first_coor;
      new_blk_coors_vector.push_back(new_blk_coors);
      size_t new_idx = new_bsdt.BlkCoorsToBlkIdx(new_blk_coors);
      old_blk_idx_mapto_new_blk_idx.insert(std::make_pair(old_idx, new_idx));
      new_blk_idx_vector.push_back(new_idx);
    }

    new_bsdt.DataBlksInsert(
        new_blk_idx_vector,
        new_blk_coors_vector,
        true,
        true
    );//note here we initial the memory, so need performence test here.

    //Assign copy task
    std::vector<RawDataCopyTask> data_copy_tasks;
    data_copy_tasks.reserve(blk_idx_data_blk_map_.size());
    for (auto &[old_idx, data_blk] : blk_idx_data_blk_map_) {
      CoorsT &blk_coors = data_blk.blk_coors;
      ShapeT &shape = data_blk.shape;
      std::pair<size_t, size_t> old_blk_first_two_coors = std::make_pair(
          blk_coors[0],
          blk_coors[1]
      );
      size_t first_dim_off_set = map_from_old_blk_first_two_coors_to_new_blk_data_off_set.at(old_blk_first_two_coors);
      size_t new_idx = old_blk_idx_mapto_new_blk_idx.at(old_idx);
      size_t dest_data_offset = new_bsdt.blk_idx_data_blk_map_.at(new_idx).data_offset;
      if (first_dim_off_set != 0) {
        size_t other_dimension = 1;
        for (size_t i = 2; i < shape.size(); i++) {
          other_dimension *= shape[i];
        }
        dest_data_offset += other_dimension * first_dim_off_set;
      }
      RawDataCopyTask task(
          blk_coors,
          data_blk.data_offset,
          data_blk.size,
          dest_data_offset,
          false
      );
      data_copy_tasks.push_back(task);
    }

#ifdef QLTEN_TIMING_MODE
    fuse_index_bsdt_pre_timer.PrintElapsed();
#endif

#ifdef QLTEN_TIMING_MODE
    Timer fuse_index_bsdt_raw_data_copy("fuse_index_bsdt_raw_data_copy");
#endif
    new_bsdt.RawDataCopyNoAdd_(data_copy_tasks, pactual_raw_data_);
#ifdef QLTEN_TIMING_MODE
    fuse_index_bsdt_raw_data_copy.PrintElapsed();
#endif
    qlten::QLFree(pactual_raw_data_);
    // right value referece copy
    ten_rank = new_bsdt.ten_rank;
    blk_shape = new_bsdt.blk_shape;
    blk_multi_dim_offsets_ = new_bsdt.blk_multi_dim_offsets_;
    blk_idx_data_blk_map_ = new_bsdt.blk_idx_data_blk_map_;
    actual_raw_data_size_ = new_bsdt.actual_raw_data_size_;
    pqlten_indexes = new_bsdt.pqlten_indexes;
    pactual_raw_data_ = new_bsdt.pactual_raw_data_;

    new_bsdt.pactual_raw_data_ = nullptr;
    new_bsdt.pqlten_indexes = nullptr;
  } else {
    (*this) = BlockSparseDataTensor<ElemT, QNT>(pqlten_indexes);
  }
}

template<typename ElemT, typename QNT>
auto BlockSparseDataTensor<ElemT, QNT>::Norm(void) {
  if constexpr (Fermionicable<QNT>::IsFermionic()) {
    if (IsScalar()) {
#ifndef USE_GPU
      return qlten::abs(*pactual_raw_data_);
#else
      ElemT single_raw_data;
      cudaMemcpy(&single_raw_data, pactual_raw_data_, sizeof(single_raw_data), cudaMemcpyDeviceToHost);
      return qlten::abs(single_raw_data);
#endif
    }
    auto tasks = GenFermionNormTask();
    return RawDataFermionNorm_(tasks);
  } else {
    return RawDataNorm_();
  }
}

template<typename ElemT, typename QNT>
auto BlockSparseDataTensor<ElemT, QNT>::Quasi2Norm(void) {
  return RawDataNorm_();
}

/**
Normalize the data tensor and return its norm.

@return The norm before the normalization.
*/
template<typename ElemT, typename QNT>
auto BlockSparseDataTensor<ElemT, QNT>::Normalize(void) {
  auto norm = Norm();
  return RawDataNormalize_(norm);
}

template<typename ElemT, typename QNT>
auto BlockSparseDataTensor<ElemT, QNT>::QuasiNormalize(void) {
  auto norm = RawDataNorm_();
  return RawDataNormalize_(norm);
}

/**
Complex conjugate.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::Conj(void) {
  if constexpr (Fermionicable<QNT>::IsFermionic()) {
    std::vector<RawDataInplaceReverseTask> tasks;
    tasks.reserve(blk_idx_data_blk_map_.size());
    for (auto &[idx, data_blk] : blk_idx_data_blk_map_) {
      if (data_blk.ReverseSign() == -1) {
        RawDataInplaceReverseTask task(data_blk.data_offset, data_blk.size);
        tasks.push_back(task);
      }
    }
    FermionicRawDataConj_(tasks);
  } else {
    RawDataConj_();
  }
}

template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::ActFermionPOps() {
  if constexpr (Fermionicable<QNT>::IsFermionic()) {
    std::vector<size_t> in_indices;
    for (size_t i = 0; i < this->pqlten_indexes->size(); i++) {
      if ((*this->pqlten_indexes)[i].GetDir() == IN) {
        in_indices.push_back(i);
      }
    }
    if (in_indices.empty()) {
      return;
    }
    for (auto &[idx, data_blk] : blk_idx_data_blk_map_) {
      if (data_blk.SelectedIndicesParity(in_indices) == -1) {
        hp_numeric::VectorScale(pactual_raw_data_ + data_blk.data_offset, data_blk.size, -1.0);
      }
    }
  }
}

/**
Add two input block sparse data tensor together and assign into this tensor.

@param a Block sparse data tensor A.
@param b Block sparse data tensor B.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::AddTwoBSDTAndAssignIn(
    const BlockSparseDataTensor &a,
    const BlockSparseDataTensor &b) {
  if (a.IsScalar() && b.IsScalar()) {
    ElemSet({}, a.ElemGet({}) + b.ElemGet({}));
    return;
  }

  auto blk_idx_data_blk_map_a = a.GetBlkIdxDataBlkMap();
  std::vector<RawDataCopyTask> raw_data_copy_tasks_a;
  for (auto &blk_idx_data_blk : blk_idx_data_blk_map_a) {
    auto data_blk = blk_idx_data_blk.second;
    DataBlkInsert(data_blk.blk_coors, false);
    raw_data_copy_tasks_a.push_back(
        RawDataCopyTask(data_blk.blk_coors, data_blk.data_offset, data_blk.size)
    );
  }

  auto blk_idx_data_blk_map_b = b.GetBlkIdxDataBlkMap();
  std::vector<RawDataCopyTask> raw_data_copy_tasks_b;
  for (auto &blk_idx_data_blk : blk_idx_data_blk_map_b) {
    auto blk_idx = blk_idx_data_blk.first;
    auto data_blk = blk_idx_data_blk.second;
    if (blk_idx_data_blk_map_a.find(blk_idx) != blk_idx_data_blk_map_a.end()) {
      raw_data_copy_tasks_b.push_back(
          RawDataCopyTask(
              data_blk.blk_coors,
              data_blk.data_offset,
              data_blk.size,
              true
          )
      );
    } else {
      DataBlkInsert(data_blk.blk_coors, false);
      raw_data_copy_tasks_b.push_back(
          RawDataCopyTask(
              data_blk.blk_coors,
              data_blk.data_offset,
              data_blk.size
          )
      );
    }
  }

  // Get data offset in destination.
  for (auto &task : raw_data_copy_tasks_a) {
    task.dest_data_offset = blk_idx_data_blk_map_[
        BlkCoorsToBlkIdx(task.src_blk_coors)
    ].data_offset;
  }
  for (auto &task : raw_data_copy_tasks_b) {
    task.dest_data_offset = blk_idx_data_blk_map_[
        BlkCoorsToBlkIdx(task.src_blk_coors)
    ].data_offset;
  }

  Allocate();
  RawDataCopy_(raw_data_copy_tasks_a, a.pactual_raw_data_);
  RawDataCopy_(raw_data_copy_tasks_b, b.pactual_raw_data_);
}

/**
Add another block sparse data tensor to this block sparse data tensor.

@param rhs Block sparse data tensor on the right hand side.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::AddAndAssignIn(
    const BlockSparseDataTensor &rhs) {
  assert(ten_rank == rhs.ten_rank);
  if (IsScalar() && rhs.IsScalar()) {
    ElemSet({}, ElemGet({}) + rhs.ElemGet({}));
    return;
  }

  // Copy block index <-> data block map and save actual raw data pointer.
  BlkIdxDataBlkMap this_blk_idx_data_blk_map(blk_idx_data_blk_map_);
  ElemT *this_pactual_raw_data_ = pactual_raw_data_;
  RawDataDiscard_();

  // Create raw data copy tasks for this tensor.
  std::vector<RawDataCopyTask> raw_data_copy_tasks_this;
  for (auto &blk_idx_data_blk : this_blk_idx_data_blk_map) {
    auto data_blk = blk_idx_data_blk.second;
    raw_data_copy_tasks_this.push_back(
        RawDataCopyTask(data_blk.blk_coors, data_blk.data_offset, data_blk.size)
    );
  }

  // Create raw data copy tasks for tensor on the right hand side.
  auto blk_idx_data_blk_map_rhs = rhs.GetBlkIdxDataBlkMap();
  std::vector<RawDataCopyTask> raw_data_copy_tasks_rhs;
  for (auto &blk_idx_data_blk : blk_idx_data_blk_map_rhs) {
    auto blk_idx = blk_idx_data_blk.first;
    auto data_blk = blk_idx_data_blk.second;
    if (blk_idx_data_blk_map_.find(blk_idx) != blk_idx_data_blk_map_.end()) {
      raw_data_copy_tasks_rhs.push_back(
          RawDataCopyTask(
              data_blk.blk_coors,
              data_blk.data_offset,
              data_blk.size,
              true
          )
      );
    } else {
      DataBlkInsert(data_blk.blk_coors, false);
      raw_data_copy_tasks_rhs.push_back(
          RawDataCopyTask(
              data_blk.blk_coors,
              data_blk.data_offset,
              data_blk.size
          )
      );
    }
  }

  // Get data offset in result block sparse data tensor.
  for (auto &task : raw_data_copy_tasks_this) {
    task.dest_data_offset = blk_idx_data_blk_map_[
        BlkCoorsToBlkIdx(task.src_blk_coors)
    ].data_offset;
  }
  for (auto &task : raw_data_copy_tasks_rhs) {
    task.dest_data_offset = blk_idx_data_blk_map_[
        BlkCoorsToBlkIdx(task.src_blk_coors)
    ].data_offset;
  }

  Allocate();
  RawDataCopy_(raw_data_copy_tasks_this, this_pactual_raw_data_);
  qlten::QLFree(this_pactual_raw_data_);
  RawDataCopy_(raw_data_copy_tasks_rhs, rhs.pactual_raw_data_);
}

/**
Multiply this block sparse data tensor by a scalar.

@param s A scalar.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::MultiplyByScalar(const ElemT s) {
  RawDataMultiplyByScalar_(s);
}

/**
Contract two block sparse data tensors follow a queue of raw data contraction
tasks.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::CtrctTwoBSDTAndAssignIn(
    const BlockSparseDataTensor &bsdt_a,
    const BlockSparseDataTensor &bsdt_b,
    std::vector<RawDataCtrctTask> &raw_data_ctrct_tasks,
    std::vector<int> &a_trans_orders,
    std::vector<int> &b_trans_orders
) {
  assert(!(bsdt_a.IsScalar() || bsdt_b.IsScalar()));
  if (raw_data_ctrct_tasks.empty()) { return; }

  Allocate();

  bool a_need_trans = !(a_trans_orders.empty());
  bool b_need_trans = !(b_trans_orders.empty());
  std::unordered_map<size_t, ElemT *> a_blk_idx_transed_data_map;
  std::unordered_map<size_t, ElemT *> b_blk_idx_transed_data_map;
  RawDataCtrctTask::SortTasksByCBlkIdx(raw_data_ctrct_tasks);

#ifdef QLTEN_TIMING_MODE
  Timer contract_mkl_timer("matrix multiplication");
  contract_mkl_timer.Suspend();
#endif

  for (auto &task : raw_data_ctrct_tasks) {
    const ElemT *a_data;
    const ElemT *b_data;
    if (a_need_trans) {
      auto poss_it = a_blk_idx_transed_data_map.find(task.a_blk_idx);
      if (poss_it != a_blk_idx_transed_data_map.end()) {
        a_data = poss_it->second;
      } else {
        const auto &a_data_blk = bsdt_a.blk_idx_data_blk_map_.at(task.a_blk_idx);
        ElemT *transed_data = (ElemT *) qlten::QLMalloc(a_data_blk.size * sizeof(ElemT));
        std::vector<int> a_blk_transed_shape = Reorder(a_data_blk.shape, a_trans_orders);
        hp_numeric::TensorTranspose(
            a_trans_orders,
            bsdt_a.ten_rank,
            bsdt_a.pactual_raw_data_ + task.a_data_offset,
            a_data_blk.shape,
            transed_data,
            a_blk_transed_shape,
            1.0
        );
        a_blk_idx_transed_data_map[task.a_blk_idx] = transed_data;
        a_data = transed_data;
      }
    } else {
      a_data = bsdt_a.pactual_raw_data_ + task.a_data_offset;
    }
    if (b_need_trans) {
      auto poss_it = b_blk_idx_transed_data_map.find(task.b_blk_idx);
      if (poss_it != b_blk_idx_transed_data_map.end()) {
        b_data = poss_it->second;
      } else {
        const auto &b_data_blk = bsdt_b.blk_idx_data_blk_map_.at(task.b_blk_idx);
        ElemT *transed_data = (ElemT *) qlten::QLMalloc(b_data_blk.size * sizeof(ElemT));
        std::vector<int> b_blk_transed_shape = Reorder(b_data_blk.shape, b_trans_orders);
        hp_numeric::TensorTranspose(
            b_trans_orders,
            bsdt_b.ten_rank,
            bsdt_b.pactual_raw_data_ + task.b_data_offset,
            b_data_blk.shape,
            transed_data,
            b_blk_transed_shape,
            1.0
        );
        b_blk_idx_transed_data_map[task.b_blk_idx] = transed_data;
        b_data = transed_data;
      }
    } else {
      b_data = bsdt_b.pactual_raw_data_ + task.b_data_offset;
    }
#ifdef QLTEN_TIMING_MODE
    contract_mkl_timer.Restart();
#endif
    RawDataTwoMatMultiplyAndAssignIn_(
        a_data,
        b_data,
        task.c_data_offset,
        task.m, task.k, task.n,
        task.f_ex_sign,
        task.beta
    );
#ifdef QLTEN_TIMING_MODE
    contract_mkl_timer.Suspend();
#endif
  }
#ifdef QLTEN_TIMING_MODE
  contract_mkl_timer.PrintElapsed();
#endif
  for (auto &blk_idx_transed_data : a_blk_idx_transed_data_map) {
    qlten::QLFree(blk_idx_transed_data.second);
  }
  for (auto &blk_idx_transed_data : b_blk_idx_transed_data_map) {
    qlten::QLFree(blk_idx_transed_data.second);
  }
}

template<typename ElemT, typename QNT>
template<bool a_ctrct_tail, bool b_ctrct_head>
void BlockSparseDataTensor<ElemT, QNT>::CtrctAccordingTask(
    const ElemT *a_raw_data,
    const ElemT *b_raw_data,
    const std::vector<RawDataCtrctTask> &raw_data_ctrct_tasks,
    const std::map<size_t, int> &a_blk_idx_sign_map,
    const std::map<size_t, int> &b_blk_idx_sign_map
) {
  Allocate();
#ifdef QLTEN_TIMING_MODE
  Timer contract_mkl_timer("matrix multiplication");
#endif

  for (const auto &task : raw_data_ctrct_tasks) {
    const ElemT *a_data = a_raw_data + task.a_data_offset;
    const ElemT *b_data = b_raw_data + task.b_data_offset;
    double over_all_f_ex_sign = task.f_ex_sign;
    if constexpr (Fermionicable<QNT>::IsFermionic()) {
      if (a_blk_idx_sign_map.size() > 0) {
        over_all_f_ex_sign
            *= a_blk_idx_sign_map.at(task.a_blk_idx);
      }
      if (b_blk_idx_sign_map.size() > 0) {
        over_all_f_ex_sign
            *= b_blk_idx_sign_map.at(task.b_blk_idx);
      }

    }
#ifndef  USE_GPU
    auto QLBlasNoTrans = CblasNoTrans;
    auto QLBlasTrans = CblasTrans;
#else
    auto QLBlasNoTrans = CUBLAS_OP_N;
    auto QLBlasTrans = CUBLAS_OP_T;
#endif
    if (a_ctrct_tail && b_ctrct_head) {
      RawDataTwoMatMultiplyAndAssignIn_(
          a_data,
          b_data,
          task.c_data_offset,
          task.m, task.k, task.n,
          over_all_f_ex_sign,
          task.beta
      );
    } else if (a_ctrct_tail && (!b_ctrct_head)) {
      hp_numeric::MatMultiply(
          over_all_f_ex_sign,
          a_data,
          QLBlasNoTrans,
          b_data,
          QLBlasTrans,
          task.m, task.k, task.n,
          task.k, task.k,
          task.beta,
          pactual_raw_data_ + task.c_data_offset
      );
    } else if ((!a_ctrct_tail) && (b_ctrct_head)) {
      hp_numeric::MatMultiply(
          over_all_f_ex_sign,
          a_data,
          QLBlasTrans,
          b_data,
          QLBlasNoTrans,
          task.m, task.k, task.n,
          task.m, task.n,
          task.beta,
          pactual_raw_data_ + task.c_data_offset
      );
    } else {
      hp_numeric::MatMultiply(
          over_all_f_ex_sign,
          a_data,
          QLBlasTrans,
          b_data,
          QLBlasTrans,
          task.m, task.k, task.n,
          task.m, task.k,
          task.beta,
          pactual_raw_data_ + task.c_data_offset
      );
    }
  }
#ifdef QLTEN_TIMING_MODE
  contract_mkl_timer.PrintElapsed();
#endif
}

// Helpers for tensor expansion
using BlkCoorsShapePair = std::pair<CoorsT, ShapeT>;
// (hash value of qn info) -> (blk coors, shape)
using QnInfoHashBlkCoorsShapeMap = std::unordered_map<
    size_t,
    BlkCoorsShapePair
>;

template<typename QNT>
inline size_t CalcDataBlkResidueDimSize(const DataBlk<QNT> &data_blk) {
  return data_blk.size / data_blk.shape[0];
}

/**
Construct tensor expansion data over the first index, from corresponding BSDTs.
The DataBlk in new tensor come from `bsdt_a` and `bsdt_b`.
The new generated DataBlk's index is the same with the index in `bsdt_a`.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::ConstructExpandedDataOnFirstIndex(
    const BlockSparseDataTensor &bsdt_a,
    const BlockSparseDataTensor &bsdt_b,
    const std::vector<bool> &is_a_first_idx_qnsct_expanded,
    const std::map<size_t, size_t> &b_idx_qnsct_coor_expanded_idx_qnsct_coor_map
) {
#ifdef QLTEN_TIMING_MODE
  Timer expand_data_blk_timer("   =============> expansion_construct_data_blk_and_prepare_raw_data_tasks");
#endif
  std::map < size_t, size_t > expanded_idx_qnsct_coor_b_idx_qnsct_coor_map;
  for (auto &elem : b_idx_qnsct_coor_expanded_idx_qnsct_coor_map) {
    expanded_idx_qnsct_coor_b_idx_qnsct_coor_map[elem.second] = elem.first;
  }
  auto blk_idx_data_blk_map_a = bsdt_a.GetBlkIdxDataBlkMap();
  auto blk_idx_data_blk_map_b = bsdt_b.GetBlkIdxDataBlkMap();

  std::map<size_t, int> blk_idx_expand_mapto_blk_map_a, blk_idx_expand_mapto_blk_map_b;
  // The new generated blk_data's index map to original blk_data index in a
  // if no corresponding original blk_data but need filled with zero, label {blk_data_idx, -1};
  // if neither corresponding original blk_data nor filled with zero, no the pair.


  // First we construct the new blk_idx_data_blk_map
  std::vector<CoorsT> blk_coors_s;
  std::vector<size_t> blk_idxs;
  blk_coors_s.reserve(blk_idx_data_blk_map_a.size() + blk_idx_data_blk_map_b.size());// reserve more
  blk_idxs.reserve(blk_idx_data_blk_map_a.size() + blk_idx_data_blk_map_b.size());
  size_t zero_piece_num = 0;//how many pieces of zeros need to set
  for (const auto &[blk_idx_a, data_blk_a] : blk_idx_data_blk_map_a) {
    size_t blk_coor_in_first_idx = data_blk_a.blk_coors[0];
    size_t blk_idx = blk_idx_a;
    blk_idx_expand_mapto_blk_map_a[blk_idx] = blk_idx_a;
    blk_coors_s.push_back(data_blk_a.blk_coors);
    blk_idxs.push_back(blk_idx);
    if (is_a_first_idx_qnsct_expanded[blk_coor_in_first_idx]) {

      std::vector<size_t> blk_coors_b = data_blk_a.blk_coors;
      blk_coors_b[0] = expanded_idx_qnsct_coor_b_idx_qnsct_coor_map[blk_coor_in_first_idx];
      size_t blk_idx_b = bsdt_b.BlkCoorsToBlkIdx(blk_coors_b);
      auto pdata_blk_b = blk_idx_data_blk_map_b.find(blk_idx_b);
      if (pdata_blk_b != blk_idx_data_blk_map_b.end()) {
        blk_idx_expand_mapto_blk_map_b[blk_idx] = pdata_blk_b->first;
        blk_idx_data_blk_map_b.erase(pdata_blk_b);
      } else {
        blk_idx_expand_mapto_blk_map_b[blk_idx] = -1;
        zero_piece_num++;
      }
    }
  }

  for (const auto &[blk_idx_b, data_blk_b] : blk_idx_data_blk_map_b) {
    const size_t blk_coor_in_first_idx_b = data_blk_b.blk_coors[0];
    size_t blk_coor_in_first_idx_expand = b_idx_qnsct_coor_expanded_idx_qnsct_coor_map.at(blk_coor_in_first_idx_b);
    std::vector<size_t> blk_coors = data_blk_b.blk_coors;
    blk_coors[0] = blk_coor_in_first_idx_expand;
    //Generate the DataBlk
    size_t blk_idx = BlkCoorsToBlkIdx(blk_coors);
    blk_idx_expand_mapto_blk_map_b[blk_idx] = blk_idx_b;
    blk_coors_s.push_back(blk_coors);
    blk_idxs.push_back(blk_idx);

    if (blk_coor_in_first_idx_expand < is_a_first_idx_qnsct_expanded.size()) {
      // auto pdata_blk_a = blk_idx_data_blk_map_a.find(blk_idx);
      blk_idx_expand_mapto_blk_map_a[blk_idx] = -1;
      zero_piece_num++;
    }
  }

  DataBlksInsert(blk_idxs, blk_coors_s, true);

  //copy and write raw data
  blk_idx_data_blk_map_b = bsdt_b.GetBlkIdxDataBlkMap(); // regenerate it
  std::vector<RawDataCopyTask> raw_data_copy_tasks_from_a, raw_data_copy_tasks_from_b;
  raw_data_copy_tasks_from_a.reserve(blk_idx_expand_mapto_blk_map_a.size());// reserve more
  raw_data_copy_tasks_from_b.reserve(blk_idx_expand_mapto_blk_map_b.size());// reserve more

  std::vector<size_t> raw_data_zero_pieces_offsets;
  std::vector<size_t> raw_data_zero_pieces_size;
  raw_data_zero_pieces_offsets.reserve(zero_piece_num);
  raw_data_zero_pieces_size.reserve(zero_piece_num);
  for (const auto &[blk_idx, data_blk] : blk_idx_data_blk_map_) {
    if (
        blk_idx_expand_mapto_blk_map_a.find(blk_idx) !=
            blk_idx_expand_mapto_blk_map_a.end()
        ) {
      int blk_idx_a = blk_idx_expand_mapto_blk_map_a[blk_idx];
      if (blk_idx_a != -1) {
        auto data_blk_a = blk_idx_data_blk_map_a.at(blk_idx);

        RawDataCopyTask task = RawDataCopyTask(
            data_blk.blk_coors,
            data_blk_a.data_offset, //raw_data_offset_in_a
            data_blk_a.size
        );
        task.dest_data_offset = data_blk.data_offset;
        raw_data_copy_tasks_from_a.push_back(task);
      } else {
        size_t filled_zero_elem_number = data_blk.size - blk_idx_data_blk_map_b[
            blk_idx_expand_mapto_blk_map_b[
                blk_idx
            ]
        ].size;
        raw_data_zero_pieces_offsets.push_back(data_blk.data_offset);
        raw_data_zero_pieces_size.push_back(filled_zero_elem_number);
      }
    }

    if (
        blk_idx_expand_mapto_blk_map_b.find(blk_idx) !=
            blk_idx_expand_mapto_blk_map_b.end()
        ) {
      int blk_idx_b = blk_idx_expand_mapto_blk_map_b[blk_idx];
      if (blk_idx_b != -1) {

        RawDataCopyTask task = RawDataCopyTask(
            blk_idx_data_blk_map_b[blk_idx_b].blk_coors,
            blk_idx_data_blk_map_b[blk_idx_b].data_offset, //raw_data_offset_in_b
            blk_idx_data_blk_map_b[blk_idx_b].size
        );
        task.dest_data_offset = data_blk.data_offset + data_blk.size -
            blk_idx_data_blk_map_b[blk_idx_b].size;
        raw_data_copy_tasks_from_b.push_back(task);
      } else {
        int blk_idx_a = blk_idx_expand_mapto_blk_map_a[blk_idx];
        auto pblk_idx_data_blk_pair_a = blk_idx_data_blk_map_a.find(
            static_cast<size_t>(blk_idx_a)
        );
        size_t filled_zero_elem_number = data_blk.size -
            pblk_idx_data_blk_pair_a->second.size;
        raw_data_zero_pieces_offsets.push_back(
            data_blk.data_offset + pblk_idx_data_blk_pair_a->second.size
        );
        raw_data_zero_pieces_size.push_back(filled_zero_elem_number);

      }
    }
  }
#ifdef QLTEN_TIMING_MODE
  expand_data_blk_timer.PrintElapsed();
#endif

#ifdef QLTEN_TIMING_MODE
  Timer expand_raw_data_set_zero_timer("   =============> expansion_raw_data_set_zeros");
#endif
  RawDataSetZeros_(raw_data_zero_pieces_offsets, raw_data_zero_pieces_size);

#ifdef QLTEN_TIMING_MODE
  expand_raw_data_set_zero_timer.PrintElapsed();
#endif

#ifdef QLTEN_TIMING_MODE
  Timer expand_raw_data_cp_timer("   =============> expansion_raw_data_copy");
#endif
  // Do data copy
  RawDataCopyNoAdd_(raw_data_copy_tasks_from_a, bsdt_a.pactual_raw_data_);
  RawDataCopyNoAdd_(raw_data_copy_tasks_from_b, bsdt_b.pactual_raw_data_);
#ifdef QLTEN_TIMING_MODE
  expand_raw_data_cp_timer.PrintElapsed();
#endif
}

/**
Construct tensor (magic changing version) expansion data over the first index, from corresponding BSDTs.
The DataBlk in new tensor come from `bsdt_a` and `bsdt_b`.
The new generated DataBlk's index is the same with the index in `bsdt_a`.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::ConstructMCExpandedDataOnFirstIndex(
    const BlockSparseDataTensor &bsdt_a,
    const BlockSparseDataTensor &bsdt_b,
    const std::map<size_t, int> &b_idx_qnsct_coor_expanded_idx_qnsct_coor_map
) {
  blk_idx_data_blk_map_ = bsdt_a.GetBlkIdxDataBlkMap();
  raw_data_size_ = bsdt_a.GetActualRawDataSize();
  std::map < size_t, size_t > blk_idx_in_b_mapto_blk_idx_in_expanded_bsdt;
  auto blk_idx_data_blk_map_b = bsdt_b.GetBlkIdxDataBlkMap();
  for (auto iter = blk_idx_data_blk_map_b.begin();
       iter != blk_idx_data_blk_map_b.cend();) {
    auto blk_idx_data_blk = (*iter);
    DataBlk<QNT> data_blk = blk_idx_data_blk.second;
    size_t first_coor_in_b = data_blk.blk_coors[0];
    if (b_idx_qnsct_coor_expanded_idx_qnsct_coor_map.at(first_coor_in_b) == -1) {
      blk_idx_data_blk_map_b.erase(iter++);
    } else {
      size_t first_coor = b_idx_qnsct_coor_expanded_idx_qnsct_coor_map.at(first_coor_in_b);
      CoorsT blk_coor = data_blk.blk_coors;
      blk_coor[0] = first_coor;
      DataBlkInsert(blk_coor, false);
      blk_idx_in_b_mapto_blk_idx_in_expanded_bsdt.insert(
          std::make_pair(blk_idx_data_blk.first, BlkCoorsToBlkIdx(blk_coor))
      );
      iter++;
    }
  }
  std::vector<RawDataCopyTask> tasks;
  tasks.reserve(blk_idx_in_b_mapto_blk_idx_in_expanded_bsdt.size());
  for (auto &blk_idx_data_blk : blk_idx_data_blk_map_b) {
    size_t blk_idx = blk_idx_in_b_mapto_blk_idx_in_expanded_bsdt.at(blk_idx_data_blk.first);//dest blk_idx
    DataBlk<QNT> data_blk = blk_idx_data_blk.second;//src data_blk
    RawDataCopyTask task(
        data_blk.blk_coors, //src_blk_coors
        data_blk.data_offset, //src_data_offset
        data_blk.size, //src_data_size
        blk_idx_data_blk_map_[blk_idx].data_offset, //dest_data_offset
        false);
    tasks.push_back(task);
  }

  Allocate(false);

  qlten::QLMemcpy(
      pactual_raw_data_,
      bsdt_a.GetActualRawDataPtr(),
      bsdt_a.GetActualRawDataSize() * sizeof(ElemT)
  );
  RawDataCopy_(tasks, bsdt_b.GetActualRawDataPtr());
}

/**
 *
 * @tparam ElemT
 * @tparam QNT
 * @param selected_data_blk_idxs  elements should be unique
 * @param critical_axe
 * @param transposed_data
 */
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::OutOfPlaceMatrixTransposeForSelectedDataBlk(
    const std::set<size_t> &selected_data_blk_idxs,
    const size_t critical_axe,
    ElemT *transposed_data
) const {
#ifndef  USE_GPU
  const size_t group_count = selected_data_blk_idxs.size();
  const ElemT **Amat_array = (const ElemT **) malloc(group_count * sizeof(ElemT *));
  ElemT **Bmat_array = (ElemT **) malloc(group_count * sizeof(ElemT *));
  size_t *rows_array = (size_t *) malloc(group_count * sizeof(size_t));
  size_t *cols_array = (size_t *) malloc(group_count * sizeof(size_t));
  auto iter = selected_data_blk_idxs.begin();
  //TODO: omp
  for (size_t i = 0; i < group_count; i++) {
    const size_t idx = (*iter);
    assert(blk_idx_data_blk_map_.find(idx) != blk_idx_data_blk_map_.end());
    const DataBlk<QNT> &data_blk = blk_idx_data_blk_map_.at(idx);
    if constexpr (Fermionicable<QNT>::IsFermionic()) {

    }
    const size_t off_set = data_blk.data_offset;
    const std::vector<size_t> &shape = data_blk.shape;
    Amat_array[i] = pactual_raw_data_ + off_set;
    Bmat_array[i] = transposed_data + off_set;
    size_t row(1), col(1);
    for (size_t j = 0; j < critical_axe; j++) {
      row *= shape[j];
    }
    for (size_t j = critical_axe; j < ten_rank; j++) {
      col *= shape[j];
    }
    rows_array[i] = row;
    cols_array[i] = col;
    iter++;
  }

  hp_numeric::MatrixTransposeBatch(
      Amat_array,
      Bmat_array,
      rows_array,
      cols_array,
      group_count
  );
  free(Amat_array);
  free(Bmat_array);
  free(rows_array);
  free(cols_array);
#else//USE_GPU
  const size_t group_count = selected_data_blk_idxs.size();
  auto iter = selected_data_blk_idxs.begin();
  for (size_t i = 0; i < group_count; i++) {
    const size_t idx = (*iter);
    const DataBlk<QNT> &data_blk = blk_idx_data_blk_map_.at(idx);
    if constexpr (Fermionicable<QNT>::IsFermionic()) {

    }
    const size_t off_set = data_blk.data_offset;
    const std::vector<size_t> &shape = data_blk.shape;
    auto a_mat = pactual_raw_data_ + off_set;
    auto b_mat = transposed_data + off_set;
    size_t rows(1), cols(1);
    for (size_t j = 0; j < critical_axe; j++) {
      rows *= shape[j];
    }
    for (size_t j = critical_axe; j < ten_rank; j++) {
      cols *= shape[j];
    }
    hp_numeric::MatrixTranspose(
        a_mat,
        rows,
        cols,
        b_mat
    );
    iter++;
  }
#endif
}

/**
Copy contents from a real block sparse data tensor.

@param real_bsdt A real block sparse data tensor.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::CopyFromReal(
    const BlockSparseDataTensor<QLTEN_Double, QNT> &real_bsdt
) {
  Clear();
  if (std::is_same<ElemT, QLTEN_Complex>::value) {
    for (auto &blk_idx_data_blk : real_bsdt.GetBlkIdxDataBlkMap()) {
      DataBlkInsert(blk_idx_data_blk.second.blk_coors, false);
    }
    if (IsScalar() && (real_bsdt.GetActualRawDataSize() != 0)) {
      raw_data_size_ = 1;
    }

    Allocate();
    RawDataDuplicateFromReal_(
        real_bsdt.GetActualRawDataPtr(),
        real_bsdt.GetActualRawDataSize()
    );
  } else {
    assert(false);    // TODO: To-be implemented!
  }
}

template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::CopyFromReal(
    const BlockSparseDataTensor<QLTEN_Float, QNT> &real_bsdt
) {
  Clear();
  if (std::is_same<ElemT, QLTEN_ComplexFloat>::value) {
    for (auto &blk_idx_data_blk : real_bsdt.GetBlkIdxDataBlkMap()) {
      DataBlkInsert(blk_idx_data_blk.second.blk_coors, false);
    }
    if (IsScalar() && (real_bsdt.GetActualRawDataSize() != 0)) {
      raw_data_size_ = 1;
    }

    Allocate();
    RawDataDuplicateFromReal_(
        real_bsdt.GetActualRawDataPtr(),
        real_bsdt.GetActualRawDataSize()
    );
  } else {
    assert(false);    // TODO: To-be implemented!
  }
}
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::CollectiveLinearCombine(
    const std::vector<const BlockSparseDataTensor *> pbsdts
) {
  const size_t tensor_num = pbsdts.size();
  for (size_t i = 0; i < tensor_num; i++) {
    const BlockSparseDataTensor<ElemT, QNT> *pbsdt = pbsdts[i];
    const decltype(blk_idx_data_blk_map_) &blk_idx_data_blk_map_out = pbsdt->GetBlkIdxDataBlkMap();
    blk_idx_data_blk_map_.insert(blk_idx_data_blk_map_out.cbegin(), blk_idx_data_blk_map_out.cend());
  }
  size_t total_data_offset = 0;
  for (auto &idx_blk : blk_idx_data_blk_map_) {
    idx_blk.second.data_offset = total_data_offset;
    total_data_offset += idx_blk.second.size;
  }
  raw_data_size_ += total_data_offset;
  Allocate(false);

  size_t data_blk_size = blk_idx_data_blk_map_.size();
#ifndef NDEBUG
  size_t out_data_blk_size = 0;
  for (size_t i = 0; i < tensor_num; i++) {
    const BlockSparseDataTensor<ElemT, QNT> *pbsdt = pbsdts[i];
    const decltype(blk_idx_data_blk_map_) &blk_idx_data_blk_map_out = pbsdt->GetBlkIdxDataBlkMap();
    out_data_blk_size += blk_idx_data_blk_map_out.size();
  }
  assert(out_data_blk_size == data_blk_size);
#endif
  std::vector<ElemT *> source_pointers(data_blk_size);
  std::vector<ElemT *> dest_pointers(data_blk_size);
  std::vector<size_t> copy_size(data_blk_size);
  size_t task_idx = 0;
  for (size_t i = 0; i < tensor_num; i++) {
    const BlockSparseDataTensor<ElemT, QNT> *pbsdt = pbsdts[i];
    const decltype(blk_idx_data_blk_map_) &blk_idx_data_blk_map_out = pbsdt->GetBlkIdxDataBlkMap();
    for (auto &[idx, datablk] : blk_idx_data_blk_map_out) {
      source_pointers[task_idx] = pbsdt->pactual_raw_data_ + datablk.data_offset;
      DataBlk<QNT> &this_data_blk = blk_idx_data_blk_map_[idx];
      dest_pointers[task_idx] = pactual_raw_data_ + this_data_blk.data_offset;
      copy_size[task_idx] = this_data_blk.size;
      task_idx++;
    }
  }
  RawDataCopy_(source_pointers, dest_pointers, copy_size);
}

template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::SymMatEVDRawDataDecomposition(
    BlockSparseDataTensor<ElemT, QNT> &u,
    BlockSparseDataTensor<qlten::QLTEN_Double, QNT> &d,
    std::vector<SymMatEVDTask> evd_tasks) const {
  u.raw_data_size_ = this->raw_data_size_;
  d.raw_data_size_ = this->raw_data_size_;
  u.Allocate(false);
  d.Allocate(true);
  const ElemT *pa = this->pactual_raw_data_;
  ElemT *pu_start = u.pactual_raw_data_;
  QLTEN_Double *pd_start = d.pactual_raw_data_;
  for (auto &task : evd_tasks) {
    hp_numeric::SymMatEVD(pa + task.data_offset,
                          task.mat_size,
                          pd_start + task.data_offset,
                          pu_start + task.data_offset);
  }
}

} /* qlten */
#endif /* ifndef QLTEN_QLTENSOR_BLK_SPAR_DATA_TEN_GLOBAL_OPERATIONS_H */
