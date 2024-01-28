// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2021-7-29
*
* Description: QuantumLiquids/tensor project. Contract two tensors, where one of the free indices in the first tensor
*              is projected onto a subspace representing a single quantum number sector.
*/

/**
 * @file ten_ctrct_1sct.h
 * @brief Contract two tensors, where one of the free indices in the first tensor
 *        is projected onto a subspace representing a single quantum number sector.
 *        This function is designed for parallel implementation of
 *        Density Matrix Renormalization Group (DMRG) algorithm.
 */
#ifndef QLTEN_TENSOR_MANIPULATION_TEN_CTRCT_ONE_SCT_H
#define QLTEN_TENSOR_MANIPULATION_TEN_CTRCT_ONE_SCT_H

#include <vector>     // vector
#include <cassert>     // assert

#include "qlten/framework/bases/executor.h"                 // Executor
#include "qlten/qltensor_all.h"
#include "qlten/tensor_manipulation/basic_operations.h"     // ToComplex
#include "qlten/qltensor/blk_spar_data_ten/data_blk_operator_seperate_ctrct.h"

#ifdef Release
#define NDEBUG
#endif

namespace qlten {

/// Forward declarations.
/// TenCtrctInitResTen only gives the Indexes of QLTensor *pc,
/// pointer to BSDT = nullptr.
template<typename TenElemT, typename QNT>
void TenCtrctInitResTen(
    const QLTensor<TenElemT, QNT> *,
    const QLTensor<TenElemT, QNT> *,
    const std::vector<std::vector<size_t>> &,
    QLTensor<TenElemT, QNT> *
);

/**
Tensor contraction executor.

@tparam TenElemT The type of tensor elements.
@tparam QNT The quantum number type of the tensors.
*/
template<typename TenElemT, typename QNT>
class TensorContraction1SectorExecutor : public Executor {
 public:
  TensorContraction1SectorExecutor(
      const QLTensor<TenElemT, QNT> *pa,
      const size_t idx_a,
      const size_t qn_sector_idx_a,
      const QLTensor<TenElemT, QNT> *pb,
      const std::vector<std::vector<size_t>> &,
      QLTensor<TenElemT, QNT> *pc
  );// split the index of pa->GetIndexes[idx_a], contract restrict on qn_sector_idx_a's qnsector

  inline void SetSelectedQNSect(const size_t qn_sector_idx_a) {
    qn_sector_idx_a_ = qn_sector_idx_a;
    ClearResBSDT_();
  }

  void Execute(void) override;

 private:
  void ClearResBSDT_(void) {
    pc_->GetBlkSparDataTen().Clear();
    raw_data_ctrct_tasks_.clear();
    a_blk_idx_data_blk_map_select_.clear();
  }

  void SelectBlkIdxDataBlkMap_() {
    const auto &bsdt_a = pa_->GetBlkSparDataTen();
    const auto &a_blk_idx_data_blk_map_total = bsdt_a.GetBlkIdxDataBlkMap();
    for (auto &[idx, data_blk] : a_blk_idx_data_blk_map_total) {
      if (data_blk.blk_coors[idx_a_] == qn_sector_idx_a_) {
        a_blk_idx_data_blk_map_select_.insert(std::make_pair(idx, data_blk));
      }
    }
  }

  inline void DataBlkGenForTenCtrct_(void) {
    raw_data_ctrct_tasks_ = pc_->GetBlkSparDataTen().DataBlkGenForTenCtrct(
        a_blk_idx_data_blk_map_select_,
        pb_->GetBlkSparDataTen().GetBlkIdxDataBlkMap(),
        axes_set_,
        saved_axes_set_,
        b_blk_idx_qnblk_info_part_hash_map_
    );
  }

  const QLTensor<TenElemT, QNT> *pa_;
  const size_t idx_a_;
  size_t qn_sector_idx_a_;
  const QLTensor<TenElemT, QNT> *pb_;
  QLTensor<TenElemT, QNT> *pc_;
  const std::vector<std::vector<size_t>> axes_set_;
  std::vector<std::vector<size_t>> saved_axes_set_;
  std::vector<int> a_trans_orders_;
  std::vector<int> b_trans_orders_;
  std::vector<RawDataCtrctTask> raw_data_ctrct_tasks_;
  std::map<size_t, qlten::DataBlk<QNT>> a_blk_idx_data_blk_map_select_;
  std::vector<size_t> b_blk_idx_qnblk_info_part_hash_map_;//TODO: rename as coor
};

/**
Initialize a tensor contraction executor.

@param pa Pointer to input tensor \f$ A \f$.
@param pb Pointer to input tensor \f$ B \f$.
@param axes_set To-be contracted tensor axes indexes. For example, {{0, 1}, {3, 2}}.
@param pc Pointer to result tensor \f$ C \f$.
*/
template<typename TenElemT, typename QNT>
TensorContraction1SectorExecutor<TenElemT, QNT>::TensorContraction1SectorExecutor(
    const QLTensor<TenElemT, QNT> *pa,
    const size_t idx_a,
    const size_t qn_sector_idx_a,
    const QLTensor<TenElemT, QNT> *pb,
    const std::vector<std::vector<size_t>> &axes_set,
    QLTensor<TenElemT, QNT> *pc
) : pa_(pa), idx_a_(idx_a), qn_sector_idx_a_(qn_sector_idx_a),
    pb_(pb), pc_(pc), axes_set_(axes_set) {
  assert(pc_->IsDefault());    // Only empty tensor can take the result
#ifndef NDEBUG
  // Check indexes matching
  auto indexesa = pa->GetIndexes();
  auto indexesb = pb->GetIndexes();
  for (size_t i = 0; i < axes_set[0].size(); ++i) {
    assert(indexesa[axes_set[0][i]] == InverseIndex(indexesb[axes_set[1][i]]));
  }
  // Check if idx_a_ is in the outer legs
  auto iter = find(axes_set[0].begin(), axes_set[0].end(), idx_a);
  assert(iter == axes_set[0].cend());
  // Check if qn_sector_idx_a_ < number of quantum number sector
  assert(qn_sector_idx_a_ < pa->GetIndexes()[idx_a_].GetQNSctNum());
#endif
  //Then we assign the DataBlk and and contract tasks
  auto &bsdt_a = pa_->GetBlkSparDataTen();
  auto &bsdt_b = pb_->GetBlkSparDataTen();
  assert(!(bsdt_a.IsScalar() || bsdt_b.IsScalar()));
  saved_axes_set_ = TenCtrctGenSavedAxesSet(
      bsdt_a.ten_rank,
      bsdt_b.ten_rank,
      axes_set_
  );
  TenCtrctNeedTransCheck(
      saved_axes_set_[0],
      axes_set[0],
      a_trans_orders_
  );

  TenCtrctNeedTransCheck(
      axes_set[1],
      saved_axes_set_[1],
      b_trans_orders_
  );
  TenCtrctInitResTen(pa_, pb_, saved_axes_set_, pc_);
  auto &b_blk_idx_data_blk_map = bsdt_b.GetBlkIdxDataBlkMap();
  b_blk_idx_qnblk_info_part_hash_map_ = bsdt_b.GenBlkIdxQNBlkCoorPartHashMap(
      b_blk_idx_data_blk_map,
      axes_set_[1]
  );
  SetStatus(ExecutorStatus::INITED);
}

/**
Allocate memory and perform raw data contraction calculation.
*/
template<typename TenElemT, typename QNT>
void TensorContraction1SectorExecutor<TenElemT, QNT>::Execute(void) {
  SetStatus(ExecutorStatus::EXEING);
  SelectBlkIdxDataBlkMap_();
  DataBlkGenForTenCtrct_();

  pc_->GetBlkSparDataTen().CtrctTwoBSDTAndAssignIn(
      pa_->GetBlkSparDataTen(),
      pb_->GetBlkSparDataTen(),
      raw_data_ctrct_tasks_,
      a_trans_orders_,
      b_trans_orders_
  );

  SetStatus(ExecutorStatus::FINISH);
}

/**
Function version for tensor contraction, with one of index restrict on only one sector

@tparam TenElemT The type of tensor elements.
@tparam QNT The quantum number type of the tensors.

@param pa Pointer to input tensor \f$ A \f$.
@param idx_a the idx_a-th leg is restrict.
@param qn_sector_idx_a restrict on qn_sector_idx_a-th quantum number sector.
@param pb Pointer to input tensor \f$ B \f$.
@param axes_set To-be contracted tensor axes indexes. For example, {{0, 1}, {3, 2}}.
@param pc Pointer to result tensor \f$ C \f$.
*/
template<typename TenElemT, typename QNT>
void Contract1Sector(
    const QLTensor<TenElemT, QNT> *pa,
    const size_t idx_a,
    const size_t qn_sector_idx_a,
    const QLTensor<TenElemT, QNT> *pb,
    const std::vector<std::vector<size_t>> &axes_set,
    QLTensor<TenElemT, QNT> *pc
) {
  TensorContraction1SectorExecutor<TenElemT, QNT> ten_ctrct_1sector_executor(
      pa,
      idx_a,
      qn_sector_idx_a,
      pb,
      axes_set,
      pc
  );
  ten_ctrct_1sector_executor.Execute();
}

template<typename QNT>
void Contract1Sector(
    const QLTensor<QLTEN_Double, QNT> *pa,
    const size_t idx_a,
    const size_t qn_sector_idx_a,
    const QLTensor<QLTEN_Complex, QNT> *pb,
    const std::vector<std::vector<size_t>> &axes_set,
    QLTensor<QLTEN_Complex, QNT> *pc
) {
  auto cplx_a = ToComplex(*pa);
  Contract1Sector(&cplx_a, idx_a, qn_sector_idx_a, pb, axes_set, pc);
}

template<typename QNT>
void Contract1Sector(
    const QLTensor<QLTEN_Complex, QNT> *pa,
    const size_t idx_a,
    const size_t qn_sector_idx_a,
    const QLTensor<QLTEN_Double, QNT> *pb,
    const std::vector<std::vector<size_t>> &axes_set,
    QLTensor<QLTEN_Complex, QNT> *pc
) {
  auto cplx_b = ToComplex(*pb);
  Contract1Sector(pa, idx_a, qn_sector_idx_a, &cplx_b, axes_set, pc);
}

} /* qlten */
#endif /* ifndef QLTEN_TENSOR_MANIPULATION_TEN_CTRCT_ONE_SCT_H */
