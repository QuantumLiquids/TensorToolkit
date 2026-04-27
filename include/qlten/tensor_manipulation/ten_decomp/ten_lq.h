// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: TensorToolkit Contributors
* Creation Date: 2026-01-04
*
* Description: QuantumLiquids/tensor project. LQ decomposition for a symmetric QLTensor.
*/

/**
@file ten_lq.h
@brief LQ decomposition for a symmetric QLTensor.

LQ decomposition factorizes a tensor T into L * Q:
  T[i0, ..., i_{n-k-1}, i_{n-k}, ..., i_{n-1}]
    = L[i0, ..., i_{n-k-1}, m(OUT)] * Q[m(IN), i_{n-k}, ..., i_{n-1}]

where:
- rdims = k is the number of right indices that Q retains
- Q is right-orthonormal (rows form an orthonormal set): Q * Q† = I
- L's last index has direction OUT, Q's first index has direction IN
- These two middle indices are InverseIndex of each other
*/
#ifndef QLTEN_TENSOR_MANIPULATION_TEN_DECOMP_TEN_LQ_H
#define QLTEN_TENSOR_MANIPULATION_TEN_DECOMP_TEN_LQ_H


#include "qlten/framework/bases/executor.h"     // Executor
#include "qlten/qltensor_all.h"
#include "qlten/tensor_manipulation/ten_decomp/ten_decomp_basic.h"    // GenIdxTenDecompDataBlkMats
#include "qlten/tensor_manipulation/ten_decomp/ten_qr.h"    // GenMidQNSects, QRDataBlkInfo

#include <algorithm>    // min
#include <stdexcept>    // runtime_error

#ifdef Release
  #define NDEBUG
#endif
#include <cassert>     // assert


namespace qlten {


using LQDataBlkInfo = QRDataBlkInfo;
using LQDataBlkInfoVec = std::vector<LQDataBlkInfo>;
using LQDataBlkInfoVecPair = std::pair<LQDataBlkInfoVec, LQDataBlkInfoVec>;


/**
Tensor LQ executor.
*/
template <typename TenElemT, typename QNT>
class TensorLQExecutor : public Executor {
public:
  TensorLQExecutor(
      const QLTensor<TenElemT, QNT> *,
      const size_t,
      const QNT &,
      QLTensor<TenElemT, QNT> *,
      QLTensor<TenElemT, QNT> *
  );

  ~TensorLQExecutor(void) = default;

  void Execute(void) override;

private:
  const QLTensor<TenElemT, QNT> *pt_;
  const size_t ldims_;
  const QNT &lqndiv_;
  QLTensor<TenElemT, QNT> *pl_;
  QLTensor<TenElemT, QNT> *pq_;
  IdxDataBlkMatMap<QNT> idx_ten_decomp_data_blk_mat_map_;
  void ConstructLQResTens_(const std::map<size_t, DataBlkMatLqRes<TenElemT>> &);
  LQDataBlkInfoVecPair CreatLQResTens_(void);
  void FillLQResTens_(
      const std::map<size_t, DataBlkMatLqRes<TenElemT>> &,
      const LQDataBlkInfoVecPair &
  );
};


/**
Initialize a tensor LQ executor.

@tparam TenElemT The element type of the tensors.
@tparam QNT The quantum number type of the tensors.

@param pt A pointer to to-be LQ decomposed tensor \f$ T \f$.
@param ldims Number of indices on the left hand side of the decomposition.
@param lqndiv Quantum number divergence of the result \f$ L \f$ tensor.
@param pl A pointer to result \f$ L \f$ tensor.
@param pq A pointer to result \f$ Q \f$ tensor.
*/
template <typename TenElemT, typename QNT>
TensorLQExecutor<TenElemT, QNT>::TensorLQExecutor(
    const QLTensor<TenElemT, QNT> *pt,
    const size_t ldims,
    const QNT &lqndiv,
    QLTensor<TenElemT, QNT> *pl,
    QLTensor<TenElemT, QNT> *pq
) : pt_(pt), ldims_(ldims), lqndiv_(lqndiv), pl_(pl), pq_(pq) {
  assert(pt_->Rank() >= 2);
  assert(ldims_ < pt_->Rank());
  assert(pl_->IsDefault());
  assert(pq_->IsDefault());

  idx_ten_decomp_data_blk_mat_map_ = GenIdxTenDecompDataBlkMats(
                                         *pt_,
                                         ldims_,
                                         lqndiv_
                                     );

  SetStatus(ExecutorStatus::INITED);
}


/**
Execute tensor LQ decomposition calculation.
*/
template <typename TenElemT, typename QNT>
void TensorLQExecutor<TenElemT, QNT>::Execute(void) {
  SetStatus(ExecutorStatus::EXEING);

  auto idx_raw_data_lq_res = pt_->GetBlkSparDataTen().DataBlkDecompLQ(
                                 idx_ten_decomp_data_blk_mat_map_
                             );
  if (idx_raw_data_lq_res.empty()) {
    throw std::runtime_error(
        "LQ failed: empty result. The input tensor has no data blocks "
        "compatible with the specified quantum number divergence (lqndiv)."
    );
  }
  ConstructLQResTens_(idx_raw_data_lq_res);
  DeleteDataBlkMatLqResMap(idx_raw_data_lq_res);

  SetStatus(ExecutorStatus::FINISH);
}


/**
LQ decomposition for a QLTensor.

@tparam TenElemT The element type of the tensors.
@tparam QNT The quantum number type of the tensors.

@param pt A pointer to the to-be LQ decomposed tensor \f$ T \f$. The rank of
       \f$ T \f$ should be larger than 1.
@param rdims Number of indices on the right hand side that Q retains.
       Must satisfy 0 < rdims < pt->Rank().
@param rqndiv Quantum number divergence of the result \f$ Q \f$ tensor.
@param pl A pointer to result \f$ L \f$ tensor (lower triangular factor).
@param pq A pointer to result \f$ Q \f$ tensor (right-orthonormal factor).
*/
template <typename TenElemT, typename QNT>
void LQ(
    const QLTensor<TenElemT, QNT> *pt,
    const size_t rdims,
    const QNT &rqndiv,
    QLTensor<TenElemT, QNT> *pl,
    QLTensor<TenElemT, QNT> *pq
) {
  assert(pt->Rank() >= 2);
  assert(rdims > 0 && rdims < pt->Rank());
  assert(pl->IsDefault());
  assert(pq->IsDefault());

  const size_t ldims = pt->Rank() - rdims;
  QNT lqndiv = Div(*pt) - rqndiv;

  TensorLQExecutor<TenElemT, QNT> executor(pt, ldims, lqndiv, pl, pq);
  executor.Execute();
}


/**
Construct LQ result tensors.
*/
template <typename TenElemT, typename QNT>
void TensorLQExecutor<TenElemT, QNT>::ConstructLQResTens_(
    const std::map<size_t, DataBlkMatLqRes<TenElemT>> &idx_raw_data_lq_res
) {
  auto l_q_data_blks_info = CreatLQResTens_();
  FillLQResTens_(idx_raw_data_lq_res, l_q_data_blks_info);
}


template <typename TenElemT, typename QNT>
LQDataBlkInfoVecPair TensorLQExecutor<TenElemT, QNT>::CreatLQResTens_(void) {
  // Initialize l, q tensors
  auto mid_qnscts = GenMidQNSects(idx_ten_decomp_data_blk_mat_map_);
  auto mid_index_out = Index<QNT>(mid_qnscts, TenIndexDirType::OUT);
  auto mid_index_in = InverseIndex(mid_index_out);
  auto t_indexes = pt_->GetIndexes();

  // L gets [left_indices..., mid_index_out]
  IndexVec<QNT> l_indexes(t_indexes.begin(), t_indexes.begin() + ldims_);
  l_indexes.push_back(mid_index_out);
  (*pl_) = QLTensor<TenElemT, QNT>(std::move(l_indexes));

  // Q gets [mid_index_in, right_indices...]
  IndexVec<QNT> q_indexes{mid_index_in};
  q_indexes.insert(
      q_indexes.end(),
      t_indexes.begin() + ldims_, t_indexes.end()
  );
  (*pq_) = QLTensor<TenElemT, QNT>(std::move(q_indexes));

  // Insert empty data blocks
  LQDataBlkInfoVec l_data_blks_info, q_data_blks_info;
  size_t mid_blk_coor = 0;
  for (
      auto &[data_blk_mat_idx, data_blk_mat] : idx_ten_decomp_data_blk_mat_map_
  ) {
    auto k = std::min(data_blk_mat.rows, data_blk_mat.cols);

    // L blocks: [left_block_coors..., mid_blk_coor], shape (row_dim, k)
    for (auto &row_sct : data_blk_mat.row_scts) {
      CoorsT l_data_blk_coors(std::get<0>(row_sct));
      l_data_blk_coors.push_back(mid_blk_coor);
      pl_->GetBlkSparDataTen().DataBlkInsert(l_data_blk_coors, false);
      l_data_blks_info.push_back(
          LQDataBlkInfo(
              l_data_blk_coors,
              data_blk_mat_idx, std::get<1>(row_sct),
              std::get<2>(row_sct), k
          )
      );
    }

    // Q blocks: [mid_blk_coor, right_block_coors...], shape (k, col_dim)
    for (auto &col_sct : data_blk_mat.col_scts) {
      CoorsT q_data_blk_coors{mid_blk_coor};
      auto rpart_blk_coors = std::get<0>(col_sct);
      q_data_blk_coors.insert(
          q_data_blk_coors.end(),
          rpart_blk_coors.begin(), rpart_blk_coors.end()
      );
      pq_->GetBlkSparDataTen().DataBlkInsert(q_data_blk_coors, false);
      q_data_blks_info.push_back(
          LQDataBlkInfo(
              q_data_blk_coors,
              data_blk_mat_idx, std::get<1>(col_sct),
              k, std::get<2>(col_sct)
          )
      );
    }

    mid_blk_coor++;
  }
  return std::make_pair(l_data_blks_info, q_data_blks_info);
}


template <typename TenElemT, typename QNT>
void TensorLQExecutor<TenElemT, QNT>::FillLQResTens_(
    const std::map<size_t, DataBlkMatLqRes<TenElemT>> &idx_lq_res_map,
    const LQDataBlkInfoVecPair &l_q_data_blks_info
) {
  // Fill L tensor (left factor, row offset pattern like QR's Q)
  pl_->GetBlkSparDataTen().Allocate();
  auto l_data_blks_info = l_q_data_blks_info.first;
  for (auto &l_data_blk_info : l_data_blks_info) {
    auto lq_res = idx_lq_res_map.at(l_data_blk_info.data_blk_mat_idx);
    pl_->GetBlkSparDataTen().DataBlkCopyLQLdata(
        l_data_blk_info.blk_coors,
        l_data_blk_info.m,
        l_data_blk_info.n,
        l_data_blk_info.offset,
        lq_res.l, lq_res.m, lq_res.k
    );
  }

  // Fill Q tensor (right factor, col offset pattern like QR's R)
  pq_->GetBlkSparDataTen().Allocate();
  auto q_data_blks_info = l_q_data_blks_info.second;
  for (auto &q_data_blk_info : q_data_blks_info) {
    auto lq_res = idx_lq_res_map.at(q_data_blk_info.data_blk_mat_idx);
    pq_->GetBlkSparDataTen().DataBlkCopyLQQdata(
        q_data_blk_info.blk_coors,
        q_data_blk_info.m,
        q_data_blk_info.n,
        q_data_blk_info.offset,
        lq_res.q, lq_res.k, lq_res.n
    );
  }
}
} /* qlten */
#endif /* ifndef QLTEN_TENSOR_MANIPULATION_TEN_DECOMP_TEN_LQ_H */
