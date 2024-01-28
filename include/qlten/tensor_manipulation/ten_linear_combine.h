// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
*         Hao-Xin Wang
* Creation Date: 2020-12-10 18:23
*
* Description: QuantumLiquids/tensor project. Perform linear combination of tensors.
*/

/**
@file ten_linear_combine.h
@brief Perform linear combination of tensors.
*/
#ifndef QLTEN_TENSOR_MANIPULATION_TEN_LINEAR_COMBINE_H
#define QLTEN_TENSOR_MANIPULATION_TEN_LINEAR_COMBINE_H

#include <unordered_set>    // unordered_set
#include <map>              // map
#include <cassert>     // assert

#include "qlten/framework/bases/executor.h"                         // Executor
#include "qlten/qltensor_all.h"
#include "qlten/utility/timer.h"

#ifdef Release
#define NDEBUG
#endif

namespace qlten {

template<typename CoefT>
using TenLinCmbDataCopyTasks = std::vector<RawDataCopyAndScaleTask<CoefT>>;

template<typename CoefT>
using TenIdxTenLinCmbDataCopyTasksMap = std::map<
    size_t,
    TenLinCmbDataCopyTasks<CoefT>
>;

/**
Tensors linear combination executor. \f$ T = a \cdot A + b \cdot B + \cdots + \beta T\f$.

@tparam TenElemT The type of tensor elements.
@tparam QNT The quantum number type of the tensors.

@note For linear combination, rank 0 tensor (scalar) is not supported.
*/
template<typename TenElemT, typename QNT>
class TensorLinearCombinationExecutor : public Executor {
 public:
  using TenT = QLTensor<TenElemT, QNT>;

  TensorLinearCombinationExecutor(
      const std::vector<TenElemT> &,
      const std::vector<TenT *> &,
      const TenElemT,
      TenT *
  );

  void Execute(void) override;

 private:
  std::vector<TenElemT> coefs_;
  std::vector<TenT *> tens_;
  TenElemT beta_;
  TenT *pres_;

  TenT actual_res_;
  TenIdxTenLinCmbDataCopyTasksMap<
      TenElemT
  > ten_idx_ten_lin_cmb_data_copy_tasks_map_;
};

/**
Initialize a tensor linear combination executor.

@param coefs Coefficients of each to-be linear combined tensor.
@param tens To-be linear combined tensors. They mush have same indexes.
@param beta \f$ \beta \f$.
@param pres The pointer to result.
*/
template<typename TenElemT, typename QNT>
TensorLinearCombinationExecutor<TenElemT, QNT>::TensorLinearCombinationExecutor(
    const std::vector<TenElemT> &coefs,
    const std::vector<TenT *> &tens,
    const TenElemT beta,
    TenT *pres
) : coefs_(coefs), tens_(tens), beta_(beta), pres_(pres) {
  assert(coefs_.size() != 0);
  assert(coefs_.size() == tens_.size());
#ifndef NDEBUG
  auto indexes = tens_[0]->GetIndexes();
  for (size_t i = 1; i < tens_.size(); ++i) {
    assert(tens_[i]->GetIndexes() == indexes);
  }
  if (!pres_->IsDefault()) {
    assert(pres_->GetIndexes() == indexes);
  }
#endif

  actual_res_ = TenT(tens_[0]->GetIndexes());

  // Deal with input result tensor and its coefficient beta
  if (beta_ != 0.0) {
    assert(!pres_->IsDefault());
    coefs_.push_back(beta_);
    tens_.push_back(pres_);
  }
  std::unordered_set<size_t> res_data_blk_idx_set;
  for (size_t i = 0; i < coefs_.size(); ++i) {
    if (coefs_[i] != 0.0) {
      ten_idx_ten_lin_cmb_data_copy_tasks_map_[i] =
          GenTenLinearCombineDataCopyTasks(
              coefs_[i],
              tens_[i],
              actual_res_,
              res_data_blk_idx_set
          );
    }
  }

  SetStatus(ExecutorStatus::INITED);
}

template<typename TenElemT, typename QNT>
void TensorLinearCombinationExecutor<TenElemT, QNT>::Execute(void) {
  SetStatus(ExecutorStatus::EXEING);

  actual_res_.GetBlkSparDataTen().Allocate();
  for (auto &ten_idx_tasks : ten_idx_ten_lin_cmb_data_copy_tasks_map_) {
    auto ten_idx = ten_idx_tasks.first;
    auto ten_bsdt_raw_data = tens_[
        ten_idx
    ]->GetBlkSparDataTen().GetActualRawDataPtr();
    auto tasks = ten_idx_tasks.second;
    for (auto &task : tasks) {
      actual_res_.GetBlkSparDataTen().DataBlkCopyAndScale(task, ten_bsdt_raw_data);
    }
  }
  (*pres_) = std::move(actual_res_);

  SetStatus(ExecutorStatus::FINISH);
}

template<typename TenElemT, typename QNT>
TenLinCmbDataCopyTasks<TenElemT> GenTenLinearCombineDataCopyTasks(
    const TenElemT coef,
    const QLTensor<TenElemT, QNT> *pten,
    QLTensor<TenElemT, QNT> &res,
    std::unordered_set<size_t> &res_data_blk_idx_set
) {
  TenLinCmbDataCopyTasks<TenElemT> tasks;
  auto idx_data_blk_map = pten->GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  auto data_blk_num = idx_data_blk_map.size();
  tasks.reserve(data_blk_num);
  bool copy_and_add;
  for (auto &idx_data_blk : idx_data_blk_map) {
    auto idx = idx_data_blk.first;
    auto data_blk = idx_data_blk.second;
    if (res_data_blk_idx_set.find(idx) != res_data_blk_idx_set.end()) {
      copy_and_add = true;
    } else {
      copy_and_add = false;
      res.GetBlkSparDataTen().DataBlkInsert(data_blk.blk_coors, false);
      res_data_blk_idx_set.insert(idx);
    }
    tasks.push_back(
        RawDataCopyAndScaleTask<TenElemT>(
            data_blk.data_offset,
            data_blk.size,
            data_blk.blk_coors,
            coef,
            copy_and_add
        )
    );
  }
  return tasks;
}

/**
Function version of tensors linear combination. \f$ T = a \cdot A + b \cdot B + \cdots + \beta T\f$.

@param coefs Coefficients of each to-be linear combined tensor.
@param tens To-be linear combined tensors. They mush have same indexes.
@param beta \f$ \beta \f$.
@param pres The pointer to result.
*/
template<typename TenElemT, typename QNT>
void LinearCombine(
    const std::vector<TenElemT> &coefs,
    const std::vector<QLTensor<TenElemT, QNT> *> &tens,
    const TenElemT beta,
    QLTensor<TenElemT, QNT> *pres
) {
  auto ten_lin_cmb_exector = TensorLinearCombinationExecutor<TenElemT, QNT>(
      coefs, tens,
      beta, pres
  );
  ten_lin_cmb_exector.Execute();
}

// Other function versions
inline std::vector<QLTEN_Complex> ToCplxVec(
    const std::vector<QLTEN_Double> &real_v
) {
  std::vector<QLTEN_Complex> cplx_v;
  cplx_v.reserve(real_v.size());
  for (auto &e : real_v) {
    cplx_v.emplace_back(e);
  }
  return cplx_v;
}

template<typename QNT>
void LinearCombine(
    const std::vector<QLTEN_Double> &coefs,
    const std::vector<QLTensor<QLTEN_Complex, QNT> *> &tens,
    const QLTEN_Complex beta,
    QLTensor<QLTEN_Complex, QNT> *pres
) {
  LinearCombine(ToCplxVec(coefs), tens, beta, pres);
}

template<typename ElemType, typename QNT>
void LinearCombine(
    const size_t size,
    const ElemType *pcoefs,
    const std::vector<QLTensor<ElemType, QNT> *> &tens,
    const ElemType beta,
    QLTensor<ElemType, QNT> *pres
) {
  std::vector<ElemType> coefs;
  coefs.resize(size);
  std::copy_n(pcoefs, size, coefs.begin());
  std::vector<QLTensor<ElemType, QNT> *> actual_tens;
  actual_tens.resize(size);
  std::copy_n(tens.begin(), size, actual_tens.begin());
  LinearCombine(coefs, actual_tens, beta, pres);
}

template<typename QNT>
void LinearCombine(
    const size_t size,
    const QLTEN_Double *pcoefs,
    const std::vector<QLTensor<QLTEN_Complex, QNT> *> &tens,
    const QLTEN_Complex beta,
    QLTensor<QLTEN_Complex, QNT> *pres
) {
  std::vector<QLTEN_Complex> coefs;
  coefs.resize(size);
  std::copy_n(pcoefs, size, coefs.begin());
  std::vector<QLTensor<QLTEN_Complex, QNT> *> actual_tens;
  actual_tens.resize(size);
  std::copy_n(tens.begin(), size, actual_tens.begin());
  LinearCombine(coefs, actual_tens, beta, pres);
}

/**
 * @param tens  tens should have different data_blks
 * @return summation of tens
 */
template<typename ElemT, typename QNT>
void CollectiveLinearCombine(
    std::vector<QLTensor<ElemT, QNT>> &tens,
    QLTensor<ElemT, QNT> &summation_tensor
) {
  assert(tens.size() > 0);
  using std::vector;
  std::vector<Index<QNT>> indexes = tens[0].GetIndexes();
  summation_tensor = QLTensor<ElemT, QNT>(indexes);
  size_t ten_num = tens.size();
  vector<const BlockSparseDataTensor<ElemT, QNT> *> pbsdts(ten_num);
  for (size_t i = 0; i < ten_num; i++) {
    pbsdts[i] = tens[i].GetBlkSparDataTenPtr();
  }
  BlockSparseDataTensor<ElemT, QNT> &bsdt = summation_tensor.GetBlkSparDataTen();
  bsdt.CollectiveLinearCombine(pbsdts);
}

} /* qlten */
#endif /* ifndef QLTEN_TENSOR_MANIPULATION_TEN_LINEAR_COMBINE_H */
