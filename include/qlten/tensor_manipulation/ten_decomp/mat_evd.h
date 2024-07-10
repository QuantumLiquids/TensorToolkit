// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2024-07-09
*
* Description: QuantumLiquids/tensor project. EVD for a Hermitian 2-leg QLTensor. No truncation.
*/


#ifndef QLTEN_TENSOR_MANIPULATION_TEN_DECOMP_MAT_EVD_H
#define QLTEN_TENSOR_MANIPULATION_TEN_DECOMP_MAT_EVD_H

#include <cassert>     // assert

#include "qlten/framework/bases/executor.h"                           // Executor
#include "qlten/qltensor_all.h"

namespace qlten {

/**
 * T = U * D * U^\dagger
 * T, U and D are stored in pt_, pu_ and pd_, respectively
 */
template<typename TenElemT, typename QNT>
class SymMatEVDExecutor : public Executor {
 public:
  SymMatEVDExecutor(
      const QLTensor<TenElemT, QNT> *,
      QLTensor<TenElemT, QNT> *,
      QLTensor<QLTEN_Double, QNT> *
  );

  ~SymMatEVDExecutor(void) = default;

  void Execute(void) override;

 private:
  const QLTensor<TenElemT, QNT> *pt_;
  QLTensor<TenElemT, QNT> *pu_;
  QLTensor<QLTEN_Double, QNT> *pd_;
  QNT qn0_;
};

template<typename TenElemT, typename QNT>
SymMatEVDExecutor<TenElemT, QNT>::SymMatEVDExecutor(const QLTensor<TenElemT, QNT> *pt,
                                                    QLTensor<TenElemT, QNT> *pu,
                                                    QLTensor<qlten::QLTEN_Double, QNT> *pd
) : pt_(pt), pu_(pu), pd_(pd) {
  assert(pt_->Rank() == 2);
  assert(pu_->IsDefault());
  assert(pd_->IsDefault());
  assert(pt_->GetIndex(0) == InverseIndex(pt_->GetIndex(1)));
  auto qn = pt_->GetIndex(0).GetQNSct(0).GetQn();
  qn0_ = qn + (-qn);
  assert(pt_->Div() == qn0_);
  // hermitian check?


  SymMatEVDInitResTen(pt_, pu_, pd_);

  SetStatus(ExecutorStatus::INITED);
}

template<typename TenElemT, typename QNT>
void SymMatEVDExecutor<TenElemT, QNT>::Execute() {
  SetStatus(ExecutorStatus::EXEING);
  // Generate Eigen value decomposition raw data task
  auto raw_data_tasks = pt_->GetBlkSparDataTen().DataBlkGenForSymMatEVD(
      pu_->GetBlkSparDataTen(),
      pd_->GetBlkSparDataTen()
  );
  // raw data operator
  pt_->GetBlkSparDataTen().SymMatEVDRawDataDecomposition(pu_->GetBlkSparDataTen(),
                                                         pd_->GetBlkSparDataTen(),
                                                         raw_data_tasks
  );
  SetStatus(ExecutorStatus::FINISH);
}

template<typename TenElemT, typename QNT>
void SymMatEVDInitResTen(
    const QLTensor<TenElemT, QNT> *pt,
    QLTensor<TenElemT, QNT> *pu,
    QLTensor<QLTEN_Double, QNT> *pd
) {
  auto indices = pt->GetIndexes();
  *pu = QLTensor<TenElemT, QNT>(indices);
  *pd = QLTensor<QLTEN_Double, QNT>(indices);
}

//wrapper
template<typename TenElemT, typename QNT>
void SymMatEVD(
    const QLTensor<TenElemT, QNT> *pt,
    QLTensor<TenElemT, QNT> *pu,
    QLTensor<QLTEN_Double, QNT> *pd
) {
  SymMatEVDExecutor<TenElemT, QNT> executor(
      pt, pu, pd
  );
  executor.Execute();
}

}//qlten

#endif //QLTEN_TENSOR_MANIPULATION_TEN_DECOMP_MAT_EVD_H
