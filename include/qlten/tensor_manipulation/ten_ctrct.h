// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-27 10:02
*
* Description: QuantumLiquids/tensor project. Contract two tensors.
*/

/**
@file ten_ctrct.h
@brief Contract two tensors.
*/
#ifndef QLTEN_TENSOR_MANIPULATION_TEN_CTRCT_H
#define QLTEN_TENSOR_MANIPULATION_TEN_CTRCT_H

#include <vector>     // vector
#include <cassert>     // assert

#include "qlten/framework/bases/executor.h"                 // Executor
#include "qlten/qltensor_all.h"
#include "qlten/tensor_manipulation/basic_operations.h"     // ToComplex

#ifdef Release
#define NDEBUG
#endif

namespace qlten {

// Forward declarations
template<typename TenElemT, typename QNT>
void TenCtrctInitResTen(
    const QLTensor<TenElemT, QNT> *,
    const QLTensor<TenElemT, QNT> *,
    const std::vector<std::vector<size_t>> &,
    QLTensor<TenElemT, QNT> *
);
inline bool TenCtrctNeedTransCheck(
    const std::vector<size_t> &,
    const std::vector<size_t> &,
    std::vector<int> &
);

/**
Tensor contraction executor.

@tparam TenElemT The type of tensor elements.
@tparam QNT The quantum number type of the tensors.
*/
template<typename TenElemT, typename QNT>
class TensorContractionExecutor : public Executor {
 public:
  TensorContractionExecutor(
      const QLTensor<TenElemT, QNT> *,
      const QLTensor<TenElemT, QNT> *,
      const std::vector<std::vector<size_t>> &,
      QLTensor<TenElemT, QNT> *
  );

  void Execute(void) override;

 private:
  const QLTensor<TenElemT, QNT> *pa_;
  const QLTensor<TenElemT, QNT> *pb_;
  QLTensor<TenElemT, QNT> *pc_;
  const std::vector<std::vector<size_t>> &axes_set_;
  bool a_need_trans_;
  bool b_need_trans_;
  std::vector<int> a_trans_orders_; //use int but not size_t to compatible with hptt
  std::vector<int> b_trans_orders_;
  std::vector<RawDataCtrctTask> raw_data_ctrct_tasks_;
};

/**
Initialize a tensor contraction executor.

@param pa Pointer to input tensor \f$ A \f$.
@param pb Pointer to input tensor \f$ B \f$.
@param axes_set To-be contracted tensor axes indexes. For example, {{0, 1}, {3, 2}}.
@param pc Pointer to result tensor \f$ C \f$.
*/
template<typename TenElemT, typename QNT>
TensorContractionExecutor<TenElemT, QNT>::TensorContractionExecutor(
    const QLTensor<TenElemT, QNT> *pa,
    const QLTensor<TenElemT, QNT> *pb,
    const std::vector<std::vector<size_t>> &axes_set,
    QLTensor<TenElemT, QNT> *pc
) : pa_(pa), pb_(pb), pc_(pc), axes_set_(axes_set) {
  assert(pc_->IsDefault());    // Only empty tensor can take the result
  // Check indexes matching
#ifndef NDEBUG
  auto &indexesa = pa->GetIndexes();
  auto &indexesb = pb->GetIndexes();
  for (size_t i = 0; i < axes_set[0].size(); ++i) {
    assert(indexesa[axes_set[0][i]] == InverseIndex(indexesb[axes_set[1][i]]));
  }
#endif
  std::vector<std::vector<size_t> > saved_axes_set = TenCtrctGenSavedAxesSet(
      pa->Rank(),
      pb->Rank(),
      axes_set
  );

  a_need_trans_ = TenCtrctNeedTransCheck(
      saved_axes_set[0],
      axes_set[0],
      a_trans_orders_
  );

  b_need_trans_ = TenCtrctNeedTransCheck(
      axes_set[1],
      saved_axes_set[1],
      b_trans_orders_
  );

  TenCtrctInitResTen(pa_, pb_, saved_axes_set, pc_);
  raw_data_ctrct_tasks_ = pc_->GetBlkSparDataTen().DataBlkGenForTenCtrct(
      pa_->GetBlkSparDataTen(),
      pb_->GetBlkSparDataTen(),
      axes_set_,
      saved_axes_set
  );

  SetStatus(ExecutorStatus::INITED);
}

/**
Allocate memory and perform raw data contraction calculation.
*/
template<typename TenElemT, typename QNT>
void TensorContractionExecutor<TenElemT, QNT>::Execute(void) {
  SetStatus(ExecutorStatus::EXEING);

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
Function version for tensor contraction.

@tparam TenElemT The type of tensor elements.
@tparam QNT The quantum number type of the tensors.

@param pa Pointer to input tensor \f$ A \f$.
@param pb Pointer to input tensor \f$ B \f$.
@param axes_set To-be contracted tensor axes indexes. For example, {{0, 1}, {3, 2}}.
@param pc Pointer to result tensor \f$ C \f$.

@note For fermionic tensors, the indices with direction IN corresponds to the |ket>,
 while OUT corresponds to the <bra|, which defines
 when last index of pa with index direction OUT, is contracted on the first
 index of pb with index direction IN, gives no additional sign for odd qn sectors.
*/
template<typename TenElemT, typename QNT>
void Contract(
    const QLTensor<TenElemT, QNT> *pa,
    const QLTensor<TenElemT, QNT> *pb,
    const std::vector<std::vector<size_t>> &axes_set,
    QLTensor<TenElemT, QNT> *pc
) {
  TensorContractionExecutor<TenElemT, QNT> ten_ctrct_executor(
      pa,
      pb,
      axes_set,
      pc
  );
  ten_ctrct_executor.Execute();
}

template<typename QNT>
void Contract(
    const QLTensor<QLTEN_Double, QNT> *pa,
    const QLTensor<QLTEN_Complex, QNT> *pb,
    const std::vector<std::vector<size_t>> &axes_set,
    QLTensor<QLTEN_Complex, QNT> *pc
) {
  auto cplx_a = ToComplex(*pa);
  Contract(&cplx_a, pb, axes_set, pc);
}

template<typename QNT>
void Contract(
    const QLTensor<QLTEN_Complex, QNT> *pa,
    const QLTensor<QLTEN_Double, QNT> *pb,
    const std::vector<std::vector<size_t>> &axes_set,
    QLTensor<QLTEN_Complex, QNT> *pc
) {
  auto cplx_b = ToComplex(*pb);
  Contract(pa, &cplx_b, axes_set, pc);
}

template<typename TenElemT1, typename TenElemT2, typename TenElemT3, typename QNT>
void Contract(
    const QLTensor<TenElemT1, QNT> *pa,
    const std::vector<size_t> &axes_a,
    const QLTensor<TenElemT2, QNT> *pb,
    const std::vector<size_t> &axes_b,
    QLTensor<TenElemT3, QNT> *pc
) {
  std::vector<std::vector<size_t>> axes_set = {axes_a, axes_b};
  Contract(pa, pb, axes_set, pc);
}

/**
Initialize tensor contraction result tensor.

@param pa Pointer to input tensor \f$ A \f$.
@param pb Pointer to input tensor \f$ B \f$.
@param axes_set To-be contracted tensor axes indexes. For example, {{0, 1}, {3, 2}}.
@param pc Pointer to result tensor \f$ C \f$.
*/
template<typename TenElemT, typename QNT>
void TenCtrctInitResTen(
    const QLTensor<TenElemT, QNT> *pa,
    const QLTensor<TenElemT, QNT> *pb,
    const std::vector<std::vector<size_t>> &saved_axes_set,
    QLTensor<TenElemT, QNT> *pc
) {
  const auto &a_ctrct_axes = saved_axes_set[0];
  const auto &b_ctrct_axes = saved_axes_set[1];
  const size_t c_rank = a_ctrct_axes.size() + b_ctrct_axes.size();
  IndexVec<QNT> c_idxs;
  c_idxs.reserve(c_rank);
  auto &a_idxs = pa->GetIndexes();
  auto &b_idxs = pb->GetIndexes();
  for (size_t saved_axes_a : a_ctrct_axes) {
    c_idxs.push_back(a_idxs[saved_axes_a]);
  }
  for (size_t saved_axes_b : b_ctrct_axes) {
    c_idxs.push_back(b_idxs[saved_axes_b]);
  }

  (*pc) = QLTensor<TenElemT, QNT>(std::move(c_idxs));
}

inline bool TenCtrctNeedTransCheck(
    const std::vector<size_t> &first_part_axes,
    const std::vector<size_t> &second_part_axes,
    std::vector<int> &trans_orders
) {
  bool need_trans(false);
  for (size_t i = 0; i < first_part_axes.size(); i++) {
    if (first_part_axes[i] != i) {
      need_trans = true;
      break;
    }
  }
  if (!need_trans) {
    const size_t first_part_axes_size = first_part_axes.size();
    for (size_t i = 0; i < second_part_axes.size(); i++) {
      if (second_part_axes[i] != i + first_part_axes_size) {
        need_trans = true;
        break;
      }
    }
  }

  if (need_trans) {
    const size_t trans_order_size = first_part_axes.size() + second_part_axes.size();
    const size_t first_part_axes_size = first_part_axes.size();
    trans_orders.resize(trans_order_size);
    /*
    trans_orders.insert(
        trans_orders.end(),
        first_part_axes.begin(),
        first_part_axes.end()
    );
    trans_orders.insert(
        trans_orders.end(),
        second_part_axes.begin(),
        second_part_axes.end()
    );
     */
    for (size_t i = 0; i < first_part_axes_size; i++) {
      trans_orders[i] = first_part_axes[i];
    }
    for (size_t i = 0; i < second_part_axes.size(); i++) {
      trans_orders[i + first_part_axes_size] = second_part_axes[i];
    }
  }
  return need_trans;

}

} /* qlten */
#endif /* ifndef QLTEN_TENSOR_MANIPULATION_TEN_CTRCT_H */
