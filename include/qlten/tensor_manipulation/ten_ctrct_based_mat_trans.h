// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2021-10-23.
*
* Description: Implementation of fast tensor contraction functions without tensor transposition.
*/

/**
@file ten_ctrct_based_mat_trans.h
@brief Implementation of a faster tensor contraction functions without involving tensor transposition operations.

       This contraction method is defined by the following constraints:
       - The contracted indices must be continuous and ordered. Specifically, for a contraction
         defined as Contract(A, B, axes1, axes2), the axes1 indices must be in ascending order with
         a step of 1, and the same applies to axes2.
       - Periodic boundary conditions for indices are allowed. For example, if A.rank() == 5,
         axes1 can be {3, 4, 0}.

       These constraints allow the transpositions required for contraction to be realized as matrix transpositions.
*/


#ifndef QLTEN_TENSOR_MANIPULATION_TEN_CTRCT_BASED_MAT_TRANS_H
#define QLTEN_TENSOR_MANIPULATION_TEN_CTRCT_BASED_MAT_TRANS_H

#include <set>

#include "qlten/framework/bases/executor.h"
#include "qlten/qltensor_all.h"

namespace qlten {

/**
 * @tparam TenElemT Type of the tensor elements.
 * @tparam QNT Type of the quantum number.
 * @tparam a_ctrct_tail Indicates whether the contraction occurs at the tail of tensor A.
 * @tparam b_ctrct_head Indicates whether the contraction occurs at the head of tensor B.
 *
 * Note:
 * - When `a_ctrct_tail` is true and `b_ctrct_head` is true, no transpose is needed for the matrix product.
 * - Otherwise, transpose the corresponding tensor(s) for the matrix product.
 * - These two parameters do not change the result of the contraction but provide hints to optimize operations and enhance performance.
 */
template<typename TenElemT, typename QNT, bool a_ctrct_tail, bool b_ctrct_head>
class MatrixBasedTensorContractionExecutor : public Executor {
 public:
  MatrixBasedTensorContractionExecutor(
      const QLTensor<TenElemT, QNT> *,
      const QLTensor<TenElemT, QNT> *,
      const size_t a_ctrct_axes_start,
      const size_t b_ctrct_axes_start,
      const size_t ctrct_axes_size,
      QLTensor<TenElemT, QNT> *
  );

  void Execute(void) override;

 private:
  void GenerateDataBlk_();

  void TransposePrepare_();

  void ExecutePost_();//clear transpose data;

  const QLTensor<TenElemT, QNT> *pa_;
  const QLTensor<TenElemT, QNT> *pb_;
  QLTensor<TenElemT, QNT> *pc_;
  size_t a_ctrct_axes_start_; //ctrct_axes include this one
  size_t b_ctrct_axes_start_;
  size_t a_ctrct_axes_end_; //ctrct_axes do not include this one
  size_t b_ctrct_axes_end_; //ctrct_axes do not include this one
  size_t ctrct_axes_size_;

  std::vector<std::vector<size_t>> saved_axes_set_;

  // trans_critical_axe == 0 imply no transpose is need for the matrix multiplication
  // If trans_critical_axe > 0, the tensor indices from 0 to a_trans_critical_axe_ - 1
  // and from a_trans_critical_axe_ to rank - 1 will be exchanged to perform the transpose.
  size_t a_trans_critical_axe_;
  size_t b_trans_critical_axe_;

  // fermion sign which haven't been count when transpose
  // only need consider saved axes.
  std::map<size_t, int> a_trans_data_blk_idx_map_to_fermion_sign_;
  std::map<size_t, int> b_trans_data_blk_idx_map_to_fermion_sign_;

  TenElemT *a_trans_data_ = nullptr;
  //TODO: more template parameter to determine if the original data is need to save,
  // so that we can save some memory.
  TenElemT *b_trans_data_ = nullptr;
  std::vector<RawDataCtrctTask> raw_data_ctrct_tasks_;
};

/**
 * @example
 *      For the general contraction: Contract(A, B, {1,2,3},{5,6,0});
 *      a_ctrct_axes_start = 1,
 *      b_ctrct_axes_start = 5,
 *      ctrct_axes_size = 3;
 *
 *      for a_ctrct_tail == true case
 *          a_trans_critical_axe_ = 4 if A.Rank()>4 or a_trans_critical_axe_ = 0 if A.Rank()==4;
 *
 *
 * @tparam TenElemT
 * @tparam QNT
 * @tparam a_ctrct_tail
 * @tparam b_ctrct_head
 * @param pa
 * @param pb
 * @param a_ctrct_axes_start
 * @param b_ctrct_axes_start
 * @param ctrct_axes_size
 * @param pc
 *
 *  question: meaning for set status?
 */
template<typename TenElemT, typename QNT, bool a_ctrct_tail, bool b_ctrct_head>
MatrixBasedTensorContractionExecutor<TenElemT, QNT, a_ctrct_tail, b_ctrct_head>::MatrixBasedTensorContractionExecutor(
    const QLTensor<TenElemT, QNT> *pa,
    const QLTensor<TenElemT, QNT> *pb,
    const size_t a_ctrct_axes_start,
    const size_t b_ctrct_axes_start,
    const size_t ctrct_axes_size,
    QLTensor<TenElemT, QNT> *pc
) : pa_(pa), pb_(pb), pc_(pc),
    a_ctrct_axes_start_(a_ctrct_axes_start),
    b_ctrct_axes_start_(b_ctrct_axes_start),
    a_ctrct_axes_end_((a_ctrct_axes_start + ctrct_axes_size) % (pa->Rank())),
    b_ctrct_axes_end_((b_ctrct_axes_start + ctrct_axes_size) % (pb->Rank())),
    ctrct_axes_size_(ctrct_axes_size),
    saved_axes_set_(2),
    a_trans_critical_axe_(a_ctrct_tail ? a_ctrct_axes_end_ : a_ctrct_axes_start),
    b_trans_critical_axe_(b_ctrct_head ? b_ctrct_axes_start : b_ctrct_axes_end_) {}

template<typename TenElemT, typename QNT, bool a_ctrct_tail, bool b_ctrct_head>
void MatrixBasedTensorContractionExecutor<TenElemT, QNT, a_ctrct_tail, b_ctrct_head>::GenerateDataBlk_() {
  using std::vector;
  const size_t a_rank(pa_->Rank()), b_rank(pb_->Rank());
  vector<vector<size_t>> ctrct_axes_set(2);
  ctrct_axes_set[0].reserve(ctrct_axes_size_);
  ctrct_axes_set[1].reserve(ctrct_axes_size_);
  saved_axes_set_[0].reserve(pa_->Rank() - ctrct_axes_size_);
  saved_axes_set_[1].reserve(pb_->Rank() - ctrct_axes_size_);
  for (size_t i = 0; i < ctrct_axes_size_; i++) {
    ctrct_axes_set[0].push_back((a_ctrct_axes_start_ + i) % a_rank);
    ctrct_axes_set[1].push_back((b_ctrct_axes_start_ + i) % b_rank);
  }
  const size_t save_axes_size_a = a_rank - ctrct_axes_size_;
  const size_t save_axes_size_b = b_rank - ctrct_axes_size_;
  for (size_t i = 0; i < save_axes_size_a; i++) {
    saved_axes_set_[0].push_back((a_ctrct_axes_end_ + i) % a_rank);
  }
  for (size_t i = 0; i < save_axes_size_b; i++) {
    saved_axes_set_[1].push_back((b_ctrct_axes_end_ + i) % b_rank);
  }
#ifndef NDEBUG
  auto &indexesa = pa_->GetIndexes();
  auto &indexesb = pb_->GetIndexes();
  for (size_t i = 0; i < ctrct_axes_size_; ++i) {
    assert(indexesa[ctrct_axes_set[0][i]] == InverseIndex(indexesb[ctrct_axes_set[1][i]]));
  }
#endif

  TenCtrctInitResTen(pa_, pb_, saved_axes_set_, pc_);  //note the order of saved_axes_set_
  raw_data_ctrct_tasks_ = pc_->GetBlkSparDataTen().DataBlkGenForTenCtrct(
      pa_->GetBlkSparDataTen(),
      pb_->GetBlkSparDataTen(),
      ctrct_axes_set,
      saved_axes_set_
  );
}

template<typename TenElemT, typename QNT, bool a_ctrct_tail, bool b_ctrct_head>
void MatrixBasedTensorContractionExecutor<TenElemT, QNT, a_ctrct_tail, b_ctrct_head>::TransposePrepare_() {
  using std::set;
  if (a_trans_critical_axe_ > 0) {
    const auto &a_bsdt = pa_->GetBlkSparDataTen();
    const size_t a_raw_data_size = a_bsdt.GetActualRawDataSize();
    a_trans_data_ = (TenElemT *) qlten::QLMalloc(a_raw_data_size * sizeof(TenElemT));
    set < size_t > selected_data_blk_idxs;
    for (auto &task : raw_data_ctrct_tasks_) {
      selected_data_blk_idxs.insert(task.a_blk_idx);
    }
    a_bsdt.OutOfPlaceMatrixTransposeForSelectedDataBlk(
        selected_data_blk_idxs,
        a_trans_critical_axe_,
        a_trans_data_
    );
    if constexpr (Fermionicable<QNT>::IsFermionic()) {
      a_trans_data_blk_idx_map_to_fermion_sign_ =
          a_bsdt.CountResidueFermionSignForMatBasedCtrct(saved_axes_set_[0],
                                                         a_trans_critical_axe_);
    }
  }
  if (b_trans_critical_axe_ > 0) {
    const auto &b_bsdt = pb_->GetBlkSparDataTen();
    const size_t b_raw_data_size = b_bsdt.GetActualRawDataSize();
    b_trans_data_ = (TenElemT *) qlten::QLMalloc(b_raw_data_size * sizeof(TenElemT));
    set < size_t > selected_data_blk_idxs;
    for (auto &task : raw_data_ctrct_tasks_) {
      selected_data_blk_idxs.insert(task.b_blk_idx);
    }
    b_bsdt.OutOfPlaceMatrixTransposeForSelectedDataBlk(
        selected_data_blk_idxs,
        b_trans_critical_axe_,
        b_trans_data_
    );
    if constexpr (Fermionicable<QNT>::IsFermionic()) {
      b_trans_data_blk_idx_map_to_fermion_sign_ =
          b_bsdt.CountResidueFermionSignForMatBasedCtrct(saved_axes_set_[1],
                                                         b_trans_critical_axe_);
    }
  }
}

template<typename TenElemT, typename QNT, bool a_ctrct_tail, bool b_ctrct_head>
void MatrixBasedTensorContractionExecutor<TenElemT, QNT, a_ctrct_tail, b_ctrct_head>::ExecutePost_() {
  qlten::QLFree(a_trans_data_);
  qlten::QLFree(b_trans_data_);
}

template<typename TenElemT, typename QNT, bool a_ctrct_tail, bool b_ctrct_head>
void MatrixBasedTensorContractionExecutor<TenElemT, QNT, a_ctrct_tail, b_ctrct_head>::Execute() {
#ifdef QLTEN_TIMING_MODE
  Timer gen_datablk_timer("mat_based_ctrct_gen_data_blk");
#endif
  GenerateDataBlk_();
#ifdef QLTEN_TIMING_MODE
  gen_datablk_timer.PrintElapsed();
  Timer trans_pre_timer("mat_based_ctrct_mat_transpose");
#endif

  TransposePrepare_();
#ifdef QLTEN_TIMING_MODE
  trans_pre_timer.PrintElapsed();
#endif

  const TenElemT *a_raw_data;
  const TenElemT *b_raw_data;
  if (a_trans_critical_axe_ > 0) {
    a_raw_data = a_trans_data_;
  } else {
    a_raw_data = pa_->GetBlkSparDataTen().GetActualRawDataPtr();
  }

  if (b_trans_critical_axe_ > 0) {
    b_raw_data = b_trans_data_;
  } else {
    b_raw_data = pb_->GetBlkSparDataTen().GetActualRawDataPtr();
  }
  auto &bsdt_c = pc_->GetBlkSparDataTen();
#ifdef QLTEN_TIMING_MODE
  Timer contract_according_task_timer("mat_based_ctrct_do_task");
#endif

  bsdt_c.template CtrctAccordingTask<a_ctrct_tail, b_ctrct_head>(
      a_raw_data,
      b_raw_data,
      raw_data_ctrct_tasks_,
      a_trans_data_blk_idx_map_to_fermion_sign_,
      b_trans_data_blk_idx_map_to_fermion_sign_
  );
#ifdef QLTEN_TIMING_MODE
  contract_according_task_timer.PrintElapsed();
#endif
  ExecutePost_();
}

/**
 *
 * @tparam TenElemT
 * @tparam QNT
 * @tparam a_ctrct_use_tail
 * @tparam b_ctrct_use_head
 * @param pa
 * @param pb
 * @param a_ctrct_axes_start
 * @param a_ctrct_axes_size
 * @param b_ctrct_axes_start
 * @param b_ctrct_axes_size
 * @param pc
 */
template<typename TenElemT, typename QNT, bool a_ctrct_tail = true, bool b_ctrct_head = true>
void Contract(
    const QLTensor<TenElemT, QNT> &pa, //use ref to make sure it is not a null pointer
    const QLTensor<TenElemT, QNT> &pb, //TODO: unify the style of code
    const size_t a_ctrct_axes_start,
    const size_t b_ctrct_axes_start,
    const size_t ctrct_axes_size,
    QLTensor<TenElemT, QNT> &pc
) {
  auto extra_contraction_executor = MatrixBasedTensorContractionExecutor<TenElemT, QNT, a_ctrct_tail, b_ctrct_head>(
      &pa, &pb, a_ctrct_axes_start, b_ctrct_axes_start, ctrct_axes_size, &pc
  );
  extra_contraction_executor.Execute();
}

template<typename TenElemT, typename QNT, bool a_ctrct_tail = true, bool b_ctrct_head = true>
void Contract(
    const QLTensor<QLTEN_Complex, QNT> &pa, //use ref to make sure it is not a null pointer
    const QLTensor<QLTEN_Double, QNT> &pb, //TODO: unify the style of code
    const size_t a_ctrct_axes_start,
    const size_t b_ctrct_axes_start,
    const size_t ctrct_axes_size,
    QLTensor<QLTEN_Complex, QNT> &pc
) {
  auto pb_complex = ToComplex(pb);
  Contract<QLTEN_Complex, QNT, a_ctrct_tail, b_ctrct_head>(pa,
                                                           pb_complex,
                                                           a_ctrct_axes_start,
                                                           b_ctrct_axes_start,
                                                           ctrct_axes_size,
                                                           pc);
}

template<typename TenElemT, typename QNT, bool a_ctrct_tail = true, bool b_ctrct_head = true>
void Contract(
    const QLTensor<QLTEN_Double, QNT> &pa, //use ref to make sure it is not a null pointer
    const QLTensor<QLTEN_Complex, QNT> &pb, //TODO: unify the style of code
    const size_t a_ctrct_axes_start,
    const size_t b_ctrct_axes_start,
    const size_t ctrct_axes_size,
    QLTensor<QLTEN_Complex, QNT> &pc
) {
  auto pa_complex = ToComplex(pa);
  Contract<QLTEN_Complex, QNT, a_ctrct_tail, b_ctrct_head>(pa_complex,
                                                           pb,
                                                           a_ctrct_axes_start,
                                                           b_ctrct_axes_start,
                                                           ctrct_axes_size,
                                                           pc);
}
}//qlten



#endif //QLTEN_TENSOR_MANIPULATION_TEN_CTRCT_BASED_MAT_TRANS_H
