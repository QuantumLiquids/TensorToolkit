// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2021-10-23.
*
* Description: tensor extra contraction function
*/

/**
@file ten_ctrct_fast.h
@brief tensor fast contraction function without tensor transpose are implemented.
       This another contraction is defined by the following:
       - the contracted indexes must be continuous. More concentrate, if it was defined by the usually language
            Contract(A, B, axes1, axes2), the axes1 must be ascendant numbers with interval 1, and so as axes2.
            We allow the period boundary of axes, for example: if A.rank() == 5, axes1 can be {3,4,0}
       - thus the transposes in contraction can be realized in matrix transpose
*/


#ifndef QLTEN_TENSOR_MANIPULATION_TEN_EXTRA_CTRCT_H
#define QLTEN_TENSOR_MANIPULATION_TEN_EXTRA_CTRCT_H

#include <set>

#include "qlten/framework/bases/executor.h"
#include "qlten/qltensor_all.h"

namespace qlten {

/**
 *
 * @tparam TenElemT
 * @tparam QNT
 * @tparam a_ctrct_tail
 * @tparam b_ctrct_head
 *
 *  develop note: if a_ctrct_tail && b_ctrct_head, then no transpose need when matrix product.
 *  else transpose the corresponding the tensor(s) when matrix product
 *  this two parameters don't change the result of contraction,
 *  but support some hints to reduce operators and increase the performance.
 */
template<typename TenElemT, typename QNT, bool a_ctrct_tail, bool b_ctrct_head>
class TensorExtraContractionExecutor : public Executor {
 public:
  TensorExtraContractionExecutor(
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

  size_t a_trans_critical_axe_;
  //if a_trans_critical_axe_ > 0, transpose happen between (a_trans_ctritical_axe_ - 1, a_trans_ctritical_axe_),
  //else a_trans_critical_axe_ == 0, no need to transpose. (this design can be reconsidered)
  size_t b_trans_critical_axe_;
  //if b_trans_critical_axe_ == 0, no need to transpose

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
TensorExtraContractionExecutor<TenElemT, QNT, a_ctrct_tail, b_ctrct_head>::TensorExtraContractionExecutor(
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
    a_trans_critical_axe_(a_ctrct_tail ? a_ctrct_axes_end_ : a_ctrct_axes_start),
    b_trans_critical_axe_(b_ctrct_head ? b_ctrct_axes_start : b_ctrct_axes_end_) {}

template<typename TenElemT, typename QNT, bool a_ctrct_tail, bool b_ctrct_head>
void TensorExtraContractionExecutor<TenElemT, QNT, a_ctrct_tail, b_ctrct_head>::GenerateDataBlk_() {
  using std::vector;
  const size_t a_rank(pa_->Rank()), b_rank(pb_->Rank());
  vector<vector<size_t>> saved_axes_set(2), ctrct_axes_set(2);
  ctrct_axes_set[0].reserve(ctrct_axes_size_);
  ctrct_axes_set[1].reserve(ctrct_axes_size_);
  saved_axes_set[0].reserve(pa_->Rank() - ctrct_axes_size_);
  saved_axes_set[1].reserve(pb_->Rank() - ctrct_axes_size_);
  for (size_t i = 0; i < ctrct_axes_size_; i++) {
    ctrct_axes_set[0].push_back((a_ctrct_axes_start_ + i) % a_rank);
    ctrct_axes_set[1].push_back((b_ctrct_axes_start_ + i) % b_rank);
  }
  const size_t save_axes_size_a = a_rank - ctrct_axes_size_;
  const size_t save_axes_size_b = b_rank - ctrct_axes_size_;
  for (size_t i = 0; i < save_axes_size_a; i++) {
    saved_axes_set[0].push_back((a_ctrct_axes_end_ + i) % a_rank);
  }
  for (size_t i = 0; i < save_axes_size_b; i++) {
    saved_axes_set[1].push_back((b_ctrct_axes_end_ + i) % b_rank);
  }
#ifndef NDEBUG
  auto &indexesa = pa_->GetIndexes();
  auto &indexesb = pb_->GetIndexes();
  for (size_t i = 0; i < ctrct_axes_size_; ++i) {
    assert(indexesa[ctrct_axes_set[0][i]] == InverseIndex(indexesb[ctrct_axes_set[1][i]]));
  }
#endif

  TenCtrctInitResTen(pa_, pb_, saved_axes_set, pc_);
  raw_data_ctrct_tasks_ = pc_->GetBlkSparDataTen().DataBlkGenForTenCtrct(
      pa_->GetBlkSparDataTen(),
      pb_->GetBlkSparDataTen(),
      ctrct_axes_set,
      saved_axes_set
  );
}

template<typename TenElemT, typename QNT, bool a_ctrct_tail, bool b_ctrct_head>
void TensorExtraContractionExecutor<TenElemT, QNT, a_ctrct_tail, b_ctrct_head>::TransposePrepare_() {
  using std::set;
  if (a_trans_critical_axe_ > 0) {
    const auto &a_bsdt = pa_->GetBlkSparDataTen();
    const size_t a_raw_data_size = a_bsdt.GetActualRawDataSize();
    a_trans_data_ = (TenElemT *) malloc(a_raw_data_size * sizeof(TenElemT));
    set < size_t > selected_data_blk_idxs;
    for (auto &task: raw_data_ctrct_tasks_) {
      selected_data_blk_idxs.insert(task.a_blk_idx);
    }
    a_bsdt.OutOfPlaceMatrixTransposeForSelectedDataBlk(
        selected_data_blk_idxs,
        a_trans_critical_axe_,
        a_trans_data_
    );
  }
  if (b_trans_critical_axe_ > 0) {
    const auto &b_bsdt = pb_->GetBlkSparDataTen();
    const size_t b_raw_data_size = b_bsdt.GetActualRawDataSize();
    b_trans_data_ = (TenElemT *) malloc(b_raw_data_size * sizeof(TenElemT));
    set < size_t > selected_data_blk_idxs;
    for (auto &task: raw_data_ctrct_tasks_) {
      selected_data_blk_idxs.insert(task.b_blk_idx);
    }
    b_bsdt.OutOfPlaceMatrixTransposeForSelectedDataBlk(
        selected_data_blk_idxs,
        b_trans_critical_axe_,
        b_trans_data_
    );
  }
}

template<typename TenElemT, typename QNT, bool a_ctrct_tail, bool b_ctrct_head>
void TensorExtraContractionExecutor<TenElemT, QNT, a_ctrct_tail, b_ctrct_head>::ExecutePost_() {
  free(a_trans_data_);
  free(b_trans_data_);
}

template<typename TenElemT, typename QNT, bool a_ctrct_tail, bool b_ctrct_head>
void TensorExtraContractionExecutor<TenElemT, QNT, a_ctrct_tail, b_ctrct_head>::Execute() {
#ifdef QLTEN_TIMING_MODE
  Timer extrac_contraction_generate_data_blk("ExtraContraction.Execute.GenerateDataBlk");
#endif
  GenerateDataBlk_();
#ifdef QLTEN_TIMING_MODE
  extrac_contraction_generate_data_blk.PrintElapsed();
  Timer extrac_contraction_tranpose("ExtraContraction.Execute.TransposePrepare");
#endif

  TransposePrepare_();
#ifdef QLTEN_TIMING_MODE
  extrac_contraction_tranpose.PrintElapsed();
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
  Timer extrac_contraction_do_ctrct_task("ExtraContraction.Execute.CtrctAccordingTask");
#endif

  bsdt_c.template CtrctAccordingTask<a_ctrct_tail, b_ctrct_head>(
      a_raw_data,
      b_raw_data,
      raw_data_ctrct_tasks_
  );
#ifdef QLTEN_TIMING_MODE
  extrac_contraction_do_ctrct_task.PrintElapsed();
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
  auto extra_contraction_executor = TensorExtraContractionExecutor<TenElemT, QNT, a_ctrct_tail, b_ctrct_head>(
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



#endif //QLTEN_TENSOR_MANIPULATION_TEN_EXTRA_CTRCT_H
