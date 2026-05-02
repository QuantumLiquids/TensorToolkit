// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-27 09:21
*
* Description: QuantumLiquids/tensor project. Basic tensor operations.
*/

/**
@file basic_operations.h
@brief Basic tensor operations.
*/
#ifndef QLTEN_TENSOR_MANIPULATION_BASIC_OPERATIONS_H
#define QLTEN_TENSOR_MANIPULATION_BASIC_OPERATIONS_H

#include <iostream>     // cout, endl
#include <iterator>     // next
#include <cassert>     // assert

#include "qlten/qltensor/qltensor.h"    // QLTensor
#include "qlten/framework/hp_numeric/blas_level1.h"   // VectorConjDot

#ifdef Release
#define NDEBUG
#endif

namespace qlten {

/**
Calculate dagger of a QLTensor.

@param t A QLTensor \f$ T \f$.

@return Daggered \f$ T^{\dagger} \f$.
*/
template<typename QLTensorT>
QLTensorT Dag(const QLTensorT &t) {
  QLTensorT t_dag(t);
  t_dag.Dag();
  return t_dag;
}

/**
Calculate the quantum number divergence of a QLTensor by call QLTensor::Div().

@param t A QLTensor.

@return The quantum number divergence.
*/
template<typename ElemT, typename QNT>
inline QNT Div(const QLTensor<ElemT, QNT> &t) { return t.Div(); }

/**
Convert a real QLTensor to a complex QLTensor.
*/
template<typename QNT>
QLTensor<QLTEN_Complex, QNT> ToComplex(
    const QLTensor<QLTEN_Double, QNT> &real_t
) {
  assert(!real_t.IsDefault());
  QLTensor<QLTEN_Complex, QNT> cplx_t(real_t.GetIndexes());
  if (cplx_t.IsScalar()) {
    cplx_t.SetElem({}, real_t.GetElem({}));
  } else {
    cplx_t.GetBlkSparDataTen().CopyFromReal(real_t.GetBlkSparDataTen());
  }
  return cplx_t;
}

/**
Convert a real QLTensor to a complex QLTensor.
*/
template<typename QNT>
QLTensor<QLTEN_ComplexFloat, QNT> ToComplex(
    const QLTensor<QLTEN_Float, QNT> &real_t
) {
  assert(!real_t.IsDefault());
  QLTensor<QLTEN_ComplexFloat, QNT> cplx_t(real_t.GetIndexes());
  if (cplx_t.IsScalar()) {
    cplx_t.SetElem({}, real_t.GetElem({}));
  } else {
    cplx_t.GetBlkSparDataTen().CopyFromReal(real_t.GetBlkSparDataTen());
  }
  return cplx_t;
}

namespace inner_product_detail {

template<typename QNT>
int FermionInnerProductSign(
    const DataBlk<QNT> &data_blk,
    const IndexVec<QNT> &indexes
) {
  const std::vector<bool> &parities = data_blk.GetBlkFermionParities();
  int in_count = 0;
  for (size_t i = 0; i < indexes.size(); ++i) {
    if (indexes[i].GetDir() == TenIndexDirType::IN && parities[i]) {
      ++in_count;
    }
  }
  return (in_count % 2 == 0) ? 1 : -1;
}

template<bool use_fermion_sign, typename ElemT, typename QNT>
ElemT InnerProductImpl(
    const QLTensor<ElemT, QNT> &bra,
    const QLTensor<ElemT, QNT> &ket
) {
  assert(!(bra.IsDefault() || ket.IsDefault()));
  assert(bra.GetIndexes() == ket.GetIndexes());

  const auto &bra_bsdt = bra.GetBlkSparDataTen();
  const auto &ket_bsdt = ket.GetBlkSparDataTen();
  const ElemT *bra_raw_data = bra_bsdt.GetActualRawDataPtr();
  const ElemT *ket_raw_data = ket_bsdt.GetActualRawDataPtr();

  if (bra.IsScalar()) {
    return hp_numeric::VectorConjDot(bra_raw_data, ket_raw_data, 1);
  }

  ElemT res(0);
  const auto &bra_blk_map = bra_bsdt.GetBlkIdxDataBlkMap();
  const auto &ket_blk_map = ket_bsdt.GetBlkIdxDataBlkMap();
  for (const auto &[blk_idx, bra_data_blk] : bra_blk_map) {
    auto ket_iter = ket_blk_map.find(blk_idx);
    if (ket_iter == ket_blk_map.end()) {
      continue;
    }

    const auto &ket_data_blk = ket_iter->second;
    assert(bra_data_blk.size == ket_data_blk.size);
    ElemT blk_res = hp_numeric::VectorConjDot(
        bra_raw_data + bra_data_blk.data_offset,
        ket_raw_data + ket_data_blk.data_offset,
        bra_data_blk.size
    );

    if constexpr (use_fermion_sign && Fermionicable<QNT>::IsFermionic()) {
      blk_res *= FermionInnerProductSign(bra_data_blk, bra.GetIndexes());
    }
    res += blk_res;
  }
  return res;
}

} /* inner_product_detail */

/**
 * Calculate the graded inner product \f$\langle bra | ket \rangle\f$ of two
 * tensors with exactly the same indices.
 *
 * For bosonic tensors this is the ordinary flattened conjugate dot product and
 * is equivalent to fully contracting \f$bra^\dagger\f$ with \f$ket\f$. For
 * fermionic tensors this follows the same graded sign convention as
 * QLTensor::Get2Norm(); therefore InnerProduct(t, t) can be indefinite.
 *
 * @param bra Tensor on the bra side.
 * @param ket Tensor on the ket side.
 * @return The scalar inner product.
 */
template<typename ElemT, typename QNT>
ElemT InnerProduct(
    const QLTensor<ElemT, QNT> &bra,
    const QLTensor<ElemT, QNT> &ket
) {
  return inner_product_detail::InnerProductImpl<true>(bra, ket);
}

/**
 * Calculate the positive-definite flattened conjugate dot product of two
 * tensors with exactly the same indices.
 *
 * This ignores fermionic grading signs and is consistent with
 * QLTensor::GetQuasi2Norm(), i.e. QuasiInnerProduct(t, t) equals
 * GetQuasi2Norm() squared.
 *
 * @param bra Tensor on the bra side.
 * @param ket Tensor on the ket side.
 * @return The scalar quasi inner product.
 */
template<typename ElemT, typename QNT>
ElemT QuasiInnerProduct(
    const QLTensor<ElemT, QNT> &bra,
    const QLTensor<ElemT, QNT> &ket
) {
  return inner_product_detail::InnerProductImpl<false>(bra, ket);
}
} /* qlten */
#endif /* ifndef QLTEN_TENSOR_MANIPULATION_BASIC_OPERATIONS_H */
