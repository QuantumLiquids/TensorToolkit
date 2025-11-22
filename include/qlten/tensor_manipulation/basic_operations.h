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
} /* qlten */
#endif /* ifndef QLTEN_TENSOR_MANIPULATION_BASIC_OPERATIONS_H */
