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

This is complementary to QR decomposition which gives left-orthonormal Q:
- QR: T = Q * R, where Q is left-orthonormal (columns orthonormal): Q† * Q = I  
- LQ: T = L * Q, where Q is right-orthonormal (rows orthonormal): Q * Q† = I

Implementation uses the mathematical relationship:
  If T† = Q' * R' (QR decomposition), then T = R'† * Q'† = L * Q
*/
#ifndef QLTEN_TENSOR_MANIPULATION_TEN_DECOMP_TEN_LQ_H
#define QLTEN_TENSOR_MANIPULATION_TEN_DECOMP_TEN_LQ_H

#include "qlten/qltensor_all.h"
#include "qlten/tensor_manipulation/basic_operations.h"    // Dag
#include "qlten/tensor_manipulation/ten_decomp/ten_qr.h"   // QR

#include <vector>
#include <numeric>    // iota

#ifdef Release
  #define NDEBUG
#endif
#include <cassert>


namespace qlten {


/**
Generate transpose axes to move the last rdims indices to the front.

@param rank Total number of indices.
@param rdims Number of right indices to move to front.
@return Transpose axes vector.
*/
inline std::vector<size_t> GenLQPreTransposeAxes(
    const size_t rank,
    const size_t rdims
) {
  std::vector<size_t> axes(rank);
  size_t ldims = rank - rdims;
  // Put right indices first: [ldims, ldims+1, ..., rank-1, 0, 1, ..., ldims-1]
  for (size_t i = 0; i < rdims; ++i) {
    axes[i] = ldims + i;
  }
  for (size_t i = 0; i < ldims; ++i) {
    axes[rdims + i] = i;
  }
  return axes;
}


/**
Generate transpose axes to restore Q from QR result.
Q' has shape [right_indices..., m], need to get [m, right_indices...]

@param rdims Number of right indices (excluding m).
@return Transpose axes vector.
*/
inline std::vector<size_t> GenLQQTransposeAxes(const size_t rdims) {
  std::vector<size_t> axes(rdims + 1);
  // Move last index (m) to first position
  axes[0] = rdims;
  for (size_t i = 0; i < rdims; ++i) {
    axes[i + 1] = i;
  }
  return axes;
}


/**
Generate transpose axes to restore L from QR result.
L' has shape [m, left_indices...], need to get [left_indices..., m]

@param ldims Number of left indices (excluding m).
@return Transpose axes vector.
*/
inline std::vector<size_t> GenLQLTransposeAxes(const size_t ldims) {
  std::vector<size_t> axes(ldims + 1);
  // Move first index (m) to last position
  for (size_t i = 0; i < ldims; ++i) {
    axes[i] = i + 1;
  }
  axes[ldims] = 0;
  return axes;
}


/**
LQ decomposition for a QLTensor.

Decomposes tensor T into L * Q where Q is right-orthonormal:
  T[i0, ..., i_{n-k-1}, i_{n-k}, ..., i_{n-1}]
    = L[i0, ..., i_{n-k-1}, m(OUT)] * Q[m(IN), i_{n-k}, ..., i_{n-1}]

@tparam TenElemT The element type of the tensors.
@tparam QNT The quantum number type of the tensors.

@param pt A pointer to the to-be LQ decomposed tensor \f$ T \f$. The rank of 
       \f$ T \f$ should be larger than 1.
@param rdims Number of indices on the right hand side that Q retains. 
       Must satisfy 0 < rdims < pt->Rank().
@param rqndiv Quantum number divergence of the result \f$ Q \f$ tensor.
@param pl A pointer to result \f$ L \f$ tensor (lower triangular factor).
@param pq A pointer to result \f$ Q \f$ tensor (right-orthonormal factor).

@note The decomposition satisfies:
  - Contract(pl, pq, {{ldims}, {0}}) ≈ T (up to numerical precision)
  - Contract(pq, Dag(pq), {right_axes, right_axes}) ≈ Identity
  - L's last index is OUT, Q's first index is IN (they are InverseIndex)
  - Div(L) + Div(Q) = Div(T), and Div(Q) = rqndiv

@par Implementation
Uses the mathematical relationship between LQ and QR:
  1. Transpose T to move right indices to front
  2. Compute Dag (invert directions, conjugate elements)  
  3. Apply QR decomposition
  4. Dag both results to restore original structure
  5. Transpose to final index order

@par Example
@code
using qlten::special_qn::U1QN;
QLTensor<QLTEN_Double, U1QN> T(...);  // 3-index tensor [i, j, k]
QLTensor<QLTEN_Double, U1QN> L, Q;

// Decompose T = L * Q where Q has 1 right index
LQ(&T, 1, U1QN(0), &L, &Q);
// Result: L[i, j, m], Q[m, k]

// Verify: L * Q ≈ T
QLTensor<QLTEN_Double, U1QN> T_restored;
Contract(&L, &Q, {{2}, {0}}, &T_restored);
@endcode
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

  const size_t rank = pt->Rank();
  const size_t ldims = rank - rdims;

  // Step 1: Transpose T to move right rdims indices to front
  // T[i0, ..., i_{ldims-1}, i_{ldims}, ..., i_{n-1}]
  //   -> T'[i_{ldims}, ..., i_{n-1}, i0, ..., i_{ldims-1}]
  auto pre_transpose_axes = GenLQPreTransposeAxes(rank, rdims);
  QLTensor<TenElemT, QNT> t_transposed(*pt);
  t_transposed.Transpose(pre_transpose_axes);

  // Step 2: Compute Dag of transposed tensor
  // This inverts all index directions and conjugates elements
  auto t_dag = Dag(t_transposed);

  // Step 3: QR decomposition
  // T†[right†, left†] = Q'[right†, m(OUT)] * R'[m(IN), left†]
  // 
  // The qndiv for QR: since Q_final = Dag(Q'), we have Div(Q) = -Div(Q')
  // We want Div(Q) = rqndiv, so Div(Q') = -rqndiv
  QLTensor<TenElemT, QNT> q_prime, r_prime;
  QNT lqndiv_for_qr = -rqndiv;
  QR(&t_dag, rdims, lqndiv_for_qr, &q_prime, &r_prime);

  // Step 4: Dag both results to restore original index directions
  // Q'' = Dag(Q') has [right_indices, m(IN)]
  // L'' = Dag(R') has [m(OUT), left_indices]
  auto q_dag = Dag(q_prime);
  auto l_dag = Dag(r_prime);

  // Step 5: Transpose Q to [m(IN), right_indices]
  auto q_transpose_axes = GenLQQTransposeAxes(rdims);
  q_dag.Transpose(q_transpose_axes);
  *pq = std::move(q_dag);

  // Step 6: Transpose L to [left_indices, m(OUT)]
  auto l_transpose_axes = GenLQLTransposeAxes(ldims);
  l_dag.Transpose(l_transpose_axes);
  *pl = std::move(l_dag);
}


} /* qlten */
#endif /* ifndef QLTEN_TENSOR_MANIPULATION_TEN_DECOMP_TEN_LQ_H */

