// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: TensorToolkit Contributors
* Creation Date: 2026-01-04
*
* Description: QuantumLiquids/tensor project. Unittests for tensor LQ decomposition.
*/

#include "gtest/gtest.h"
#include "qlten/qltensor_all.h"
#include "qlten/tensor_manipulation/ten_decomp/ten_lq.h"    // LQ
#include "qlten/tensor_manipulation/ten_decomp/ten_qr.h"    // QR for comparison
#include "qlten/tensor_manipulation/ten_ctrct.h"            // Contract
#include "qlten/tensor_manipulation/basic_operations.h"     // Dag
#include "../testing_utility.h"                             // kEpsilon

using namespace qlten;
using U1QN = QN<U1QNVal>;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DQLTensor = QLTensor<QLTEN_Double, U1QN>;
using ZQLTensor = QLTensor<QLTEN_Complex, U1QN>;
using FQLTensor = QLTensor<QLTEN_Float, U1QN>;
using CQLTensor = QLTensor<QLTEN_ComplexFloat, U1QN>;


/**
 * Check if a tensor is approximately identity matrix.
 */
template<typename ElemT, typename QNT>
void CheckIsIdTen(const QLTensor<ElemT, QNT> &t) {
  double epsilon = kEpsilon;
  if constexpr (std::is_same_v<ElemT, float> || std::is_same_v<ElemT, std::complex<float>>
#ifdef USE_GPU
      || std::is_same_v<ElemT, cuda::std::complex<float>>
#endif
  ) {
    epsilon = 2.0e-5;
  }
  auto shape = t.GetShape();
  EXPECT_EQ(shape.size(), 2);
  EXPECT_EQ(shape[0], shape[1]);
  for (size_t i = 0; i < shape[0]; ++i) {
    QLTEN_Complex elem = t.GetElem({i, i});
    EXPECT_NEAR(elem.real(), 1.0, epsilon);
    EXPECT_NEAR(elem.imag(), 0.0, epsilon);
  }
}


/**
 * Check if two tensors are approximately equal.
 */
template<typename ElemT, typename QNT>
void CheckTwoTenClose(const QLTensor<ElemT, QNT> &t1, const QLTensor<ElemT, QNT> &t2) {
  double epsilon = kEpsilon;
  if constexpr (std::is_same_v<ElemT, float> || std::is_same_v<ElemT, std::complex<float>>
#ifdef USE_GPU
      || std::is_same_v<ElemT, cuda::std::complex<float>>
#endif
  ) {
    epsilon = 2.0e-5;
  }
  EXPECT_EQ(t1.GetIndexes(), t2.GetIndexes());
  for (auto &coors : GenAllCoors(t1.GetShape())) {
    QLTEN_Complex elem1 = t1.GetElem(coors);
    QLTEN_Complex elem2 = t2.GetElem(coors);
    EXPECT_NEAR(elem1.real(), elem2.real(), epsilon);
    EXPECT_NEAR(elem1.imag(), elem2.imag(), epsilon);
  }
}


struct TestLq : public testing::Test {
  std::string qn_nm = "qn";
  U1QN qn0 = U1QN({QNCard(qn_nm, U1QNVal(0))});
  U1QN qnp1 = U1QN({QNCard(qn_nm, U1QNVal(1))});
  U1QN qnp2 = U1QN({QNCard(qn_nm, U1QNVal(2))});
  U1QN qnm1 = U1QN({QNCard(qn_nm, U1QNVal(-1))});
  U1QN qnm2 = U1QN({QNCard(qn_nm, U1QNVal(-2))});

  size_t d_s = 3;
  QNSctT qnsct0_s = QNSctT(qn0, d_s);
  QNSctT qnsctp1_s = QNSctT(qnp1, d_s);
  QNSctT qnsctm1_s = QNSctT(qnm1, d_s);
  IndexT idx_in_s = IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, IN);
  IndexT idx_out_s = IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, OUT);

  DQLTensor dten_2d_s = DQLTensor({idx_in_s, idx_out_s});
  DQLTensor dten_3d_s = DQLTensor({idx_in_s, idx_out_s, idx_out_s});
  DQLTensor dten_4d_s = DQLTensor({idx_in_s, idx_out_s, idx_out_s, idx_out_s});

  ZQLTensor zten_2d_s = ZQLTensor({idx_in_s, idx_out_s});
  ZQLTensor zten_3d_s = ZQLTensor({idx_in_s, idx_out_s, idx_out_s});
  ZQLTensor zten_4d_s = ZQLTensor({idx_in_s, idx_out_s, idx_out_s, idx_out_s});

  FQLTensor ften_2d_s = FQLTensor({idx_in_s, idx_out_s});
  FQLTensor ften_3d_s = FQLTensor({idx_in_s, idx_out_s, idx_out_s});

  CQLTensor cten_2d_s = CQLTensor({idx_in_s, idx_out_s});
  CQLTensor cten_3d_s = CQLTensor({idx_in_s, idx_out_s, idx_out_s});
};


/**
 * Run a test case for LQ decomposition.
 *
 * Tests:
 * 1. L * Q ≈ T (reconstruction)
 * 2. Q * Q† = I (right orthonormality, contracting over right indices)
 * 3. Index directions: L's last is OUT, Q's first is IN
 * 4. Middle indices are InverseIndex of each other
 */
template<typename TenElemT, typename QNT>
void RunTestLqCase(
    QLTensor<TenElemT, QNT> &t,
    const size_t rdims,
    const QNT *random_div = nullptr
) {
  if (random_div != nullptr) {
    qlten::SetRandomSeed(0);
    t.Random(*random_div);
  }

  QLTensor<TenElemT, QNT> l, q;
  std::string qn_nm = "qn";
  U1QN qn0 = U1QN({QNCard(qn_nm, U1QNVal(0))});

  LQ(&t, rdims, qn0, &l, &q);

  // Check index directions
  size_t rank = t.Rank();
  size_t ldims = rank - rdims;

  // L's last index should be OUT
  EXPECT_EQ(l.GetIndexes().back().GetDir(), TenIndexDirType::OUT);
  // Q's first index should be IN
  EXPECT_EQ(q.GetIndexes().front().GetDir(), TenIndexDirType::IN);

  // L's last index and Q's first index should be InverseIndex
  auto l_mid_idx = l.GetIndexes().back();
  auto q_mid_idx = q.GetIndexes().front();
  EXPECT_EQ(InverseIndex(l_mid_idx), q_mid_idx);

  // Check right orthonormality: Q * Q† = I
  // Contract Q[m, r1, r2, ...] with Dag(Q)[m, r1, r2, ...] over right indices
  QLTensor<TenElemT, QNT> temp;
  auto q_dag = Dag(q);
  std::vector<size_t> right_ctrct_axes;
  for (size_t i = 1; i <= rdims; ++i) {
    right_ctrct_axes.push_back(i);
  }
  Contract(
      &q, &q_dag,
      {right_ctrct_axes, right_ctrct_axes},
      &temp
  );
  CheckIsIdTen(temp);

  // Check reconstruction: L * Q ≈ T
  QLTensor<TenElemT, QNT> t_restored;
  Contract(&l, &q, {{ldims}, {0}}, &t_restored);
  CheckTwoTenClose(t_restored, t);
}


// ============================================================================
// 2D Tensor Tests
// ============================================================================

TEST_F(TestLq, 2DCase) {
  // rdims = 1 means Q has 1 right index
  RunTestLqCase(dten_2d_s, 1, &qn0);
  RunTestLqCase(dten_2d_s, 1, &qnp1);
  RunTestLqCase(dten_2d_s, 1, &qnm1);
  RunTestLqCase(dten_2d_s, 1, &qnp2);
  RunTestLqCase(dten_2d_s, 1, &qnm2);

  RunTestLqCase(zten_2d_s, 1, &qn0);
  RunTestLqCase(zten_2d_s, 1, &qnp1);
  RunTestLqCase(zten_2d_s, 1, &qnm1);
  RunTestLqCase(zten_2d_s, 1, &qnp2);
  RunTestLqCase(zten_2d_s, 1, &qnm2);
}


// ============================================================================
// 3D Tensor Tests
// ============================================================================

TEST_F(TestLq, 3DCase) {
  // rdims = 1: L[i, j, m], Q[m, k]
  RunTestLqCase(dten_3d_s, 1, &qn0);
  RunTestLqCase(dten_3d_s, 1, &qnp1);
  RunTestLqCase(dten_3d_s, 1, &qnp2);
  RunTestLqCase(dten_3d_s, 1, &qnm1);
  RunTestLqCase(dten_3d_s, 1, &qnm2);

  // rdims = 2: L[i, m], Q[m, j, k]
  RunTestLqCase(dten_3d_s, 2, &qn0);
  RunTestLqCase(dten_3d_s, 2, &qnp1);
  RunTestLqCase(dten_3d_s, 2, &qnp2);
  RunTestLqCase(dten_3d_s, 2, &qnm1);
  RunTestLqCase(dten_3d_s, 2, &qnm2);

  RunTestLqCase(zten_3d_s, 1, &qn0);
  RunTestLqCase(zten_3d_s, 1, &qnp1);
  RunTestLqCase(zten_3d_s, 1, &qnp2);
  RunTestLqCase(zten_3d_s, 1, &qnm1);
  RunTestLqCase(zten_3d_s, 1, &qnm2);
  RunTestLqCase(zten_3d_s, 2, &qn0);
  RunTestLqCase(zten_3d_s, 2, &qnp1);
  RunTestLqCase(zten_3d_s, 2, &qnp2);
  RunTestLqCase(zten_3d_s, 2, &qnm1);
  RunTestLqCase(zten_3d_s, 2, &qnm2);
}


// ============================================================================
// 4D Tensor Tests
// ============================================================================

TEST_F(TestLq, 4DCase) {
  // rdims = 1: L[i, j, k, m], Q[m, l]
  RunTestLqCase(dten_4d_s, 1, &qn0);
  RunTestLqCase(dten_4d_s, 1, &qnp1);
  RunTestLqCase(dten_4d_s, 1, &qnp2);
  RunTestLqCase(dten_4d_s, 1, &qnm1);
  RunTestLqCase(dten_4d_s, 1, &qnm2);

  // rdims = 2: L[i, j, m], Q[m, k, l]
  RunTestLqCase(dten_4d_s, 2, &qn0);
  RunTestLqCase(dten_4d_s, 2, &qnp1);
  RunTestLqCase(dten_4d_s, 2, &qnp2);
  RunTestLqCase(dten_4d_s, 2, &qnm1);
  RunTestLqCase(dten_4d_s, 2, &qnm2);

  // rdims = 3: L[i, m], Q[m, j, k, l]
  RunTestLqCase(dten_4d_s, 3, &qn0);
  RunTestLqCase(dten_4d_s, 3, &qnp1);
  RunTestLqCase(dten_4d_s, 3, &qnp2);
  RunTestLqCase(dten_4d_s, 3, &qnm1);
  RunTestLqCase(dten_4d_s, 3, &qnm2);

  RunTestLqCase(zten_4d_s, 1, &qn0);
  RunTestLqCase(zten_4d_s, 1, &qnp1);
  RunTestLqCase(zten_4d_s, 1, &qnp2);
  RunTestLqCase(zten_4d_s, 1, &qnm1);
  RunTestLqCase(zten_4d_s, 1, &qnm2);
  RunTestLqCase(zten_4d_s, 2, &qn0);
  RunTestLqCase(zten_4d_s, 2, &qnp1);
  RunTestLqCase(zten_4d_s, 2, &qnp2);
  RunTestLqCase(zten_4d_s, 2, &qnm1);
  RunTestLqCase(zten_4d_s, 2, &qnm2);
  RunTestLqCase(zten_4d_s, 3, &qn0);
  RunTestLqCase(zten_4d_s, 3, &qnp1);
  RunTestLqCase(zten_4d_s, 3, &qnp2);
  RunTestLqCase(zten_4d_s, 3, &qnm1);
  RunTestLqCase(zten_4d_s, 3, &qnm2);
}


// ============================================================================
// Float Precision Tests
// ============================================================================

TEST_F(TestLq, 2DCaseFloat) {
  RunTestLqCase(ften_2d_s, 1, &qn0);
  RunTestLqCase(ften_2d_s, 1, &qnp1);
  RunTestLqCase(ften_2d_s, 1, &qnm1);

  RunTestLqCase(cten_2d_s, 1, &qn0);
  RunTestLqCase(cten_2d_s, 1, &qnp1);
  RunTestLqCase(cten_2d_s, 1, &qnm1);
}

TEST_F(TestLq, 3DCaseFloat) {
  RunTestLqCase(ften_3d_s, 1, &qn0);
  RunTestLqCase(ften_3d_s, 2, &qn0);

  RunTestLqCase(cten_3d_s, 1, &qn0);
  RunTestLqCase(cten_3d_s, 2, &qn0);
}


// ============================================================================
// Duality Test: LQ(T) should be related to Dag(QR(Dag(T)))
// ============================================================================

template<typename TenElemT, typename QNT>
void RunDualityTest(
    QLTensor<TenElemT, QNT> &t,
    const size_t rdims,
    const QNT &div,
    const QNT &qn0
) {
  qlten::SetRandomSeed(0);
  t.Random(div);

  // Perform LQ
  QLTensor<TenElemT, QNT> l_lq, q_lq;
  LQ(&t, rdims, qn0, &l_lq, &q_lq);

  // The relationship: if LQ(T) = L * Q
  // then Dag(T) with appropriate QR gives related decomposition
  // 
  // Specifically, for T = L * Q (LQ):
  //   - L[left, m(OUT)], Q[m(IN), right]
  //   - Q is right-orthonormal
  //
  // For T† = Q' * R' (QR on transposed/dagged input):
  //   - Q' is left-orthonormal
  //
  // The duality: L = Dag(R'), Q = Dag(Q') (with appropriate transposes)

  // Verify by reconstruction: L * Q should equal T
  size_t ldims = t.Rank() - rdims;
  QLTensor<TenElemT, QNT> t_restored;
  Contract(&l_lq, &q_lq, {{ldims}, {0}}, &t_restored);
  CheckTwoTenClose(t_restored, t);

  // Verify divergence conservation
  QNT div_l = l_lq.Div();
  QNT div_q = q_lq.Div();
  QNT div_t = t.Div();
  EXPECT_EQ(div_l + div_q, div_t);
}

TEST_F(TestLq, DualityTest) {
  RunDualityTest(dten_3d_s, 1, qn0, qn0);
  RunDualityTest(dten_3d_s, 2, qn0, qn0);
  RunDualityTest(dten_3d_s, 1, qnp1, qn0);
  RunDualityTest(dten_3d_s, 2, qnm1, qn0);

  RunDualityTest(zten_3d_s, 1, qn0, qn0);
  RunDualityTest(zten_3d_s, 2, qn0, qn0);
}


// ============================================================================
// MPS Right Canonicalization Use Case Test
// ============================================================================

/**
 * Test the typical MPS right canonicalization use case.
 *
 * For an MPS tensor A[i, s, j]:
 * - Left canonicalization: QR with ldims=2 gives Q[i, s, m] * R[m, j]
 * - Right canonicalization: LQ with rdims=2 gives L[i, m] * Q[m, s, j]
 *
 * This test verifies that LQ works correctly for this use case.
 */
TEST_F(TestLq, MPSRightCanonTest) {
  // Create a 3-index MPS-like tensor
  DQLTensor mps_tensor({idx_in_s, idx_out_s, idx_out_s});
  qlten::SetRandomSeed(42);
  mps_tensor.Random(qn0);

  // Right canonicalization: LQ with rdims=2
  // A[i, s, j] = L[i, m] * Q[m, s, j]
  DQLTensor l, q;
  LQ(&mps_tensor, 2, qn0, &l, &q);

  // Verify shapes
  auto l_shape = l.GetShape();
  auto q_shape = q.GetShape();
  auto t_shape = mps_tensor.GetShape();

  // L should have shape [dim_i, m]
  EXPECT_EQ(l_shape.size(), 2);
  EXPECT_EQ(l_shape[0], t_shape[0]);

  // Q should have shape [m, dim_s, dim_j]
  EXPECT_EQ(q_shape.size(), 3);
  EXPECT_EQ(q_shape[1], t_shape[1]);
  EXPECT_EQ(q_shape[2], t_shape[2]);

  // Middle dimensions should match
  EXPECT_EQ(l_shape[1], q_shape[0]);

  // Verify reconstruction
  DQLTensor restored;
  Contract(&l, &q, {{1}, {0}}, &restored);
  CheckTwoTenClose(restored, mps_tensor);

  // Verify Q is right-orthonormal
  DQLTensor id_check;
  auto q_dag = Dag(q);
  Contract(&q, &q_dag, {{1, 2}, {1, 2}}, &id_check);
  CheckIsIdTen(id_check);
}

