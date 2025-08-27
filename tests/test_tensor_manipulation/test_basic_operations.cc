// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Haoxin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2020-11-27 09:33
*
* Description: QuantumLiquids/tensor project. Unittests for basic tensor operations.
*/

#include "gtest/gtest.h"
#include "qlten/qltensor_all.h"      // QLTensor, Index, QN, U1QNVal, QNSectorVec
#include "qlten/tensor_manipulation/basic_operations.h"


using namespace qlten;

using U1QN = QN<U1QNVal>;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DQLTensor = QLTensor<QLTEN_Double, U1QN>;
using ZQLTensor = QLTensor<QLTEN_Complex, U1QN>;


struct TestBasicTensorOperations : public testing::Test {
  std::string qn_nm = "qn";
  U1QN qn0 =  U1QN({QNCard(qn_nm, U1QNVal( 0))});
  U1QN qnp1 = U1QN({QNCard(qn_nm, U1QNVal( 1))});
  U1QN qnp2 = U1QN({QNCard(qn_nm, U1QNVal( 2))});
  U1QN qnm1 = U1QN({QNCard(qn_nm, U1QNVal(-1))});
  QNSctT qnsct0_s =  QNSctT(qn0,  4);
  QNSctT qnsctp1_s = QNSctT(qnp1, 5);
  QNSctT qnsctm1_s = QNSctT(qnm1, 3);
  IndexT idx_in_s =  IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, TenIndexDirType::IN);
  IndexT idx_out_s = IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, TenIndexDirType::OUT);

  DQLTensor dten_default = DQLTensor();
  DQLTensor dten_scalar = DQLTensor(IndexVec<U1QN>{});
  DQLTensor dten_1d_s = DQLTensor({idx_out_s});
  DQLTensor dten_2d_s = DQLTensor({idx_in_s, idx_out_s});
  DQLTensor dten_3d_s = DQLTensor({idx_in_s, idx_out_s, idx_out_s});
  ZQLTensor zten_default = ZQLTensor();
  ZQLTensor zten_scalar = ZQLTensor(IndexVec<U1QN>{});
  ZQLTensor zten_1d_s = ZQLTensor({idx_out_s});
  ZQLTensor zten_2d_s = ZQLTensor({idx_in_s, idx_out_s});
  ZQLTensor zten_3d_s = ZQLTensor({idx_in_s, idx_out_s, idx_out_s});
};


template <typename QLTensorT>
void RunTestTensorDagCase(const QLTensorT &t) {
  if (t.IsDefault()) {
    // Do nothing
  } else if (t.IsScalar()) {
    auto t_dag = Dag(t);
    EXPECT_EQ(t_dag.GetElem({}), CalcConj(t.GetElem({})));
  } else {
    auto t_dag = Dag(t);
    for (size_t i = 0; i < t.Rank(); ++i) {
      EXPECT_EQ(t_dag.GetIndexes()[i], InverseIndex(t.GetIndexes()[i]));
    }
    for (auto &coor : GenAllCoors(t.GetShape())) {
      EXPECT_EQ(t_dag.GetElem(coor), CalcConj(t.GetElem(coor)));
    }
  }
}


TEST_F(TestBasicTensorOperations, TestDag) {
  RunTestTensorDagCase(dten_default);
  dten_1d_s.Random(qn0);
  RunTestTensorDagCase(dten_1d_s);
  dten_2d_s.Random(qn0);
  RunTestTensorDagCase(dten_2d_s);
  dten_3d_s.Random(qn0);
  RunTestTensorDagCase(dten_3d_s);

  RunTestTensorDagCase(zten_default);
  zten_1d_s.Random(qn0);
  RunTestTensorDagCase(zten_1d_s);
  zten_2d_s.Random(qn0);
  RunTestTensorDagCase(zten_2d_s);
  zten_3d_s.Random(qn0);
  RunTestTensorDagCase(zten_3d_s);
}


template <typename ElemT, typename QNT>
void RunTestTensorDivCase(QLTensor<ElemT, QNT> &t, const QNT &div) {
  t.Random(div);
  EXPECT_EQ(Div(t), div);
}


TEST_F(TestBasicTensorOperations, TestDiv) {
  dten_scalar.Random(U1QN());
  EXPECT_EQ(Div(dten_scalar), U1QN());
  RunTestTensorDivCase(dten_1d_s, qn0);
  RunTestTensorDivCase(dten_2d_s, qn0);
  RunTestTensorDivCase(dten_2d_s, qnp1);
  RunTestTensorDivCase(dten_2d_s, qnm1);
  RunTestTensorDivCase(dten_2d_s, qnp2);
  RunTestTensorDivCase(dten_3d_s, qn0);
  RunTestTensorDivCase(dten_3d_s, qnp1);
  RunTestTensorDivCase(dten_3d_s, qnm1);
  RunTestTensorDivCase(dten_3d_s, qnp2);

  zten_scalar.Random(U1QN());
  EXPECT_EQ(Div(zten_scalar), U1QN());
  RunTestTensorDivCase(zten_1d_s, qn0);
  RunTestTensorDivCase(zten_2d_s, qn0);
  RunTestTensorDivCase(zten_2d_s, qnp1);
  RunTestTensorDivCase(zten_2d_s, qnm1);
  RunTestTensorDivCase(zten_2d_s, qnp2);
  RunTestTensorDivCase(zten_3d_s, qn0);
  RunTestTensorDivCase(zten_3d_s, qnp1);
  RunTestTensorDivCase(zten_3d_s, qnm1);
  RunTestTensorDivCase(zten_3d_s, qnp2);
}


template <typename QNT>
void RunTestRealTensorToComplexCase(
    QLTensor<QLTEN_Double, QNT> &t,
    const QNT & div,
    unsigned int rand_seed) {
  qlten::SetRandomSeed(rand_seed);
  t.Random(div);
  auto zten = ToComplex(t);
  for (auto &coors : GenAllCoors(t.GetShape())) {
    EXPECT_DOUBLE_EQ(zten.GetElem(coors).real(), t.GetElem(coors));
    EXPECT_DOUBLE_EQ(zten.GetElem(coors).imag(), 0.0);
  }
}


TEST_F(TestBasicTensorOperations, TestToComplex) {
  dten_scalar.Random(U1QN());
  auto zten = ToComplex(dten_scalar);
  EXPECT_DOUBLE_EQ(zten.GetElem({}).real(), dten_scalar.GetElem({}));
  EXPECT_DOUBLE_EQ(zten.GetElem({}).imag(), 0.0);

  RunTestRealTensorToComplexCase(dten_1d_s, qn0, 0);
  RunTestRealTensorToComplexCase(dten_1d_s, qn0, 1);
  RunTestRealTensorToComplexCase(dten_1d_s, qnp1, 0);
  RunTestRealTensorToComplexCase(dten_1d_s, qnp1, 1);
  RunTestRealTensorToComplexCase(dten_2d_s, qn0, 0);
  RunTestRealTensorToComplexCase(dten_2d_s, qn0, 1);
  RunTestRealTensorToComplexCase(dten_2d_s, qnp1, 0);
  RunTestRealTensorToComplexCase(dten_2d_s, qnp1, 1);
  RunTestRealTensorToComplexCase(dten_3d_s, qn0, 0);
  RunTestRealTensorToComplexCase(dten_3d_s, qn0, 1);
  RunTestRealTensorToComplexCase(dten_3d_s, qnp1, 0);
  RunTestRealTensorToComplexCase(dten_3d_s, qnp1, 1);
}
