// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2021-07-09 11:11
*
* Description: QuantumLiquids/tensor project. Unittests for tensor QR.
*/

#include "gtest/gtest.h"
#include "qlten/qltensor_all.h"
#include "qlten/tensor_manipulation/ten_decomp/ten_qr.h"    // QR
#include "qlten/tensor_manipulation/ten_ctrct.h"            // Contract
#include "../testing_utility.h"                            // kEpsilon

using namespace qlten;
using U1QN = QN<U1QNVal>;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DQLTensor = QLTensor<QLTEN_Double, U1QN>;
using ZQLTensor = QLTensor<QLTEN_Complex, U1QN>;

template<typename TenT>
void CheckIsIdTen(const TenT &t) {
  auto shape = t.GetShape();
  EXPECT_EQ(shape.size(), 2);
  EXPECT_EQ(shape[0], shape[1]);
  for (size_t i = 0; i < shape[0]; ++i) {
    QLTEN_Complex elem = t.GetElem({i, i});
    EXPECT_NEAR(elem.real(), 1.0, kEpsilon);
    EXPECT_NEAR(elem.imag(), 0.0, kEpsilon);
  }
}

template<typename TenT>
void CheckTwoTenClose(const TenT &t1, const TenT &t2) {
  EXPECT_EQ(t1.GetIndexes(), t2.GetIndexes());
  for (auto &coors : GenAllCoors(t1.GetShape())) {
    QLTEN_Complex elem1 = t1.GetElem(coors);
    QLTEN_Complex elem2 = t2.GetElem(coors);
    EXPECT_NEAR(elem1.real(), elem2.real(), kEpsilon);
    EXPECT_NEAR(elem1.imag(), elem2.imag(), kEpsilon);
  }
}

struct TestQr : public testing::Test {
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

  DQLTensor dten_1d_s = DQLTensor({idx_out_s});
  DQLTensor dten_2d_s = DQLTensor({idx_in_s, idx_out_s});
  DQLTensor dten_3d_s = DQLTensor({idx_in_s, idx_out_s, idx_out_s});
  DQLTensor dten_4d_s = DQLTensor({idx_in_s, idx_out_s, idx_out_s, idx_out_s});

  ZQLTensor zten_1d_s = ZQLTensor({idx_out_s});
  ZQLTensor zten_2d_s = ZQLTensor({idx_in_s, idx_out_s});
  ZQLTensor zten_3d_s = ZQLTensor({idx_in_s, idx_out_s, idx_out_s});
  ZQLTensor zten_4d_s = ZQLTensor({idx_in_s, idx_out_s, idx_out_s, idx_out_s});
};

template<typename TenElemT, typename QNT>
void RunTestQrCase(
    QLTensor<TenElemT, QNT> &t,
    const size_t ldims,
    const QNT *random_div = nullptr
) {
  if (random_div != nullptr) {
    qlten::SetRandomSeed(0);
    t.Random(*random_div);
  }
  QLTensor<TenElemT, QNT> q, r;
  std::string qn_nm = "qn";
  U1QN qn0 = U1QN({QNCard(qn_nm, U1QNVal(0))});

  QR(&t, ldims, qn0, &q, &r);

  QLTensor<TenElemT, QNT> temp;
  auto q_dag = Dag(q);
  std::vector<size_t> cano_check_u_ctrct_axes;
  for (size_t i = 0; i < ldims; ++i) { cano_check_u_ctrct_axes.push_back(i); }
  Contract(
      &q, &q_dag,
      {cano_check_u_ctrct_axes, cano_check_u_ctrct_axes},
      &temp
  );
  CheckIsIdTen(temp);

  QLTensor<TenElemT, QNT> t_restored;
  qlten::Contract(&q, &r, {{ldims}, {0}}, &t_restored);
  CheckTwoTenClose(t_restored, t);
}

TEST_F(TestQr, 2DCase) {
  RunTestQrCase(dten_2d_s, 1, &qn0);
  RunTestQrCase(dten_2d_s, 1, &qnp1);
  RunTestQrCase(dten_2d_s, 1, &qnm1);
  RunTestQrCase(dten_2d_s, 1, &qnp2);
  RunTestQrCase(dten_2d_s, 1, &qnm2);

  RunTestQrCase(zten_2d_s, 1, &qn0);
  RunTestQrCase(zten_2d_s, 1, &qnp1);
  RunTestQrCase(zten_2d_s, 1, &qnm1);
  RunTestQrCase(zten_2d_s, 1, &qnp2);
  RunTestQrCase(zten_2d_s, 1, &qnm2);
}

TEST_F(TestQr, 3DCase) {
  RunTestQrCase(dten_3d_s, 1, &qn0);
  RunTestQrCase(dten_3d_s, 1, &qnp1);
  RunTestQrCase(dten_3d_s, 1, &qnp2);
  RunTestQrCase(dten_3d_s, 1, &qnm1);
  RunTestQrCase(dten_3d_s, 1, &qnm2);
  RunTestQrCase(dten_3d_s, 2, &qn0);
  RunTestQrCase(dten_3d_s, 2, &qnp1);
  RunTestQrCase(dten_3d_s, 2, &qnp2);
  RunTestQrCase(dten_3d_s, 2, &qnm1);
  RunTestQrCase(dten_3d_s, 2, &qnm2);

  RunTestQrCase(zten_3d_s, 1, &qn0);
  RunTestQrCase(zten_3d_s, 1, &qnp1);
  RunTestQrCase(zten_3d_s, 1, &qnp2);
  RunTestQrCase(zten_3d_s, 1, &qnm1);
  RunTestQrCase(zten_3d_s, 1, &qnm2);
  RunTestQrCase(zten_3d_s, 2, &qn0);
  RunTestQrCase(zten_3d_s, 2, &qnp1);
  RunTestQrCase(zten_3d_s, 2, &qnp2);
  RunTestQrCase(zten_3d_s, 2, &qnm1);
  RunTestQrCase(zten_3d_s, 2, &qnm2);
}

TEST_F(TestQr, 4DCase) {
  RunTestQrCase(dten_4d_s, 1, &qn0);
  RunTestQrCase(dten_4d_s, 1, &qnp1);
  RunTestQrCase(dten_4d_s, 1, &qnp2);
  RunTestQrCase(dten_4d_s, 1, &qnm1);
  RunTestQrCase(dten_4d_s, 1, &qnm2);
  RunTestQrCase(dten_4d_s, 2, &qn0);
  RunTestQrCase(dten_4d_s, 2, &qnp1);
  RunTestQrCase(dten_4d_s, 2, &qnp2);
  RunTestQrCase(dten_4d_s, 2, &qnm1);
  RunTestQrCase(dten_4d_s, 2, &qnm2);
  RunTestQrCase(dten_4d_s, 3, &qn0);
  RunTestQrCase(dten_4d_s, 3, &qnp1);
  RunTestQrCase(dten_4d_s, 3, &qnp2);
  RunTestQrCase(dten_4d_s, 3, &qnm1);
  RunTestQrCase(dten_4d_s, 3, &qnm2);

  RunTestQrCase(zten_4d_s, 1, &qn0);
  RunTestQrCase(zten_4d_s, 1, &qnp1);
  RunTestQrCase(zten_4d_s, 1, &qnp2);
  RunTestQrCase(zten_4d_s, 1, &qnm1);
  RunTestQrCase(zten_4d_s, 1, &qnm2);
  RunTestQrCase(zten_4d_s, 2, &qn0);
  RunTestQrCase(zten_4d_s, 2, &qnp1);
  RunTestQrCase(zten_4d_s, 2, &qnp2);
  RunTestQrCase(zten_4d_s, 2, &qnm1);
  RunTestQrCase(zten_4d_s, 2, &qnm2);
  RunTestQrCase(zten_4d_s, 3, &qn0);
  RunTestQrCase(zten_4d_s, 3, &qnp1);
  RunTestQrCase(zten_4d_s, 3, &qnp2);
  RunTestQrCase(zten_4d_s, 3, &qnm1);
  RunTestQrCase(zten_4d_s, 3, &qnm2);
}
