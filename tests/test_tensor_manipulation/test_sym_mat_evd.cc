// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2024-07-10
*
* Description: QuantumLiquids/tensor project. Unittests for Hermitian Matrix EVD.
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "../testing_utility.h"

using namespace qlten;

using U1QN = special_qn::U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DQLTensor = QLTensor<QLTEN_Double, U1QN>;
using ZQLTensor = QLTensor<QLTEN_Complex, U1QN>;

struct TestSvd : public testing::Test {
  U1QN qn0 = U1QN(0);
  U1QN qnp1 = U1QN(1);
  U1QN qnp2 = U1QN(2);
  U1QN qnm1 = U1QN(-1);
  U1QN qnm2 = U1QN(-2);

  size_t d_s = 3;
  QNSctT qnsct0_s = QNSctT(qn0, d_s);
  QNSctT qnsctp1_s = QNSctT(qnp1, d_s);
  QNSctT qnsctm1_s = QNSctT(qnm1, d_s);
  IndexT idx_in_s = IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, IN);
  IndexT idx_out_s = IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, OUT);

  DQLTensor dten_2d_s = DQLTensor({idx_in_s, idx_out_s});

  ZQLTensor zten_2d_s = ZQLTensor({idx_in_s, idx_out_s});
};

template<typename TenT>
void CheckIsIdTen(const TenT &t) {
  auto shape = t.GetShape();
  EXPECT_EQ(shape.size(), 2);
  EXPECT_EQ(shape[0], shape[1]);
  for (size_t i = 0; i < shape[0]; ++i) {
    for (size_t j = 0; j < shape[1]; ++j) {
      QLTEN_Complex elem = t.GetElem({i, j});
      if (i == j) {
        EXPECT_NEAR(elem.real(), 1.0, 1E-14);
      } else {
        EXPECT_NEAR(elem.real(), 0.0, 1E-14);
      }
      EXPECT_NEAR(elem.imag(), 0.0, 1E-14);
    }
  }
}

template<typename TenElemT, typename QNT>
void RunTestSymMatEVDCase(
    QLTensor<TenElemT, QNT> &t) {
  QLTensor<TenElemT, QNT> u;
  QLTensor<QLTEN_Double, QNT> d;
  SymMatEVD(&t, &u, &d);

  // Canonical check
  QLTensor<TenElemT, QNT> temp1, temp2, temp3, temp4;
  auto u_dag = Dag(u);
  Contract(&u, &u_dag, {{0}, {0}}, &temp1);
  CheckIsIdTen(temp1);
  Contract(&u, &u_dag, {{1}, {1}}, &temp2);
  CheckIsIdTen(temp2);

  Contract(&u, &d, {{1}, {0}}, &temp3);
  Contract(&temp3, &u_dag, {{1}, {1}}, &temp4);
  for (auto &coors : GenAllCoors(t.GetShape())) {
    GtestExpectNear(t.GetElem(coors), temp4.GetElem(coors), 1E-14);
  }
}

TEST_F(TestSvd, 2DCase) {
  dten_2d_s.Random(qn0);
  for (size_t i = 0; i < dten_2d_s.GetShape()[0]; i++) {
    for (size_t j = 0; j < dten_2d_s.GetShape()[1]; j++) {
      if (dten_2d_s.GetElem({i, j}) == 0 && dten_2d_s.GetElem({j, i}) == 0) {
        continue;
      }
      auto ave_elem = (dten_2d_s.GetElem({i, j}) + dten_2d_s.GetElem({j, i})) / 2.0;
      dten_2d_s({i, j}) = ave_elem;
      dten_2d_s({j, i}) = ave_elem;
    }
  }
  RunTestSymMatEVDCase(dten_2d_s);

  zten_2d_s.Random(qn0);
  for (size_t i = 0; i < zten_2d_s.GetShape()[0]; i++) {
    for (size_t j = 0; j < zten_2d_s.GetShape()[1]; j++) {
      if (zten_2d_s.GetElem({i, j}) == std::complex<double>(0.0)
          && zten_2d_s.GetElem({j, i}) == std::complex<double>(0.0)) {
        continue;
      }
      if (i == j) {
        zten_2d_s({i, i}) = zten_2d_s.GetElem({i, i}).real();
      } else {
        zten_2d_s({i, j}) = qlten::conj(zten_2d_s.GetElem({j, i}));
      }

    }
  }
  RunTestSymMatEVDCase(zten_2d_s);
}