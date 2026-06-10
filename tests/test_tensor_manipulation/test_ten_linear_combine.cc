// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-08-11 09:38
*
* Description: QuantumLiquids/tensor project. Unittests for tensor linear combination functions.
*/

#include "gtest/gtest.h"
#include "qlten/qltensor_all.h"
#include "qlten/tensor_manipulation/ten_linear_combine.h"
#include "qlten/utility/utils_inl.h"

#include "../testing_utility.h"

using namespace qlten;

using U1QN = QN<U1QNVal>;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DQLTensor = QLTensor<QLTEN_Double, U1QN>;
using ZQLTensor = QLTensor<QLTEN_Complex, U1QN>;

using namespace qlten;

struct TestLinearCombination : public testing::Test {
  std::string qn_nm = "qn";
  U1QN qn0 = U1QN({QNCard(qn_nm, U1QNVal(0))});
  U1QN qnp1 = U1QN({QNCard(qn_nm, U1QNVal(1))});
  U1QN qnp2 = U1QN({QNCard(qn_nm, U1QNVal(2))});
  U1QN qnm1 = U1QN({QNCard(qn_nm, U1QNVal(-1))});
  int d_s = 3;
  QNSctT qnsct0_s = QNSctT(qn0, d_s);
  QNSctT qnsctp1_s = QNSctT(qnp1, d_s);
  QNSctT qnsctm1_s = QNSctT(qnm1, d_s);
  IndexT idx_in_s = IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, IN);
  IndexT idx_out_s = IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, OUT);

  DQLTensor dten_default = DQLTensor();
  DQLTensor dten_1d_s = DQLTensor({idx_out_s});
  DQLTensor dten_2d_s = DQLTensor({idx_in_s, idx_out_s});
  DQLTensor dten_3d_s = DQLTensor({idx_in_s, idx_out_s, idx_out_s});

  ZQLTensor zten_default = ZQLTensor();
  ZQLTensor zten_1d_s = ZQLTensor({idx_out_s});
  ZQLTensor zten_2d_s = ZQLTensor({idx_in_s, idx_out_s});
  ZQLTensor zten_3d_s = ZQLTensor({idx_in_s, idx_out_s, idx_out_s});
};

template<typename TenElemT, typename QNT>
void RunTestLinearCombinationCase(
    const std::vector<TenElemT> &coefs,
    const std::vector<QLTensor<TenElemT, QNT> *> &pts,
    QLTensor<TenElemT, QNT> *res,
    const TenElemT beta = 0.0
) {
  QLTensor<TenElemT, QNT> bnchmrk;
  if (!(beta == TenElemT(0.0))) {
    bnchmrk = beta * (*res);
  }
  auto nt = pts.size();
  for (size_t i = 0; i < nt; ++i) {
    auto temp = coefs[i] * (*pts[i]);
    if (i == 0 && beta == TenElemT(0.0)) {
      bnchmrk = temp;
    } else {
      bnchmrk += temp;
    }
  }

  LinearCombine(coefs, pts, beta, res);

  EXPECT_EQ(*res, bnchmrk);
}

template<typename ElemT, typename QNT>
void ExpectTensorNear(
    const QLTensor<ElemT, QNT> &actual,
    const QLTensor<ElemT, QNT> &expected
) {
  ASSERT_EQ(actual.GetIndexes(), expected.GetIndexes());
  for (auto &coors : GenAllCoors(actual.GetShape())) {
    GtestExpectNear(actual.GetElem(coors), expected.GetElem(coors), kEpsilon);
  }
}

TEST_F(TestLinearCombination, 1TenCases) {
  DQLTensor dten_res;
  dten_res = dten_1d_s;
  dten_1d_s.Random(qn0);
  RunTestLinearCombinationCase({drand()}, {&dten_1d_s}, &dten_res);
  RunTestLinearCombinationCase({drand()}, {&dten_1d_s}, &dten_res, drand());
  dten_1d_s.Random(qnp1);
  RunTestLinearCombinationCase({drand()}, {&dten_1d_s}, &dten_res);
  RunTestLinearCombinationCase({drand()}, {&dten_1d_s}, &dten_res, drand());
  dten_res = dten_2d_s;
  dten_2d_s.Random(qn0);
  RunTestLinearCombinationCase({drand()}, {&dten_2d_s}, &dten_res);
  RunTestLinearCombinationCase({drand()}, {&dten_2d_s}, &dten_res, drand());
  dten_2d_s.Random(qnm1);
  RunTestLinearCombinationCase({drand()}, {&dten_2d_s}, &dten_res);
  RunTestLinearCombinationCase({drand()}, {&dten_2d_s}, &dten_res, drand());
  dten_res = dten_3d_s;
  dten_3d_s.Random(qn0);
  RunTestLinearCombinationCase({drand()}, {&dten_3d_s}, &dten_res);
  RunTestLinearCombinationCase({drand()}, {&dten_3d_s}, &dten_res, drand());
  dten_3d_s.Random(qnp2);
  RunTestLinearCombinationCase({drand()}, {&dten_3d_s}, &dten_res);
  RunTestLinearCombinationCase({drand()}, {&dten_3d_s}, &dten_res, drand());

  ZQLTensor zten_res;
  zten_res = zten_1d_s;
  zten_1d_s.Random(qn0);
  RunTestLinearCombinationCase({zrand()}, {&zten_1d_s}, &zten_res);
  RunTestLinearCombinationCase({zrand()}, {&zten_1d_s}, &zten_res, zrand());
  zten_1d_s.Random(qnp1);
  RunTestLinearCombinationCase({zrand()}, {&zten_1d_s}, &zten_res);
  RunTestLinearCombinationCase({zrand()}, {&zten_1d_s}, &zten_res, zrand());
  zten_res = zten_2d_s;
  zten_2d_s.Random(qn0);
  RunTestLinearCombinationCase({zrand()}, {&zten_2d_s}, &zten_res);
  RunTestLinearCombinationCase({zrand()}, {&zten_2d_s}, &zten_res, zrand());
  zten_2d_s.Random(qnm1);
  RunTestLinearCombinationCase({zrand()}, {&zten_2d_s}, &zten_res);
  RunTestLinearCombinationCase({zrand()}, {&zten_2d_s}, &zten_res, zrand());
  zten_res = zten_3d_s;
  zten_3d_s.Random(qn0);
  RunTestLinearCombinationCase({zrand()}, {&zten_3d_s}, &zten_res);
  RunTestLinearCombinationCase({zrand()}, {&zten_3d_s}, &zten_res, zrand());
  zten_3d_s.Random(qnp2);
  RunTestLinearCombinationCase({zrand()}, {&zten_3d_s}, &zten_res);
  RunTestLinearCombinationCase({zrand()}, {&zten_3d_s}, &zten_res, zrand());
}

TEST_F(TestLinearCombination, 2TenCases) {
  DQLTensor dten_res;
  auto dten_1d_s2 = dten_1d_s;
  auto dten_2d_s2 = dten_2d_s;
  auto dten_3d_s2 = dten_3d_s;
  dten_res = dten_1d_s;
  dten_1d_s.Random(qn0);
  dten_1d_s2.Random(qn0);
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_1d_s, &dten_1d_s2},
      &dten_res
  );
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_1d_s, &dten_1d_s2},
      &dten_res,
      drand()
  );
  dten_1d_s.Random(qn0);
  dten_1d_s2.Random(qnp1);
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_1d_s, &dten_1d_s2},
      &dten_res
  );
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_1d_s, &dten_1d_s2},
      &dten_res,
      drand()
  );
  dten_1d_s.Random(qnm1);
  dten_1d_s2.Random(qnp2);
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_1d_s, &dten_1d_s2},
      &dten_res
  );
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_1d_s, &dten_1d_s2},
      &dten_res,
      drand()
  );

  dten_res = dten_2d_s;
  dten_2d_s.Random(qn0);
  dten_2d_s2.Random(qn0);
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_2d_s, &dten_2d_s2},
      &dten_res
  );
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_2d_s, &dten_2d_s2},
      &dten_res,
      drand()
  );
  dten_2d_s.Random(qn0);
  dten_2d_s2.Random(qnp1);
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_2d_s, &dten_2d_s2},
      &dten_res
  );
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_2d_s, &dten_2d_s2},
      &dten_res,
      drand()
  );
  dten_2d_s.Random(qnm1);
  dten_2d_s2.Random(qnp2);
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_2d_s, &dten_2d_s2},
      &dten_res
  );
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_2d_s, &dten_2d_s2},
      &dten_res,
      drand()
  );

  dten_res = dten_3d_s;
  dten_3d_s.Random(qn0);
  dten_3d_s2.Random(qn0);
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_3d_s, &dten_3d_s2},
      &dten_res
  );
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_3d_s, &dten_3d_s2},
      &dten_res,
      drand()
  );
  dten_3d_s.Random(qn0);
  dten_3d_s2.Random(qnp1);
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_3d_s, &dten_3d_s2},
      &dten_res,
      drand()
  );
  dten_3d_s.Random(qnm1);
  dten_3d_s2.Random(qnp2);
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_3d_s, &dten_3d_s2},
      &dten_res
  );
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_3d_s, &dten_3d_s2},
      &dten_res,
      drand()
  );

  ZQLTensor zten_res;
  auto zten_1d_s2 = zten_1d_s;
  auto zten_2d_s2 = zten_2d_s;
  auto zten_3d_s2 = zten_3d_s;
  zten_res = zten_1d_s;
  zten_1d_s.Random(qn0);
  zten_1d_s2.Random(qn0);
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_1d_s, &zten_1d_s2},
      &zten_res
  );
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_1d_s, &zten_1d_s2},
      &zten_res,
      zrand()
  );
  zten_1d_s.Random(qn0);
  zten_1d_s2.Random(qnp1);
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_1d_s, &zten_1d_s2},
      &zten_res
  );
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_1d_s, &zten_1d_s2},
      &zten_res,
      zrand()
  );
  zten_1d_s.Random(qnm1);
  zten_1d_s2.Random(qnp2);
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_1d_s, &zten_1d_s2},
      &zten_res
  );
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_1d_s, &zten_1d_s2},
      &zten_res,
      zrand()
  );

  zten_res = zten_2d_s;
  zten_2d_s.Random(qn0);
  zten_2d_s2.Random(qn0);
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_2d_s, &zten_2d_s2},
      &zten_res
  );
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_2d_s, &zten_2d_s2},
      &zten_res,
      zrand()
  );
  zten_2d_s.Random(qn0);
  zten_2d_s2.Random(qnp1);
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_2d_s, &zten_2d_s2},
      &zten_res
  );
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_2d_s, &zten_2d_s2},
      &zten_res,
      zrand()
  );
  zten_2d_s.Random(qnm1);
  zten_2d_s2.Random(qnp2);
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_2d_s, &zten_2d_s2},
      &zten_res
  );
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_2d_s, &zten_2d_s2},
      &zten_res,
      zrand()
  );

  zten_res = zten_3d_s;
  zten_3d_s.Random(qn0);
  zten_3d_s2.Random(qn0);
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_3d_s, &zten_3d_s2},
      &zten_res
  );
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_3d_s, &zten_3d_s2},
      &zten_res,
      zrand()
  );
  zten_3d_s.Random(qn0);
  zten_3d_s2.Random(qnp1);
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_3d_s, &zten_3d_s2},
      &zten_res
  );
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_3d_s, &zten_3d_s2},
      &zten_res,
      zrand()
  );
  zten_3d_s.Random(qnm1);
  zten_3d_s2.Random(qnp2);
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_3d_s, &zten_3d_s2},
      &zten_res
  );
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_3d_s, &zten_3d_s2},
      &zten_res,
      zrand()
  );
}

TEST_F(TestLinearCombination, AddScaledAssignScalars) {
  DQLTensor dy(IndexVec<U1QN>{});
  DQLTensor dx(IndexVec<U1QN>{});
  dy() = 2.0;
  dx() = 3.0;
  AddScaledAssign(dy, 4.0, dx);
  EXPECT_DOUBLE_EQ(dy(), 14.0);

  ZQLTensor zy(IndexVec<U1QN>{});
  ZQLTensor zx(IndexVec<U1QN>{});
  zy() = QLTEN_Complex(2.0, -1.0);
  zx() = QLTEN_Complex(3.0, 2.0);
  Axpy(QLTEN_Complex(0.5, -1.0), zx, zy);
  EXPECT_COMPLEX_EQ(zy(), QLTEN_Complex(5.5, -3.0));
}

TEST_F(TestLinearCombination, AddScaledAssignUsesExistingLayoutInPlace) {
  dten_2d_s.Random(qn0);
  auto y = dten_2d_s;
  auto x = dten_2d_s;
  x.Random(qn0);
  const QLTEN_Double alpha = -1.25;
  auto expected = y + alpha * x;

  const auto *raw_data_before = y.GetBlkSparDataTen().GetActualRawDataPtr();
  AddScaledAssign(y, alpha, x);

  ExpectTensorNear(y, expected);
  EXPECT_EQ(y.GetBlkSparDataTen().GetActualRawDataPtr(), raw_data_before);
}

TEST_F(TestLinearCombination, AddScaledAssignInsertsMissingBlocks) {
  dten_2d_s.Random(qn0);
  auto y = dten_2d_s;
  auto x = dten_2d_s;
  x.Random(qnp1);
  const QLTEN_Double alpha = 0.75;
  auto expected = y + alpha * x;

  AddScaledAssign(y, alpha, x);

  ExpectTensorNear(y, expected);
}
