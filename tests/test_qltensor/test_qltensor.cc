// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-17 15:19
*
* Description: QuantumLiquids/tensor project. Unittests for QLTensor object.
*/

#include <fstream>    // ifstream, ofstream
#include "gtest/gtest.h"
#include <type_traits>
#include <complex>

//#define PLAIN_TRANSPOSE 1

#include "qlten/qltensor_all.h"      // QLTensor, Index, QN, U1QNVal, QNSectorVec
#include "qlten/utility/utils_inl.h" // GenAllCoors

#include "../testing_utility.h"      // RandInt, RandUnsignedInt, TransCoors

using namespace qlten;

using U1QN = QN<U1QNVal>;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DQLTensor = QLTensor<QLTEN_Double, U1QN>;
using ZQLTensor = QLTensor<QLTEN_Complex, U1QN>;

struct TestQLTensor : public testing::Test {
  std::string qn_nm = "qn";
  U1QN qn0 = U1QN({QNCard(qn_nm, U1QNVal(0))});
  U1QN qnp1 = U1QN({QNCard(qn_nm, U1QNVal(1))});
  U1QN qnp2 = U1QN({QNCard(qn_nm, U1QNVal(2))});
  U1QN qnm1 = U1QN({QNCard(qn_nm, U1QNVal(-1))});
  QNSctT qnsctm1_s = QNSctT(qnm1, 3);
  QNSctT qnsct0_s = QNSctT(qn0, 4);
  QNSctT qnsctp1_s = QNSctT(qnp1, 5);
  QNSctT qnsct0_l = QNSctT(qn0, 10);
  QNSctT qnsctp1_l = QNSctT(qnp1, 8);
  QNSctT qnsctm1_l = QNSctT(qnm1, 12);
  IndexT idx_in_s = IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, TenIndexDirType::IN);
  IndexT idx_out_s = IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, TenIndexDirType::OUT);
  IndexT idx_in_l = IndexT({qnsctm1_l, qnsct0_l, qnsctp1_l}, TenIndexDirType::IN);
  IndexT idx_out_l = IndexT({qnsctm1_l, qnsct0_l, qnsctp1_l}, TenIndexDirType::OUT);

  DQLTensor dten_default = DQLTensor();
  DQLTensor dten_scalar = DQLTensor(IndexVec<U1QN>{});
  DQLTensor dten_1d_s = DQLTensor({idx_out_s});
  DQLTensor dten_1d_l = DQLTensor({idx_out_l});
  DQLTensor dten_2d_s = DQLTensor({idx_in_s, idx_out_s});
  DQLTensor dten_2d_l = DQLTensor({idx_in_l, idx_out_l});
  DQLTensor dten_3d_s = DQLTensor({idx_in_s, idx_out_s, idx_out_s});
  DQLTensor dten_3d_l = DQLTensor({idx_in_l, idx_out_l, idx_out_l});
  ZQLTensor zten_default = ZQLTensor();
  ZQLTensor zten_scalar = ZQLTensor(IndexVec<U1QN>{});
  ZQLTensor zten_1d_s = ZQLTensor({idx_out_s});
  ZQLTensor zten_1d_l = ZQLTensor({idx_out_l});
  ZQLTensor zten_2d_s = ZQLTensor({idx_in_s, idx_out_s});
  ZQLTensor zten_2d_l = ZQLTensor({idx_in_l, idx_out_l});
  ZQLTensor zten_3d_s = ZQLTensor({idx_in_s, idx_out_s, idx_out_s});
  ZQLTensor zten_3d_l = ZQLTensor({idx_in_l, idx_out_l, idx_out_l});
};

template<typename QLTensorT>
void RunTestQLTensorCommonConstructorCase(
    const QLTensorT &ten,
    const std::vector<IndexT> &indexes) {
  EXPECT_EQ(ten.GetIndexes(), indexes);

  size_t size = 1;
  for (size_t i = 0; i < ten.Rank(); i++) {
    auto dim = ten.GetShape()[i];
    EXPECT_EQ(dim, indexes[i].dim());
    size *= dim;
  }

  if (ten.IsDefault()) {
    EXPECT_EQ(ten.size(), 0);
  } else {
    EXPECT_EQ(ten.size(), size);
  }
}

TEST_F(TestQLTensor, TestCommonConstructor) {
  EXPECT_TRUE(dten_default.IsDefault());
  RunTestQLTensorCommonConstructorCase(dten_scalar, {});
  RunTestQLTensorCommonConstructorCase(dten_1d_s, {idx_out_s});
  RunTestQLTensorCommonConstructorCase(dten_2d_s, {idx_in_s, idx_out_s});
  RunTestQLTensorCommonConstructorCase(
      dten_3d_s,
      {idx_in_s, idx_out_s, idx_out_s});

  EXPECT_TRUE(zten_default.IsDefault());
  RunTestQLTensorCommonConstructorCase(zten_scalar, {});
  RunTestQLTensorCommonConstructorCase(zten_1d_s, {idx_out_s});
  RunTestQLTensorCommonConstructorCase(zten_2d_s, {idx_in_s, idx_out_s});
  RunTestQLTensorCommonConstructorCase(
      zten_3d_s,
      {idx_in_s, idx_out_s, idx_out_s});
}

template<typename ElemT, typename QNT>
void RunTestQLTensorElemAssignmentCase(
    const QLTensor<ElemT, QNT> &t_init,
    const std::vector<ElemT> elems,
    const std::vector<std::vector<size_t>> coors) {
  auto t = t_init;
  for (size_t i = 0; i < elems.size(); ++i) {
    t(coors[i]) = elems[i];
  }
  for (auto coor : GenAllCoors(t.GetShape())) {
    auto coor_it = std::find(coors.cbegin(), coors.cend(), coor);
    if (coor_it != coors.end()) {
      auto elem_idx = std::distance(coors.cbegin(), coor_it);
      EXPECT_EQ(t(coor), elems[elem_idx]);
    } else {
      EXPECT_EQ(t(coor), ElemT(0.0));
    }
  }
  Show(t);
}

TEST_F(TestQLTensor, TestElemAssignment) {
  Show(dten_default);

  DQLTensor dten_scalar2(dten_scalar);
  auto scalar = drand();
  dten_scalar2() = scalar;
  EXPECT_EQ(dten_scalar2(), scalar);
  RunTestQLTensorElemAssignmentCase(dten_scalar, {drand()}, {{}});

  DQLTensor dten_1d_s2(dten_1d_s);
  auto elem0 = drand();
  dten_1d_s2(0) = elem0;
  EXPECT_EQ(dten_1d_s2(0), elem0);
  EXPECT_EQ(dten_1d_s2(1), 0.0);
  auto elem1 = drand();
  dten_1d_s2(1) = elem1;
  EXPECT_EQ(dten_1d_s2(1), elem1);
  RunTestQLTensorElemAssignmentCase(dten_1d_s, {drand()}, {{0}});
  RunTestQLTensorElemAssignmentCase(dten_1d_s, {drand()}, {{1}});
  RunTestQLTensorElemAssignmentCase(dten_1d_s, {drand(), drand()}, {{0},
                                                                    {1}});
  RunTestQLTensorElemAssignmentCase(dten_1d_s, {drand(), drand()}, {{1},
                                                                    {2}});

  DQLTensor dten_2d_s2(dten_2d_s);
  dten_2d_s2(2, 3) = elem0;
  EXPECT_EQ(dten_2d_s2(2, 3), elem0);
  EXPECT_EQ(dten_2d_s2(1, 7), 0.0);
  dten_2d_s2(1, 7) = elem1;
  EXPECT_EQ(dten_2d_s2(1, 7), elem1);
  RunTestQLTensorElemAssignmentCase(dten_2d_s, {drand()}, {{0, 0}});
  RunTestQLTensorElemAssignmentCase(dten_2d_s, {drand()}, {{2, 3}});
  RunTestQLTensorElemAssignmentCase(
      dten_2d_s,
      {drand(), drand()},
      {{2, 3},
       {3, 7}});

  RunTestQLTensorElemAssignmentCase(dten_3d_s, {drand()}, {{0, 0, 0}});
  RunTestQLTensorElemAssignmentCase(dten_3d_s, {drand()}, {{2, 3, 4}});
  RunTestQLTensorElemAssignmentCase(
      dten_3d_s,
      {drand(), drand()},
      {{2, 3, 5},
       {8, 7, 10}});

  RunTestQLTensorElemAssignmentCase(
      zten_1d_s,
      {zrand()},
      {{0}});
  RunTestQLTensorElemAssignmentCase(
      zten_1d_s,
      {zrand()},
      {{1}});
  RunTestQLTensorElemAssignmentCase(
      zten_1d_s,
      {zrand(), zrand()},
      {{0},
       {1}});
  RunTestQLTensorElemAssignmentCase(
      zten_1d_s,
      {zrand(), zrand()},
      {{1},
       {2}});
  RunTestQLTensorElemAssignmentCase(
      zten_2d_s,
      {zrand()},
      {{0, 0}});
  RunTestQLTensorElemAssignmentCase(
      zten_2d_s,
      {zrand()},
      {{2, 3}});
  RunTestQLTensorElemAssignmentCase(
      zten_2d_s,
      {zrand(), zrand()},
      {{2, 3},
       {3, 7}});
  RunTestQLTensorElemAssignmentCase(zten_3d_s, {zrand()}, {{0, 0, 0}});
  RunTestQLTensorElemAssignmentCase(zten_3d_s, {zrand()}, {{2, 3, 4}});
  RunTestQLTensorElemAssignmentCase(
      zten_3d_s,
      {zrand(), zrand()},
      {{2, 3, 5},
       {8, 7, 10}});
}

template<typename ElemT, typename QNT>
void RunTestQLTensorRandomCase(
    QLTensor<ElemT, QNT> &t,
    const QNT &div,
    const std::vector<std::vector<QNSector<QNT>>> &qnscts_set) {
  std::vector<std::vector<QNSector<QNT>>> had_qnscts_set;
  qlten::SetRandomSeed(0);
  t.Random(div);

  EXPECT_EQ(t.GetQNBlkNum(), qnscts_set.size());
  EXPECT_EQ(t.Div(), div);
#ifndef  USE_GPU
  if (t.IsScalar()) {
    qlten::SetRandomSeed(0);
    EXPECT_EQ(t.GetElem({}), RandT<ElemT>());
  }
#endif
  // TODO: Check each element in the random tensor.
}

TEST_F(TestQLTensor, Random) {
  RunTestQLTensorRandomCase(dten_scalar, U1QN(), {});
  RunTestQLTensorRandomCase(dten_1d_s, qn0, {{qnsct0_s}});
  RunTestQLTensorRandomCase(dten_1d_s, qnp1, {{qnsctp1_s}});
  RunTestQLTensorRandomCase(dten_1d_l, qn0, {{qnsct0_l}});
  RunTestQLTensorRandomCase(dten_1d_l, qnp1, {{qnsctp1_l}});
  RunTestQLTensorRandomCase(
      dten_2d_s,
      qn0,
      {
          {qnsctm1_s, qnsctm1_s},
          {qnsct0_s, qnsct0_s},
          {qnsctp1_s, qnsctp1_s}
      });
  RunTestQLTensorRandomCase(
      dten_2d_s,
      qnp1,
      {
          {qnsctm1_s, qnsct0_s},
          {qnsct0_s, qnsctp1_s}
      });
  RunTestQLTensorRandomCase(
      dten_2d_s,
      qnm1,
      {
          {qnsct0_s, qnsctm1_s},
          {qnsctp1_s, qnsct0_s}
      });
  RunTestQLTensorRandomCase(
      dten_2d_s,
      qnp2,
      {
          {qnsctm1_s, qnsctp1_s}
      });
  RunTestQLTensorRandomCase(
      dten_2d_l,
      qn0,
      {
          {qnsctm1_l, qnsctm1_l},
          {qnsct0_l, qnsct0_l},
          {qnsctp1_l, qnsctp1_l}
      });
  RunTestQLTensorRandomCase(
      dten_2d_l,
      qnp1,
      {
          {qnsctm1_l, qnsct0_l},
          {qnsct0_l, qnsctp1_l}
      });
  RunTestQLTensorRandomCase(
      dten_2d_l,
      qnm1,
      {
          {qnsct0_l, qnsctm1_l},
          {qnsctp1_l, qnsct0_l}
      });
  RunTestQLTensorRandomCase(
      dten_2d_l,
      qnp2,
      {
          {qnsctm1_l, qnsctp1_l}
      });
  RunTestQLTensorRandomCase(
      dten_3d_s,
      qn0,
      {
          {qnsctm1_s, qnsctm1_s, qnsct0_s},
          {qnsctm1_s, qnsct0_s, qnsctm1_s},
          {qnsct0_s, qnsct0_s, qnsct0_s},
          {qnsct0_s, qnsctp1_s, qnsctm1_s},
          {qnsct0_s, qnsctm1_s, qnsctp1_s},
          {qnsctp1_s, qnsctp1_s, qnsct0_s},
          {qnsctp1_s, qnsct0_s, qnsctp1_s}
      });
  RunTestQLTensorRandomCase(
      dten_3d_s,
      qnp1,
      {
          {qnsctm1_s, qnsct0_s, qnsct0_s},
          {qnsctm1_s, qnsctm1_s, qnsctp1_s},
          {qnsctm1_s, qnsctp1_s, qnsctm1_s},
          {qnsct0_s, qnsctp1_s, qnsct0_s},
          {qnsct0_s, qnsct0_s, qnsctp1_s},
          {qnsctp1_s, qnsctp1_s, qnsctp1_s}
      });
  RunTestQLTensorRandomCase(
      dten_3d_s,
      qnp2,
      {
          {qnsctm1_s, qnsctp1_s, qnsct0_s},
          {qnsctm1_s, qnsct0_s, qnsctp1_s},
          {qnsct0_s, qnsctp1_s, qnsctp1_s}
      });
  RunTestQLTensorRandomCase(
      dten_3d_l,
      qn0,
      {
          {qnsctm1_l, qnsctm1_l, qnsct0_l},
          {qnsctm1_l, qnsct0_l, qnsctm1_l},
          {qnsct0_l, qnsct0_l, qnsct0_l},
          {qnsct0_l, qnsctp1_l, qnsctm1_l},
          {qnsct0_l, qnsctm1_l, qnsctp1_l},
          {qnsctp1_l, qnsctp1_l, qnsct0_l},
          {qnsctp1_l, qnsct0_l, qnsctp1_l}
      });
  RunTestQLTensorRandomCase(
      dten_3d_l,
      qnp1,
      {
          {qnsctm1_l, qnsct0_l, qnsct0_l},
          {qnsctm1_l, qnsctm1_l, qnsctp1_l},
          {qnsctm1_l, qnsctp1_l, qnsctm1_l},
          {qnsct0_l, qnsctp1_l, qnsct0_l},
          {qnsct0_l, qnsct0_l, qnsctp1_l},
          {qnsctp1_l, qnsctp1_l, qnsctp1_l}
      });
  RunTestQLTensorRandomCase(
      dten_3d_l,
      qnp2,
      {
          {qnsctm1_l, qnsctp1_l, qnsct0_l},
          {qnsctm1_l, qnsct0_l, qnsctp1_l},
          {qnsct0_l, qnsctp1_l, qnsctp1_l}
      });

  RunTestQLTensorRandomCase(zten_scalar, U1QN(), {});
  RunTestQLTensorRandomCase(zten_1d_s, qn0, {{qnsct0_s}});
  RunTestQLTensorRandomCase(zten_1d_s, qnp1, {{qnsctp1_s}});
  RunTestQLTensorRandomCase(zten_1d_l, qn0, {{qnsct0_l}});
  RunTestQLTensorRandomCase(zten_1d_l, qnp1, {{qnsctp1_l}});
  RunTestQLTensorRandomCase(
      zten_2d_s,
      qn0,
      {
          {qnsctm1_s, qnsctm1_s},
          {qnsct0_s, qnsct0_s},
          {qnsctp1_s, qnsctp1_s}
      });
  RunTestQLTensorRandomCase(
      zten_2d_s,
      qnp1,
      {
          {qnsctm1_s, qnsct0_s},
          {qnsct0_s, qnsctp1_s}
      });
  RunTestQLTensorRandomCase(
      zten_2d_s,
      qnm1,
      {
          {qnsct0_s, qnsctm1_s},
          {qnsctp1_s, qnsct0_s}
      });
  RunTestQLTensorRandomCase(
      zten_2d_s,
      qnp2,
      {
          {qnsctm1_s, qnsctp1_s}
      });
  RunTestQLTensorRandomCase(
      zten_2d_l,
      qn0,
      {
          {qnsctm1_l, qnsctm1_l},
          {qnsct0_l, qnsct0_l},
          {qnsctp1_l, qnsctp1_l}
      });
  RunTestQLTensorRandomCase(
      zten_2d_l,
      qnp1,
      {
          {qnsctm1_l, qnsct0_l},
          {qnsct0_l, qnsctp1_l}
      });
  RunTestQLTensorRandomCase(
      zten_2d_l,
      qnm1,
      {
          {qnsct0_l, qnsctm1_l},
          {qnsctp1_l, qnsct0_l}
      });
  RunTestQLTensorRandomCase(
      zten_2d_l,
      qnp2,
      {
          {qnsctm1_l, qnsctp1_l}
      });
  RunTestQLTensorRandomCase(
      zten_3d_s,
      qn0,
      {
          {qnsctm1_s, qnsctm1_s, qnsct0_s},
          {qnsctm1_s, qnsct0_s, qnsctm1_s},
          {qnsct0_s, qnsct0_s, qnsct0_s},
          {qnsct0_s, qnsctp1_s, qnsctm1_s},
          {qnsct0_s, qnsctm1_s, qnsctp1_s},
          {qnsctp1_s, qnsctp1_s, qnsct0_s},
          {qnsctp1_s, qnsct0_s, qnsctp1_s}
      });
  RunTestQLTensorRandomCase(
      zten_3d_s,
      qnp1,
      {
          {qnsctm1_s, qnsct0_s, qnsct0_s},
          {qnsctm1_s, qnsctm1_s, qnsctp1_s},
          {qnsctm1_s, qnsctp1_s, qnsctm1_s},
          {qnsct0_s, qnsctp1_s, qnsct0_s},
          {qnsct0_s, qnsct0_s, qnsctp1_s},
          {qnsctp1_s, qnsctp1_s, qnsctp1_s}
      });
  RunTestQLTensorRandomCase(
      zten_3d_s,
      qnp2,
      {
          {qnsctm1_s, qnsctp1_s, qnsct0_s},
          {qnsctm1_s, qnsct0_s, qnsctp1_s},
          {qnsct0_s, qnsctp1_s, qnsctp1_s}
      });
  RunTestQLTensorRandomCase(
      zten_3d_l,
      qn0,
      {
          {qnsctm1_l, qnsctm1_l, qnsct0_l},
          {qnsctm1_l, qnsct0_l, qnsctm1_l},
          {qnsct0_l, qnsct0_l, qnsct0_l},
          {qnsct0_l, qnsctp1_l, qnsctm1_l},
          {qnsct0_l, qnsctm1_l, qnsctp1_l},
          {qnsctp1_l, qnsctp1_l, qnsct0_l},
          {qnsctp1_l, qnsct0_l, qnsctp1_l}
      });
  RunTestQLTensorRandomCase(
      zten_3d_l,
      qnp1,
      {
          {qnsctm1_l, qnsct0_l, qnsct0_l},
          {qnsctm1_l, qnsctm1_l, qnsctp1_l},
          {qnsctm1_l, qnsctp1_l, qnsctm1_l},
          {qnsct0_l, qnsctp1_l, qnsct0_l},
          {qnsct0_l, qnsct0_l, qnsctp1_l},
          {qnsctp1_l, qnsctp1_l, qnsctp1_l}
      });
  RunTestQLTensorRandomCase(
      zten_3d_l,
      qnp2,
      {
          {qnsctm1_l, qnsctp1_l, qnsct0_l},
          {qnsctm1_l, qnsct0_l, qnsctp1_l},
          {qnsct0_l, qnsctp1_l, qnsctp1_l}
      });
}

template<typename QLTensorT>
void RunTestQLTensorEqCase(
    const QLTensorT &lhs, const QLTensorT &rhs, bool test_eq = true) {
  if (test_eq) {
    EXPECT_TRUE(lhs == rhs);
  } else {
    EXPECT_TRUE(lhs != rhs);
  }
}

TEST_F(TestQLTensor, TestEq) {
  RunTestQLTensorEqCase(dten_default, dten_default);

  RunTestQLTensorEqCase(dten_scalar, dten_scalar);
  dten_scalar.Random(U1QN());
  decltype(dten_scalar) dten_scalar2(dten_scalar.GetIndexes());
  RunTestQLTensorEqCase(dten_scalar, dten_scalar2, false);

  RunTestQLTensorEqCase(dten_1d_s, dten_1d_s);
  dten_1d_s.Random(qn0);
  RunTestQLTensorEqCase(dten_1d_s, dten_1d_s);
  decltype(dten_1d_s) dten_1d_s2(dten_1d_s.GetIndexes());
  dten_1d_s2.Random(qnp1);
  RunTestQLTensorEqCase(dten_1d_s, dten_1d_s2, false);
  RunTestQLTensorEqCase(dten_1d_s, dten_1d_l, false);
  RunTestQLTensorEqCase(dten_1d_s, dten_2d_s, false);

  RunTestQLTensorEqCase(zten_default, zten_default);

  RunTestQLTensorEqCase(zten_scalar, zten_scalar);
  zten_scalar.Random(U1QN());
  decltype(zten_scalar) zten_scalar2(zten_scalar.GetIndexes());
  RunTestQLTensorEqCase(zten_scalar, zten_scalar2, false);

  RunTestQLTensorEqCase(zten_1d_s, zten_1d_s);
  zten_1d_s.Random(qn0);
  RunTestQLTensorEqCase(zten_1d_s, zten_1d_s);
  decltype(zten_1d_s) zten_1d_s2(zten_1d_s.GetIndexes());
  zten_1d_s2.Random(qnp1);
  RunTestQLTensorEqCase(zten_1d_s, zten_1d_s2, false);
  RunTestQLTensorEqCase(zten_1d_s, zten_1d_l, false);
  RunTestQLTensorEqCase(zten_1d_s, zten_2d_s, false);
}

template<typename QLTensorT>
void RunTestQLTensorCopyAndMoveConstructorsCase(const QLTensorT &t) {
  QLTensorT qlten_cpy(t);
  EXPECT_EQ(qlten_cpy, t);
  auto qlten_cpy2 = t;
  EXPECT_EQ(qlten_cpy2, t);

  QLTensorT qlten_tomove(t);    // Copy it.
  QLTensorT qlten_moved(std::move(qlten_tomove));
  EXPECT_EQ(qlten_moved, t);
  EXPECT_EQ(qlten_tomove.GetBlkSparDataTenPtr(), nullptr);
  QLTensorT qlten_tomove2(t);
  auto qlten_moved2 = std::move(qlten_tomove2);
  EXPECT_EQ(qlten_moved2, t);
  EXPECT_EQ(qlten_tomove2.GetBlkSparDataTenPtr(), nullptr);
}

TEST_F(TestQLTensor, TestCopyAndMoveConstructors) {
  RunTestQLTensorCopyAndMoveConstructorsCase(dten_default);
  dten_1d_s.Random(qn0);
  RunTestQLTensorCopyAndMoveConstructorsCase(dten_1d_s);
  dten_2d_s.Random(qn0);
  RunTestQLTensorCopyAndMoveConstructorsCase(dten_2d_s);
  dten_3d_s.Random(qn0);
  RunTestQLTensorCopyAndMoveConstructorsCase(dten_3d_s);

  RunTestQLTensorCopyAndMoveConstructorsCase(zten_default);
  zten_1d_s.Random(qn0);
  RunTestQLTensorCopyAndMoveConstructorsCase(zten_1d_s);
  zten_2d_s.Random(qn0);
  RunTestQLTensorCopyAndMoveConstructorsCase(zten_2d_s);
  zten_3d_s.Random(qn0);
  RunTestQLTensorCopyAndMoveConstructorsCase(zten_3d_s);
}

template<typename QLTensorT>
void RunTestQLTensorTransposeCase(
    const QLTensorT &t, const std::vector<size_t> &axes) {
  auto transed_t = t;
  transed_t.Transpose(axes);
  if (t.IsScalar()) {
    EXPECT_EQ(transed_t, t);
  } else {
    for (size_t i = 0; i < axes.size(); ++i) {
      EXPECT_EQ(transed_t.GetIndexes()[i], t.GetIndexes()[axes[i]]);
      EXPECT_EQ(transed_t.GetShape()[i], t.GetShape()[axes[i]]);
    }
    for (auto &coors : GenAllCoors(t.GetShape())) {
      EXPECT_EQ(
          transed_t.GetElem(TransCoors(coors, axes)),
          t.GetElem(coors)
      );
    }
  }
}

TEST_F(TestQLTensor, TestTranspose) {
  dten_scalar.Random(U1QN());
  RunTestQLTensorTransposeCase(dten_scalar, {});
  dten_1d_s.Random(qn0);
  RunTestQLTensorTransposeCase(dten_1d_s, {0});
  dten_2d_s.Random(qn0);
  RunTestQLTensorTransposeCase(dten_2d_s, {0, 1});
  RunTestQLTensorTransposeCase(dten_2d_s, {1, 0});
  dten_2d_s.Random(qnp1);
  RunTestQLTensorTransposeCase(dten_2d_s, {0, 1});
  RunTestQLTensorTransposeCase(dten_2d_s, {1, 0});
  dten_3d_s.Random(qn0);
  RunTestQLTensorTransposeCase(dten_3d_s, {0, 1, 2});
  RunTestQLTensorTransposeCase(dten_3d_s, {1, 0, 2});
  RunTestQLTensorTransposeCase(dten_3d_s, {2, 0, 1});
  dten_3d_s.Random(qnp1);
  RunTestQLTensorTransposeCase(dten_3d_s, {0, 1, 2});
  RunTestQLTensorTransposeCase(dten_3d_s, {1, 0, 2});
  RunTestQLTensorTransposeCase(dten_3d_s, {2, 0, 1});

  zten_scalar.Random(U1QN());
  RunTestQLTensorTransposeCase(zten_scalar, {});
  zten_1d_s.Random(qn0);
  RunTestQLTensorTransposeCase(zten_1d_s, {0});
  zten_2d_s.Random(qn0);
  RunTestQLTensorTransposeCase(zten_2d_s, {0, 1});
  RunTestQLTensorTransposeCase(zten_2d_s, {1, 0});
  zten_2d_s.Random(qnp1);
  RunTestQLTensorTransposeCase(zten_2d_s, {0, 1});
  RunTestQLTensorTransposeCase(zten_2d_s, {1, 0});
  zten_3d_s.Random(qn0);
  RunTestQLTensorTransposeCase(zten_3d_s, {0, 1, 2});
  RunTestQLTensorTransposeCase(zten_3d_s, {1, 0, 2});
  RunTestQLTensorTransposeCase(zten_3d_s, {2, 0, 1});
  zten_3d_s.Random(qnp1);
  RunTestQLTensorTransposeCase(zten_3d_s, {0, 1, 2});
  RunTestQLTensorTransposeCase(zten_3d_s, {1, 0, 2});
  RunTestQLTensorTransposeCase(zten_3d_s, {2, 0, 1});
}

template<typename QLTensorT>
void RunTestQLTensorNormalizeCase(QLTensorT &t) {

  std::vector<double> elem_norms;
  for (auto &coors : GenAllCoors(t.GetShape())) {
    elem_norms.push_back(qlten::norm(t.GetElem(coors)));
  }
  std::sort(elem_norms.begin(), elem_norms.end());// ascent order
  auto norm2 = 0.0;
  for (auto norm : elem_norms) {
    norm2 += norm;
  }
  auto norm = t.Normalize();
  EXPECT_NEAR(1, qlten::sqrt(norm2) / norm, 1e-14);
  norm2 = 0.0;
  for (auto &coors : GenAllCoors(t.GetShape())) {
    norm2 += qlten::norm(t.GetElem(coors));
  }
  EXPECT_NEAR(norm2, 1.0, kEpsilon);
}

TEST_F(TestQLTensor, TestNormalize) {
  dten_scalar.Random(U1QN());
  auto dscalar = dten_scalar.GetElem({});
  auto dnorm = dten_scalar.Normalize();
  EXPECT_DOUBLE_EQ(dnorm, qlten::abs(dscalar));
  EXPECT_DOUBLE_EQ(dten_scalar.GetElem({}), 1.0);

  dten_1d_s.Random(qn0);
  RunTestQLTensorNormalizeCase(dten_1d_s);
  dten_1d_s.Random(qnp1);
  RunTestQLTensorNormalizeCase(dten_1d_s);
  dten_2d_s.Random(qn0);
  RunTestQLTensorNormalizeCase(dten_2d_s);
  dten_3d_s.Random(qn0);
  RunTestQLTensorNormalizeCase(dten_3d_s);

  zten_scalar.Random(U1QN());
  auto zscalar = zten_scalar.GetElem({});
  auto znorm = zten_scalar.Normalize();
  EXPECT_DOUBLE_EQ(znorm, qlten::abs(zscalar));
  EXPECT_COMPLEX_EQ(zten_scalar.GetElem({}), zscalar / znorm);

  zten_1d_s.Random(qn0);
  RunTestQLTensorNormalizeCase(zten_1d_s);
  zten_1d_s.Random(qnp1);
  RunTestQLTensorNormalizeCase(zten_1d_s);
  zten_2d_s.Random(qn0);
  RunTestQLTensorNormalizeCase(zten_2d_s);
  zten_3d_s.Random(qn0);
  RunTestQLTensorNormalizeCase(zten_3d_s);
}

template<typename QLTensorT>
void RunTestQLTensorQuasi2NormCase(const QLTensorT &t) {
  // For bosonic tensors, Quasi2Norm should be the same as Get2Norm
  auto quasi_norm = t.GetQuasi2Norm();
  auto norm = t.Get2Norm();
  EXPECT_NEAR(quasi_norm, norm, kEpsilon);
  
  // Verify that Quasi2Norm is the square root of sum of element squares
  double expected_norm = 0.0;
  for (auto &coors : GenAllCoors(t.GetShape())) {
    expected_norm += qlten::norm(t.GetElem(coors));
  }
  expected_norm = qlten::sqrt(expected_norm);
  EXPECT_NEAR(quasi_norm, expected_norm, kEpsilon);
}

TEST_F(TestQLTensor, TestQuasi2Norm) {
  // Test scalar tensors
  dten_scalar.Random(U1QN());
  auto dscalar = dten_scalar.GetElem({});
  auto dquasi_norm = dten_scalar.GetQuasi2Norm();
  EXPECT_DOUBLE_EQ(dquasi_norm, qlten::abs(dscalar));

  zten_scalar.Random(U1QN());
  auto zscalar = zten_scalar.GetElem({});
  auto zquasi_norm = zten_scalar.GetQuasi2Norm();
  EXPECT_DOUBLE_EQ(zquasi_norm, qlten::abs(zscalar));

  // Test 1D tensors
  dten_1d_s.Random(qn0);
  RunTestQLTensorQuasi2NormCase(dten_1d_s);
  dten_1d_s.Random(qnp1);
  RunTestQLTensorQuasi2NormCase(dten_1d_s);

  zten_1d_s.Random(qn0);
  RunTestQLTensorQuasi2NormCase(zten_1d_s);
  zten_1d_s.Random(qnp1);
  RunTestQLTensorQuasi2NormCase(zten_1d_s);

  // Test 2D tensors
  dten_2d_s.Random(qn0);
  RunTestQLTensorQuasi2NormCase(dten_2d_s);
  dten_2d_s.Random(qnp1);
  RunTestQLTensorQuasi2NormCase(dten_2d_s);

  zten_2d_s.Random(qn0);
  RunTestQLTensorQuasi2NormCase(zten_2d_s);
  zten_2d_s.Random(qnp1);
  RunTestQLTensorQuasi2NormCase(zten_2d_s);

  // Test 3D tensors
  dten_3d_s.Random(qn0);
  RunTestQLTensorQuasi2NormCase(dten_3d_s);
  dten_3d_s.Random(qnp1);
  RunTestQLTensorQuasi2NormCase(dten_3d_s);

  zten_3d_s.Random(qn0);
  RunTestQLTensorQuasi2NormCase(zten_3d_s);
  zten_3d_s.Random(qnp1);
  RunTestQLTensorQuasi2NormCase(zten_3d_s);
}

template<typename QLTensorT>
void RunTestQLTensorSumCase(const QLTensorT &lhs, const QLTensorT &rhs) {
  auto sum1 = lhs + rhs;
  QLTensorT sum2(lhs);
  sum2 += rhs;
  for (auto &coors : GenAllCoors(lhs.GetShape())) {
    auto elem_sum = lhs.GetElem(coors) + rhs.GetElem(coors);
    EXPECT_EQ(sum1.GetElem(coors), elem_sum);
    EXPECT_EQ(sum2.GetElem(coors), elem_sum);
  }
  EXPECT_EQ(sum1, sum2);
}

TEST_F(TestQLTensor, TestSummation) {
  dten_scalar.Random(U1QN());
  RunTestQLTensorSumCase(dten_scalar, dten_scalar);

  DQLTensor dten_1d_s1(dten_1d_s);
  dten_1d_s1.Random(qn0);
  RunTestQLTensorSumCase(dten_1d_s1, dten_1d_s1);
  DQLTensor dten_1d_s2(dten_1d_s);
  dten_1d_s2.Random(qnp1);
  RunTestQLTensorSumCase(dten_1d_s1, dten_1d_s2);

  DQLTensor dten_2d_s1(dten_2d_s);
  dten_2d_s1.Random(qn0);
  RunTestQLTensorSumCase(dten_2d_s1, dten_2d_s1);
  DQLTensor dten_2d_s2(dten_2d_s);
  dten_2d_s2.Random(qnp1);
  RunTestQLTensorSumCase(dten_2d_s1, dten_2d_s2);

  DQLTensor dten_3d_s1(dten_3d_s);
  dten_3d_s1.Random(qn0);
  RunTestQLTensorSumCase(dten_3d_s1, dten_3d_s1);
  DQLTensor dten_3d_s2(dten_3d_s);
  dten_3d_s2.Random(qnp1);
  RunTestQLTensorSumCase(dten_3d_s1, dten_3d_s2);

  zten_scalar.Random(U1QN());
  RunTestQLTensorSumCase(zten_scalar, zten_scalar);

  ZQLTensor zten_1d_s1(zten_1d_s);
  zten_1d_s1.Random(qn0);
  RunTestQLTensorSumCase(zten_1d_s1, zten_1d_s1);
  ZQLTensor zten_1d_s2(zten_1d_s);
  zten_1d_s2.Random(qnp1);
  RunTestQLTensorSumCase(zten_1d_s1, zten_1d_s2);

  ZQLTensor zten_2d_s1(zten_2d_s);
  zten_2d_s1.Random(qn0);
  RunTestQLTensorSumCase(zten_2d_s1, zten_2d_s1);
  ZQLTensor zten_2d_s2(zten_2d_s);
  zten_2d_s2.Random(qnp1);
  RunTestQLTensorSumCase(zten_2d_s1, zten_2d_s2);

  ZQLTensor zten_3d_s1(zten_3d_s);
  zten_3d_s1.Random(qn0);
  RunTestQLTensorSumCase(zten_3d_s1, zten_3d_s1);
  ZQLTensor zten_3d_s2(zten_3d_s);
  zten_3d_s2.Random(qnp1);
  RunTestQLTensorSumCase(zten_3d_s1, zten_3d_s2);
}

template<typename ElemT, typename QNT>
void RunTestQLTensorDotMultiCase(
    const QLTensor<ElemT, QNT> &t, const ElemT scalar) {
  auto multied_t = scalar * t;
  QLTensor<ElemT, QNT> multied_t2(t);
  multied_t2 *= scalar;
  for (auto &coors : GenAllCoors(t.GetShape())) {
    GtestNear(multied_t.GetElem(coors), scalar * t.GetElem(coors), kEpsilon);
    GtestNear(multied_t2.GetElem(coors), scalar * t.GetElem(coors), kEpsilon);
  }
}

TEST_F(TestQLTensor, TestDotMultiplication) {
  dten_scalar.Random(U1QN());
  auto rand_d = drand();
  auto multied_ten = rand_d * dten_scalar;
  EXPECT_DOUBLE_EQ(multied_ten.GetElem({}), rand_d * dten_scalar.GetElem({}));

  dten_1d_s.Random(qn0);
  RunTestQLTensorDotMultiCase(dten_1d_s, drand());
  dten_2d_s.Random(qn0);
  RunTestQLTensorDotMultiCase(dten_2d_s, drand());
  dten_3d_s.Random(qn0);
  RunTestQLTensorDotMultiCase(dten_3d_s, drand());

  zten_scalar.Random(U1QN());
  auto rand_z = zrand();
  auto multied_ten_z = rand_z * zten_scalar;
  EXPECT_COMPLEX_EQ(
      multied_ten_z.GetElem({}),
      rand_z * zten_scalar.GetElem({})
  );

  zten_1d_s.Random(qn0);
  RunTestQLTensorDotMultiCase(zten_1d_s, zrand());
  zten_2d_s.Random(qn0);
  RunTestQLTensorDotMultiCase(zten_2d_s, zrand());
  zten_3d_s.Random(qn0);
  RunTestQLTensorDotMultiCase(zten_3d_s, zrand());
}

template<typename QLTensorT>
void RunTestQLTensorFileIOCase(const QLTensorT &t) {
  std::string file = "test.qlten";
  std::ofstream out(file, std::ofstream::binary);
  out << t;
  out.close();
  std::ifstream in(file, std::ifstream::binary);
  QLTensorT t_cpy;
  in >> t_cpy;
  in.close();
  std::remove(file.c_str());
  EXPECT_EQ(t_cpy, t);
}

TEST_F(TestQLTensor, FileIO) {
  dten_scalar.Random(U1QN());
  RunTestQLTensorFileIOCase(dten_scalar);
  dten_1d_s.Random(qn0);
  RunTestQLTensorFileIOCase(dten_1d_s);
  dten_1d_s.Random(qnp1);
  RunTestQLTensorFileIOCase(dten_1d_s);
  dten_2d_s.Random(qn0);
  RunTestQLTensorFileIOCase(dten_2d_s);
  dten_2d_s.Random(qnp1);
  RunTestQLTensorFileIOCase(dten_2d_s);
  dten_3d_s.Random(qn0);
  RunTestQLTensorFileIOCase(dten_3d_s);
  dten_3d_s.Random(qnp1);
  RunTestQLTensorFileIOCase(dten_3d_s);

  zten_scalar.Random(U1QN());
  RunTestQLTensorFileIOCase(zten_scalar);
  zten_1d_s.Random(qn0);
  RunTestQLTensorFileIOCase(zten_1d_s);
  zten_1d_s.Random(qnp1);
  RunTestQLTensorFileIOCase(zten_1d_s);
  zten_2d_s.Random(qn0);
  RunTestQLTensorFileIOCase(zten_2d_s);
  zten_2d_s.Random(qnp1);
  RunTestQLTensorFileIOCase(zten_2d_s);
  zten_3d_s.Random(qn0);
  RunTestQLTensorFileIOCase(zten_3d_s);
  zten_3d_s.Random(qnp1);
  RunTestQLTensorFileIOCase(zten_3d_s);
}

template<typename QLTensorT>
void RunTestQLTensorElementWiseOperationCase(QLTensorT t, bool real_ten = true) {
  t.ElementWiseInv(1e-13);
  t.ElementWiseSqrt();
  t.ElementWiseSquare();
  if (real_ten) {
    t.ElementWiseSign();
  }
  t.ElementWiseClipTo(0.05);

#ifndef  USE_GPU
  std::uniform_real_distribution<double> u_double(0, 1);
  std::random_device rd;
  std::mt19937 gen(rd());
  t.ElementWiseRandomizeMagnitudePreservePhase(u_double, gen);
#endif
}

#ifndef USE_GPU
template<typename ElemT, typename QNT>
void CheckRandomizeMagnitudePreservePhase(QLTensor<ElemT, QNT> t) {
  QLTensor<ElemT, QNT> before(t);
  double max_abs = 0.0;
  const auto &shape = before.GetShape();
  std::vector<size_t> coor(shape.size(), 0);
  auto advance = [&](std::vector<size_t> &c) -> bool {
    for (int k = int(c.size()) - 1; k >= 0; --k) {
      if (++c[static_cast<size_t>(k)] < shape[static_cast<size_t>(k)]) return true;
      c[static_cast<size_t>(k)] = 0;
    }
    return false;
  };
  if (shape.empty()) {
    max_abs = std::abs(before.GetElem({}));
  } else {
    do {
      max_abs = std::max(max_abs, (double)std::abs(before.GetElem(coor)));
    } while (advance(coor));
  }
  const double tol = 1e-3 * max_abs;

  std::uniform_real_distribution<double> u_double(0, 1);
  std::random_device rd;
  std::mt19937 gen(rd());
  t.ElementWiseRandomizeMagnitudePreservePhase(u_double, gen);

  std::fill(coor.begin(), coor.end(), 0);
  auto check_scalar = [&](const ElemT &oldv, const ElemT &newv) {
    constexpr double eps = 1e-12;
    if constexpr (std::is_same_v<ElemT, QLTEN_Double>) {
      double a = oldv;
      double b = newv;
      if (std::abs(a) > tol) {
        EXPECT_EQ(std::signbit(a), std::signbit(b));
        EXPECT_GE(std::abs(b), 0.0);
        EXPECT_LE(std::abs(b), 1.0);
      } else {
        EXPECT_NEAR(b, a, eps);
      }
    } else {
      std::complex<double> a = oldv;
      std::complex<double> b = newv;
      double amag = std::abs(a);
      if (amag > tol) {
        auto prod = b * std::conj(a);
        EXPECT_NEAR(prod.imag(), 0.0, 1e-10);
        EXPECT_GE(prod.real(), -1e-12);
        EXPECT_GE(std::abs(b), 0.0);
        EXPECT_LE(std::abs(b), 1.0);
      } else {
        EXPECT_NEAR(b.real(), a.real(), eps);
        EXPECT_NEAR(b.imag(), a.imag(), eps);
      }
    }
  };

  if (shape.empty()) {
    check_scalar(before.GetElem({}), t.GetElem({}));
  } else {
    do {
      check_scalar(before.GetElem(coor), t.GetElem(coor));
    } while (advance(coor));
  }
}
#endif

TEST_F(TestQLTensor, ElementWiseOperation) {
  dten_scalar.Random(U1QN());
  RunTestQLTensorElementWiseOperationCase(dten_scalar);

  DQLTensor dten_1d_s1(dten_1d_s);
  dten_1d_s1.Random(qn0);
  RunTestQLTensorElementWiseOperationCase(dten_1d_s1);
  DQLTensor dten_1d_s2(dten_1d_s);
  dten_1d_s2.Random(qnp1);
  RunTestQLTensorElementWiseOperationCase(dten_1d_s2);

  DQLTensor dten_2d_s1(dten_2d_s);
  dten_2d_s1.Random(qn0);
  RunTestQLTensorElementWiseOperationCase(dten_2d_s1);
  DQLTensor dten_2d_s2(dten_2d_s);
  dten_2d_s2.Random(qnp1);
  RunTestQLTensorElementWiseOperationCase(dten_2d_s2);

  DQLTensor dten_3d_s1(dten_3d_s);
  dten_3d_s1.Random(qn0);
  RunTestQLTensorElementWiseOperationCase(dten_3d_s1);
  DQLTensor dten_3d_s2(dten_3d_s);
  dten_3d_s2.Random(qnp1);
  RunTestQLTensorElementWiseOperationCase(dten_3d_s2);

  zten_scalar.Random(U1QN());
  RunTestQLTensorElementWiseOperationCase(zten_scalar, false);

  ZQLTensor zten_1d_s1(zten_1d_s);
  zten_1d_s1.Random(qn0);
  RunTestQLTensorElementWiseOperationCase(zten_1d_s1, false);
  ZQLTensor zten_1d_s2(zten_1d_s);
  zten_1d_s2.Random(qnp1);
  RunTestQLTensorElementWiseOperationCase(zten_1d_s2, false);

  ZQLTensor zten_2d_s1(zten_2d_s);
  zten_2d_s1.Random(qn0);
  RunTestQLTensorElementWiseOperationCase(zten_2d_s1);
  ZQLTensor zten_2d_s2(zten_2d_s);
  zten_2d_s2.Random(qnp1);
  RunTestQLTensorElementWiseOperationCase(zten_2d_s2, false);

  ZQLTensor zten_3d_s1(zten_3d_s);
  zten_3d_s1.Random(qn0);
  RunTestQLTensorElementWiseOperationCase(zten_3d_s1, false);
  ZQLTensor zten_3d_s2(zten_3d_s);
  zten_3d_s2.Random(qnp1);
  RunTestQLTensorElementWiseOperationCase(zten_3d_s2, false);
}

#ifndef USE_GPU
TEST_F(TestQLTensor, ElementWiseRandomizeMagnitudePreservePhaseBehavior) {
  // Real tensors
  DQLTensor d1(dten_1d_s); d1.Random(qn0); CheckRandomizeMagnitudePreservePhase<QLTEN_Double, U1QN>(d1);
  DQLTensor d2(dten_2d_s); d2.Random(qn0); CheckRandomizeMagnitudePreservePhase<QLTEN_Double, U1QN>(d2);
  DQLTensor d3(dten_3d_s); d3.Random(qn0); CheckRandomizeMagnitudePreservePhase<QLTEN_Double, U1QN>(d3);
  // Complex tensors
  ZQLTensor z1(zten_1d_s); z1.Random(qn0); CheckRandomizeMagnitudePreservePhase<QLTEN_Complex, U1QN>(z1);
  ZQLTensor z2(zten_2d_s); z2.Random(qn0); CheckRandomizeMagnitudePreservePhase<QLTEN_Complex, U1QN>(z2);
  ZQLTensor z3(zten_3d_s); z3.Random(qn0); CheckRandomizeMagnitudePreservePhase<QLTEN_Complex, U1QN>(z3);
}
#endif

TEST_F(TestQLTensor, ElementWiseSquare) {
  // Test scalar tensor
  dten_scalar.Random(U1QN());
  auto original_val = dten_scalar.GetElem({});
  dten_scalar.ElementWiseSquare();
  EXPECT_NEAR(dten_scalar.GetElem({}), original_val * original_val, 1e-10);

  // Test 1D tensor
  DQLTensor dten_1d_test(dten_1d_s);
  dten_1d_test.Random(qn0);
  std::vector<QLTEN_Double> original_vals;
  for (size_t i = 0; i < dten_1d_test.GetShape()[0]; ++i) {
    original_vals.push_back(dten_1d_test.GetElem({i}));
  }
  dten_1d_test.ElementWiseSquare();
  for (size_t i = 0; i < dten_1d_test.GetShape()[0]; ++i) {
    EXPECT_NEAR(dten_1d_test.GetElem({i}), original_vals[i] * original_vals[i], 1e-10);
  }

  // Test 2D tensor
  DQLTensor dten_2d_test(dten_2d_s);
  dten_2d_test.Random(qn0);
  std::vector<std::vector<QLTEN_Double>> original_vals_2d;
  for (size_t i = 0; i < dten_2d_test.GetShape()[0]; ++i) {
    original_vals_2d.emplace_back();
    for (size_t j = 0; j < dten_2d_test.GetShape()[1]; ++j) {
      original_vals_2d[i].push_back(dten_2d_test.GetElem({i, j}));
    }
  }
  dten_2d_test.ElementWiseSquare();
  for (size_t i = 0; i < dten_2d_test.GetShape()[0]; ++i) {
    for (size_t j = 0; j < dten_2d_test.GetShape()[1]; ++j) {
      EXPECT_NEAR(dten_2d_test.GetElem({i, j}), 
                  original_vals_2d[i][j] * original_vals_2d[i][j], 1e-10);
    }
  }

  // Test complex tensor
  zten_scalar.Random(U1QN());
  auto original_zval = zten_scalar.GetElem({});
  zten_scalar.ElementWiseSquare();
  EXPECT_NEAR(zten_scalar.GetElem({}).real(), 
              original_zval.real() * original_zval.real() - original_zval.imag() * original_zval.imag(), 1e-10);
  EXPECT_NEAR(zten_scalar.GetElem({}).imag(), 
              2.0 * original_zval.real() * original_zval.imag(), 1e-10);

  // Test outer function (by copy + call member function)
  DQLTensor dten_copy_test(dten_1d_s);
  dten_copy_test.Random(qn0);
  auto result = ElementWiseSquare(dten_copy_test);
  EXPECT_NE(result.GetActualDataSize(), 0);
  EXPECT_EQ(result.Rank(), dten_copy_test.Rank());
  EXPECT_EQ(result.GetShape(), dten_copy_test.GetShape());
}

TEST_F(TestQLTensor, ElementWiseClipTo) {
  // Test scalar tensor
  dten_scalar.Random(U1QN());
  auto original_val = dten_scalar.GetElem({});
  (void)original_val;
  double limit = 0.5;
  dten_scalar.ElementWiseClipTo(limit);
  EXPECT_LE(std::abs(dten_scalar.GetElem({})), limit);

  // Test 1D tensor
  DQLTensor dten_1d_test(dten_1d_s);
  dten_1d_test.Random(qn0);
  limit = 0.3;
  dten_1d_test.ElementWiseClipTo(limit);
  for (size_t i = 0; i < dten_1d_test.GetShape()[0]; ++i) {
    EXPECT_LE(std::abs(dten_1d_test.GetElem({i})), limit);
  }

  // Test 2D tensor
  DQLTensor dten_2d_test(dten_2d_s);
  dten_2d_test.Random(qn0);
  limit = 0.2;
  dten_2d_test.ElementWiseClipTo(limit);
  for (size_t i = 0; i < dten_2d_test.GetShape()[0]; ++i) {
    for (size_t j = 0; j < dten_2d_test.GetShape()[1]; ++j) {
      EXPECT_LE(std::abs(dten_2d_test.GetElem({i, j})), limit);
    }
  }

  // Test complex tensor - verify phase preservation
  zten_scalar.Random(U1QN());
  auto original_complex = zten_scalar.GetElem({});
  double original_phase = qlten::arg(original_complex);
  limit = 0.4;
  zten_scalar.ElementWiseClipTo(limit);
  auto clipped_complex = zten_scalar.GetElem({});
  // Magnitude should be clipped
  EXPECT_LE(qlten::abs(clipped_complex), limit + 1e-10);
  // Phase should be preserved (for non-zero original values)
  if (qlten::abs(original_complex) > 1e-10) {
    double clipped_phase = qlten::arg(clipped_complex);
    EXPECT_NEAR(original_phase, clipped_phase, 1e-10);
  }
}


TEST_F(TestQLTensor, ElementWiseInv) {
  // Test scalar tensor
  dten_scalar.Random(U1QN());
  auto original_val = dten_scalar.GetElem({});
  dten_scalar.ElementWiseInv();
  EXPECT_NEAR(dten_scalar.GetElem({}), 1.0 / original_val, 1e-10);

  // Test 1D tensor
  DQLTensor dten_1d_test(dten_1d_s);
  dten_1d_test.Random(qn0);
  std::vector<QLTEN_Double> original_vals;
  for (size_t i = 0; i < dten_1d_test.GetShape()[0]; ++i) {
    original_vals.push_back(dten_1d_test.GetElem({i}));
  }
  double tolerance = 1e-13;
  dten_1d_test.ElementWiseInv(tolerance);
  for (size_t i = 0; i < dten_1d_test.GetShape()[0]; ++i) {
    if (std::abs(original_vals[i]) > tolerance) {
      EXPECT_NEAR(dten_1d_test.GetElem({i}), 1.0 / original_vals[i], 1e-10);
    } else {
      EXPECT_EQ(dten_1d_test.GetElem({i}), 0.0);
    }
  }

  // Test with tolerance
  DQLTensor dten_1d_tol_test(dten_1d_s);
  dten_1d_tol_test.Random(qn0);
  std::vector<QLTEN_Double> original_vals_tol;
  for (size_t i = 0; i < dten_1d_tol_test.GetShape()[0]; ++i) {
    original_vals_tol.push_back(dten_1d_tol_test.GetElem({i}));
  }
  double tolerance_tol = 1e-6;
  dten_1d_tol_test.ElementWiseInv(tolerance_tol);
  for (size_t i = 0; i < dten_1d_tol_test.GetShape()[0]; ++i) {
    if (std::abs(original_vals_tol[i]) > tolerance_tol) {
      EXPECT_NEAR(dten_1d_tol_test.GetElem({i}), 1.0 / original_vals_tol[i], 1e-10);
    } else {
      EXPECT_EQ(dten_1d_tol_test.GetElem({i}), 0.0);
    }
  }

  // Test complex tensor
  zten_scalar.Random(U1QN());
  auto original_zval = zten_scalar.GetElem({});
  zten_scalar.ElementWiseInv();
  auto expected_inv = 1.0 / original_zval;
  EXPECT_NEAR(zten_scalar.GetElem({}).real(), expected_inv.real(), 1e-10);
  EXPECT_NEAR(zten_scalar.GetElem({}).imag(), expected_inv.imag(), 1e-10);

  // Test outer function (by copy + call member function)
  DQLTensor dten_copy_test(dten_1d_s);
  dten_copy_test.Random(qn0);
  auto result = ElementWiseInv(dten_copy_test);
  EXPECT_NE(result.GetActualDataSize(), 0);
  EXPECT_EQ(result.Rank(), dten_copy_test.Rank());
  EXPECT_EQ(result.GetShape(), dten_copy_test.GetShape());
}

TEST_F(TestQLTensor, ElementWiseSqrt) {
  // Test scalar tensor
  dten_scalar.Random(U1QN());
  auto original_val = dten_scalar.GetElem({});
  dten_scalar.ElementWiseSqrt();
  EXPECT_NEAR(dten_scalar.GetElem({}), std::sqrt(original_val), 1e-10);

  // Test 1D tensor
  DQLTensor dten_1d_test(dten_1d_s);
  dten_1d_test.Random(qn0);
  std::vector<QLTEN_Double> original_vals;
  for (size_t i = 0; i < dten_1d_test.GetShape()[0]; ++i) {
    original_vals.push_back(dten_1d_test.GetElem({i}));
  }
  dten_1d_test.ElementWiseSqrt();
  for (size_t i = 0; i < dten_1d_test.GetShape()[0]; ++i) {
    EXPECT_NEAR(dten_1d_test.GetElem({i}), std::sqrt(original_vals[i]), 1e-10);
  }

  // Test 2D tensor
  DQLTensor dten_2d_test(dten_2d_s);
  dten_2d_test.Random(qn0);
  std::vector<std::vector<QLTEN_Double>> original_vals_2d;
  for (size_t i = 0; i < dten_2d_test.GetShape()[0]; ++i) {
    original_vals_2d.emplace_back();
    for (size_t j = 0; j < dten_2d_test.GetShape()[1]; ++j) {
      original_vals_2d[i].push_back(dten_2d_test.GetElem({i, j}));
    }
  }
  dten_2d_test.ElementWiseSqrt();
  for (size_t i = 0; i < dten_2d_test.GetShape()[0]; ++i) {
    for (size_t j = 0; j < dten_2d_test.GetShape()[1]; ++j) {
      EXPECT_NEAR(dten_2d_test.GetElem({i, j}), 
                  std::sqrt(original_vals_2d[i][j]), 1e-10);
    }
  }

  // Test complex tensor
  zten_scalar.Random(U1QN());
  auto original_zval = zten_scalar.GetElem({});
  zten_scalar.ElementWiseSqrt();
  auto expected_sqrt = qlten::sqrt(original_zval);
  EXPECT_NEAR(zten_scalar.GetElem({}).real(), expected_sqrt.real(), 1e-10);
  EXPECT_NEAR(zten_scalar.GetElem({}).imag(), expected_sqrt.imag(), 1e-10);

  // Test outer function (by copy + call member function)
  DQLTensor dten_copy_test(dten_1d_s);
  dten_copy_test.Random(qn0);
  auto result = ElementWiseSqrt(dten_copy_test);
  EXPECT_NE(result.GetActualDataSize(), 0);
  EXPECT_EQ(result.Rank(), dten_copy_test.Rank());
  EXPECT_EQ(result.GetShape(), dten_copy_test.GetShape());
}

template<typename ElemT, typename QNT>
void RunTestQLTensorSetElemByQNSectorCase(
    QLTensor<ElemT, QNT>& t,
    const std::vector<QNT>& qn_sector,
    const std::vector<size_t>& blk_coors,
    const ElemT value
) {
  t.SetElemByQNSector(qn_sector, blk_coors, value);
  
  // Convert QN sector and block coordinates to global coordinates
  std::vector<size_t> global_coors(t.Rank());
  for (size_t i = 0; i < t.Rank(); ++i) {
    const auto& idx = t.GetIndex(i);
    size_t offset = 0;
    for (size_t j = 0; j < idx.GetQNSctNum(); ++j) {
      if (idx.GetQNSct(j).GetQn() == qn_sector[i]) {
        global_coors[i] = offset + blk_coors[i];
        break;
      }
      offset += idx.GetQNSct(j).dim();
    }
  }
  
  // Check if the element was set correctly
  EXPECT_EQ(t.GetElem(global_coors), value);
}

TEST_F(TestQLTensor, TestSetElemByQNSector) {
  // Test scalar tensor
  dten_scalar.Random(U1QN());
  auto scalar_val = drand();
  RunTestQLTensorSetElemByQNSectorCase(dten_scalar, {}, {}, scalar_val);
  
  // Test 1D tensor
  dten_1d_s.Random(qn0);
  RunTestQLTensorSetElemByQNSectorCase(
      dten_1d_s,
      {qn0},
      {1},
      drand()
  );
  RunTestQLTensorSetElemByQNSectorCase(
      dten_1d_s,
      {qnp1},
      {2},
      drand()
  );
  
  // Test 2D tensor
  dten_2d_s.Random(qn0);
  RunTestQLTensorSetElemByQNSectorCase(
      dten_2d_s,
      {qnm1, qnm1},
      {1, 2},
      drand()
  );
  RunTestQLTensorSetElemByQNSectorCase(
      dten_2d_s,
      {qn0, qnp1},
      {2, 1},
      drand()
  );
  
  // Test 3D tensor
  dten_3d_s.Random(qn0);
  RunTestQLTensorSetElemByQNSectorCase(
      dten_3d_s,
      {qnm1, qn0, qnp1},
      {1, 2, 3},
      drand()
  );
  RunTestQLTensorSetElemByQNSectorCase(
      dten_3d_s,
      {qn0, qnp1, qn0},
      {2, 1, 2},
      drand()
  );
  
  // Test complex tensors
  zten_scalar.Random(U1QN());
  auto scalar_val_z = zrand();
  RunTestQLTensorSetElemByQNSectorCase(zten_scalar, {}, {}, scalar_val_z);
  
  zten_2d_s.Random(qn0);
  RunTestQLTensorSetElemByQNSectorCase(
      zten_2d_s,
      {qnm1, qnm1},
      {1, 2},
      zrand()
  );
  RunTestQLTensorSetElemByQNSectorCase(
      zten_2d_s,
      {qn0, qnp1},
      {2, 1},
      zrand()
  );
  
  // Test creating new blocks
  DQLTensor new_block_ten({idx_in_s, idx_out_s});
  auto new_val = drand();
  RunTestQLTensorSetElemByQNSectorCase(
      new_block_ten,
      {qnm1, qnp1},
      {1, 2},
      new_val
  );
  
  // Test error cases
  EXPECT_THROW(
      dten_2d_s.SetElemByQNSector({qnm1}, {1, 2}, drand()),
      std::invalid_argument
  );
  EXPECT_THROW(
      dten_2d_s.SetElemByQNSector({qnm1, qnm1}, {1}, drand()),
      std::invalid_argument  
  );

  // Test normal case
  auto val = drand();
  dten_2d_s.SetElemByQNSector({qnm1, qnm1}, {1, 1}, val);
  EXPECT_EQ(dten_2d_s.GetElem({1, 1}), val);
}

