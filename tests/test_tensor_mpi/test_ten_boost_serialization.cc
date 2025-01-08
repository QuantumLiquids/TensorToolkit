// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2021-7-26
*
* Description: QuantumLiquids/tensor project. Unittests for Boost Serialization of QLTensor.
* Note: The serialization functions only serializes the wraps of tensor, rather the raw data
*/

#include <fstream>                            // ifstream, ofstream

#include "gtest/gtest.h"
#include "qlten/qltensor_all.h"               // QLTensor, Index, QN, U1QNVal, QNSectorVec
#include "qlten/utility/utils_inl.h"          // GenAllCoors

using namespace qlten;

using special_qn::U1QN;
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
  QNSctT qnsct0_s = QNSctT(qn0, 4);
  QNSctT qnsctp1_s = QNSctT(qnp1, 5);
  QNSctT qnsctm1_s = QNSctT(qnm1, 3);
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
  void SetUp() override {
    dten_scalar() = 0.3;
    zten_scalar() = 1.33;
    dten_1d_s({2}) = 0.38;
    dten_1d_l({4}) = 0.12;
    zten_1d_s({1}) = 0.10086;
    zten_1d_l({2}) = 0.78;
    dten_2d_s(0, 0) = 0.4;
    zten_2d_s(0, 1) = -0.55;
    dten_2d_l(1, 3) = 0.99;
    zten_2d_l(2, 0) = -0.32;
    dten_3d_s(0, 1, 2) = 1.5;
    zten_3d_s(1, 0, 1) = 2.75;
    dten_3d_l(2, 2, 1) = 1.23;
    zten_3d_l(1, 0, 2) = -0.98;
  }
};

template<typename ElemT, typename QNT>
void TestSerializationWriteAndReadQLTensor(
    const QLTensor<ElemT, QNT> &tensor
) {
  std::string filename = "serialization_test_file.qlten";
  std::ofstream ofs(filename);
  boost::archive::binary_oarchive oa(ofs);
  oa << tensor;
  ofs.close();

  std::ifstream ifs(filename);
  boost::archive::binary_iarchive ia(ifs);

  DQLTensor load_ten;
  ia >> load_ten;
  ifs.close();
  EXPECT_EQ(load_ten.GetIndexes(), load_ten.GetIndexes());
  EXPECT_EQ(load_ten.GetQNBlkNum(), load_ten.GetQNBlkNum());
  if (!tensor.IsDefault() && !tensor.IsScalar() && tensor.GetActualDataSize() != 0) {
    EXPECT_EQ(load_ten.Div(), load_ten.Div());
  }
}

TEST_F(TestQLTensor, TestSerializationWriteAndRead) {
  TestSerializationWriteAndReadQLTensor(dten_scalar);
  TestSerializationWriteAndReadQLTensor(dten_1d_s);
  TestSerializationWriteAndReadQLTensor(dten_1d_l);
  TestSerializationWriteAndReadQLTensor(dten_2d_s);
  TestSerializationWriteAndReadQLTensor(dten_2d_l);
  TestSerializationWriteAndReadQLTensor(dten_3d_s);
  TestSerializationWriteAndReadQLTensor(dten_3d_l);

  TestSerializationWriteAndReadQLTensor(zten_scalar);
  TestSerializationWriteAndReadQLTensor(zten_1d_s);
  TestSerializationWriteAndReadQLTensor(zten_1d_l);
  TestSerializationWriteAndReadQLTensor(zten_2d_s);
  TestSerializationWriteAndReadQLTensor(zten_2d_l);
  TestSerializationWriteAndReadQLTensor(zten_3d_s);
  TestSerializationWriteAndReadQLTensor(zten_3d_l);
}

//helper
IndexT RandIndex(const unsigned qn_sct_num,            // the number of QN sectors
                 const unsigned max_dim_in_one_qn_sct, // maximum dimension in each QN sector
                 const TenIndexDirType dir) {
  QNSectorVec<U1QN> qnsv(qn_sct_num);
  for (size_t i = 0; i < qn_sct_num; i++) {
    auto qn = U1QN({QNCard("qn", U1QNVal(i))});
    srand((unsigned) time(NULL));
    unsigned degeneracy = rand() % max_dim_in_one_qn_sct + 1;
    qnsv[i] = QNSector(qn, degeneracy);
  }
  return Index(qnsv, dir);
}

TEST_F(TestQLTensor, TestSerializationRandom4DTensor) {
  auto index1_in = RandIndex(5, 4, qlten::IN);
  auto index2_in = RandIndex(4, 6, qlten::IN);
  auto index1_out = RandIndex(3, 3, qlten::OUT);
  auto index2_out = RandIndex(2, 5, qlten::OUT);

  DQLTensor t1({index2_out, index1_in, index2_in, index1_out});
  t1.Random(qn0);
  TestSerializationWriteAndReadQLTensor(t1);
}