// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author:  Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2021-07-22
*
* Description: QuantumLiquids/tensor project. Unittests for tensor fuse index.
*/

#include "gtest/gtest.h"
#include "qlten/qltensor_all.h"
#include "qlten/tensor_manipulation/ten_fuse_index.h"
#include "qlten/tensor_manipulation/ten_ctrct.h"

#include "qlten/utility/timer.h"

using namespace qlten;
using U1QN = QN<U1QNVal>;
using QNSctT = QNSector<U1QN>;
using IndexT = Index<U1QN>;
using DQLTensor = QLTensor<QLTEN_Double, U1QN>;

namespace {

enum class PhysicalParity { kEven, kOdd };

struct PhysTag {
  size_t site;
  PhysicalParity parity;
};

bool operator==(const PhysTag &lhs, const PhysTag &rhs) {
  return lhs.site == rhs.site && lhs.parity == rhs.parity;
}

}  // namespace

std::string qn_nm = "qn";
U1QN qn0 = U1QN({QNCard(qn_nm, U1QNVal(0))});
U1QN qnp1 = U1QN({QNCard(qn_nm, U1QNVal(1))});
U1QN qnm1 = U1QN({QNCard(qn_nm, U1QNVal(-1))});
U1QN qnp2 = U1QN({QNCard(qn_nm, U1QNVal(2))});

QNSctT qnsct0_1 = QNSctT(qn0, 1);
QNSctT qnsct0_2 = QNSctT(qn0, 2);
QNSctT qnsct0_4 = QNSctT(qn0, 4);
QNSctT qnsctp1_1 = QNSctT(qnp1, 1);
QNSctT qnsctp1_2 = QNSctT(qnp1, 2);
QNSctT qnsctp1_4 = QNSctT(qnp1, 4);
QNSctT qnsctm1_1 = QNSctT(qnm1, 1);
QNSctT qnsctp2_1 = QNSctT(qnp2, 1);

IndexT idx_in0 = IndexT({qnsct0_1}, IN);
IndexT idx_out0 = IndexT({qnsct0_1}, OUT);
IndexT idx_out0_2 = IndexT({qnsct0_2}, OUT);
IndexT idx_out0_4 = IndexT({qnsct0_4}, OUT);

IndexT idx_in1 = IndexT({qnsct0_2, qnsctp1_1}, IN);
IndexT idx_in1plus1 = IndexT({qnsct0_4, qnsctp1_4, qnsctp2_1}, IN);
IndexT idx_out1 = IndexT({qnsctp1_1, qnsct0_1, qnsctm1_1}, OUT);

template<typename TenT>
void RunTestTenFuseIndexCase(
    TenT &a,
    const size_t idx1,
    const size_t idx2,
    TenT &correct_res
) {
  a.FuseIndex(idx1, idx2);
  // a.Show();
  // correct_res.Show();
  EXPECT_TRUE(a == correct_res);
}

TEST(TestFuseIndexNaive, TestCase) {
  DQLTensor ten0 = DQLTensor({idx_out0_2, idx_out0_2, idx_in0});
  DQLTensor ten1 = DQLTensor({idx_out0_4, idx_in0});
  ten0({0, 0, 0}) = 0.5;
  ten0({0, 1, 0}) = 0.7;
  ten0({1, 0, 0}) = 0.2;
  ten0({1, 1, 0}) = 1.2;

  ten1({0, 0}) = 0.5;
  ten1({1, 0}) = 0.7;
  ten1({2, 0}) = 0.2;
  ten1({3, 0}) = 1.2;
  RunTestTenFuseIndexCase(ten0, 0, 1, ten1);

  DQLTensor ten2 = DQLTensor({idx_in1, idx_in1});
  ten2({1, 2}) = 0.5;
  ten2({2, 0}) = 4.5;
  ten2({2, 1}) = 2.3;
  DQLTensor ten3 = DQLTensor({idx_in1plus1});
  ten3(5) = 0.5;
  ten3(6) = 4.5;
  ten3(7) = 2.3;
  RunTestTenFuseIndexCase(ten2, 0, 1, ten3);

}

//helper
IndexT RandIndex(const unsigned qn_sct_num,  //how many quantum number sectors?
                 const unsigned max_dim_in_one_qn_sct, // maximum dimension in every quantum number sector?
                 const TenIndexDirType dir) {
  QNSectorVec<U1QN> qnsv(qn_sct_num);
  for (size_t i = 0; i < qn_sct_num; i++) {
    auto qn = U1QN({QNCard("qn", U1QNVal(i))});
    qlten::SetRandomSeed(static_cast<unsigned long long>(time(NULL)));
    unsigned degeneracy = static_cast<unsigned>(qlten::RandUint32() % max_dim_in_one_qn_sct) + 1;
    qnsv[i] = QNSector(qn, degeneracy);
  }
  return Index(qnsv, dir);
}

template<typename TenElemT, typename QNT>
void RunTestTenFuseIndexBenchMarkByIndexCombinerCase(
    QLTensor<TenElemT, QNT> &a,
    const size_t idx1,
    const size_t idx2
) {
  a.ConciseShow();
  using TenT = QLTensor<TenElemT, QNT>;
  TenT correct_res;
  Index<QNT> index1 = a.GetIndexes()[idx1];
  Index<QNT> index2 = a.GetIndexes()[idx2];
  TenT index_combine = IndexCombine<TenElemT, QNT>(
      InverseIndex(index1), InverseIndex(index2),
      index2.GetDir());

  Contract(&index_combine, &a, {{0, 1}, {idx1, idx2}}, &correct_res);
  a.FuseIndex(idx1, idx2);
  EXPECT_TRUE(a == correct_res);
}

TEST(TESTFuseIndexRandom, 3DCase) {
  auto index1_in = RandIndex(5, 4, qlten::IN);
  auto index1_out = RandIndex(6, 3, qlten::OUT);
  DQLTensor t1({index1_in, index1_in, index1_out});
  t1.Random(qn0);
  RunTestTenFuseIndexBenchMarkByIndexCombinerCase(t1, 0, 1);
}

TEST(TESTFuseIndexRandom, 4DCase) {
  auto index1_in = RandIndex(5, 4, qlten::IN);
  auto index2_in = RandIndex(4, 6, qlten::IN);
  auto index1_out = RandIndex(3, 3, qlten::OUT);
  auto index2_out = RandIndex(2, 5, qlten::OUT);

  DQLTensor t1({index2_out, index1_in, index2_in, index1_out});
  t1.Random(qn0);
  RunTestTenFuseIndexBenchMarkByIndexCombinerCase(t1, 1, 2);
}

// Test that IndexFusionInfo is correctly returned and contains valid information
TEST(TestIndexFusionInfo, BasicTest) {
  DQLTensor ten0 = DQLTensor({idx_out0_2, idx_out0_2, idx_in0});
  ten0({0, 0, 0}) = 0.5;
  ten0({0, 1, 0}) = 0.7;
  ten0({1, 0, 0}) = 0.2;
  ten0({1, 1, 0}) = 1.2;

  // Get IndexFusionInfo from FuseIndex
  IndexFusionInfo<U1QN> fusion_info = ten0.FuseIndex(0, 1);

  // Verify IndexFusionInfo contains correct original indices
  EXPECT_EQ(fusion_info.left_index, idx_out0_2);
  EXPECT_EQ(fusion_info.right_index, idx_out0_2);

  // Verify fused index has correct dimension
  EXPECT_EQ(fusion_info.fused_index.dim(), idx_out0_2.dim() * idx_out0_2.dim());

  // Verify fused index direction matches original
  EXPECT_EQ(fusion_info.fused_index.GetDir(), idx_out0_2.GetDir());

  // Verify tensor has the fused index
  EXPECT_EQ(ten0.GetIndexes()[0], fusion_info.fused_index);

  // Verify sector offset metadata was recorded
  EXPECT_FALSE(fusion_info.sector_offsets.empty());
}

TEST(TestFuseIndexLineage, ReturnsFusionInfoAndFusedLineage) {
  DQLTensor ten({idx_out0_2, idx_out0_2, idx_out0_2, idx_out0_2});
  ten({0, 0, 0, 0}) = 1.0;
  ten({1, 0, 1, 0}) = 2.0;

  DQLTensor expected = ten;
  IndexFusionInfo<U1QN> expected_fusion = expected.FuseIndex(1, 3);

  const IndexLineage<PhysTag> left_lineage = {{{1, PhysicalParity::kOdd}}};
  const IndexLineage<PhysTag> right_lineage = {{{3, PhysicalParity::kEven}}};
  const FuseIndexResult<U1QN, PhysTag> result =
      ten.FuseIndex(1, 3, left_lineage, right_lineage);

  EXPECT_TRUE(ten == expected);
  EXPECT_EQ(result.index_fusion.left_index, expected_fusion.left_index);
  EXPECT_EQ(result.index_fusion.right_index, expected_fusion.right_index);
  EXPECT_EQ(result.index_fusion.fused_index, expected_fusion.fused_index);
  EXPECT_EQ(result.index_fusion.sector_offsets, expected_fusion.sector_offsets);

  const IndexLineage<PhysTag> expected_lineage = {
      {{1, PhysicalParity::kOdd}, {3, PhysicalParity::kEven}}
  };
  EXPECT_EQ(result.fused_lineage, expected_lineage);
}
