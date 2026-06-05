// SPDX-License-Identifier: LGPL-3.0-only
/*
 * Description: Unittests for tensor primitive cost estimators.
 */

#include "gtest/gtest.h"

#include "qlten/tensor_manipulation/tensor_op_cost.h"

using namespace qlten;
using special_qn::U1QN;

namespace {

using QNSctT = QNSector<U1QN>;
using IndexT = Index<U1QN>;
using Tensor = QLTensor<QLTEN_Double, U1QN>;

TEST(TensorOpCost, EstimateContractCostReportsExecutableTasksAndOutputLayout) {
  const U1QN qn0(0);
  const IndexT lhs_free({QNSctT(qn0, 2)}, TenIndexDirType::OUT);
  const IndexT contracted_lhs({QNSctT(qn0, 3)}, TenIndexDirType::OUT);
  const IndexT contracted_rhs({QNSctT(qn0, 3)}, TenIndexDirType::IN);
  const IndexT rhs_free({QNSctT(qn0, 4)}, TenIndexDirType::OUT);

  Tensor lhs({lhs_free, contracted_lhs});
  Tensor rhs({contracted_rhs, rhs_free});
  lhs.Fill(qn0, 1.0);
  rhs.Fill(qn0, 1.0);

  const auto cost = EstimateContractCost(lhs, rhs, {{1}, {0}});

  EXPECT_DOUBLE_EQ(cost.flops, 48.0);
  EXPECT_EQ(cost.gemm_count, 1U);
  EXPECT_EQ(cost.executable_block_pair_count, 1U);
  EXPECT_EQ(cost.output_block_count, 1U);
  EXPECT_EQ(cost.output_raw_elem_count, 8U);
  EXPECT_EQ(cost.read_bytes, 18U * sizeof(QLTEN_Double));
  EXPECT_EQ(cost.write_bytes, 8U * sizeof(QLTEN_Double));
  EXPECT_EQ(cost.output_layout.indexes, (IndexVec<U1QN>{lhs_free, rhs_free}));
  ASSERT_EQ(cost.output_layout.blocks.size(), 1U);
  EXPECT_EQ(cost.output_layout.blocks.begin()->second.shape, (ShapeT{2, 4}));
}

TEST(TensorOpCost, MakeTensorShellFromLayoutPreservesTopologyWithoutRawData) {
  const U1QN qn0(0);
  const IndexT left({QNSctT(qn0, 2)}, TenIndexDirType::OUT);
  const IndexT right({QNSctT(qn0, 3)}, TenIndexDirType::OUT);

  Tensor tensor({left, right});
  tensor.Fill(qn0, 1.0);
  const auto layout = MakeBlockSparseTensorLayout(tensor);

  const auto shell = MakeTensorShellFromLayout<QLTEN_Double>(layout);

  EXPECT_EQ(MakeBlockSparseTensorLayout(shell), layout);
  EXPECT_FALSE(shell.HasActualData());
}

TEST(TensorOpCost, EstimatorCanComposeFromPreviousOutputLayoutShell) {
  const U1QN qn0(0);
  const IndexT lhs_free({QNSctT(qn0, 2)}, TenIndexDirType::OUT);
  const IndexT contracted_lhs({QNSctT(qn0, 3)}, TenIndexDirType::OUT);
  const IndexT contracted_rhs({QNSctT(qn0, 3)}, TenIndexDirType::IN);
  const IndexT intermediate_axis({QNSctT(qn0, 4)}, TenIndexDirType::OUT);
  const IndexT final_axis({QNSctT(qn0, 5)}, TenIndexDirType::OUT);

  Tensor lhs({lhs_free, contracted_lhs});
  Tensor rhs({contracted_rhs, intermediate_axis});
  Tensor rank2_op({InverseIndex(intermediate_axis), final_axis});
  lhs.Fill(qn0, 1.0);
  rhs.Fill(qn0, 1.0);
  rank2_op.Fill(qn0, 1.0);

  const auto contract_cost = EstimateContractCost(lhs, rhs, {{1}, {0}});
  const auto intermediate_shell =
      MakeTensorShellFromLayout<QLTEN_Double>(contract_cost.output_layout);
  const auto rank2_cost = EstimateRank2AxisApplyCost(
      intermediate_shell, rank2_op, 1, dmrg::Rank2AxisApplyGemmMode::kLoop);

  EXPECT_EQ(rank2_cost.output_layout.indexes,
            (IndexVec<U1QN>{lhs_free, final_axis}));
  EXPECT_EQ(rank2_cost.output_raw_elem_count, 10U);
  EXPECT_FALSE(intermediate_shell.HasActualData());
}

TEST(TensorOpCost, EstimateRank2AxisApplyCostDistinguishesLoopAndBatchMode) {
  const U1QN qn0(0);
  const IndexT left({QNSctT(qn0, 2)}, TenIndexDirType::OUT);
  const IndexT input_axis({QNSctT(qn0, 3)}, TenIndexDirType::OUT);
  const IndexT output_axis({QNSctT(qn0, 5)}, TenIndexDirType::OUT);
  const IndexT right({QNSctT(qn0, 4)}, TenIndexDirType::OUT);

  Tensor input({left, input_axis, right});
  Tensor rank2_op({InverseIndex(input_axis), output_axis});
  input.Fill(qn0, 1.0);
  rank2_op.Fill(qn0, 1.0);

  const auto loop_cost = EstimateRank2AxisApplyCost(
      input, rank2_op, 1, dmrg::Rank2AxisApplyGemmMode::kLoop);
  const auto batch_cost = EstimateRank2AxisApplyCost(
      input, rank2_op, 1, dmrg::Rank2AxisApplyGemmMode::kBatch);

  EXPECT_DOUBLE_EQ(loop_cost.flops, 240.0);
  EXPECT_EQ(loop_cost.gemm_count, 2U);
  EXPECT_EQ(loop_cost.executable_block_pair_count, 1U);
  EXPECT_EQ(loop_cost.output_raw_elem_count, 40U);
  EXPECT_EQ(loop_cost.output_layout.indexes,
            (IndexVec<U1QN>{left, output_axis, right}));
  EXPECT_EQ(loop_cost.temp_peak_bytes, 0U);

  EXPECT_DOUBLE_EQ(batch_cost.flops, loop_cost.flops);
  EXPECT_EQ(batch_cost.gemm_count, loop_cost.gemm_count);
  EXPECT_EQ(batch_cost.temp_peak_bytes, 15U * sizeof(QLTEN_Double));
}

TEST(TensorOpCost, EstimateTensorLinearCombineCostBuildsUnionLayout) {
  const U1QN qn0(0);
  const U1QN qn1(1);
  const IndexT index({QNSctT(qn0, 2), QNSctT(qn1, 3)},
                     TenIndexDirType::OUT);

  Tensor first({index});
  Tensor second({index});
  first.Fill(qn0, 1.0);
  second.Fill(qn1, 1.0);

  const auto first_layout = MakeBlockSparseTensorLayout(first);
  const auto second_layout = MakeBlockSparseTensorLayout(second);
  const auto layout_cost =
      EstimateTensorLinearCombineCost<QLTEN_Double>(
          {first_layout, second_layout});
  const auto tensor_cost =
      EstimateTensorLinearCombineCost(
          std::vector<const Tensor *>{&first, &second});

  EXPECT_EQ(layout_cost.output_block_count, 2U);
  EXPECT_EQ(layout_cost.output_raw_elem_count, 5U);
  EXPECT_EQ(layout_cost.read_bytes, 5U * sizeof(QLTEN_Double));
  EXPECT_EQ(layout_cost.write_bytes, 5U * sizeof(QLTEN_Double));
  EXPECT_EQ(layout_cost.output_layout.indexes, (IndexVec<U1QN>{index}));
  EXPECT_EQ(tensor_cost.output_layout.blocks.size(),
            layout_cost.output_layout.blocks.size());
}

TEST(TensorOpCost, EstimateAxisMetadataApplyCostUsesInputLayoutForSectorScalars) {
  const U1QN qn0(0);
  const IndexT left({QNSctT(qn0, 2)}, TenIndexDirType::OUT);
  const IndexT axis({QNSctT(qn0, 3)}, TenIndexDirType::OUT);
  const IndexT right({QNSctT(qn0, 4)}, TenIndexDirType::OUT);

  Tensor input({left, axis, right});
  input.Fill(qn0, 1.0);

  const dmrg::AxisSectorScalar<QLTEN_Double, U1QN> scale{1, {2.0}};
  const auto cost = EstimateAxisMetadataApplyCost(input, scale);

  EXPECT_EQ(cost.gemm_count, 0U);
  EXPECT_EQ(cost.output_block_count, input.GetQNBlkNum());
  EXPECT_EQ(cost.output_raw_elem_count, 24U);
  EXPECT_EQ(cost.read_bytes, 24U * sizeof(QLTEN_Double));
  EXPECT_EQ(cost.write_bytes, 24U * sizeof(QLTEN_Double));
  EXPECT_EQ(cost.output_layout.indexes, input.GetIndexes());
}

TEST(TensorOpCost, EstimateContractAccumulateCostReadsExistingOutput) {
  const U1QN qn0(0);
  const IndexT lhs_free({QNSctT(qn0, 2)}, TenIndexDirType::OUT);
  const IndexT contracted_lhs({QNSctT(qn0, 3)}, TenIndexDirType::OUT);
  const IndexT contracted_rhs({QNSctT(qn0, 3)}, TenIndexDirType::IN);
  const IndexT rhs_free({QNSctT(qn0, 4)}, TenIndexDirType::OUT);

  Tensor lhs({lhs_free, contracted_lhs});
  Tensor rhs({contracted_rhs, rhs_free});
  Tensor existing_output({lhs_free, rhs_free});
  lhs.Fill(qn0, 1.0);
  rhs.Fill(qn0, 1.0);
  existing_output.Fill(qn0, 0.0);

  const auto contract_cost = EstimateContractCost(lhs, rhs, {{1}, {0}});
  const auto accumulate_cost =
      EstimateContractAccumulateCost(lhs, rhs, {{1}, {0}}, existing_output);

  EXPECT_DOUBLE_EQ(accumulate_cost.flops, contract_cost.flops);
  EXPECT_EQ(accumulate_cost.gemm_count, contract_cost.gemm_count);
  EXPECT_EQ(accumulate_cost.output_raw_elem_count,
            contract_cost.output_raw_elem_count);
  EXPECT_GT(accumulate_cost.read_bytes, contract_cost.read_bytes);
  EXPECT_EQ(accumulate_cost.output_layout.indexes,
            contract_cost.output_layout.indexes);
}

TEST(TensorOpCost, ContractAccumulateExpansionCountsExistingBlockCopyTraffic) {
  const U1QN qn0(0);
  const U1QN qn1(1);
  const IndexT output_index({QNSctT(qn0, 2), QNSctT(qn1, 3)},
                            TenIndexDirType::OUT);
  const IndexT contracted_lhs({QNSctT(qn0, 1)}, TenIndexDirType::OUT);
  const IndexT contracted_rhs({QNSctT(qn0, 1)}, TenIndexDirType::IN);

  Tensor lhs({output_index, contracted_lhs});
  Tensor rhs({contracted_rhs});
  Tensor existing_output({output_index});
  lhs.Fill(qn0, 1.0);
  rhs.Fill(qn0, 1.0);
  existing_output.Fill(qn1, 2.0);

  const auto contract_cost = EstimateContractCost(lhs, rhs, {{1}, {0}});
  const auto accumulate_cost =
      EstimateContractAccumulateCost(lhs, rhs, {{1}, {0}}, existing_output);
  const size_t existing_copy_bytes = 3U * sizeof(QLTEN_Double);

  EXPECT_EQ(accumulate_cost.output_block_count, 2U);
  EXPECT_EQ(accumulate_cost.output_raw_elem_count, 5U);
  EXPECT_EQ(accumulate_cost.read_bytes,
            contract_cost.read_bytes + existing_copy_bytes);
  EXPECT_EQ(accumulate_cost.write_bytes,
            contract_cost.write_bytes + existing_copy_bytes);
}

TEST(TensorOpCost, ContractAccumulateStartSizeWrapsAxesLikeExecutor) {
  const U1QN qn0(0);
  const IndexT wrapped_first({QNSctT(qn0, 2)}, TenIndexDirType::OUT);
  const IndexT free_axis({QNSctT(qn0, 3)}, TenIndexDirType::OUT);
  const IndexT wrapped_second({QNSctT(qn0, 4)}, TenIndexDirType::OUT);

  Tensor lhs({wrapped_first, free_axis, wrapped_second});
  Tensor rhs({InverseIndex(wrapped_second), InverseIndex(wrapped_first)});
  Tensor existing_output({free_axis});
  lhs.Fill(qn0, 1.0);
  rhs.Fill(qn0, 1.0);
  existing_output.Fill(qn0, 0.0);

  const auto cost = EstimateContractAccumulateCost(
      lhs, rhs, 2, 0, 2, existing_output);

  EXPECT_EQ(cost.output_layout.indexes, (IndexVec<U1QN>{free_axis}));
  EXPECT_EQ(cost.output_raw_elem_count, 3U);
}

}  // namespace
