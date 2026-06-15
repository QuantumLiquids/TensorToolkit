// SPDX-License-Identifier: LGPL-3.0-only
/*
* Description: Unittests for DMRG axis-local bosonic tensor operations.
*/

#define QLTEN_COUNT_FLOPS 1

#include <functional>  // multiplies
#include <numeric>     // accumulate

#include "gtest/gtest.h"
#include "qlten/qltensor_all.h"
#include "qlten/tensor_manipulation/dmrg/axis_ops.h"
#include "qlten/tensor_manipulation/ten_ctrct.h"

#include "../testing_utility.h"

using namespace qlten;
using special_qn::U1QN;

namespace {

using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using Tensor = QLTensor<QLTEN_Double, U1QN>;

struct DmrgAxisOpsTest : public testing::Test {
  U1QN qn0 = U1QN(0);

  QNSctT bond_sct = QNSctT(qn0, 2);
  QNSctT phys_in_sct0 = QNSctT(qn0, 2);
  QNSctT phys_in_sct1 = QNSctT(qn0, 3);
  QNSctT phys_out_sct0 = QNSctT(qn0, 3);
  QNSctT phys_out_sct1 = QNSctT(qn0, 2);

  IndexT left = IndexT({bond_sct}, TenIndexDirType::IN);
  IndexT right = IndexT({bond_sct}, TenIndexDirType::OUT);
  IndexT phys_in = IndexT({phys_in_sct0, phys_in_sct1}, TenIndexDirType::OUT);
  IndexT phys_out = IndexT({phys_out_sct0, phys_out_sct1}, TenIndexDirType::OUT);

  Tensor MakeInput() {
    Tensor tensor({left, phys_in, right});
    tensor.Fill(qn0, 0.0);
    for (size_t l = 0; l < left.dim(); ++l) {
      for (size_t p = 0; p < phys_in.dim(); ++p) {
        for (size_t r = 0; r < right.dim(); ++r) {
          tensor.SetElem({l, p, r}, 100.0 * l + 10.0 * p + r + 1.0);
        }
      }
    }
    return tensor;
  }
};

using Rank4Tensor = QLTensor<QLTEN_Double, U1QN>;

void FillStoredBlocks(Rank4Tensor &tensor, const double base) {
  auto &bsdt = tensor.GetBlkSparDataTen();
  for (const auto &entry : bsdt.GetBlkIdxDataBlkMap()) {
    const auto &block = entry.second;
    CoorsT sector_offsets(block.shape.size(), 0);
    for (size_t axis = 0; axis < block.shape.size(); ++axis) {
      const auto &index = tensor.GetIndex(axis);
      for (size_t sector = 0; sector < block.blk_coors[axis]; ++sector) {
        sector_offsets[axis] += index.GetQNSct(sector).dim();
      }
    }
    size_t local_linear_index = 0;
    for (const auto &local_coors : GenAllCoors(block.shape)) {
      CoorsT coors(local_coors.size());
      for (size_t axis = 0; axis < local_coors.size(); ++axis) {
        coors[axis] = sector_offsets[axis] + local_coors[axis];
      }
      tensor.SetElem(
          coors,
          base + static_cast<double>(entry.first + 1) * 0.25 +
          static_cast<double>(local_linear_index + 1) * 0.03125
      );
      ++local_linear_index;
    }
  }
}

std::vector<size_t> ReferenceTransposeOrder(
    const size_t rank,
    const size_t target_axis
) {
  std::vector<size_t> order;
  order.reserve(rank);
  for (size_t axis = 0; axis < target_axis; ++axis) {
    order.push_back(axis + 1);
  }
  order.push_back(0);
  for (size_t axis = target_axis + 1; axis < rank; ++axis) {
    order.push_back(axis);
  }
  return order;
}

void ExpectTensorElementsNear(
    const Rank4Tensor &actual,
    const Rank4Tensor &expected,
    const size_t target_axis
) {
  ASSERT_EQ(actual.GetIndexes(), expected.GetIndexes());
  for (const auto &coors : GenAllCoors(expected.GetShape())) {
    EXPECT_NEAR(actual.GetElem(coors), expected.GetElem(coors), 1e-12)
        << "target_axis=" << target_axis;
  }
}

Rank4Tensor MakeNontrivialRank4Input(const U1QN &div) {
  const auto qnm1 = U1QN(-1);
  const auto qnp1 = U1QN(1);
  const IndexT state_index(
      {QNSctT(qnm1, 1), QNSctT(div, 2), QNSctT(qnp1, 1)},
      TenIndexDirType::OUT);
  Rank4Tensor input({state_index, state_index, state_index, state_index});
  input.Fill(div, 0.0);
  FillStoredBlocks(input, 10.0);
  return input;
}

std::vector<size_t> MoveHeadToTailOrder(const size_t rank) {
  std::vector<size_t> order;
  order.reserve(rank);
  for (size_t axis = 1; axis < rank; ++axis) {
    order.push_back(axis);
  }
  order.push_back(0);
  return order;
}

size_t ExpectedRank2GemmFlops(
    const Rank4Tensor &input,
    const Rank4Tensor &rank2_op,
    const size_t target_axis
) {
  size_t expected = 0;
  const auto &input_blocks =
      input.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  const auto &op_blocks =
      rank2_op.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  for (const auto &input_entry : input_blocks) {
    const auto &input_blk = input_entry.second;
    const size_t inner_size =
        std::accumulate(input_blk.shape.begin() + target_axis + 1,
                        input_blk.shape.end(),
                        size_t(1),
                        std::multiplies<size_t>());
    const size_t outer_size =
        std::accumulate(input_blk.shape.begin(),
                        input_blk.shape.begin() + target_axis,
                        size_t(1),
                        std::multiplies<size_t>());
    const size_t input_axis_dim = input_blk.shape[target_axis];
    for (const auto &op_entry : op_blocks) {
      const auto &op_blk = op_entry.second;
      if (op_blk.blk_coors[0] != input_blk.blk_coors[target_axis]) {
        continue;
      }
      const size_t output_axis_dim = op_blk.shape[1];
      expected += outer_size * output_axis_dim * inner_size *
                  (2 * input_axis_dim + 2);
    }
  }
  return expected;
}

size_t ExpectedRank2BlockPairCount(
    const Rank4Tensor &input,
    const Rank4Tensor &rank2_op,
    const size_t target_axis
) {
  size_t expected = 0;
  const auto &input_blocks =
      input.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  const auto &op_blocks =
      rank2_op.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  for (const auto &input_entry : input_blocks) {
    const auto &input_blk = input_entry.second;
    for (const auto &op_entry : op_blocks) {
      const auto &op_blk = op_entry.second;
      if (op_blk.blk_coors[0] == input_blk.blk_coors[target_axis]) {
        ++expected;
      }
    }
  }
  return expected;
}

TEST_F(DmrgAxisOpsTest, ApplyAxisDiagonalDefaultOutputScalesEachDegeneracy) {
  auto input = MakeInput();
  dmrg::AxisDiagonalOp<QLTEN_Double, U1QN> op{
      phys_in,
      {{2.0, -1.0}, {0.5, 3.0, -4.0}}
  };

  Tensor output;
  dmrg::ApplyAxisDiagonal(input, 1, op, output, 1.5);

  ASSERT_EQ(output.GetIndexes(), input.GetIndexes());
  for (size_t l = 0; l < left.dim(); ++l) {
    for (size_t p = 0; p < phys_in.dim(); ++p) {
      const auto sct_offset = phys_in.CoorToBlkCoorDataCoor(p);
      const double scale = 1.5 * op.values[sct_offset.first][sct_offset.second];
      for (size_t r = 0; r < right.dim(); ++r) {
        EXPECT_DOUBLE_EQ(output.GetElem({l, p, r}),
                         scale * input.GetElem({l, p, r}));
      }
    }
  }
}

TEST_F(DmrgAxisOpsTest, ApplyAxisDiagonalAccumulatesIntoExistingOutput) {
  auto input = MakeInput();
  Tensor output({left, phys_in, right});
  output.Fill(qn0, 5.0);
  dmrg::AxisDiagonalOp<QLTEN_Double, U1QN> op{
      phys_in,
      {{2.0, -1.0}, {0.5, 3.0, -4.0}}
  };

  dmrg::ApplyAxisDiagonal(input, 1, op, output, 2.0, 3.0);

  for (size_t l = 0; l < left.dim(); ++l) {
    for (size_t p = 0; p < phys_in.dim(); ++p) {
      const auto sct_offset = phys_in.CoorToBlkCoorDataCoor(p);
      const double scale = op.values[sct_offset.first][sct_offset.second];
      for (size_t r = 0; r < right.dim(); ++r) {
        EXPECT_DOUBLE_EQ(output.GetElem({l, p, r}),
                         15.0 + 2.0 * scale * input.GetElem({l, p, r}));
      }
    }
  }
}

TEST_F(DmrgAxisOpsTest, ApplyAxisDiagonalIncrementsFlopCount) {
  auto input = MakeInput();
  dmrg::AxisDiagonalOp<QLTEN_Double, U1QN> op{
      phys_in,
      {{2.0, -1.0}, {0.5, 3.0, -4.0}}
  };

  Tensor output;
  flop = 0;
  dmrg::ApplyAxisDiagonal(input, 1, op, output, 1.5);

  EXPECT_EQ(flop, 40);
}

TEST_F(DmrgAxisOpsTest, ApplyAxisDiagonalInPlacePreservesTopology) {
  auto tensor = MakeInput();
  const auto input_blocks = tensor.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  dmrg::AxisDiagonalOp<QLTEN_Double, U1QN> op{
      phys_in,
      {{0.0, -1.0}, {0.5, 3.0, -4.0}}
  };
  const auto before = tensor;

  dmrg::ApplyAxisDiagonalInPlace(tensor, 1, op);

  EXPECT_EQ(tensor.GetBlkSparDataTen().GetBlkIdxDataBlkMap().size(),
            input_blocks.size());
  for (size_t l = 0; l < left.dim(); ++l) {
    for (size_t p = 0; p < phys_in.dim(); ++p) {
      const auto sct_offset = phys_in.CoorToBlkCoorDataCoor(p);
      const double scale = op.values[sct_offset.first][sct_offset.second];
      for (size_t r = 0; r < right.dim(); ++r) {
        EXPECT_DOUBLE_EQ(tensor.GetElem({l, p, r}),
                         scale * before.GetElem({l, p, r}));
      }
    }
  }
}

TEST_F(DmrgAxisOpsTest, ScaleAxisSectorsInPlaceScalesWholeBlocks) {
  auto tensor = MakeInput();
  const auto before = tensor;
  dmrg::AxisSectorScalar<QLTEN_Double, U1QN> scale{1, {2.0, -3.0}};

  dmrg::ScaleAxisSectorsInPlace(tensor, scale);

  for (size_t l = 0; l < left.dim(); ++l) {
    for (size_t p = 0; p < phys_in.dim(); ++p) {
      const auto sct_offset = phys_in.CoorToBlkCoorDataCoor(p);
      for (size_t r = 0; r < right.dim(); ++r) {
        EXPECT_DOUBLE_EQ(tensor.GetElem({l, p, r}),
                         scale.values_by_sector[sct_offset.first] *
                             before.GetElem({l, p, r}));
      }
    }
  }
}

TEST_F(DmrgAxisOpsTest, ApplyAxisMonomialScattersAndAccumulatesEntries) {
  auto input = MakeInput();
  dmrg::AxisMonomialOp<QLTEN_Double, U1QN> op{
      phys_in,
      phys_out,
      qn0,
      {
          {0, 0, 1, 1, 5.0},
          {0, 1, 1, 1, 7.0},
          {1, 0, 0, 2, 3.0},
          {1, 2, 0, 0, -2.0},
      }
  };

  Tensor output;
  dmrg::ApplyAxisMonomial(input, 1, op, output, 1.5);

  ASSERT_EQ(output.GetIndex(0), left);
  ASSERT_EQ(output.GetIndex(1), phys_out);
  ASSERT_EQ(output.GetIndex(2), right);

  for (size_t l = 0; l < left.dim(); ++l) {
    for (size_t r = 0; r < right.dim(); ++r) {
      EXPECT_DOUBLE_EQ(output.GetElem({l, 0, r}),
                       1.5 * -2.0 * input.GetElem({l, 4, r}));
      EXPECT_DOUBLE_EQ(output.GetElem({l, 1, r}), 0.0);
      EXPECT_DOUBLE_EQ(output.GetElem({l, 2, r}),
                       1.5 * 3.0 * input.GetElem({l, 2, r}));
      EXPECT_DOUBLE_EQ(output.GetElem({l, 3, r}), 0.0);
      EXPECT_DOUBLE_EQ(output.GetElem({l, 4, r}),
                       1.5 * (5.0 * input.GetElem({l, 0, r}) +
                              7.0 * input.GetElem({l, 1, r})));
    }
  }
}

TEST_F(DmrgAxisOpsTest, ApplyAxisMonomialIncrementsFlopCount) {
  auto input = MakeInput();
  dmrg::AxisMonomialOp<QLTEN_Double, U1QN> op{
      phys_in,
      phys_out,
      qn0,
      {
          {0, 0, 1, 1, 5.0},
          {0, 1, 1, 1, 7.0},
          {1, 0, 0, 2, 3.0},
          {1, 2, 0, 0, -2.0},
      }
  };

  Tensor output;
  flop = 0;
  dmrg::ApplyAxisMonomial(input, 1, op, output, 1.5);

  EXPECT_EQ(flop, 32);
}

TEST_F(DmrgAxisOpsTest, ApplyRank2ToAxisPreserveOrderMatchesDenseReference) {
  const auto qnm1 = U1QN(-1);
  const auto qnp1 = U1QN(1);
  const IndexT state_index(
      {QNSctT(qnm1, 1), QNSctT(qn0, 2), QNSctT(qnp1, 1)},
      TenIndexDirType::OUT);
  const IndexT op_output_index(
      {QNSctT(qnm1, 2), QNSctT(qn0, 1), QNSctT(qnp1, 2)},
      TenIndexDirType::OUT);
  Rank4Tensor input({state_index, state_index, state_index, state_index});
  input.Fill(qn0, 0.0);
  FillStoredBlocks(input, 10.0);

  Rank4Tensor rank2_op({InverseIndex(state_index), op_output_index});
  rank2_op.Fill(qn0, 0.0);
  FillStoredBlocks(rank2_op, 0.5);

  for (size_t target_axis = 0; target_axis < input.Rank(); ++target_axis) {
    Rank4Tensor output;
    dmrg::Rank2AxisApplyStats stats;
    dmrg::ApplyRank2ToAxisPreserveOrder(
        input, rank2_op, target_axis, output, &stats);

    auto expected_indexes = input.GetIndexes();
    expected_indexes[target_axis] = op_output_index;
    EXPECT_EQ(output.GetIndexes(), expected_indexes);
    EXPECT_EQ(stats.layout_copy_calls, 0U);
    EXPECT_EQ(stats.layout_transpose_calls, 0U);
    EXPECT_DOUBLE_EQ(stats.layout_gb, 0.0);

    Rank4Tensor reference;
    Contract(&rank2_op, &input, {{0}, {target_axis}}, &reference);
    reference.Transpose(ReferenceTransposeOrder(input.Rank(), target_axis));
    ExpectTensorElementsNear(output, reference, target_axis);
  }
}

TEST_F(DmrgAxisOpsTest, ApplyRank2ToAxisPreserveOrderUsesGemmFlopAccounting) {
  const auto qnm1 = U1QN(-1);
  const auto qnp1 = U1QN(1);
  const IndexT state_index(
      {QNSctT(qnm1, 1), QNSctT(qn0, 2), QNSctT(qnp1, 1)},
      TenIndexDirType::OUT);
  const IndexT op_output_index(
      {QNSctT(qnm1, 2), QNSctT(qn0, 1), QNSctT(qnp1, 2)},
      TenIndexDirType::OUT);
  Rank4Tensor input({state_index, state_index, state_index, state_index});
  input.Fill(qn0, 0.0);
  FillStoredBlocks(input, 10.0);

  Rank4Tensor rank2_op({InverseIndex(state_index), op_output_index});
  rank2_op.Fill(qn0, 0.0);
  FillStoredBlocks(rank2_op, 0.5);

  Rank4Tensor output;
  flop = 0;
  dmrg::ApplyRank2ToAxisPreserveOrder(input, rank2_op, 2, output);

  EXPECT_EQ(flop, ExpectedRank2GemmFlops(input, rank2_op, 2));
}

TEST_F(DmrgAxisOpsTest, ApplyRank2ToBoundaryAxesReportsGemmFastPathStats) {
  const auto qnm1 = U1QN(-1);
  const auto qnp1 = U1QN(1);
  const IndexT state_index(
      {QNSctT(qnm1, 1), QNSctT(qn0, 2), QNSctT(qnp1, 1)},
      TenIndexDirType::OUT);
  const IndexT op_output_index(
      {QNSctT(qnm1, 2), QNSctT(qn0, 1), QNSctT(qnp1, 2)},
      TenIndexDirType::OUT);
  Rank4Tensor input({state_index, state_index, state_index, state_index});
  input.Fill(qn0, 0.0);
  FillStoredBlocks(input, 10.0);

  Rank4Tensor rank2_op({InverseIndex(state_index), op_output_index});
  rank2_op.Fill(qn0, 0.0);
  FillStoredBlocks(rank2_op, 0.5);

  for (const size_t target_axis : {size_t(0), input.Rank() - 1}) {
    Rank4Tensor output;
    dmrg::AxisOpStats stats;
    dmrg::ApplyRank2ToAxisPreserveOrder(
        input, rank2_op, target_axis, output, &stats);

    EXPECT_EQ(stats.output_tensor_rebuilds, 1U);
    EXPECT_EQ(stats.gemm_calls,
              ExpectedRank2BlockPairCount(input, rank2_op, target_axis));
    EXPECT_EQ(stats.boundary_axis_fast_path_hits, stats.gemm_calls);
    EXPECT_EQ(stats.rank2_topology_block_pair_visits, stats.gemm_calls);
    EXPECT_GT(stats.rank2_sector_index_hits, 0U);
    EXPECT_EQ(stats.layout_copy_calls, 0U);
    EXPECT_EQ(stats.layout_transpose_calls, 0U);
    EXPECT_EQ(stats.transpose_prepare_calls, 0U);
    EXPECT_EQ(stats.raw_data_copy_bytes, 0U);
  }
}

TEST_F(DmrgAxisOpsTest, ApplyRank2ToMiddleAxisCanUseBatchGemmStrategy) {
  const auto qnm1 = U1QN(-1);
  const auto qnp1 = U1QN(1);
  const IndexT state_index(
      {QNSctT(qnm1, 1), QNSctT(qn0, 2), QNSctT(qnp1, 1)},
      TenIndexDirType::OUT);
  const IndexT op_output_index(
      {QNSctT(qnm1, 2), QNSctT(qn0, 1), QNSctT(qnp1, 2)},
      TenIndexDirType::OUT);
  Rank4Tensor input({state_index, state_index, state_index, state_index});
  input.Fill(qn0, 0.0);
  FillStoredBlocks(input, 10.0);

  Rank4Tensor rank2_op({InverseIndex(state_index), op_output_index});
  rank2_op.Fill(qn0, 0.0);
  FillStoredBlocks(rank2_op, 0.5);

  Rank4Tensor loop_output;
  dmrg::AxisOpStats loop_stats;
  dmrg::ApplyRank2ToAxisPreserveOrder(
      input, rank2_op, 2, loop_output, &loop_stats);

  Rank4Tensor batch_output;
  dmrg::AxisOpStats batch_stats;
  dmrg::ApplyRank2ToAxisPreserveOrder<
      dmrg::Rank2AxisApplyGemmMode::kBatch>(
      input, rank2_op, 2, batch_output, &batch_stats);

  ExpectTensorElementsNear(batch_output, loop_output, 2);
  EXPECT_EQ(loop_stats.batch_gemm_calls, 0U);
  EXPECT_GT(batch_stats.batch_gemm_calls, 0U);
  EXPECT_EQ(batch_stats.batched_gemm_items, loop_stats.gemm_calls);
  EXPECT_EQ(batch_stats.gemm_calls, loop_stats.gemm_calls);
  EXPECT_EQ(batch_stats.boundary_axis_fast_path_hits, 0U);
  EXPECT_EQ(batch_stats.layout_copy_calls, 0U);
  EXPECT_EQ(batch_stats.layout_transpose_calls, 0U);
#ifdef USE_GPU
  EXPECT_EQ(batch_stats.batch_gemm_fallback_calls, 0U);
  EXPECT_EQ(batch_stats.batch_gemm_workspace_bytes, 0U);
#endif
}

TEST_F(DmrgAxisOpsTest, ApplyTwoRank2ToAxesPreserveOrderMatchesSequential) {
  const auto qnm1 = U1QN(-1);
  const auto qnp1 = U1QN(1);
  const IndexT state_index(
      {QNSctT(qnm1, 1), QNSctT(qn0, 2), QNSctT(qnp1, 1)},
      TenIndexDirType::OUT);
  const IndexT op1_output_index(
      {QNSctT(qnm1, 2), QNSctT(qn0, 1), QNSctT(qnp1, 2)},
      TenIndexDirType::OUT);
  const IndexT op2_output_index(
      {QNSctT(qnm1, 1), QNSctT(qn0, 3), QNSctT(qnp1, 1)},
      TenIndexDirType::OUT);
  Rank4Tensor input({state_index, state_index, state_index, state_index});
  input.Fill(qn0, 0.0);
  FillStoredBlocks(input, 10.0);

  Rank4Tensor op1({InverseIndex(state_index), op1_output_index});
  op1.Fill(qn0, 0.0);
  FillStoredBlocks(op1, 0.5);
  Rank4Tensor op2({InverseIndex(state_index), op2_output_index});
  op2.Fill(qn0, 0.0);
  FillStoredBlocks(op2, 1.5);

  Rank4Tensor first_output;
  Rank4Tensor reference;
  dmrg::ApplyRank2ToAxisPreserveOrder(input, op1, 0, first_output);
  dmrg::ApplyRank2ToAxisPreserveOrder(first_output, op2, 1, reference);

  Rank4Tensor output;
  dmrg::AxisOpStats stats;
  dmrg::ApplyTwoRank2ToAxesPreserveOrder(
      input, op1, 0, op2, 1, output, &stats);

  ExpectTensorElementsNear(output, reference, 0);
  EXPECT_EQ(output.GetIndex(0), op1_output_index);
  EXPECT_EQ(output.GetIndex(1), op2_output_index);
  EXPECT_EQ(output.GetIndex(2), state_index);
  EXPECT_EQ(output.GetIndex(3), state_index);
#ifdef QLTEN_DMRG_ENABLE_SCALAR_TWO_RANK2_FALLBACK
  EXPECT_EQ(stats.two_rank2_block_gemm_hits, 0U);
  EXPECT_EQ(stats.two_rank2_block_workspace_bytes, 0U);
  EXPECT_EQ(stats.two_rank2_fused_hits, 1U);
  EXPECT_GT(stats.fused_element_updates, 0U);
  EXPECT_EQ(stats.gemm_calls, 0U);
#else
  EXPECT_EQ(stats.two_rank2_block_gemm_hits, 1U);
  EXPECT_GT(stats.two_rank2_block_workspace_bytes, 0U);
  EXPECT_EQ(stats.two_rank2_fused_hits, 0U);
  EXPECT_EQ(stats.fused_element_updates, 0U);
  EXPECT_GT(stats.gemm_calls, 0U);
#endif
  EXPECT_EQ(stats.two_rank2_intermediate_rebuilds, 0U);
  EXPECT_EQ(stats.output_tensor_rebuilds, 1U);
  EXPECT_EQ(stats.layout_transpose_calls, 0U);
}

TEST_F(DmrgAxisOpsTest, ApplyAxisDiagonalPreserveOrderCanConsumeOwnedTensor) {
  auto input = MakeInput();
  dmrg::AxisDiagonalOp<QLTEN_Double, U1QN> op{
      phys_in,
      {{0.0, -1.0}, {0.5, 3.0, -4.0}}
  };

  Tensor out_of_place;
  dmrg::ApplyAxisDiagonalPreserveOrder(
      input, 1, op, out_of_place, dmrg::AxisApplyStorageMode::kOutOfPlace);

  Tensor owned = input;
  dmrg::AxisOpStats stats;
  dmrg::ApplyAxisDiagonalPreserveOrder(
      owned, 1, op, owned,
      dmrg::AxisApplyStorageMode::kConsumeOwnedInPlace, &stats);

  ASSERT_EQ(owned.GetIndexes(), out_of_place.GetIndexes());
  EXPECT_EQ(stats.in_place_scale_hits, 1U);
  EXPECT_EQ(stats.output_tensor_rebuilds, 0U);
  EXPECT_EQ(stats.raw_data_copy_bytes, 0U);
  for (const auto &coors : GenAllCoors(out_of_place.GetShape())) {
    EXPECT_DOUBLE_EQ(owned.GetElem(coors), out_of_place.GetElem(coors));
  }
}

TEST_F(DmrgAxisOpsTest, ApplyAxisSectorScalarsPreserveOrderCopyAndConsume) {
  auto input = MakeInput();
  const auto input_blocks = input.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  dmrg::AxisSectorScalar<QLTEN_Double, U1QN> scale{1, {2.0, -3.0}};

  Tensor copied;
  dmrg::AxisOpStats copied_stats;
  dmrg::ApplyAxisSectorScalarsPreserveOrder(
      input, scale, copied, dmrg::AxisApplyStorageMode::kOutOfPlace,
      &copied_stats);

  Tensor owned = input;
  dmrg::AxisOpStats owned_stats;
  dmrg::ApplyAxisSectorScalarsPreserveOrder(
      owned, scale, owned, dmrg::AxisApplyStorageMode::kConsumeOwnedInPlace,
      &owned_stats);

  ASSERT_EQ(copied.GetIndexes(), input.GetIndexes());
  ASSERT_EQ(owned.GetIndexes(), input.GetIndexes());
  EXPECT_EQ(copied.GetBlkSparDataTen().GetBlkIdxDataBlkMap().size(),
            input_blocks.size());
  EXPECT_EQ(owned.GetBlkSparDataTen().GetBlkIdxDataBlkMap().size(),
            input_blocks.size());
  EXPECT_EQ(copied_stats.output_tensor_rebuilds, 1U);
  EXPECT_EQ(copied_stats.raw_data_copy_bytes,
            input.GetBlkSparDataTen().GetActualRawDataSize() *
                sizeof(QLTEN_Double));
  EXPECT_GT(copied_stats.direct_scaled_copy_calls, 0U);
  EXPECT_EQ(copied_stats.in_place_scale_hits, 0U);
  EXPECT_EQ(owned_stats.output_tensor_rebuilds, 0U);
  EXPECT_EQ(owned_stats.raw_data_copy_bytes, 0U);
  EXPECT_EQ(owned_stats.in_place_scale_hits, 1U);
  for (const auto &coors : GenAllCoors(copied.GetShape())) {
    EXPECT_DOUBLE_EQ(owned.GetElem(coors), copied.GetElem(coors));
  }

  Tensor empty(input.GetIndexes());
  Tensor empty_output;
  dmrg::AxisOpStats empty_stats;
  dmrg::ApplyAxisSectorScalarsPreserveOrder(
      empty, scale, empty_output, dmrg::AxisApplyStorageMode::kOutOfPlace,
      &empty_stats);
  EXPECT_EQ(empty_output.GetIndexes(), empty.GetIndexes());
  EXPECT_TRUE(empty_output.GetBlkSparDataTen().GetBlkIdxDataBlkMap().empty());
  EXPECT_EQ(empty_stats.raw_data_copy_bytes, 0U);
}

TEST_F(DmrgAxisOpsTest, ApplyRank2ThenDiagonalConsumesOwnedOutput) {
  const auto qnm1 = U1QN(-1);
  const auto qnp1 = U1QN(1);
  const IndexT state_index(
      {QNSctT(qnm1, 1), QNSctT(qn0, 2), QNSctT(qnp1, 1)},
      TenIndexDirType::OUT);
  const IndexT op_output_index(
      {QNSctT(qnm1, 2), QNSctT(qn0, 1), QNSctT(qnp1, 2)},
      TenIndexDirType::OUT);
  Rank4Tensor input({state_index, state_index, state_index, state_index});
  input.Fill(qn0, 0.0);
  FillStoredBlocks(input, 10.0);

  Rank4Tensor rank2_op({InverseIndex(state_index), op_output_index});
  rank2_op.Fill(qn0, 0.0);
  FillStoredBlocks(rank2_op, 0.5);
  dmrg::AxisDiagonalOp<QLTEN_Double, U1QN> diag{
      state_index,
      {{2.0}, {-1.0, 0.5}, {3.0}}
  };

  Rank4Tensor reference;
  dmrg::ApplyRank2ToAxisPreserveOrder(input, rank2_op, 0, reference);
  dmrg::ApplyAxisDiagonalInPlace(reference, 1, diag);

  Rank4Tensor output;
  dmrg::AxisOpStats stats;
  dmrg::ApplyRank2ThenAxisDiagonalPreserveOrder(
      input, rank2_op, 0, 1, diag, output, &stats);

  ExpectTensorElementsNear(output, reference, 0);
  EXPECT_EQ(stats.output_tensor_rebuilds, 1U);
  EXPECT_EQ(stats.in_place_scale_hits, 1U);
  EXPECT_EQ(stats.raw_data_copy_bytes, 0U);
}

TEST_F(DmrgAxisOpsTest, ApplyRank2ThenSectorScalarsConsumesOwnedOutput) {
  const auto qnm1 = U1QN(-1);
  const auto qnp1 = U1QN(1);
  const IndexT state_index(
      {QNSctT(qnm1, 1), QNSctT(qn0, 2), QNSctT(qnp1, 1)},
      TenIndexDirType::OUT);
  const IndexT op_output_index(
      {QNSctT(qnm1, 2), QNSctT(qn0, 1), QNSctT(qnp1, 2)},
      TenIndexDirType::OUT);
  Rank4Tensor input({state_index, state_index, state_index, state_index});
  input.Fill(qn0, 0.0);
  FillStoredBlocks(input, 10.0);

  Rank4Tensor rank2_op({InverseIndex(state_index), op_output_index});
  rank2_op.Fill(qn0, 0.0);
  FillStoredBlocks(rank2_op, 0.5);
  dmrg::AxisSectorScalar<QLTEN_Double, U1QN> scale{1, {2.0, -3.0, 0.5}};

  Rank4Tensor reference;
  dmrg::ApplyRank2ToAxisPreserveOrder(input, rank2_op, 0, reference);
  dmrg::ScaleAxisSectorsInPlace(reference, scale);

  Rank4Tensor output;
  dmrg::AxisOpStats stats;
  dmrg::ApplyRank2ThenAxisSectorScalarsPreserveOrder(
      input, rank2_op, 0, scale, output, &stats);

  ExpectTensorElementsNear(output, reference, 0);
  EXPECT_EQ(stats.output_tensor_rebuilds, 1U);
  EXPECT_EQ(stats.in_place_scale_hits, 1U);
  EXPECT_EQ(stats.raw_data_copy_bytes, 0U);
}

TEST_F(DmrgAxisOpsTest, ApplyTwoRank2ThenSectorScalarsConsumesOwnedOutput) {
  const auto qnm1 = U1QN(-1);
  const auto qnp1 = U1QN(1);
  const IndexT state_index(
      {QNSctT(qnm1, 1), QNSctT(qn0, 2), QNSctT(qnp1, 1)},
      TenIndexDirType::OUT);
  const IndexT op1_output_index(
      {QNSctT(qnm1, 2), QNSctT(qn0, 1), QNSctT(qnp1, 2)},
      TenIndexDirType::OUT);
  const IndexT op2_output_index(
      {QNSctT(qnm1, 1), QNSctT(qn0, 3), QNSctT(qnp1, 1)},
      TenIndexDirType::OUT);
  Rank4Tensor input({state_index, state_index, state_index, state_index});
  input.Fill(qn0, 0.0);
  FillStoredBlocks(input, 10.0);

  Rank4Tensor op1({InverseIndex(state_index), op1_output_index});
  op1.Fill(qn0, 0.0);
  FillStoredBlocks(op1, 0.5);
  Rank4Tensor op2({InverseIndex(state_index), op2_output_index});
  op2.Fill(qn0, 0.0);
  FillStoredBlocks(op2, 1.5);
  dmrg::AxisSectorScalar<QLTEN_Double, U1QN> scale{2, {2.0, -3.0, 0.5}};

  Rank4Tensor reference;
  dmrg::ApplyTwoRank2ToAxesPreserveOrder(input, op1, 0, op2, 1, reference);
  dmrg::ScaleAxisSectorsInPlace(reference, scale);

  Rank4Tensor output;
  dmrg::AxisOpStats stats;
  dmrg::ApplyTwoRank2ThenAxisSectorScalarsPreserveOrder(
      input, op1, 0, op2, 1, scale, output, &stats);

  ExpectTensorElementsNear(output, reference, 0);
  EXPECT_EQ(stats.output_tensor_rebuilds, 1U);
#ifdef QLTEN_DMRG_ENABLE_SCALAR_TWO_RANK2_FALLBACK
  EXPECT_EQ(stats.two_rank2_block_gemm_hits, 0U);
  EXPECT_EQ(stats.two_rank2_block_workspace_bytes, 0U);
  EXPECT_EQ(stats.two_rank2_fused_hits, 1U);
  EXPECT_EQ(stats.gemm_calls, 0U);
#else
  EXPECT_EQ(stats.two_rank2_block_gemm_hits, 1U);
  EXPECT_GT(stats.two_rank2_block_workspace_bytes, 0U);
  EXPECT_EQ(stats.two_rank2_fused_hits, 0U);
  EXPECT_GT(stats.gemm_calls, 0U);
#endif
  EXPECT_EQ(stats.two_rank2_intermediate_rebuilds, 0U);
  EXPECT_EQ(stats.in_place_scale_hits, 1U);
  EXPECT_EQ(stats.raw_data_copy_bytes, 0U);
}

TEST_F(DmrgAxisOpsTest, MoveHeadAxisToTailMatchesTensorTranspose) {
  const auto input = MakeNontrivialRank4Input(qn0);
  const auto block_count =
      input.GetBlkSparDataTen().GetBlkIdxDataBlkMap().size();
  const auto raw_size = input.GetBlkSparDataTen().GetActualRawDataSize();

  Tensor output;
  dmrg::AxisLayoutMoveStats stats;
  flop = 123;
  dmrg::MoveHeadAxisToTail(input, output, &stats);

  auto reference = input;
  reference.Transpose(MoveHeadToTailOrder(input.Rank()));

  ExpectTensorElementsNear(output, reference, 0);
  EXPECT_EQ(stats.layout_transpose_calls, 0U);
  EXPECT_EQ(stats.block_matrix_transpose_calls, block_count);
  EXPECT_DOUBLE_EQ(
      stats.layout_gb,
      2.0 * static_cast<double>(raw_size * sizeof(QLTEN_Double)) / 1.0e9);
  EXPECT_EQ(flop, 123U);
}

TEST_F(DmrgAxisOpsTest, MoveHeadAxisToTailRejectsInvalidInputs) {
  Tensor output;
  const Tensor default_input;
  EXPECT_THROW(dmrg::MoveHeadAxisToTail(default_input, output),
               std::invalid_argument);

  Tensor scalar(IndexVec<U1QN>{});
  scalar() = 1.0;
  EXPECT_THROW(dmrg::MoveHeadAxisToTail(scalar, output),
               std::invalid_argument);

  auto input = MakeNontrivialRank4Input(qn0);
  EXPECT_THROW(dmrg::MoveHeadAxisToTail(input, input),
               std::invalid_argument);
}

TEST_F(DmrgAxisOpsTest, ProjectAxisCompactsSectorsAndReordersDegeneracies) {
  auto input = MakeInput();
  dmrg::AxisProjector<U1QN> projector{
      phys_in,
      {{1}, {2, 0}}
  };

  Tensor output;
  dmrg::ProjectAxis(input, 1, projector, output);

  ASSERT_EQ(output.GetIndex(0), left);
  ASSERT_EQ(output.GetIndex(2), right);
  EXPECT_EQ(output.GetIndex(1).GetDir(), phys_in.GetDir());
  ASSERT_EQ(output.GetIndex(1).GetQNSctNum(), 2);
  EXPECT_EQ(output.GetIndex(1).GetQNSct(0).GetDegeneracy(), 1);
  EXPECT_EQ(output.GetIndex(1).GetQNSct(1).GetDegeneracy(), 2);

  const auto &output_blocks = output.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  ASSERT_EQ(output_blocks.size(), 2);
  auto block_it = output_blocks.begin();
  EXPECT_EQ(block_it->second.shape, (ShapeT{2, 1, 2}));
  ++block_it;
  EXPECT_EQ(block_it->second.shape, (ShapeT{2, 2, 2}));

  for (size_t l = 0; l < left.dim(); ++l) {
    for (size_t r = 0; r < right.dim(); ++r) {
      EXPECT_DOUBLE_EQ(output.GetElem({l, 0, r}), input.GetElem({l, 1, r}));
      EXPECT_DOUBLE_EQ(output.GetElem({l, 1, r}), input.GetElem({l, 4, r}));
      EXPECT_DOUBLE_EQ(output.GetElem({l, 2, r}), input.GetElem({l, 2, r}));
    }
  }
}

TEST_F(DmrgAxisOpsTest, ProjectAxisIncrementsFlopCount) {
  auto input = MakeInput();
  dmrg::AxisProjector<U1QN> projector{
      phys_in,
      {{1}, {2, 0}}
  };

  Tensor output;
  flop = 0;
  dmrg::ProjectAxis(input, 1, projector, output);

  EXPECT_EQ(flop, 12);
}

TEST_F(DmrgAxisOpsTest, ProjectAxisRejectsInvalidProjectors) {
  auto input = MakeInput();

  Tensor output;
  EXPECT_THROW(
      dmrg::ProjectAxis(input, 1, dmrg::AxisProjector<U1QN>{phys_in, {{0}}},
                        output),
      std::invalid_argument);
  EXPECT_THROW(
      dmrg::ProjectAxis(input, 1, dmrg::AxisProjector<U1QN>{phys_in, {{}, {}}},
                        output),
      std::invalid_argument);
  EXPECT_THROW(
      dmrg::ProjectAxis(input, 1, dmrg::AxisProjector<U1QN>{phys_in, {{0, 0}, {}}},
                        output),
      std::invalid_argument);
}

TEST_F(DmrgAxisOpsTest, ProjectAxisRejectsInputOutputAlias) {
  auto tensor = MakeInput();
  dmrg::AxisProjector<U1QN> projector{
      phys_in,
      {{1}, {2, 0}}
  };

  EXPECT_THROW(dmrg::ProjectAxis(tensor, 1, projector, tensor),
               std::invalid_argument);
}

}  // namespace
