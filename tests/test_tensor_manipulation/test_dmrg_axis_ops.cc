// SPDX-License-Identifier: LGPL-3.0-only
/*
* Description: Unittests for DMRG axis-local bosonic tensor operations.
*/

#define QLTEN_COUNT_FLOPS 1

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
  auto *data = bsdt.GetActualRawDataPtr();
  for (const auto &entry : bsdt.GetBlkIdxDataBlkMap()) {
    const auto &block = entry.second;
    for (size_t i = 0; i < block.size; ++i) {
      data[block.data_offset + i] =
          base + static_cast<double>(entry.first + 1) * 0.25 +
          static_cast<double>(i + 1) * 0.03125;
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
