// SPDX-License-Identifier: LGPL-3.0-only
/*
* Description: Unittests for DMRG axis-local bosonic tensor operations.
*/

#include "gtest/gtest.h"
#include "qlten/qltensor_all.h"
#include "qlten/tensor_manipulation/dmrg/axis_ops.h"

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
