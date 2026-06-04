// SPDX-License-Identifier: LGPL-3.0-only
/*
* Description: Unittests for MPI QLTensor sum reductions.
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"

#include <vector>

using namespace qlten;

namespace {

using U1QN = special_qn::U1QN;
using DQLTensor = QLTensor<QLTEN_Double, U1QN>;
using ZQLTensor = QLTensor<QLTEN_Complex, U1QN>;

const std::string qn_nm = "qn";
const U1QN qnm1({QNCard(qn_nm, U1QNVal(-1))});
const U1QN qn0({QNCard(qn_nm, U1QNVal(0))});
const U1QN qnp1({QNCard(qn_nm, U1QNVal(1))});

IndexVec<U1QN> MakeIndexes() {
  const QNSectorVec<U1QN> qnscts{
      QNSector<U1QN>(qnm1, 2),
      QNSector<U1QN>(qn0, 2),
      QNSector<U1QN>(qnp1, 2)};
  return {
      Index<U1QN>(qnscts, TenIndexDirType::IN),
      Index<U1QN>(qnscts, TenIndexDirType::OUT)};
}

IndexVec<U1QN> MakeMismatchedIndexes() {
  const QNSectorVec<U1QN> qnscts{
      QNSector<U1QN>(qnm1, 2),
      QNSector<U1QN>(qn0, 3),
      QNSector<U1QN>(qnp1, 2)};
  return {
      Index<U1QN>(qnscts, TenIndexDirType::IN),
      Index<U1QN>(qnscts, TenIndexDirType::OUT)};
}

QLTEN_Double MakeValue(
    const QLTEN_Double base,
    const size_t row,
    const size_t col
) {
  return base + static_cast<QLTEN_Double>(10 * row + col);
}

QLTEN_Complex MakeValue(
    const QLTEN_Complex base,
    const size_t row,
    const size_t col
) {
  return base + QLTEN_Complex(
      static_cast<QLTEN_Double>(10 * row + col),
      -static_cast<QLTEN_Double>(row + 10 * col));
}

template<typename Tensor>
void AddBlockContribution(
    Tensor &tensor,
    const size_t sector,
    const typename Tensor::value_type base
) {
  const size_t begin = 2 * sector;
  for (size_t row = 0; row < 2; ++row) {
    for (size_t col = 0; col < 2; ++col) {
      const std::vector<size_t> coors{begin + row, begin + col};
      tensor.SetElem(coors, tensor.GetElem(coors) + MakeValue(base, row, col));
    }
  }
}

template<typename Tensor>
Tensor MakeTensorFromContributions(
    const std::vector<std::pair<size_t, typename Tensor::value_type>> &contribs
) {
  Tensor tensor(MakeIndexes());
  for (const auto &contrib : contribs) {
    AddBlockContribution(tensor, contrib.first, contrib.second);
  }
  return tensor;
}

std::vector<std::pair<size_t, QLTEN_Double>> IdenticalDoubleContribs(
    const int rank
) {
  const QLTEN_Double r = static_cast<QLTEN_Double>(rank + 1);
  return {{0, r}, {2, 100.0 + r}};
}

std::vector<std::pair<size_t, QLTEN_Complex>> IdenticalComplexContribs(
    const int rank
) {
  const QLTEN_Double r = static_cast<QLTEN_Double>(rank + 1);
  return {
      {0, QLTEN_Complex(r, 10.0 + r)},
      {2, QLTEN_Complex(100.0 + r, -10.0 - r)}};
}

std::vector<std::pair<size_t, QLTEN_Double>> DisjointDoubleContribs(
    const int rank
) {
  return {{static_cast<size_t>(rank), 10.0 * static_cast<QLTEN_Double>(rank + 1)}};
}

std::vector<std::pair<size_t, QLTEN_Double>> PartialOverlapDoubleContribs(
    const int rank
) {
  switch (rank) {
    case 0:
      return {{0, 1.0}, {1, 10.0}};
    case 1:
      return {{1, 20.0}, {2, 200.0}};
    default:
      return {{0, 3.0}, {2, 300.0}};
  }
}

std::vector<std::pair<size_t, QLTEN_Double>> RootZeroDoubleContribs(
    const int rank
) {
  if (rank == hp_numeric::kMPIMasterRank) {
    return {};
  }
  return {{static_cast<size_t>(rank), 10.0 * static_cast<QLTEN_Double>(rank + 1)}};
}

std::vector<std::pair<size_t, QLTEN_Double>> ShiftedPartialOverlapDoubleContribs(
    const int rank
) {
  auto contribs = PartialOverlapDoubleContribs(rank);
  for (auto &contrib : contribs) {
    contrib.second += 1000.0;
  }
  return contribs;
}

std::vector<std::pair<size_t, QLTEN_Double>> SameRawSizeDifferentTopologyDoubleContribs(
    const int rank
) {
  switch (rank) {
    case 0:
      return {{1, 11.0}, {2, 22.0}};
    case 1:
      return {{0, 33.0}, {1, 44.0}};
    default:
      return {{1, 55.0}, {2, 66.0}};
  }
}

template<typename Tensor, typename ContributionFn>
Tensor MakeExpected(const int mpi_size, ContributionFn contrib_fn) {
  Tensor expected(MakeIndexes());
  for (int rank = 0; rank < mpi_size; ++rank) {
    for (const auto &contrib : contrib_fn(rank)) {
      AddBlockContribution(expected, contrib.first, contrib.second);
    }
  }
  return expected;
}

struct TestMPITenReduceSum : public testing::Test {
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank = 0;
  int mpi_size = 0;

  void SetUp() override {
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &mpi_size);
    if (rank != hp_numeric::kMPIMasterRank) {
      auto &listeners = ::testing::UnitTest::GetInstance()->listeners();
      delete listeners.Release(listeners.default_result_printer());
    }
  }
};

TEST_F(TestMPITenReduceSum, RootReduceSumsIdenticalDoubleTopology) {
  const DQLTensor local = MakeTensorFromContributions<DQLTensor>(
      IdenticalDoubleContribs(rank));
  DQLTensor result;

  MPI_ReduceSum(
      local,
      rank == hp_numeric::kMPIMasterRank ? &result : nullptr,
      hp_numeric::kMPIMasterRank,
      comm);

  if (rank == hp_numeric::kMPIMasterRank) {
    EXPECT_EQ(result, MakeExpected<DQLTensor>(mpi_size, IdenticalDoubleContribs));
  }
}

TEST_F(TestMPITenReduceSum, AllreduceSumsIdenticalComplexTopology) {
  const ZQLTensor local = MakeTensorFromContributions<ZQLTensor>(
      IdenticalComplexContribs(rank));
  ZQLTensor result;

  MPI_AllreduceSum(local, &result, comm);

  EXPECT_EQ(result, MakeExpected<ZQLTensor>(mpi_size, IdenticalComplexContribs));
}

TEST_F(TestMPITenReduceSum, AllreduceBuildsUnionForDisjointBlocks) {
  ASSERT_GE(mpi_size, 3);
  const DQLTensor local = MakeTensorFromContributions<DQLTensor>(
      DisjointDoubleContribs(rank));
  DQLTensor result;

  MPI_AllreduceSum(local, &result, comm);

  EXPECT_EQ(result, MakeExpected<DQLTensor>(mpi_size, DisjointDoubleContribs));
}

TEST_F(TestMPITenReduceSum, RootReduceSumsPartiallyOverlappingBlocks) {
  ASSERT_GE(mpi_size, 3);
  const DQLTensor local = MakeTensorFromContributions<DQLTensor>(
      PartialOverlapDoubleContribs(rank));
  DQLTensor result;

  MPI_ReduceSum(
      local,
      rank == hp_numeric::kMPIMasterRank ? &result : nullptr,
      hp_numeric::kMPIMasterRank,
      comm);

  if (rank == hp_numeric::kMPIMasterRank) {
    EXPECT_EQ(result, MakeExpected<DQLTensor>(mpi_size, PartialOverlapDoubleContribs));
  }
}

TEST_F(TestMPITenReduceSum, RootReduceConvenienceAPISupportsRootZero) {
  ASSERT_GE(mpi_size, 3);
  const DQLTensor local = MakeTensorFromContributions<DQLTensor>(
      RootZeroDoubleContribs(rank));
  DQLTensor result;

  if (rank == hp_numeric::kMPIMasterRank) {
    MPI_ReduceSum(
        local,
        &result,
        hp_numeric::kMPIMasterRank,
        comm,
        MPIReduceSumRootMode::RootZero);
  } else {
    MPI_ReduceSum(
        local,
        static_cast<DQLTensor *>(nullptr),
        hp_numeric::kMPIMasterRank,
        comm);
  }

  if (rank == hp_numeric::kMPIMasterRank) {
    EXPECT_EQ(result, MakeExpected<DQLTensor>(mpi_size, RootZeroDoubleContribs));
  }
}

TEST_F(TestMPITenReduceSum, PlanCachesAndReducesStablePartialOverlapTopology) {
  ASSERT_GE(mpi_size, 3);
  DQLTensor local_template = MakeTensorFromContributions<DQLTensor>(
      PartialOverlapDoubleContribs(rank));
  auto plan = MakeMPIReduceSumPlan(local_template, hp_numeric::kMPIMasterRank, comm);
  EXPECT_FALSE(plan.IsSameBlockTopology());

  DQLTensor result1;
  const DQLTensor local1 = MakeTensorFromContributions<DQLTensor>(
      PartialOverlapDoubleContribs(rank));
  plan.ReduceSum(
      local1,
      rank == hp_numeric::kMPIMasterRank ? &result1 : nullptr);

  if (rank == hp_numeric::kMPIMasterRank) {
    EXPECT_EQ(result1, MakeExpected<DQLTensor>(mpi_size, PartialOverlapDoubleContribs));
  }

  DQLTensor result2;
  const DQLTensor local2 = MakeTensorFromContributions<DQLTensor>(
      ShiftedPartialOverlapDoubleContribs(rank));
  plan.ReduceSum(
      local2,
      rank == hp_numeric::kMPIMasterRank ? &result2 : nullptr);

  if (rank == hp_numeric::kMPIMasterRank) {
    EXPECT_EQ(result2, MakeExpected<DQLTensor>(mpi_size, ShiftedPartialOverlapDoubleContribs));
  }
}

TEST_F(TestMPITenReduceSum, PlanUsesFastPathForStableIdenticalTopology) {
  const DQLTensor local_template = MakeTensorFromContributions<DQLTensor>(
      IdenticalDoubleContribs(rank));
  auto plan = MakeMPIReduceSumPlan(local_template, hp_numeric::kMPIMasterRank, comm);
  EXPECT_TRUE(plan.IsSameBlockTopology());

  DQLTensor result;
  const DQLTensor local = MakeTensorFromContributions<DQLTensor>(
      IdenticalDoubleContribs(rank));
  plan.ReduceSum(
      local,
      rank == hp_numeric::kMPIMasterRank ? &result : nullptr);

  if (rank == hp_numeric::kMPIMasterRank) {
    EXPECT_EQ(result, MakeExpected<DQLTensor>(mpi_size, IdenticalDoubleContribs));
  }
}

TEST_F(TestMPITenReduceSum, PlanRootZeroDoesNotRequireRootSubtotalBlocks) {
  ASSERT_GE(mpi_size, 3);
  const DQLTensor local_template = MakeTensorFromContributions<DQLTensor>(
      RootZeroDoubleContribs(rank));
  auto plan = MakeMPIReduceSumPlan(local_template, hp_numeric::kMPIMasterRank, comm);
  DQLTensor result;

  const DQLTensor local = MakeTensorFromContributions<DQLTensor>(
      RootZeroDoubleContribs(rank));
  if (rank == hp_numeric::kMPIMasterRank) {
    plan.ReduceSum(local, &result, MPIReduceSumRootMode::RootZero);
  } else {
    plan.ReduceSum(local, nullptr);
  }

  if (rank == hp_numeric::kMPIMasterRank) {
    EXPECT_EQ(result, MakeExpected<DQLTensor>(mpi_size, RootZeroDoubleContribs));
  }
}

TEST_F(TestMPITenReduceSum, PlanRootInPlaceReducesRootLocalContributionThroughResultBuffer) {
  ASSERT_GE(mpi_size, 3);
  const DQLTensor local_template = MakeTensorFromContributions<DQLTensor>(
      PartialOverlapDoubleContribs(rank));
  auto plan = MakeMPIReduceSumPlan(local_template, hp_numeric::kMPIMasterRank, comm);
  DQLTensor result;

  const DQLTensor local = MakeTensorFromContributions<DQLTensor>(
      PartialOverlapDoubleContribs(rank));
  if (rank == hp_numeric::kMPIMasterRank) {
    plan.ReduceSum(
        local,
        &result,
        MPIReduceSumRootMode::InPlaceLocalContribution);
  } else {
    plan.ReduceSum(local, nullptr);
  }

  if (rank == hp_numeric::kMPIMasterRank) {
    EXPECT_EQ(result, MakeExpected<DQLTensor>(mpi_size, PartialOverlapDoubleContribs));
  }
}

TEST_F(TestMPITenReduceSum, PlanRejectsMissingRootResultCollectively) {
  const DQLTensor local_template = MakeTensorFromContributions<DQLTensor>(
      IdenticalDoubleContribs(rank));
  auto plan = MakeMPIReduceSumPlan(local_template, hp_numeric::kMPIMasterRank, comm);
  const DQLTensor local = MakeTensorFromContributions<DQLTensor>(
      IdenticalDoubleContribs(rank));

  EXPECT_THROW(plan.ReduceSum(local, nullptr), std::runtime_error);
}

TEST_F(TestMPITenReduceSum, PlanRejectsAliasedRootResultCollectively) {
  DQLTensor local = MakeTensorFromContributions<DQLTensor>(
      IdenticalDoubleContribs(rank));
  auto plan = MakeMPIReduceSumPlan(local, hp_numeric::kMPIMasterRank, comm);

  EXPECT_THROW(
      plan.ReduceSum(
          local,
          rank == hp_numeric::kMPIMasterRank ? &local : nullptr),
      std::runtime_error);
}

TEST_F(TestMPITenReduceSum, PlanRejectsSameRawSizeWithDifferentBlockTopology) {
  ASSERT_GE(mpi_size, 3);
  const DQLTensor local_template = MakeTensorFromContributions<DQLTensor>(
      PartialOverlapDoubleContribs(rank));
  auto plan = MakeMPIReduceSumPlan(local_template, hp_numeric::kMPIMasterRank, comm);
  DQLTensor result;

  const DQLTensor mismatched = MakeTensorFromContributions<DQLTensor>(
      SameRawSizeDifferentTopologyDoubleContribs(rank));
  EXPECT_THROW(
      plan.ReduceSum(
          mismatched,
          rank == hp_numeric::kMPIMasterRank ? &result : nullptr),
      std::runtime_error);
}

TEST_F(TestMPITenReduceSum, AllreducePreservesEmptyBlockTensorLayout) {
  const DQLTensor local(MakeIndexes());
  DQLTensor result;

  MPI_AllreduceSum(local, &result, comm);

  EXPECT_FALSE(result.IsDefault());
  EXPECT_EQ(result.GetIndexes(), MakeIndexes());
  EXPECT_EQ(result.GetQNBlkNum(), 0);
  EXPECT_EQ(result.GetActualDataSize(), 0);
}

TEST_F(TestMPITenReduceSum, AllreducePreservesDefaultTensor) {
  const DQLTensor local;
  DQLTensor result(MakeIndexes());

  MPI_AllreduceSum(local, &result, comm);

  EXPECT_TRUE(result.IsDefault());
}

TEST_F(TestMPITenReduceSum, AllreduceThrowsOnMismatchedTensorLayout) {
  DQLTensor local(rank == 1 ? MakeMismatchedIndexes() : MakeIndexes());
  local.SetElem({0, 0}, 1.0);
  DQLTensor result;

  EXPECT_THROW(MPI_AllreduceSum(local, &result, comm), std::runtime_error);
}

}  // namespace

int main(int argc, char *argv[]) {
  MPI_Init(nullptr, nullptr);
  ::testing::InitGoogleTest(&argc, argv);
  const int test_err = RUN_ALL_TESTS();
  MPI_Finalize();
  return test_err;
}
