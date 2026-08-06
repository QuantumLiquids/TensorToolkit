// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Codex
*
* Description: Tests for lifetime-tracked nonblocking QLTensor MPI sends.
*/

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "qlten/qlten.h"

using namespace qlten;

namespace {

using U1QN = special_qn::U1QN;
using DQLTensor = QLTensor<QLTEN_Double, U1QN>;
using ZQLTensor = QLTensor<QLTEN_Complex, U1QN>;

constexpr int kReceiverReadyTag = 30000;

U1QN ZeroQN() {
  return U1QN({QNCard("nonblocking_mpi_qn", U1QNVal(0))});
}

Index<U1QN> MakeIndex(const size_t dimension, const TenIndexDirType direction) {
  return Index<U1QN>({QNSector<U1QN>(ZeroQN(), dimension)}, direction);
}

template<typename Tensor>
Tensor MakeRank2Tensor(const size_t dimension, const unsigned long long seed) {
  Tensor tensor({MakeIndex(dimension, IN), MakeIndex(dimension, OUT)});
  SetRandomSeed(seed);
  tensor.Random(ZeroQN());
  return tensor;
}

template<typename Tensor>
Tensor MakeRank4Tensor(const size_t dimension, const unsigned long long seed) {
  Tensor tensor({MakeIndex(dimension, IN),
                 MakeIndex(dimension, OUT),
                 MakeIndex(dimension, IN),
                 MakeIndex(dimension, OUT)});
  SetRandomSeed(seed);
  tensor.Random(ZeroQN());
  return tensor;
}

void SendReceiverReady(const int receiver, const MPI_Comm &comm) {
  const int ready = 1;
  HANDLE_MPI_ERROR(::MPI_Send(
      &ready, 1, MPI_INT, receiver, kReceiverReadyTag, comm));
}

void WaitUntilSenderIsReady(const MPI_Comm &comm) {
  int ready = 0;
  HANDLE_MPI_ERROR(::MPI_Recv(
      &ready,
      1,
      MPI_INT,
      hp_numeric::kMPIMasterRank,
      kReceiverReadyTag,
      comm,
      MPI_STATUS_IGNORE));
  ASSERT_EQ(ready, 1);
}

class TestNonblockingTensorMPI : public testing::Test {
 protected:
  MPI_Comm comm_ = MPI_COMM_WORLD;
  int rank_ = 0;

  void SetUp() override { HANDLE_MPI_ERROR(::MPI_Comm_rank(comm_, &rank_)); }

  void Barrier() { HANDLE_MPI_ERROR(::MPI_Barrier(comm_)); }
};

static_assert(!std::is_copy_constructible_v<hp_numeric::MPISendRequest>);
static_assert(!std::is_copy_assignable_v<hp_numeric::MPISendRequest>);
static_assert(std::is_move_constructible_v<hp_numeric::MPISendRequest>);
static_assert(std::is_move_assignable_v<hp_numeric::MPISendRequest>);

TEST_F(TestNonblockingTensorMPI, DefaultAndEmptyScalarUseBlockingReceiver) {
  constexpr int default_tag = 101;
  constexpr int scalar_tag = 102;

  if (rank_ == hp_numeric::kMPIMasterRank) {
    DQLTensor default_tensor;
    auto default_request = default_tensor.MPI_Isend(1, default_tag, comm_);
    EXPECT_EQ(default_request.PayloadBytes(), sizeof(bool));
    EXPECT_EQ(default_request.RequestCount(), 1u);
    default_request.Wait();
    EXPECT_TRUE(default_request.Test());

    ZQLTensor scalar{IndexVec<U1QN>{}};
    auto scalar_request = scalar.MPI_Isend(1, scalar_tag, comm_);
    EXPECT_EQ(scalar_request.RequestCount(), 4u);
    EXPECT_GT(scalar_request.PayloadBytes(), sizeof(QLTEN_Complex));
    (void) scalar_request.Test();
    scalar_request.Wait();
    scalar_request.Wait();
    EXPECT_TRUE(scalar_request.IsComplete());
  } else if (rank_ == 1) {
    DQLTensor received_default;
    received_default.MPI_Recv(
        hp_numeric::kMPIMasterRank, default_tag, comm_);
    EXPECT_TRUE(received_default.IsDefault());

    ZQLTensor received_scalar;
    received_scalar.MPI_Recv(
        hp_numeric::kMPIMasterRank, scalar_tag, comm_);
    EXPECT_EQ(received_scalar, ZQLTensor(IndexVec<U1QN>{}));
  }
  Barrier();
}

TEST_F(TestNonblockingTensorMPI, RealAndComplexTensorShellsRemainCompatible) {
  constexpr int real_tag = 110;
  constexpr int complex_tag = 111;
  const auto real_tensor = MakeRank2Tensor<DQLTensor>(32, 123456ULL);
  const auto complex_tensor = MakeRank4Tensor<ZQLTensor>(5, 789012ULL);

  if (rank_ == hp_numeric::kMPIMasterRank) {
    auto real_request = real_tensor.MPI_Isend(1, real_tag, comm_);
    EXPECT_FALSE(real_request.IsComplete());
    real_request.Wait();

    auto complex_request = complex_tensor.MPI_Isend(1, complex_tag, comm_);
    (void) complex_request.Test();
    complex_request.Wait();
    EXPECT_TRUE(complex_request.Test());
  } else if (rank_ == 1) {
    DQLTensor received_real;
    received_real.MPI_Recv(
        hp_numeric::kMPIMasterRank, real_tag, comm_);
    EXPECT_EQ(received_real, real_tensor);

    ZQLTensor received_complex;
    received_complex.MPI_Recv(
        hp_numeric::kMPIMasterRank, complex_tag, comm_);
    EXPECT_EQ(received_complex, complex_tensor);
  }
  Barrier();
}

TEST_F(TestNonblockingTensorMPI, ActiveHandleMovePreservesChunkBuffers) {
  constexpr int tensor_tag = 120;
  const auto tensor = MakeRank2Tensor<DQLTensor>(1024, 246810ULL);
  const size_t raw_request_count =
      hp_numeric::MPILargeMessageRequestCount<QLTEN_Double>(
          tensor.GetActualDataSize());
  ASSERT_GT(raw_request_count, 1u);

  if (rank_ == hp_numeric::kMPIMasterRank) {
    auto request = tensor.MPI_Isend(1, tensor_tag, comm_);
    EXPECT_EQ(request.RequestCount(), 3u + raw_request_count);
    EXPECT_LT(request.StagingBytes(), request.PayloadBytes());

    auto moved_request = std::move(request);
    EXPECT_TRUE(request.IsComplete());
    EXPECT_EQ(request.RequestCount(), 0u);
    EXPECT_FALSE(moved_request.IsComplete());

    SendReceiverReady(1, comm_);
    moved_request.Wait();
    moved_request.Wait();
    EXPECT_TRUE(moved_request.IsComplete());
  } else if (rank_ == 1) {
    WaitUntilSenderIsReady(comm_);
    DQLTensor received;
    received.MPI_Recv(
        hp_numeric::kMPIMasterRank, tensor_tag, comm_);
    EXPECT_EQ(received, tensor);
  }
  Barrier();
}

TEST_F(TestNonblockingTensorMPI, MultipleDestinationsCanRemainOutstanding) {
  constexpr int first_tag = 130;
  constexpr int second_tag = 131;
  const auto first_tensor = MakeRank2Tensor<DQLTensor>(512, 135791ULL);
  const auto second_tensor = MakeRank2Tensor<ZQLTensor>(384, 975318ULL);

  if (rank_ == hp_numeric::kMPIMasterRank) {
    std::vector<hp_numeric::MPISendRequest> requests;
    requests.reserve(2);
    requests.push_back(first_tensor.MPI_Isend(1, first_tag, comm_));
    requests.push_back(second_tensor.MPI_Isend(2, second_tag, comm_));
    EXPECT_FALSE(requests[0].IsComplete());
    EXPECT_FALSE(requests[1].IsComplete());

    SendReceiverReady(1, comm_);
    SendReceiverReady(2, comm_);
    for (auto &request : requests) { request.Wait(); }
  } else if (rank_ == 1) {
    WaitUntilSenderIsReady(comm_);
    DQLTensor received;
    received.MPI_Recv(
        hp_numeric::kMPIMasterRank, first_tag, comm_);
    EXPECT_EQ(received, first_tensor);
  } else if (rank_ == 2) {
    WaitUntilSenderIsReady(comm_);
    ZQLTensor received;
    received.MPI_Recv(
        hp_numeric::kMPIMasterRank, second_tag, comm_);
    EXPECT_EQ(received, second_tensor);
  }
  Barrier();
}

TEST_F(TestNonblockingTensorMPI,
       SameDestinationAndBaseTagPreserveTensorStreamOrder) {
  constexpr int tensor_tag = 135;
  const auto first_tensor = MakeRank2Tensor<DQLTensor>(512, 101010ULL);
  const auto second_tensor = MakeRank2Tensor<DQLTensor>(384, 202020ULL);
  const auto third_tensor = MakeRank4Tensor<DQLTensor>(5, 303030ULL);

  if (rank_ == hp_numeric::kMPIMasterRank) {
    std::vector<hp_numeric::MPISendRequest> requests;
    requests.reserve(3);
    requests.push_back(first_tensor.MPI_Isend(1, tensor_tag, comm_));
    requests.push_back(second_tensor.MPI_Isend(1, tensor_tag, comm_));
    requests.push_back(third_tensor.MPI_Isend(1, tensor_tag, comm_));
    for (const auto &request : requests) {
      EXPECT_FALSE(request.IsComplete());
    }

    SendReceiverReady(1, comm_);
    for (auto &request : requests) { request.Wait(); }
  } else if (rank_ == 1) {
    WaitUntilSenderIsReady(comm_);
    DQLTensor received_first;
    DQLTensor received_second;
    DQLTensor received_third;
    received_first.MPI_Recv(
        hp_numeric::kMPIMasterRank, tensor_tag, comm_);
    received_second.MPI_Recv(
        hp_numeric::kMPIMasterRank, tensor_tag, comm_);
    received_third.MPI_Recv(
        hp_numeric::kMPIMasterRank, tensor_tag, comm_);
    EXPECT_EQ(received_first, first_tensor);
    EXPECT_EQ(received_second, second_tensor);
    EXPECT_EQ(received_third, third_tensor);
  }
  Barrier();
}

TEST_F(TestNonblockingTensorMPI, IncompleteHandleDestructorWaitsSafely) {
  constexpr int tensor_tag = 140;
  const auto tensor = MakeRank2Tensor<DQLTensor>(512, 556677ULL);

  if (rank_ == hp_numeric::kMPIMasterRank) {
    {
      auto request = tensor.MPI_Isend(1, tensor_tag, comm_);
      EXPECT_FALSE(request.IsComplete());
      SendReceiverReady(1, comm_);
    }
  } else if (rank_ == 1) {
    WaitUntilSenderIsReady(comm_);
    DQLTensor received;
    received.MPI_Recv(
        hp_numeric::kMPIMasterRank, tensor_tag, comm_);
    EXPECT_EQ(received, tensor);
  }
  Barrier();
}

TEST_F(TestNonblockingTensorMPI,
       IncompleteHandleDestructorWaitsDuringExceptionUnwinding) {
  constexpr int tensor_tag = 145;
  const auto tensor = MakeRank2Tensor<DQLTensor>(512, 667788ULL);

  if (rank_ == hp_numeric::kMPIMasterRank) {
    bool caught = false;
    try {
      auto request = tensor.MPI_Isend(1, tensor_tag, comm_);
      EXPECT_FALSE(request.IsComplete());
      SendReceiverReady(1, comm_);
      throw std::runtime_error("test exception during active tensor send");
    } catch (const std::runtime_error &error) {
      caught = true;
      EXPECT_STREQ(error.what(), "test exception during active tensor send");
    }
    EXPECT_TRUE(caught);
  } else if (rank_ == 1) {
    WaitUntilSenderIsReady(comm_);
    DQLTensor received;
    received.MPI_Recv(
        hp_numeric::kMPIMasterRank, tensor_tag, comm_);
    EXPECT_EQ(received, tensor);
  }
  Barrier();
}

TEST_F(TestNonblockingTensorMPI, InvalidDerivedTagRangeFailsBeforeSending) {
  const auto tensor = MakeRank2Tensor<DQLTensor>(8, 112233ULL);
  if (rank_ == hp_numeric::kMPIMasterRank) {
    const int tag_upper_bound = hp_numeric::GetMPITagUpperBound(comm_);
    EXPECT_THROW(
        {
          auto request = tensor.MPI_Isend(1, tag_upper_bound, comm_);
          (void) request;
        },
        std::out_of_range);
    EXPECT_THROW(
        {
          auto request = tensor.MPI_Isend(1, MPI_ANY_TAG, comm_);
          (void) request;
        },
        std::invalid_argument);
  }
  Barrier();
}

}  // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  const int test_result = RUN_ALL_TESTS();
  MPI_Finalize();
  return test_result;
}
