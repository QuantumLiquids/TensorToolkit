// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Codex
*
* Description: Tests for lifetime-tracked nonblocking QLTensor MPI transfers.
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

using DQLTensorRecvHandle = QLTensorMPIRecvHandle<QLTEN_Double, U1QN>;
static_assert(!std::is_copy_constructible_v<DQLTensorRecvHandle>);
static_assert(!std::is_copy_assignable_v<DQLTensorRecvHandle>);
static_assert(std::is_move_constructible_v<DQLTensorRecvHandle>);
static_assert(std::is_move_assignable_v<DQLTensorRecvHandle>);
static_assert(std::is_default_constructible_v<DQLTensorRecvHandle>);

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

TEST_F(TestNonblockingTensorMPI,
       BlockingAndNonblockingSendersUseNonblockingReceiver) {
  constexpr int blocking_tag = 201;
  constexpr int nonblocking_tag = 202;
  const auto real_tensor = MakeRank2Tensor<DQLTensor>(32, 410001ULL);
  const auto complex_tensor = MakeRank4Tensor<ZQLTensor>(5, 410002ULL);

  if (rank_ == hp_numeric::kMPIMasterRank) {
    real_tensor.MPI_Send(1, blocking_tag, comm_);
    auto send_request = complex_tensor.MPI_Isend(1, nonblocking_tag, comm_);
    send_request.Wait();
  } else if (rank_ == 1) {
    DQLTensor received_real;
    auto real_request = received_real.MPI_Irecv(
        hp_numeric::kMPIMasterRank, blocking_tag, comm_);
    EXPECT_EQ(real_request.Status().MPI_SOURCE, hp_numeric::kMPIMasterRank);
    EXPECT_EQ(real_request.Status().MPI_TAG, blocking_tag);
    EXPECT_EQ(
        real_request.RequestCount(),
        hp_numeric::MPILargeMessageRequestCount<QLTEN_Double>(
            real_tensor.GetActualDataSize()));
    (void) real_request.Test();
    real_request.Wait();
    real_request.Wait();
    EXPECT_TRUE(real_request.Test());
    EXPECT_EQ(received_real, real_tensor);

    ZQLTensor received_complex;
    auto complex_request = received_complex.MPI_Irecv(
        hp_numeric::kMPIMasterRank, nonblocking_tag, comm_);
    EXPECT_EQ(complex_request.Status().MPI_SOURCE,
              hp_numeric::kMPIMasterRank);
    EXPECT_EQ(complex_request.Status().MPI_TAG, nonblocking_tag);
    complex_request.Wait();
    EXPECT_EQ(received_complex, complex_tensor);
  }
  Barrier();
}

TEST_F(TestNonblockingTensorMPI,
       DefaultAndEmptyScalarUseNonblockingReceiver) {
  constexpr int default_tag = 205;
  constexpr int scalar_tag = 206;

  if (rank_ == hp_numeric::kMPIMasterRank) {
    DQLTensor default_tensor;
    default_tensor.MPI_Send(1, default_tag, comm_);

    ZQLTensor scalar{IndexVec<U1QN>{}};
    auto send_request = scalar.MPI_Isend(1, scalar_tag, comm_);
    send_request.Wait();
  } else if (rank_ == 1) {
    DQLTensor received_default;
    auto default_request = received_default.MPI_Irecv(
        hp_numeric::kMPIMasterRank, default_tag, comm_);
    EXPECT_TRUE(default_request.IsComplete());
    EXPECT_TRUE(default_request.Test());
    EXPECT_EQ(default_request.RequestCount(), 0u);
    EXPECT_EQ(default_request.PayloadBytes(), sizeof(bool));
    EXPECT_EQ(default_request.StagingBytes(), 0u);
    EXPECT_TRUE(received_default.IsDefault());

    ZQLTensor received_scalar;
    auto scalar_request = received_scalar.MPI_Irecv(
        hp_numeric::kMPIMasterRank, scalar_tag, comm_);
    EXPECT_FALSE(scalar_request.IsComplete());
    EXPECT_EQ(scalar_request.RequestCount(), 1u);
    EXPECT_EQ(scalar_request.StagingBytes(), sizeof(MPI_Request));
    scalar_request.Wait();
    EXPECT_EQ(received_scalar, ZQLTensor(IndexVec<U1QN>{}));
  }
  Barrier();
}

TEST_F(TestNonblockingTensorMPI,
       LargeReceiveUsesMultipleDirectRawDataRequests) {
  constexpr int tensor_tag = 210;
  const auto tensor = MakeRank2Tensor<DQLTensor>(1024, 420001ULL);
  const size_t raw_request_count =
      hp_numeric::MPILargeMessageRequestCount<QLTEN_Double>(
          tensor.GetActualDataSize());
  ASSERT_GT(raw_request_count, 1u);

  if (rank_ == hp_numeric::kMPIMasterRank) {
    auto send_request = tensor.MPI_Isend(1, tensor_tag, comm_);
    send_request.Wait();
  } else if (rank_ == 1) {
    DQLTensor received;
    auto receive_request = received.MPI_Irecv(
        hp_numeric::kMPIMasterRank, tensor_tag, comm_);
    EXPECT_EQ(receive_request.RequestCount(), raw_request_count);
    EXPECT_EQ(receive_request.StagingBytes(),
              raw_request_count * sizeof(MPI_Request));
    EXPECT_GT(receive_request.PayloadBytes(),
              tensor.GetActualDataSize() * sizeof(QLTEN_Double));
    receive_request.Wait();
    EXPECT_EQ(received, tensor);
  }
  Barrier();
}

TEST_F(TestNonblockingTensorMPI,
       ActiveReceiveHandleMoveConstructionAndAssignmentAreSafe) {
  constexpr int first_tag = 212;
  constexpr int second_tag = 213;
  constexpr int third_tag = 214;
  const auto first_tensor = MakeRank2Tensor<DQLTensor>(512, 430001ULL);
  const auto second_tensor = MakeRank2Tensor<DQLTensor>(512, 430002ULL);
  const auto third_tensor = MakeRank2Tensor<DQLTensor>(512, 430003ULL);

  if (rank_ == hp_numeric::kMPIMasterRank) {
    std::vector<hp_numeric::MPISendRequest> send_requests;
    send_requests.reserve(3);
    send_requests.push_back(first_tensor.MPI_Isend(1, first_tag, comm_));
    send_requests.push_back(second_tensor.MPI_Isend(1, second_tag, comm_));
    send_requests.push_back(third_tensor.MPI_Isend(1, third_tag, comm_));
    for (auto &request : send_requests) { request.Wait(); }
  } else if (rank_ == 1) {
    DQLTensor received_first;
    DQLTensor received_second;
    DQLTensor received_third;

    auto original = received_first.MPI_Irecv(
        hp_numeric::kMPIMasterRank, first_tag, comm_);
    EXPECT_FALSE(original.IsComplete());
    DQLTensorRecvHandle moved(std::move(original));
    EXPECT_TRUE(original.IsComplete());
    EXPECT_EQ(original.RequestCount(), 0u);

    auto overwritten = received_second.MPI_Irecv(
        hp_numeric::kMPIMasterRank, second_tag, comm_);
    auto replacement = received_third.MPI_Irecv(
        hp_numeric::kMPIMasterRank, third_tag, comm_);
    EXPECT_FALSE(overwritten.IsComplete());
    EXPECT_FALSE(replacement.IsComplete());
    overwritten = std::move(replacement);
    EXPECT_TRUE(replacement.IsComplete());
    EXPECT_EQ(received_second, second_tensor);

    moved.Wait();
    overwritten.Wait();
    EXPECT_EQ(received_first, first_tensor);
    EXPECT_EQ(received_third, third_tensor);
  }
  Barrier();
}

TEST_F(TestNonblockingTensorMPI,
       MultipleOutstandingReceivesFromDifferentSources) {
  constexpr int tensor_tag = 216;
  const auto first_tensor = MakeRank2Tensor<DQLTensor>(384, 440001ULL);
  const auto second_tensor = MakeRank2Tensor<DQLTensor>(448, 440002ULL);

  if (rank_ == hp_numeric::kMPIMasterRank) {
    auto request = first_tensor.MPI_Isend(1, tensor_tag, comm_);
    request.Wait();
  } else if (rank_ == 2) {
    auto request = second_tensor.MPI_Isend(1, tensor_tag, comm_);
    request.Wait();
  } else if (rank_ == 1) {
    DQLTensor received_first;
    DQLTensor received_second;
    auto first_request = received_first.MPI_Irecv(
        hp_numeric::kMPIMasterRank, tensor_tag, comm_);
    auto second_request = received_second.MPI_Irecv(2, tensor_tag, comm_);
    first_request.Wait();
    second_request.Wait();
    EXPECT_EQ(received_first, first_tensor);
    EXPECT_EQ(received_second, second_tensor);
  }
  Barrier();
}

TEST_F(TestNonblockingTensorMPI,
       SameSourceAndBaseTagPreserveReceiveStreamOrder) {
  constexpr int tensor_tag = 218;
  const auto first_tensor = MakeRank2Tensor<DQLTensor>(384, 450001ULL);
  const auto second_tensor = MakeRank2Tensor<DQLTensor>(320, 450002ULL);
  const auto third_tensor = MakeRank4Tensor<DQLTensor>(5, 450003ULL);

  if (rank_ == hp_numeric::kMPIMasterRank) {
    std::vector<hp_numeric::MPISendRequest> send_requests;
    send_requests.reserve(3);
    send_requests.push_back(first_tensor.MPI_Isend(1, tensor_tag, comm_));
    send_requests.push_back(second_tensor.MPI_Isend(1, tensor_tag, comm_));
    send_requests.push_back(third_tensor.MPI_Isend(1, tensor_tag, comm_));
    for (auto &request : send_requests) { request.Wait(); }
  } else if (rank_ == 1) {
    DQLTensor received_first;
    DQLTensor received_second;
    DQLTensor received_third;
    auto first_request = received_first.MPI_Irecv(
        hp_numeric::kMPIMasterRank, tensor_tag, comm_);
    auto second_request = received_second.MPI_Irecv(
        hp_numeric::kMPIMasterRank, tensor_tag, comm_);
    auto third_request = received_third.MPI_Irecv(
        hp_numeric::kMPIMasterRank, tensor_tag, comm_);
    first_request.Wait();
    second_request.Wait();
    third_request.Wait();
    EXPECT_EQ(received_first, first_tensor);
    EXPECT_EQ(received_second, second_tensor);
    EXPECT_EQ(received_third, third_tensor);
  }
  Barrier();
}

TEST_F(TestNonblockingTensorMPI,
       IncompleteReceiveHandleDestructorWaitsSafely) {
  constexpr int tensor_tag = 220;
  const auto tensor = MakeRank2Tensor<DQLTensor>(512, 460001ULL);

  if (rank_ == hp_numeric::kMPIMasterRank) {
    tensor.MPI_Send(1, tensor_tag, comm_);
  } else if (rank_ == 1) {
    DQLTensor received;
    {
      auto request = received.MPI_Irecv(
          hp_numeric::kMPIMasterRank, tensor_tag, comm_);
      EXPECT_FALSE(request.IsComplete());
    }
    EXPECT_EQ(received, tensor);
  }
  Barrier();
}

TEST_F(TestNonblockingTensorMPI,
       ReceiveHandleWaitsDuringRealExceptionUnwinding) {
  constexpr int tensor_tag = 222;
  const auto tensor = MakeRank2Tensor<DQLTensor>(512, 470001ULL);

  if (rank_ == hp_numeric::kMPIMasterRank) {
    tensor.MPI_Send(1, tensor_tag, comm_);
  } else if (rank_ == 1) {
    DQLTensor received;
    bool caught = false;
    try {
      auto request = received.MPI_Irecv(
          hp_numeric::kMPIMasterRank, tensor_tag, comm_);
      EXPECT_FALSE(request.IsComplete());
      throw std::runtime_error("test exception during active tensor receive");
    } catch (const std::runtime_error &error) {
      caught = true;
      EXPECT_STREQ(error.what(),
                   "test exception during active tensor receive");
    }
    EXPECT_TRUE(caught);
    EXPECT_EQ(received, tensor);
  }
  Barrier();
}

TEST_F(TestNonblockingTensorMPI,
       WildcardReceiveResolvesSourceTagAndStatus) {
  constexpr int tensor_tag = 224;
  const auto tensor = MakeRank2Tensor<ZQLTensor>(256, 480001ULL);

  if (rank_ == 2) {
    tensor.MPI_Send(1, tensor_tag, comm_);
  } else if (rank_ == 1) {
    ZQLTensor received;
    auto request = received.MPI_Irecv(MPI_ANY_SOURCE, MPI_ANY_TAG, comm_);
    EXPECT_EQ(request.Status().MPI_SOURCE, 2);
    EXPECT_EQ(request.Status().MPI_TAG, tensor_tag);
    request.Wait();
    EXPECT_EQ(request.Status().MPI_SOURCE, 2);
    EXPECT_EQ(request.Status().MPI_TAG, tensor_tag);
    EXPECT_EQ(received, tensor);
  }
  Barrier();
}

TEST_F(TestNonblockingTensorMPI,
       DefaultReceiveAcceptsTagUpperBound) {
  const int tag_upper_bound = hp_numeric::GetMPITagUpperBound(comm_);

  if (rank_ == hp_numeric::kMPIMasterRank) {
    DQLTensor default_tensor;
    auto send_request = default_tensor.MPI_Isend(1, tag_upper_bound, comm_);
    send_request.Wait();
  } else if (rank_ == 1) {
    DQLTensor received;
    auto receive_request = received.MPI_Irecv(
        hp_numeric::kMPIMasterRank, tag_upper_bound, comm_);
    EXPECT_TRUE(receive_request.IsComplete());
    EXPECT_EQ(receive_request.Status().MPI_SOURCE,
              hp_numeric::kMPIMasterRank);
    EXPECT_EQ(receive_request.Status().MPI_TAG, tag_upper_bound);
    EXPECT_TRUE(received.IsDefault());
  }
  Barrier();
}

TEST_F(TestNonblockingTensorMPI,
       InvalidReceiveDestinationAndBaseTagFailBeforeReceiving) {
  if (rank_ == hp_numeric::kMPIMasterRank) {
    DQLTensor nondefault(IndexVec<U1QN>{});
    EXPECT_THROW(
        {
          auto request = nondefault.MPI_Irecv(1, 226, comm_);
          (void) request;
        },
        std::invalid_argument);

    DQLTensor invalid_negative_tag;
    EXPECT_THROW(
        {
          auto request = invalid_negative_tag.MPI_Irecv(1, -2, comm_);
          (void) request;
        },
        std::invalid_argument);
  }
  Barrier();
}

TEST_F(TestNonblockingTensorMPI,
       InvalidDerivedReceiveTagFailsAfterShellReception) {
  const int tag_upper_bound = hp_numeric::GetMPITagUpperBound(comm_);
  const auto tensor = MakeRank2Tensor<DQLTensor>(8, 490001ULL);

  if (rank_ == hp_numeric::kMPIMasterRank) {
    DQLTensor received;
    EXPECT_THROW(
        {
          auto request = received.MPI_Irecv(
              1, tag_upper_bound, comm_);
          (void) request;
        },
        std::out_of_range);
  } else if (rank_ == 1) {
    const bool is_default = false;
    HANDLE_MPI_ERROR(::MPI_Send(
        &is_default,
        1,
        MPI_CXX_BOOL,
        hp_numeric::kMPIMasterRank,
        tag_upper_bound,
        comm_));
    const std::string shell = tensor.SerializeShell();
    hp_numeric::MPI_Send(
        shell.size(), hp_numeric::kMPIMasterRank, tag_upper_bound, comm_);
    HANDLE_MPI_ERROR(::MPI_Send(
        shell.data(),
        static_cast<int>(shell.size()),
        MPI_CHAR,
        hp_numeric::kMPIMasterRank,
        tag_upper_bound,
        comm_));
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
