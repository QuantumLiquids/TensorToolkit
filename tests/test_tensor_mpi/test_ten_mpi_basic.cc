// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2021-7-26
*
* Description: QuantumLiquids/tensor project. Unittests for Boost MPI of QLTensor.
* Usage:  mpirun -np 2 path_to_executable/test_ten_comm_mpi
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"                  // QLTensor, Index, QN, U1QNVal, QNSectorVec
#include "qlten/utility/utils_inl.h"      // GenAllCoors
#include "qlten/utility/timer.h"
#include <cstdint>
#include <cstring>
#include <vector>

using namespace qlten;

using U1QN = special_qn::U1QN;
using U1U1QN = special_qn::U1U1QN;

using DQLTensor2 = QLTensor<QLTEN_Double, U1U1QN>;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DQLTensor = QLTensor<QLTEN_Double, U1QN>;
using ZQLTensor = QLTensor<QLTEN_Complex, U1QN>;

const std::string qn_nm = "qn_nm";
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

template<typename T>
void AppendPod(std::vector<char> &buffer, const T &value) {
  static_assert(std::is_trivially_copyable_v<T>);
  const char *ptr = reinterpret_cast<const char *>(&value);
  buffer.insert(buffer.end(), ptr, ptr + sizeof(T));
}

inline uint64_t LoadU64(const std::vector<char> &buffer, const size_t offset) {
  uint64_t value = 0;
  std::memcpy(&value, buffer.data() + offset, sizeof(value));
  return value;
}

inline void StoreU64(std::vector<char> &buffer, const size_t offset, const uint64_t value) {
  std::memcpy(buffer.data() + offset, &value, sizeof(value));
}

template<typename Tensor>
Tensor MakeDeterministicTensor(const unsigned long long seed) {
  Tensor tensor({idx_in_s, idx_out_s});
  qlten::SetRandomSeed(seed);
  tensor.Random(qn0);
  return tensor;
}

template<typename Tensor>
void ExpectPackedRoundTrip(const Tensor &tensor) {
  std::vector<char> buffer;
  tensor.PackForMPI(buffer);

  Tensor unpacked;
  const char *cursor = buffer.data();
  const char *end = cursor + buffer.size();
  ASSERT_NO_THROW(unpacked.UnpackForMPI(cursor, end));
  EXPECT_EQ(cursor, end);
  EXPECT_EQ(unpacked, tensor);
}

template<typename Tensor>
void ExpectPackedUnpackThrows(const std::vector<char> &buffer) {
  Tensor unpacked;
  const char *cursor = buffer.data();
  const char *end = cursor + buffer.size();
  EXPECT_THROW(unpacked.UnpackForMPI(cursor, end), std::runtime_error);
}

template<typename Tensor>
void ExpectPackedShellRoundTrip(
    const Tensor &tensor,
    const size_t expected_unpacked_data_size
) {
  std::vector<char> buffer;
  tensor.PackShellForMPI(buffer);

  Tensor unpacked;
  const char *cursor = buffer.data();
  const char *end = cursor + buffer.size();
  ASSERT_NO_THROW(unpacked.UnpackShellForMPI(cursor, end));
  EXPECT_EQ(cursor, end);

  EXPECT_EQ(unpacked.IsDefault(), tensor.IsDefault());
  EXPECT_EQ(unpacked.Rank(), tensor.Rank());
  EXPECT_EQ(unpacked.GetShape(), tensor.GetShape());
  EXPECT_EQ(unpacked.GetIndexes(), tensor.GetIndexes());
  if (tensor.IsDefault()) { return; }

  EXPECT_EQ(unpacked.IsScalar(), tensor.IsScalar());
  EXPECT_EQ(unpacked.GetQNBlkNum(), tensor.GetQNBlkNum());
  EXPECT_EQ(unpacked.GetActualDataSize(), expected_unpacked_data_size);
}

template<typename Tensor>
void ExpectPackedShellUnpackThrows(const std::vector<char> &buffer) {
  Tensor unpacked;
  const char *cursor = buffer.data();
  const char *end = cursor + buffer.size();
  EXPECT_THROW(unpacked.UnpackShellForMPI(cursor, end), std::runtime_error);
}

template<typename Tensor>
void TestPackedShellRawDataBcast(const Tensor &expected, const MPI_Comm &comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);

  std::vector<char> shell_buffer;
  uint64_t shell_buffer_size = 0;
  Tensor tensor;
  if (rank == hp_numeric::kMPIMasterRank) {
    tensor = expected;
    tensor.PackShellForMPI(shell_buffer);
    shell_buffer_size = static_cast<uint64_t>(shell_buffer.size());
  }

  HANDLE_MPI_ERROR(::MPI_Bcast(
      &shell_buffer_size,
      1,
      MPI_UINT64_T,
      hp_numeric::kMPIMasterRank,
      comm));
  if (rank != hp_numeric::kMPIMasterRank) {
    shell_buffer.resize(static_cast<size_t>(shell_buffer_size));
  }
  hp_numeric::MPI_Bcast(
      shell_buffer.data(),
      static_cast<size_t>(shell_buffer_size),
      hp_numeric::kMPIMasterRank,
      comm);

  if (rank != hp_numeric::kMPIMasterRank) {
    const char *cursor = shell_buffer.data();
    const char *end = cursor + shell_buffer.size();
    ASSERT_NO_THROW(tensor.UnpackShellForMPI(cursor, end));
    ASSERT_EQ(cursor, end);
  }

  if (!tensor.IsDefault()) {
    tensor.GetBlkSparDataTen().RawDataMPIBcast(comm, hp_numeric::kMPIMasterRank);
  }
  EXPECT_EQ(tensor, expected);
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

struct TestMPITenData : public testing::Test {
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank;

  DQLTensor dten1;
  DQLTensor dten2;

  ZQLTensor zten1;
  ZQLTensor zten2;

  void SetUp(void) {
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    MPI_Comm_rank(comm, &rank);
    if (rank != hp_numeric::kMPIMasterRank) {
      delete listeners.Release(listeners.default_result_printer());
    } else {
      auto index1_in = RandIndex(50, 400, qlten::IN);
      auto index2_in = RandIndex(4, 1, qlten::IN);
      auto index1_out = RandIndex(4, 1, qlten::OUT);
      auto index2_out = RandIndex(50, 400, qlten::OUT);

      dten1 = DQLTensor({index2_out, index1_in, index2_in, index1_out});

      dten2 = DQLTensor({index2_out, index1_in, index2_in, index1_out});
      dten2.Random(qn0);
      dten2.ConciseShow();

      zten1 = ZQLTensor({index2_out, index1_in, index2_in, index1_out});

      zten2 = ZQLTensor({index2_out, index1_in, index2_in, index1_out});
      zten2.Random(qn0);
      zten2.ConciseShow();
    }
  }
};

template<typename ElemT, typename QNT>
void TestTensorSerialization(
    QLTensor<ElemT, QNT> &tensor
) {
  auto buffer = tensor.SerializeShell();
  DQLTensor ten_cp;
  ten_cp.DeserializeShell(buffer);
  EXPECT_EQ(ten_cp.GetActualDataSize(), tensor.GetActualDataSize());
}

TEST_F(TestMPITenData, PrintErrMessage) {
  std::string err_msg = "Rank" + std::to_string(rank) + "Test printing info.\n";
  hp_numeric::GatherAndPrintErrorMessages(err_msg, comm);
}

TEST_F(TestMPITenData, Serialization) {
  if (rank == hp_numeric::kMPIMasterRank) {
    TestTensorSerialization(dten1);
    TestTensorSerialization(dten2);
    TestTensorSerialization(zten1);
    TestTensorSerialization(zten2);
  }
}

TEST_F(TestMPITenData, PackedWireFormatRoundTrip) {
  ExpectPackedRoundTrip(DQLTensor());
  ExpectPackedRoundTrip(ZQLTensor());
  ExpectPackedRoundTrip(DQLTensor(IndexVec<U1QN>{}));
  ExpectPackedRoundTrip(ZQLTensor(IndexVec<U1QN>{}));
  ExpectPackedRoundTrip(MakeDeterministicTensor<DQLTensor>(123456ULL));
  ExpectPackedRoundTrip(MakeDeterministicTensor<ZQLTensor>(789012ULL));
}

TEST_F(TestMPITenData, PackedShellWireFormatRoundTrip) {
  ExpectPackedShellRoundTrip(DQLTensor(), 0);
  ExpectPackedShellRoundTrip(ZQLTensor(), 0);

  DQLTensor dscalar(IndexVec<U1QN>{});
  ZQLTensor zscalar(IndexVec<U1QN>{});
  ExpectPackedShellRoundTrip(dscalar, 1);
  ExpectPackedShellRoundTrip(zscalar, 1);

  auto dten = MakeDeterministicTensor<DQLTensor>(123456ULL);
  auto zten = MakeDeterministicTensor<ZQLTensor>(789012ULL);
  ExpectPackedShellRoundTrip(dten, dten.GetActualDataSize());
  ExpectPackedShellRoundTrip(zten, zten.GetActualDataSize());
}

TEST_F(TestMPITenData, PackedWireFormatRejectsMalformedBuffers) {
  std::vector<char> mismatched_count_buffer;
  MakeDeterministicTensor<DQLTensor>(123456ULL).PackForMPI(mismatched_count_buffer);
  const uint64_t shell_size = LoadU64(mismatched_count_buffer, 1);
  const size_t raw_count_offset = 1 + sizeof(uint64_t) + static_cast<size_t>(shell_size);
  StoreU64(
      mismatched_count_buffer,
      raw_count_offset,
      LoadU64(mismatched_count_buffer, raw_count_offset) + 1);
  ExpectPackedUnpackThrows<DQLTensor>(mismatched_count_buffer);

  std::vector<char> truncated_shell_buffer;
  MakeDeterministicTensor<DQLTensor>(123456ULL).PackForMPI(truncated_shell_buffer);
  const uint64_t shell_size_for_truncation = LoadU64(truncated_shell_buffer, 1);
  ASSERT_GT(shell_size_for_truncation, 0u);
  truncated_shell_buffer.resize(
      1 + sizeof(uint64_t) + static_cast<size_t>(shell_size_for_truncation) - 1);
  ExpectPackedUnpackThrows<DQLTensor>(truncated_shell_buffer);

  std::vector<char> truncated_raw_buffer;
  MakeDeterministicTensor<DQLTensor>(123456ULL).PackForMPI(truncated_raw_buffer);
  ASSERT_FALSE(truncated_raw_buffer.empty());
  truncated_raw_buffer.pop_back();
  ExpectPackedUnpackThrows<DQLTensor>(truncated_raw_buffer);

  std::vector<char> invalid_default_shell_buffer;
  AppendPod<uint8_t>(invalid_default_shell_buffer, 1);
  AppendPod<uint64_t>(invalid_default_shell_buffer, 1);
  AppendPod<uint64_t>(invalid_default_shell_buffer, 0);
  ExpectPackedUnpackThrows<DQLTensor>(invalid_default_shell_buffer);

  std::vector<char> invalid_default_data_buffer;
  AppendPod<uint8_t>(invalid_default_data_buffer, 1);
  AppendPod<uint64_t>(invalid_default_data_buffer, 0);
  AppendPod<uint64_t>(invalid_default_data_buffer, 1);
  ExpectPackedUnpackThrows<DQLTensor>(invalid_default_data_buffer);
}

TEST_F(TestMPITenData, PackedShellWireFormatRejectsMalformedBuffers) {
  std::vector<char> mismatched_count_buffer;
  MakeDeterministicTensor<DQLTensor>(123456ULL).PackShellForMPI(mismatched_count_buffer);
  const uint64_t shell_size = LoadU64(mismatched_count_buffer, 1);
  const size_t raw_count_offset = 1 + sizeof(uint64_t) + static_cast<size_t>(shell_size);
  StoreU64(
      mismatched_count_buffer,
      raw_count_offset,
      LoadU64(mismatched_count_buffer, raw_count_offset) + 1);
  ExpectPackedShellUnpackThrows<DQLTensor>(mismatched_count_buffer);

  std::vector<char> truncated_shell_buffer;
  MakeDeterministicTensor<DQLTensor>(123456ULL).PackShellForMPI(truncated_shell_buffer);
  const uint64_t shell_size_for_truncation = LoadU64(truncated_shell_buffer, 1);
  ASSERT_GT(shell_size_for_truncation, 0u);
  truncated_shell_buffer.resize(
      1 + sizeof(uint64_t) + static_cast<size_t>(shell_size_for_truncation) - 1);
  ExpectPackedShellUnpackThrows<DQLTensor>(truncated_shell_buffer);

  std::vector<char> invalid_default_shell_buffer;
  AppendPod<uint8_t>(invalid_default_shell_buffer, 1);
  AppendPod<uint64_t>(invalid_default_shell_buffer, 1);
  AppendPod<uint64_t>(invalid_default_shell_buffer, 0);
  ExpectPackedShellUnpackThrows<DQLTensor>(invalid_default_shell_buffer);

  std::vector<char> invalid_default_data_buffer;
  AppendPod<uint8_t>(invalid_default_data_buffer, 1);
  AppendPod<uint64_t>(invalid_default_data_buffer, 0);
  AppendPod<uint64_t>(invalid_default_data_buffer, 1);
  ExpectPackedShellUnpackThrows<DQLTensor>(invalid_default_data_buffer);
}

TEST_F(TestMPITenData, PackedShellWireFormatSupportsRawDataMPIBcast) {
  TestPackedShellRawDataBcast(MakeDeterministicTensor<DQLTensor>(123456ULL), comm);
  TestPackedShellRawDataBcast(MakeDeterministicTensor<ZQLTensor>(789012ULL), comm);
}

TEST_F(TestMPITenData, PackedShellWireFormatSupportsEmptyScalarRawDataMPIBcast) {
  TestPackedShellRawDataBcast(DQLTensor(IndexVec<U1QN>{}), comm);
  TestPackedShellRawDataBcast(ZQLTensor(IndexVec<U1QN>{}), comm);
}

#ifdef USE_GPU
TEST_F(TestMPITenData, PackedWireFormatRejectsGPUBuilds) {
  std::vector<char> buffer;
  auto tensor = MakeDeterministicTensor<DQLTensor>(123456ULL);
  EXPECT_THROW(tensor.PackForMPI(buffer), std::runtime_error);

  DQLTensor unpacked;
  const char *cursor = nullptr;
  EXPECT_THROW(unpacked.UnpackForMPI(cursor, cursor), std::runtime_error);
}
#endif

///< only master has the tensor
template<typename ElemT, typename QNT>
void TestTensorCommunication(
    QLTensor<ElemT, QNT> &tensor,
    const MPI_Comm &comm
) {
  using Tensor = QLTensor<ElemT, QNT>;
  const size_t assist_proc = 1;
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == hp_numeric::kMPIMasterRank) {
    Tensor recv_bsdt_ten;
    Timer mpi_ten_passing_timer("mpi_ten_send_recv");
    tensor.MPI_Send(assist_proc, 35, comm);
    recv_bsdt_ten = Tensor(tensor.GetIndexes());
    auto &bsdt = recv_bsdt_ten.GetBlkSparDataTen();
    bsdt.MPI_Recv(comm, assist_proc, 3);
    mpi_ten_passing_timer.PrintElapsed();
    EXPECT_EQ(tensor, recv_bsdt_ten);
    tensor.MPI_Bcast(hp_numeric::kMPIMasterRank, comm);
  } else if (rank == assist_proc) {
    Tensor recv_ten, recv_bcast_ten;
    recv_ten.MPI_Recv(MPI_ANY_SOURCE, MPI_ANY_TAG, comm);
    auto &bsdt = recv_ten.GetBlkSparDataTen();
    bsdt.MPI_Send(comm, hp_numeric::kMPIMasterRank, 3);
    recv_bcast_ten.MPI_Bcast(hp_numeric::kMPIMasterRank, comm);
    EXPECT_EQ(recv_ten, recv_bcast_ten);
  } else {
    Tensor recv_bcast_ten;
    recv_bcast_ten.MPI_Bcast(hp_numeric::kMPIMasterRank, comm);
  }
  return;
}

TEST_F(TestMPITenData, TensorCommuniation) {
  TestTensorCommunication(dten1, comm);
  TestTensorCommunication(dten2, comm);
  TestTensorCommunication(zten1, comm);
  TestTensorCommunication(zten2, comm);
}

int main(int argc, char *argv[]) {
  MPI_Init(nullptr, nullptr);
  ::testing::InitGoogleTest(&argc, argv);
  int test_err = RUN_ALL_TESTS();
  MPI_Finalize();
  return test_err;
}
