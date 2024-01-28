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

namespace mpi = boost::mpi;

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

//helper
IndexT RandIndex(const unsigned qn_sct_num,  //how many quantum number sectors?
                 const unsigned max_dim_in_one_qn_sct, // maximum dimension in every quantum number sector?
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

struct TestBoostMPI : public testing::Test {
  mpi::communicator world;

  void SetUp(void) {
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    if (world.rank() != 0) {
      delete listeners.Release(listeners.default_result_printer());
    }
  }
};

///< only master has the tensor
template<typename ElemT, typename QNT>
void TestTensorCommunication(
    const QLTensor<ElemT, QNT> &tensor,
    const mpi::communicator &world
) {
  using Tensor = QLTensor<ElemT, QNT>;
  const size_t assist_proc = 1;
  if (world.rank() == kMPIMasterRank) {
    Tensor recv_bsdt_ten;
    Timer mpi_ten_passing_timer("mpi_ten_send_bsdt_recv");
    send_qlten(world, assist_proc, 35, tensor);
    recv_bsdt_ten = Tensor(tensor.GetIndexes());
    auto &bsdt = recv_bsdt_ten.GetBlkSparDataTen();
    bsdt.MPIRecv(world, assist_proc, 3);
    mpi_ten_passing_timer.PrintElapsed();
    EXPECT_EQ(tensor, recv_bsdt_ten);
    SendBroadCastQLTensor(world, tensor, kMPIMasterRank);
  } else if (world.rank() == assist_proc) {
    Tensor recv_ten, recv_bcast_ten;
    recv_qlten(world, mpi::any_source, mpi::any_tag, recv_ten);
    auto &bsdt = recv_ten.GetBlkSparDataTen();
    bsdt.MPISend(world, kMPIMasterRank, 3);
    RecvBroadCastQLTensor(world, recv_bcast_ten, kMPIMasterRank);
    EXPECT_EQ(recv_ten, recv_bcast_ten);
  } else {
    Tensor recv_bcast_ten;
    RecvBroadCastQLTensor(world, recv_bcast_ten, kMPIMasterRank);
  }
  return;
}

TEST_F(TestBoostMPI, SendRecvBroadCastRandomTensor) {
  DQLTensor dten1;
  if (world.rank() == 0) {
    auto index1_in = RandIndex(50, 400, qlten::IN);
    auto index2_in = RandIndex(4, 1, qlten::IN);
    auto index1_out = RandIndex(4, 1, qlten::OUT);
    auto index2_out = RandIndex(50, 400, qlten::OUT);

    dten1 = DQLTensor({index2_out, index1_in, index2_in, index1_out});
    dten1.Random(qn0);
    dten1.ConciseShow();
  }
  TestTensorCommunication(dten1, world);

  std::cout << std::endl;

  ZQLTensor zten1;
  if (world.rank() == 0) {
    auto index1_in = RandIndex(50, 400, qlten::IN);
    auto index2_in = RandIndex(4, 1, qlten::IN);
    auto index1_out = RandIndex(4, 1, qlten::OUT);
    auto index2_out = RandIndex(50, 400, qlten::OUT);

    zten1 = ZQLTensor({index2_out, index1_in, index2_in, index1_out});
    zten1.Random(qn0);
    zten1.ConciseShow();
  }
  TestTensorCommunication(zten1, world);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  boost::mpi::environment env;
  return RUN_ALL_TESTS();;
}