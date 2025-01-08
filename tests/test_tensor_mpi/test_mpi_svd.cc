// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang<wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2021-08-22
* 
* Description: QuantumLiquids/tensor project. Unittests for MPI tensor SVD.
*/

#include <vector>
#include <iostream>

#include "gtest/gtest.h"
#include "qlten/qltensor_all.h"
#include "qlten/tensor_manipulation/ten_decomp/ten_svd.h"   // SVD
#include "qlten/tensor_manipulation/basic_operations.h"     // Dag
#include "qlten/mpi_tensor_manipulation/ten_decomp/mpi_svd.h" //MPISVD
#include "qlten/utility/utils_inl.h"
#include "qlten/utility/timer.h"

using namespace qlten;
using namespace std;

using U1QN = special_qn::U1QN;
using U1U1QN = special_qn::U1U1QN;

using IndexT1 = Index<U1QN>;
using IndexT = Index<U1U1QN>;

using QNSctT = QNSector<U1U1QN>;
using QNSctVecT = QNSectorVec<U1U1QN>;

using DQLTensor1 = QLTensor<QLTEN_Double, U1QN>;
using ZQLTensor1 = QLTensor<QLTEN_Complex, U1QN>;

using DQLTensor2 = QLTensor<QLTEN_Double, U1U1QN>;

//helper
Index<U1QN> RandIndex(const unsigned qn_sct_num,  //how many quantum number sectors?
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

struct TestMPISvd : public testing::Test {
  const U1QN qn0 = U1QN(0);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank;

  DQLTensor1 dstate;
  ZQLTensor1 zstate;
  void SetUp(void) {
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    MPI_Comm_rank(comm, &rank);
    if (rank != kMPIMasterRank) {
      delete listeners.Release(listeners.default_result_printer());
    }
    if (rank == kMPIMasterRank) {
      auto index1_in = RandIndex(20, 30, qlten::IN);
      auto index2_in = RandIndex(4, 5, qlten::IN);
      auto index1_out = RandIndex(4, 5, qlten::OUT);
      auto index2_out = RandIndex(20, 30, qlten::OUT);
      dstate = DQLTensor1({index2_out, index1_in, index2_in, index1_out});
      dstate.Random(qn0);
      zstate = ToComplex(dstate);

      std::cout << "Randomly generate double and complex tensors." << "\n";
      cout << "Concise Infos of the double tensor: \n";
      dstate.ConciseShow();
    }
  }
};

template<typename TenElemT, typename QNT>
void RunTestSvdCase(
    QLTensor<TenElemT, QNT> &t,
    const size_t &svd_ldims,
    const QNT left_div,
    const double &trunc_err,
    const size_t &dmin,
    const size_t &dmax,
    const MPI_Comm &comm) {
  using Tensor = QLTensor<TenElemT, QNT>;
  using DTensor = QLTensor<QLTEN_Double, QNT>;
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == kMPIMasterRank) {
    Tensor u, vt, u2, vt2;
    DTensor s, s2;
    double actual_trunc_err, actual_trunc_err2;
    size_t D, D2;

    Timer parallel_svd_timer("Parallel SVD");
    MPISVDMaster(&t,
                 svd_ldims, left_div,
                 trunc_err, dmin, dmax,
                 &u, &s, &vt, &actual_trunc_err, &D,
                 comm
    );
    parallel_svd_timer.PrintElapsed();

    Timer single_processor_svd_timer("Single processor SVD");
    SVD(&t,
        svd_ldims, left_div,
        trunc_err, dmin, dmax,
        &u2, &s2, &vt2, &actual_trunc_err2, &D2
    );
    single_processor_svd_timer.PrintElapsed();
    EXPECT_EQ(D, D2);
    EXPECT_NEAR(actual_trunc_err2, actual_trunc_err, 1e-13);
    DTensor ds_diff = s + (-s2);
    EXPECT_NEAR(ds_diff.Normalize() / s.Normalize(), 0.0, 1e-13);
  } else {
    MPISVDSlave<TenElemT>(comm);
  }
}

TEST_F(TestMPISvd, SVD_RandState) {
  size_t dmax = 1;
  if (rank == kMPIMasterRank) {
    dmax = dstate.GetIndexes()[0].dim();
  }
  RunTestSvdCase(dstate, 2, qn0, 1e-8, 1, dmax, comm);
  RunTestSvdCase(zstate, 2, qn0, 1e-8, 1, dmax, comm);
#ifdef ACTUALCOMBAT
  DQLTensor2 state_load;
  if (world.rank() == kMPIMasterRank) {
    std::string file = "state_load.qlten";
    if (access(file.c_str(), 4) != 0) {
      std::cout << "The progress doesn't access to read the file " << file << "!" << std::endl;
      exit(1);
    }
    std::ifstream ifs(file, std::ios::binary);
    if (!ifs.good()) {
      std::cout << "The progress can not read the file " << file << " correctly!" << std::endl;
      exit(1);
    }
    ifs >> state_load;
    std::cout << "The progress has loaded the tensors." << std::endl;
    cout << "Concise Info of tensors: \n";
    cout << "state_load.qlten:";
    state_load.ConciseShow();
  }
  if (world.rank() == kMPIMasterRank) {
    dmax = state_load.GetIndexes()[0].dim();
  }
  RunTestSvdCase(state_load, 2, U1U1QN(0, 0), 1e-8, 1, dmax, world);
#endif

}

int main(int argc, char *argv[]) {
  MPI_Init(NULL, NULL);
  int result = 0;
  ::testing::InitGoogleTest(&argc, argv);
  size_t thread_num;
  if (argc == 1) {// no input parameter
    thread_num = 12;
  } else {
    thread_num = atoi(argv[1]);
  }
  hp_numeric::SetTensorManipulationThreads(thread_num);
  result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}