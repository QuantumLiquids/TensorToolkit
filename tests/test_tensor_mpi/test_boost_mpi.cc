// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2021-7-26
*
* Description: QuantumLiquids/tensor project. Unittests for Boost MPI.
* Usage:  mpirun -np 2 path_to_executable/test_boost_mpi
*/


#include "gtest/gtest.h"
#include "boost/mpi.hpp"
#include "qlten/utility/timer.h"

namespace mpi = boost::mpi;

using namespace qlten;

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

TEST_F(TestBoostMPI, SendRecvRawData) {
  size_t num_data = 1e6;
  using datatype = double;
  double *B = new datatype[num_data];
  if (world.rank() == 0) {
    std::cout << "data size :" << num_data << "\n";
    Timer transfer_array_boost("send_and_recv_array_by_boost");
    world.send(1, 0, B, num_data);
    world.recv(1, 2, B, num_data);
    double time = transfer_array_boost.PrintElapsed();
    double speed = double(2 * num_data * sizeof(datatype)) / 1000.0 / 1000.0 / 1000.0 / time; // unit : GB/s
    std::cout << "Speed : " << speed << "GB/s" << std::endl;
    std::cout << "The speed should almost the same with bare C MPI API." << std::endl;
  } else if (world.rank() == 1) {
    world.recv(0, 0, B, num_data);
    world.send(0, 2, B, num_data);
  }
  delete[] B;
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  boost::mpi::environment env;
  return RUN_ALL_TESTS();
}