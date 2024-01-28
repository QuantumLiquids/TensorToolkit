// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2021-7-29
*
* Description: QuantumLiquids/tensor project.  Unittests for tensor contraction with restriction on 1 sector
*/

#include <iostream>
#include <fstream>

#include "gtest/gtest.h"
#include "qlten/qltensor_all.h"
#include "qlten/tensor_manipulation/ten_ctrct.h"            // Contract
#include "qlten/tensor_manipulation/basic_operations.h"     // Dag
#include "qlten/tensor_manipulation/ten_ctrct_1sct.h"
#include "qlten/tensor_manipulation/ten_linear_combine.h"

#include "../testing_utility.h"

#include "qlten/utility/timer.h"
using namespace qlten;

using special_qn::U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DQLTensor = QLTensor<QLTEN_Double, U1QN>;
using ZQLTensor = QLTensor<QLTEN_Complex, U1QN>;

struct TestContraction : public testing::Test {
  std::string qn_nm = "qn";
  U1QN qn0 = U1QN({QNCard(qn_nm, U1QNVal(0))});
  U1QN qnp1 = U1QN({QNCard(qn_nm, U1QNVal(1))});
  U1QN qnp2 = U1QN({QNCard(qn_nm, U1QNVal(2))});
  U1QN qnm1 = U1QN({QNCard(qn_nm, U1QNVal(-1))});
  int d_s = 3;
  QNSctT qnsct0_s = QNSctT(qn0, d_s);
  QNSctT qnsctp1_s = QNSctT(qnp1, d_s);
  QNSctT qnsctm1_s = QNSctT(qnm1, d_s);
  int d_l = 10;
  QNSctT qnsct0_l = QNSctT(qn0, d_l);
  QNSctT qnsctp1_l = QNSctT(qnp1, d_l);
  QNSctT qnsctm1_l = QNSctT(qnm1, d_l);
  IndexT idx_in_s = IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, TenIndexDirType::IN);
  IndexT idx_out_s = IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, TenIndexDirType::OUT);
  IndexT idx_in_l = IndexT({qnsctm1_l, qnsct0_l, qnsctp1_l}, TenIndexDirType::IN);
  IndexT idx_out_l = IndexT({qnsctm1_l, qnsct0_l, qnsctp1_l}, TenIndexDirType::OUT);

  DQLTensor dten_1d_s = DQLTensor({idx_out_s});
  DQLTensor dten_1d_l = DQLTensor({idx_out_l});
  DQLTensor dten_2d_s = DQLTensor({idx_in_s, idx_out_s});
  DQLTensor dten_2d_l = DQLTensor({idx_in_l, idx_out_l});
  DQLTensor dten_3d_s = DQLTensor({idx_in_s, idx_out_s, idx_out_s});
  DQLTensor dten_3d_s3 = DQLTensor({idx_in_s, idx_in_s, idx_out_s});
  DQLTensor dten_3d_s4 = DQLTensor({idx_out_s, idx_in_s, idx_in_s});
  DQLTensor dten_3d_l = DQLTensor({idx_in_l, idx_out_l, idx_out_l});

  ZQLTensor zten_1d_s = ZQLTensor({idx_out_s});
  ZQLTensor zten_1d_l = ZQLTensor({idx_out_l});
  ZQLTensor zten_2d_s = ZQLTensor({idx_in_s, idx_out_s});
  ZQLTensor zten_2d_l = ZQLTensor({idx_in_l, idx_out_l});
  ZQLTensor zten_3d_s = ZQLTensor({idx_in_s, idx_out_s, idx_out_s});
  ZQLTensor zten_3d_s3 = ZQLTensor({idx_in_s, idx_in_s, idx_out_s});
  ZQLTensor zten_3d_s4 = ZQLTensor({idx_out_s, idx_in_s, idx_in_s});
  ZQLTensor zten_3d_l = ZQLTensor({idx_in_l, idx_out_l, idx_out_l});
};

TEST_F(TestContraction, 2DCase) {
  auto dten_2d_s2 = dten_2d_s;
  dten_2d_s.Random(qn0);
  dten_2d_s2.Random(qn0);

  DQLTensor dres_ten1, dres_ten2, dres_ten3;
  Contract1Sector(&dten_2d_s, 0, 0, &dten_2d_s2, {{1}, {0}}, &dres_ten1);
  Contract1Sector(&dten_2d_s, 0, 1, &dten_2d_s2, {{1}, {0}}, &dres_ten2);
  Contract1Sector(&dten_2d_s, 0, 2, &dten_2d_s2, {{1}, {0}}, &dres_ten3);
//   dres_ten1.Show();

  DQLTensor dres_ten_2d_s;
  Contract(&dten_2d_s, &dten_2d_s2, {{1}, {0}}, &dres_ten_2d_s);
//   dres_ten_2d_s.Show();
  EXPECT_EQ(dres_ten_2d_s, (dres_ten1 + dres_ten2 + dres_ten3));
}

TEST_F(TestContraction, 3DCase) {
  dten_3d_s.Random(qn0);
  dten_3d_s3.Random(qn0);

  size_t split_idx = 0;
  size_t num_qn = dten_3d_s.GetIndexes()[split_idx].GetQNSctNum();
  std::vector<DQLTensor *> split_tens(num_qn);
  for (size_t i = 0; i < num_qn; i++) {
    split_tens[i] = new DQLTensor();
    Contract1Sector(&dten_3d_s, split_idx, i, &dten_3d_s3, {{2}, {0}}, split_tens[i]);
  }
  std::vector<QLTEN_Double> coefs(num_qn, 1.0);
  DQLTensor sum_ten;
  LinearCombine(coefs, split_tens, 0.0, &sum_ten);

  DQLTensor res_ten;
  Contract(&dten_3d_s, &dten_3d_s3, {{2}, {0}}, &res_ten);
  EXPECT_EQ(sum_ten, res_ten);



  //check Collective Linear Combine
  std::vector<DQLTensor> split_tens2(num_qn);
  for (size_t i = 0; i < num_qn; i++) {
    split_tens2[i] = *split_tens[i];
  }
  DQLTensor sum_ten2;
  CollectiveLinearCombine(split_tens2, sum_ten2);

  EXPECT_EQ(sum_ten2, sum_ten);

  for (size_t i = 0; i < num_qn; i++) {
    delete split_tens[i];
  }

  /* not support from Aug 15
  //split for b
  split_idx = 1;
  num_qn = dten_3d_s3.GetIndexes()[split_idx].GetQNSctNum();
  split_tens = std::vector<DQLTensor *>(num_qn);
  for(size_t i=0;i<num_qn;i++){
      split_tens[i] = new DQLTensor();
      Contract1Sector(&dten_3d_s, &dten_3d_s3,split_idx, i ,{{2},{0}},split_tens[i]);
  }
  coefs = std::vector<QLTEN_Double>(num_qn,1.0);
  sum_ten = DQLTensor();
  LinearCombine(coefs, split_tens, 0.0, &sum_ten );

  EXPECT_EQ(sum_ten, res_ten);
  for(size_t i=0;i<num_qn;i++){
    delete split_tens[i];
  }
  */



}

#ifdef ACTUALCOMBAT
TEST(ActualCombat, SSHHubbardD15000){
  using U1U1QN = QN<U1QNVal,U1QNVal>;
  using DQLTensor2 = QLTensor<QLTEN_Double,U1U1QN>;
  std::string file = "mps_ten_l.qlten";
  std::ifstream ifs(file);
  if(!ifs.good()){
    std::cout << "opening file " << file <<" fails." <<std::endl;
    exit(1);
  }
  DQLTensor2 mps1;
  ifs >> mps1;
  ifs.close();

  file = "mps_ten_r.qlten";
  ifs.open(file);
  if(!ifs.good()){
    std::cout << "opening file " << file <<" fails." <<std::endl;
    exit(1);
  }
  DQLTensor2 mps2;
  ifs >> mps2;
  ifs.close();
  using std::vector;
  vector<vector<size_t>> contract_axes = {{2},{0}};
  DQLTensor2 initial_state;

  hp_numeric::SetTensorManipulationTotalThreads(20);
  std::cout << "\n";
  Timer contract_timer("directly contract");
  Contract(&mps1, &mps2, contract_axes, &initial_state);
  contract_timer.PrintElapsed();
  std::cout << "\n";

  
  Timer contract_split_timer("split contract");
  size_t split_idx = 0;
  size_t num_qn = mps1.GetIndexes()[split_idx].GetQNSctNum();
  std::vector<DQLTensor2> split_tens( num_qn );
  for(size_t i=0;i<num_qn;i++){
      Contract1Sector(&mps1, split_idx, i ,&mps2,contract_axes,&split_tens[i]);
  }
  // std::cout << "OK1" <<std::endl;
  DQLTensor2 sum_ten;
  Timer sum_timer("summation");
  CollectiveLinearCombine(split_tens, sum_ten);
  sum_timer.PrintElapsed();
  contract_split_timer.PrintElapsed();
  std::cout << "\n";
  EXPECT_EQ(initial_state, sum_ten);

  std::cout << "So we get an example of an initial state for Lanczos" <<std::endl;
  initial_state.ConciseShow();





  //Lanczos matrix*vector
  file = "lenv.qlten";
  ifs.open(file);
  if(!ifs.good()){
    std::cout << "opening file " << file <<" fails." <<std::endl;
    exit(1);
  }
  
  DQLTensor2 lenv;
  ifs >> lenv;
  ifs.close();


  file = "renv.qlten";
  ifs.open(file);
  if(!ifs.good()){
    std::cout << "opening file " << file <<" fails." <<std::endl;
    exit(1);
  }
  DQLTensor2 renv;
  ifs >> renv;
  ifs.close();

  file = "mpo_ten_l.qlten";
  ifs.open(file);
  if(!ifs.good()){
    std::cout << "opening file " << file <<" fails." <<std::endl;
    exit(1);
  }
  DQLTensor2 mpo1;
  ifs >> mpo1;
  ifs.close();

  file = "mpo_ten_r.qlten";
  ifs.open(file);
  if(!ifs.good()){
    std::cout << "opening file " << file <<" fails." <<std::endl;
    exit(1);
  }
  DQLTensor2 mpo2;
  ifs >> mpo2;
  ifs.close();


  DQLTensor2 next_state;
  std::cout << "\n";
  Timer lanczos_mat_vec_timer("lanczos_mat_vec");
  DQLTensor2 temp1, temp2;
  Contract(&lenv, &initial_state, {{0},{0}}, &temp1 );
  Contract(&temp1, &mpo1, {{0, 2}, {0, 1}}, &temp2);
  temp1 = DQLTensor2();
  Contract(&temp2, &mpo2,  {{4, 1}, {0, 1}}, &temp1);
  Contract(&temp1, &renv, {{4, 1}, {1, 0}}, &next_state);
  lanczos_mat_vec_timer.PrintElapsed();
  std::cout << "\n";

  sum_ten = DQLTensor2();


  
  split_idx = 2; //of lenv
  num_qn = lenv.GetIndexes()[split_idx].GetQNSctNum();
  split_tens = std::vector<DQLTensor2>( num_qn );
  Timer split_lanczos_mat_vec_timer("split_lanczos_mat_vec");
  for(size_t i=0;i<num_qn;i++){
    DQLTensor2 temp1, temp2;
    Contract1Sector(&lenv,split_idx, i, &initial_state, {{0},{0}}, &temp1 );
    Contract(&temp1, &mpo1, {{0, 2}, {0, 1}}, &temp2);
    temp1 = DQLTensor2();
    Contract(&temp2, &mpo2,  {{4, 1}, {0, 1}}, &temp1);
    Contract(&temp1, &renv, {{4, 1}, {1, 0}}, &split_tens[i]);
  }
  sum_ten =DQLTensor2();
  sum_timer.ClearAndRestart();
  CollectiveLinearCombine(split_tens, sum_ten);
  sum_timer.PrintElapsed();
  split_lanczos_mat_vec_timer.PrintElapsed();
  std::cout << "\n";
  std::cout << "difference(norm) between two tensors get from two method:\n";
  DQLTensor2 diff = next_state + (-sum_ten);
  double diff_norm = diff.Normalize();
  std::cout << diff_norm << std::endl;
  assert(fabs(diff_norm) < 1e-10);
}
#endif
