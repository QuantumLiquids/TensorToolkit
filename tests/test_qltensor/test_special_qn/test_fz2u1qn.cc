/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2025-02-24
*
* Description: QuantumLiquids/tensor project. Unit test for fermion Z2 \prod bosonic U(1) QN class.
*/


#include "gtest/gtest.h"
#include <sstream>
#include "qlten/qltensor/special_qn/fz2u1qn.h" // test target

using namespace qlten::special_qn;

TEST(fZ2U1QNTest, Equality) {
  auto qn0 = fZ2U1QN(0,0);
  auto qn1 = fZ2U1QN(0,0);
  EXPECT_EQ(qn0, qn1);
}


TEST(fZ2U1QNTest, InEquality) {
  auto qn0 = fZ2U1QN(0,0);
  auto qn1 = fZ2U1QN(1, -1);
  EXPECT_NE(qn0, qn1);
}

