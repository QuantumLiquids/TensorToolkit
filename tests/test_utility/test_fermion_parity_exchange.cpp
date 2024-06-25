/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2024-96-23
*
* Description: QuantumLiquids/tensor project. Unit test for U(1) quantum number value.
*/


#include "gtest/gtest.h"
#include "qlten/utility/utils_inl.h"

using namespace qlten;

struct TestFermionicInplaceReorder : public testing::Test {
  std::vector<int> object1 = {-4, -2, -5, -7};
  std::vector<size_t> order1 = {2, 1, 3, 0};
  std::vector<bool> signs1 = {true, false, true, false};
  std::vector<int> object1_reorder_res = {-5, -2, -7, -4};
  bool exchange_sign_res1 = -1;

  std::vector<double> object2 = {3.5, 7.8, 1.2, -0.01};
  std::vector<size_t> order2 = {3, 1, 0, 2};
  std::vector<bool> signs2 = {true, true, true, false};
  std::vector<double> object2_reorder_res = {-0.01, 7.8, 3.5, 1.2};
  bool exchange_sign_res2 = -1;
};

TEST_F(TestFermionicInplaceReorder, FermionicInplaceReorder) {
  bool exchange_sign1 = FermionicInplaceReorder(object1, order1, signs1);
  EXPECT_EQ(exchange_sign1, exchange_sign_res1);
  EXPECT_EQ(object1, object1_reorder_res);

  bool exchange_sign2 = FermionicInplaceReorder(object2, order2, signs2);
  EXPECT_EQ(exchange_sign2, exchange_sign_res2);
  EXPECT_EQ(object2, object2_reorder_res);
}