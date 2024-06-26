// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 20201-9-28
*
* Description: QuantumLiquids/tensor project. Unit test for U(1) QN class.
*/

#include "gtest/gtest.h"
#include <sstream>
#include "qlten/qltensor/special_qn/fu1qn.h"   // test target
//#include "qlten/qlten.h"                       // Test contract case.
#include "../../testing_utility.h"             // RandInt

using namespace qlten::special_qn;

// Test case for default constructor and basic operators
TEST(fU1QNTest, DefaultConstructorAndOperators) {
  fU1QN qn1;
  EXPECT_EQ(qn1.dim(), 1);  // Check the dimension
  EXPECT_EQ(qn1.IsFermionParityEven(), true);  // Check if initially even
  EXPECT_EQ(qn1.IsFermionParityOdd(), false); // Check if initially odd

  fU1QN qn2(3);  // Create another instance with value 3
  EXPECT_EQ(qn2.dim(), 1);
  EXPECT_EQ(qn2.IsFermionParityEven(), false);
  EXPECT_EQ(qn2.IsFermionParityOdd(), true);

  // Test addition and subtraction
  fU1QN sum = qn1 + qn2;
  EXPECT_EQ(sum, qn2);
  EXPECT_EQ((qn1 - qn2), fU1QN(-3));
}

// Test case for serialization and deserialization
TEST(fU1QNTest, Serialization) {
  fU1QN qn(2);
  std::stringstream ss;
  ss << qn;

  fU1QN deserialized_qn;
  ss >> deserialized_qn;

  EXPECT_EQ(qn, deserialized_qn);
}

// Test case for unary negation operator
TEST(fU1QNTest, UnaryNegationOperator) {
  fU1QN qn(5);
  fU1QN negated_qn = -qn;
  EXPECT_EQ(negated_qn, fU1QN(-5));
}

// Test case for assignment operator
TEST(fU1QNTest, AssignmentOperator) {
  fU1QN qn1(7);
  fU1QN qn2 = qn1;
  EXPECT_EQ(qn1, qn2);
}

// Test case for custom zero element creation
TEST(fU1QNTest, ZeroElement) {
  fU1QN zero = fU1QN::Zero();
  EXPECT_EQ(zero, fU1QN(0));
}

// Test case for stream read and write operations
TEST(fU1QNTest, StreamOperations) {
  fU1QN qn(4);
  std::stringstream ss;
  qn.StreamWrite(ss);

  fU1QN deserialized_qn;
  deserialized_qn.StreamRead(ss);

  EXPECT_EQ(qn, deserialized_qn);
}

// Test case for hash calculation
TEST(fU1QNTest, HashCalculation) {
  fU1QN qn(3);
  size_t hash = qn.Hash();
  EXPECT_NE(hash, 0);  // Ensure hash is not zero
}

// Test case for comparison operators
TEST(fU1QNTest, ComparisonOperators) {
  fU1QN qn1(5);
  fU1QN qn2(5);
  fU1QN qn3(7);

  EXPECT_EQ(qn1, qn2);  // Check equality
  EXPECT_NE(qn1, qn3);  // Check inequality
}

// Test case for showing the object
TEST(fU1QNTest, ShowFunction) {
  fU1QN qn(6);
  std::stringstream ss;
  testing::internal::CaptureStdout();
  qn.Show();
  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(output, "fU1QN:  6\n");
}

// Additional tests can be added to cover more edge cases and functionalities

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}




