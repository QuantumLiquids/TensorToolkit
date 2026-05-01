// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2026-05-01
*
* Description: QuantumLiquids/tensor project. Unittests for diagonal tensor product.
*/

#include <atomic>
#include <cstdlib>
#include <functional>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>

#include "gtest/gtest.h"
#include "qlten/qltensor_all.h"
#include "qlten/tensor_manipulation/ten_diagonal_tensor_product.h"

using namespace qlten;

using U1QN = QN<U1QNVal>;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using DQLTensor = QLTensor<QLTEN_Double, U1QN>;
using ZQLTensor = QLTensor<QLTEN_Complex, U1QN>;

namespace {

std::atomic<bool> g_count_allocations(false);
std::atomic<size_t> g_allocation_count(0);

}  // namespace

void *operator new(std::size_t size) {
  if (g_count_allocations.load(std::memory_order_relaxed)) {
    g_allocation_count.fetch_add(1, std::memory_order_relaxed);
  }
  if (void *ptr = std::malloc(size)) {
    return ptr;
  }
  throw std::bad_alloc();
}

void *operator new[](std::size_t size) {
  if (g_count_allocations.load(std::memory_order_relaxed)) {
    g_allocation_count.fetch_add(1, std::memory_order_relaxed);
  }
  if (void *ptr = std::malloc(size)) {
    return ptr;
  }
  throw std::bad_alloc();
}

void operator delete(void *ptr) noexcept {
  std::free(ptr);
}

void operator delete[](void *ptr) noexcept {
  std::free(ptr);
}

void operator delete(void *ptr, std::size_t) noexcept {
  std::free(ptr);
}

void operator delete[](void *ptr, std::size_t) noexcept {
  std::free(ptr);
}

namespace {

U1QN MakeQN(const int qn) {
  return U1QN({QNCard("qn", U1QNVal(qn))});
}

void ExpectInvalidArgumentContains(
    const std::function<void()> &fn,
    const std::string &message_part
) {
  try {
    fn();
    FAIL() << "Expected std::invalid_argument containing: " << message_part;
  } catch (const std::invalid_argument &e) {
    EXPECT_NE(std::string(e.what()).find(message_part), std::string::npos)
        << e.what();
  }
}

void SetAllElements(DQLTensor &ten, const QLTEN_Double value) {
  for (const auto &coors : GenAllCoors(ten.GetShape())) {
    ten.SetElem(coors, value);
  }
}

void FillLeftDiagonal(DQLTensor &left_op) {
  const auto lb_dim = left_op.GetIndex(0).dim();
  const auto ls_dim = left_op.GetIndex(1).dim();
  for (size_t lb = 0; lb < lb_dim; ++lb) {
    for (size_t ls = 0; ls < ls_dim; ++ls) {
      left_op.SetElem(
          {lb, ls, lb, ls},
          QLTEN_Double(10 * (lb + 1) + (ls + 1))
      );
    }
  }
}

void FillRightDiagonal(DQLTensor &right_op) {
  const auto rs_dim = right_op.GetIndex(0).dim();
  const auto rb_dim = right_op.GetIndex(1).dim();
  for (size_t rs = 0; rs < rs_dim; ++rs) {
    for (size_t rb = 0; rb < rb_dim; ++rb) {
      right_op.SetElem(
          {rs, rb, rs, rb},
          QLTEN_Double(100 * (rs + 1) + 3 * (rb + 1))
      );
    }
  }
}

void FillLeftDiagonal(ZQLTensor &left_op) {
  const auto lb_dim = left_op.GetIndex(0).dim();
  const auto ls_dim = left_op.GetIndex(1).dim();
  for (size_t lb = 0; lb < lb_dim; ++lb) {
    for (size_t ls = 0; ls < ls_dim; ++ls) {
      left_op.SetElem(
          {lb, ls, lb, ls},
          QLTEN_Complex(10 * (lb + 1) + (ls + 1), 2 * lb + ls + 1)
      );
    }
  }
}

void FillRightDiagonal(ZQLTensor &right_op) {
  const auto rs_dim = right_op.GetIndex(0).dim();
  const auto rb_dim = right_op.GetIndex(1).dim();
  for (size_t rs = 0; rs < rs_dim; ++rs) {
    for (size_t rb = 0; rb < rb_dim; ++rb) {
      right_op.SetElem(
          {rs, rb, rs, rb},
          QLTEN_Complex(100 * (rs + 1) + 3 * (rb + 1), rs + 4 * rb + 2)
      );
    }
  }
}

template<typename ElemT>
void ExpectComplexNear(const ElemT &actual, const ElemT &expected) {
  EXPECT_DOUBLE_EQ(actual.real(), expected.real());
  EXPECT_DOUBLE_EQ(actual.imag(), expected.imag());
}

}  // namespace

TEST(TestDiagonalTensorProduct, SingleSectorAccumulatesOuterProduct) {
  const auto qn0 = MakeQN(0);
  const IndexT lb_out({QNSctT(qn0, 2)}, TenIndexDirType::OUT);
  const IndexT ls_out({QNSctT(qn0, 3)}, TenIndexDirType::OUT);
  const IndexT rs_out({QNSctT(qn0, 2)}, TenIndexDirType::OUT);
  const IndexT rb_out({QNSctT(qn0, 2)}, TenIndexDirType::OUT);

  DQLTensor left_op(
      {lb_out, ls_out, InverseIndex(lb_out), InverseIndex(ls_out)});
  DQLTensor right_op(
      {InverseIndex(rs_out), InverseIndex(rb_out), rs_out, rb_out});
  DQLTensor out({lb_out, ls_out, rs_out, rb_out});

  FillLeftDiagonal(left_op);
  FillRightDiagonal(right_op);
  out.Fill(qn0, QLTEN_Double(0));

  DiagonalTensorProductAccumulate(left_op, right_op, out);

  for (const auto &coors : GenAllCoors(out.GetShape())) {
    const auto expected =
        left_op.GetElem({coors[0], coors[1], coors[0], coors[1]}) *
        right_op.GetElem({coors[2], coors[3], coors[2], coors[3]});
    EXPECT_DOUBLE_EQ(out.GetElem(coors), expected);
  }
}

TEST(TestDiagonalTensorProduct, AppliesAlphaAndBeta) {
  const auto qn0 = MakeQN(0);
  const IndexT lb_out({QNSctT(qn0, 2)}, TenIndexDirType::OUT);
  const IndexT ls_out({QNSctT(qn0, 2)}, TenIndexDirType::OUT);
  const IndexT rs_out({QNSctT(qn0, 2)}, TenIndexDirType::OUT);
  const IndexT rb_out({QNSctT(qn0, 1)}, TenIndexDirType::OUT);

  DQLTensor left_op(
      {lb_out, ls_out, InverseIndex(lb_out), InverseIndex(ls_out)});
  DQLTensor right_op(
      {InverseIndex(rs_out), InverseIndex(rb_out), rs_out, rb_out});
  DQLTensor out({lb_out, ls_out, rs_out, rb_out});

  FillLeftDiagonal(left_op);
  FillRightDiagonal(right_op);
  out.Fill(qn0, QLTEN_Double(3));

  DiagonalTensorProductAccumulate(
      left_op, right_op, out, QLTEN_Double(2), QLTEN_Double(-0.5));

  for (const auto &coors : GenAllCoors(out.GetShape())) {
    const auto expected = QLTEN_Double(-1.5) +
        QLTEN_Double(2) *
            left_op.GetElem({coors[0], coors[1], coors[0], coors[1]}) *
            right_op.GetElem({coors[2], coors[3], coors[2], coors[3]});
    EXPECT_DOUBLE_EQ(out.GetElem(coors), expected);
  }
}

TEST(TestDiagonalTensorProduct, MultiBlockSparseLayoutMapsByBlockCoordinates) {
  const auto qn0 = MakeQN(0);
  const auto qn1 = MakeQN(1);
  const IndexT lb_out(
      {QNSctT(qn0, 2), QNSctT(qn1, 1)}, TenIndexDirType::OUT);
  const IndexT ls_out(
      {QNSctT(qn0, 1), QNSctT(qn1, 2)}, TenIndexDirType::OUT);
  const IndexT rs_out(
      {QNSctT(qn0, 1), QNSctT(qn1, 2)}, TenIndexDirType::OUT);
  const IndexT rb_out(
      {QNSctT(qn0, 2), QNSctT(qn1, 1)}, TenIndexDirType::OUT);

  DQLTensor left_op(
      {lb_out, ls_out, InverseIndex(lb_out), InverseIndex(ls_out)});
  DQLTensor right_op(
      {InverseIndex(rs_out), InverseIndex(rb_out), rs_out, rb_out});
  DQLTensor out({lb_out, ls_out, rs_out, rb_out});

  for (size_t lb = 0; lb < lb_out.dim(); ++lb) {
    for (size_t ls = 0; ls < ls_out.dim(); ++ls) {
      if (lb >= 2 && ls == 0) {
        continue;
      }
      left_op.SetElem(
          {lb, ls, lb, ls},
          QLTEN_Double(1 + 10 * lb + ls)
      );
    }
  }
  for (size_t rs = 0; rs < rs_out.dim(); ++rs) {
    for (size_t rb = 0; rb < rb_out.dim(); ++rb) {
      if (rs == 0 && rb >= 2) {
        continue;
      }
      right_op.SetElem(
          {rs, rb, rs, rb},
          QLTEN_Double(2 + 100 * rs + 5 * rb)
      );
    }
  }
  SetAllElements(out, QLTEN_Double(1));

  DiagonalTensorProductAccumulate(
      left_op, right_op, out, QLTEN_Double(1.25), QLTEN_Double(0.5));

  for (const auto &coors : GenAllCoors(out.GetShape())) {
    const auto expected = QLTEN_Double(0.5) +
        QLTEN_Double(1.25) *
            left_op.GetElem({coors[0], coors[1], coors[0], coors[1]}) *
            right_op.GetElem({coors[2], coors[3], coors[2], coors[3]});
    EXPECT_DOUBLE_EQ(out.GetElem(coors), expected);
  }
}

TEST(TestDiagonalTensorProduct, ThrowsClearDiagnosticsForMismatches) {
  const auto qn0 = MakeQN(0);
  const IndexT space({QNSctT(qn0, 2)}, TenIndexDirType::OUT);
  const IndexT bad_space({QNSctT(qn0, 3)}, TenIndexDirType::OUT);

  DQLTensor rank2({space, InverseIndex(space)});
  DQLTensor left_op(
      {space, space, InverseIndex(space), InverseIndex(space)});
  DQLTensor right_op(
      {InverseIndex(space), InverseIndex(space), space, space});
  DQLTensor out({space, space, space, space});
  DQLTensor bad_out({bad_space, space, space, space});

  ExpectInvalidArgumentContains(
      [&]() { DiagonalTensorProductAccumulate(rank2, right_op, out); },
      "rank-4"
  );
  ExpectInvalidArgumentContains(
      [&]() { DiagonalTensorProductAccumulate(left_op, right_op, bad_out); },
      "out index 0"
  );
}

TEST(TestDiagonalTensorProduct, LargeDoubleBlockAppliesBetaOnBlasPath) {
  const auto qn0 = MakeQN(0);
  const IndexT lb_out({QNSctT(qn0, 8)}, TenIndexDirType::OUT);
  const IndexT ls_out({QNSctT(qn0, 2)}, TenIndexDirType::OUT);
  const IndexT rs_out({QNSctT(qn0, 4)}, TenIndexDirType::OUT);
  const IndexT rb_out({QNSctT(qn0, 2)}, TenIndexDirType::OUT);

  DQLTensor left_op(
      {lb_out, ls_out, InverseIndex(lb_out), InverseIndex(ls_out)});
  DQLTensor right_op(
      {InverseIndex(rs_out), InverseIndex(rb_out), rs_out, rb_out});
  DQLTensor out({lb_out, ls_out, rs_out, rb_out});

  FillLeftDiagonal(left_op);
  FillRightDiagonal(right_op);
  out.Fill(qn0, QLTEN_Double(7));

  DiagonalTensorProductAccumulate(
      left_op, right_op, out, QLTEN_Double(-1.5), QLTEN_Double(0.25));

  for (const auto &coors : GenAllCoors(out.GetShape())) {
    const auto expected = QLTEN_Double(1.75) -
        QLTEN_Double(1.5) *
            left_op.GetElem({coors[0], coors[1], coors[0], coors[1]}) *
            right_op.GetElem({coors[2], coors[3], coors[2], coors[3]});
    EXPECT_DOUBLE_EQ(out.GetElem(coors), expected);
  }
}

TEST(TestDiagonalTensorProduct, LargeComplexBlockUsesUnconjugatedGer) {
  const auto qn0 = MakeQN(0);
  const IndexT lb_out({QNSctT(qn0, 4)}, TenIndexDirType::OUT);
  const IndexT ls_out({QNSctT(qn0, 4)}, TenIndexDirType::OUT);
  const IndexT rs_out({QNSctT(qn0, 3)}, TenIndexDirType::OUT);
  const IndexT rb_out({QNSctT(qn0, 3)}, TenIndexDirType::OUT);

  ZQLTensor left_op(
      {lb_out, ls_out, InverseIndex(lb_out), InverseIndex(ls_out)});
  ZQLTensor right_op(
      {InverseIndex(rs_out), InverseIndex(rb_out), rs_out, rb_out});
  ZQLTensor out({lb_out, ls_out, rs_out, rb_out});

  FillLeftDiagonal(left_op);
  FillRightDiagonal(right_op);
  out.Fill(qn0, QLTEN_Complex(2.0, -3.0));

  const QLTEN_Complex alpha(0.25, -0.75);
  const QLTEN_Complex beta(-0.5, 0.5);
  DiagonalTensorProductAccumulate(left_op, right_op, out, alpha, beta);

  for (const auto &coors : GenAllCoors(out.GetShape())) {
    const auto expected = beta * QLTEN_Complex(2.0, -3.0) +
        alpha *
            left_op.GetElem({coors[0], coors[1], coors[0], coors[1]}) *
            right_op.GetElem({coors[2], coors[3], coors[2], coors[3]});
    ExpectComplexNear(out.GetElem(coors), expected);
  }
}

TEST(TestDiagonalTensorProduct, LargeMissingOperatorBlockOnlyScalesByBeta) {
  const auto qn0 = MakeQN(0);
  const auto qn1 = MakeQN(1);
  const IndexT lb_out(
      {QNSctT(qn0, 8), QNSctT(qn1, 2)}, TenIndexDirType::OUT);
  const IndexT ls_out(
      {QNSctT(qn0, 2), QNSctT(qn1, 2)}, TenIndexDirType::OUT);
  const IndexT rs_out({QNSctT(qn0, 4)}, TenIndexDirType::OUT);
  const IndexT rb_out({QNSctT(qn0, 2)}, TenIndexDirType::OUT);

  DQLTensor left_op(
      {lb_out, ls_out, InverseIndex(lb_out), InverseIndex(ls_out)});
  DQLTensor right_op(
      {InverseIndex(rs_out), InverseIndex(rb_out), rs_out, rb_out});
  DQLTensor out({lb_out, ls_out, rs_out, rb_out});

  for (size_t lb = 0; lb < 8; ++lb) {
    for (size_t ls = 0; ls < 2; ++ls) {
      left_op.SetElem({lb, ls, lb, ls}, QLTEN_Double(3 + lb + ls));
    }
  }
  FillRightDiagonal(right_op);
  SetAllElements(out, QLTEN_Double(5));
  out.SetElem({8, 2, 0, 0}, std::numeric_limits<QLTEN_Double>::quiet_NaN());

  DiagonalTensorProductAccumulate(
      left_op, right_op, out, QLTEN_Double(2), QLTEN_Double(0));

  for (const auto &coors : GenAllCoors(out.GetShape())) {
    const bool left_diag_exists = coors[0] < 8 && coors[1] < 2;
    const auto expected = left_diag_exists
        ? QLTEN_Double(2) *
                left_op.GetElem({coors[0], coors[1], coors[0], coors[1]}) *
                right_op.GetElem({coors[2], coors[3], coors[2], coors[3]})
        : QLTEN_Double(0);
    EXPECT_DOUBLE_EQ(out.GetElem(coors), expected);
  }
}

TEST(TestDiagonalTensorProduct, LargeBlasPathDoesNotPackDiagonalVectors) {
  const auto qn0 = MakeQN(0);
  const IndexT lb_out({QNSctT(qn0, 8)}, TenIndexDirType::OUT);
  const IndexT ls_out({QNSctT(qn0, 2)}, TenIndexDirType::OUT);
  const IndexT rs_out({QNSctT(qn0, 4)}, TenIndexDirType::OUT);
  const IndexT rb_out({QNSctT(qn0, 2)}, TenIndexDirType::OUT);

  DQLTensor left_op(
      {lb_out, ls_out, InverseIndex(lb_out), InverseIndex(ls_out)});
  DQLTensor right_op(
      {InverseIndex(rs_out), InverseIndex(rb_out), rs_out, rb_out});
  DQLTensor out({lb_out, ls_out, rs_out, rb_out});

  FillLeftDiagonal(left_op);
  FillRightDiagonal(right_op);
  out.Fill(qn0, QLTEN_Double(1));

  DiagonalTensorProductAccumulate(
      left_op, right_op, out, QLTEN_Double(0.5), QLTEN_Double(0.5));
  g_allocation_count.store(0, std::memory_order_relaxed);
  g_count_allocations.store(true, std::memory_order_relaxed);
  DiagonalTensorProductAccumulate(
      left_op, right_op, out, QLTEN_Double(0.5), QLTEN_Double(0.5));
  g_count_allocations.store(false, std::memory_order_relaxed);

  EXPECT_EQ(g_allocation_count.load(std::memory_order_relaxed), 0);
}
