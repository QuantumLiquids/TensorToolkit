// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-29 09:33
*
* Description: QuantumLiquids/tensor project. Unittests for tensor contraction.
*/

#define QLTEN_COUNT_FLOPS 1


//#define PLAIN_TRANSPOSE 1



#include "qlten/qltensor_all.h"
#include "qlten/tensor_manipulation/ten_ctrct.h"            // Contract
#include "qlten/tensor_manipulation/basic_operations.h"     // Dag
#include "qlten/tensor_manipulation/contract_contiguous_axes.h"      // ContractContiguousAxes (+ legacy Contract wrapper)
#include "gtest/gtest.h"
#include "../testing_utility.h"
#include "qlten/utility/timer.h"

//Additional add mkl/aocl/openblas for CUDA Benchmark.
//Note add it at last otherwise some complex number type transfer error
#include "qlten/framework/hp_numeric/backend_selector.h"

using namespace qlten;

using U1QN = QN<U1QNVal>;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DQLTensor = QLTensor<QLTEN_Double, U1QN>;
using ZQLTensor = QLTensor<QLTEN_Complex, U1QN>;
using FQLTensor = QLTensor<QLTEN_Float, U1QN>;
using CFQLTensor = QLTensor<QLTEN_ComplexFloat, U1QN>;

template <typename T>
double GetEpsilon() { return kEpsilon; }

template <>
double GetEpsilon<QLTEN_Float>() { return 1e-6; }

template <>
double GetEpsilon<QLTEN_ComplexFloat>() { return 1e-6; }

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

  FQLTensor ften_1d_s = FQLTensor({idx_out_s});
  FQLTensor ften_1d_l = FQLTensor({idx_out_l});
  FQLTensor ften_2d_s = FQLTensor({idx_in_s, idx_out_s});
  FQLTensor ften_2d_l = FQLTensor({idx_in_l, idx_out_l});
  FQLTensor ften_3d_s = FQLTensor({idx_in_s, idx_out_s, idx_out_s});
  FQLTensor ften_3d_s3 = FQLTensor({idx_in_s, idx_in_s, idx_out_s});
  FQLTensor ften_3d_s4 = FQLTensor({idx_out_s, idx_in_s, idx_in_s});
  FQLTensor ften_3d_l = FQLTensor({idx_in_l, idx_out_l, idx_out_l});

  CFQLTensor cften_1d_s = CFQLTensor({idx_out_s});
  CFQLTensor cften_1d_l = CFQLTensor({idx_out_l});
  CFQLTensor cften_2d_s = CFQLTensor({idx_in_s, idx_out_s});
  CFQLTensor cften_2d_l = CFQLTensor({idx_in_l, idx_out_l});
  CFQLTensor cften_3d_s = CFQLTensor({idx_in_s, idx_out_s, idx_out_s});
  CFQLTensor cften_3d_s3 = CFQLTensor({idx_in_s, idx_in_s, idx_out_s});
  CFQLTensor cften_3d_s4 = CFQLTensor({idx_out_s, idx_in_s, idx_in_s});
  CFQLTensor cften_3d_l = CFQLTensor({idx_in_l, idx_out_l, idx_out_l});
};

// NOTE: AOCL's cblas.h only declares the legacy CBLAS_ORDER enum; MKL/OpenBLAS
// continue to accept it as an alias, so use it here to avoid backend branches.
inline void CblasGemm(
    const CBLAS_ORDER Layout,
    const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
    const size_t m, const size_t n, const size_t k,
    const qlten::QLTEN_Double alpha,
    const qlten::QLTEN_Double *a, const size_t lda,
    const qlten::QLTEN_Double *b, const size_t ldb,
    const qlten::QLTEN_Double beta,
    qlten::QLTEN_Double *c, const size_t ldc) {
  cblas_dgemm(
      Layout,
      transa, transb,
      m, n, k,
      alpha,
      a, lda,
      b, ldb,
      beta,
      c, ldc);
}

inline void CblasGemm(
    const CBLAS_ORDER Layout,
    const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
    const size_t m, const size_t n, const size_t k,
    const qlten::QLTEN_Complex alpha,
    const qlten::QLTEN_Complex *a, const size_t lda,
    const qlten::QLTEN_Complex *b, const size_t ldb,
    const qlten::QLTEN_Complex beta,
    qlten::QLTEN_Complex *c, const size_t ldc) {
  cblas_zgemm(
      Layout,
      transa, transb,
      m, n, k,
      &alpha,
      a, lda,
      b, ldb,
      &beta,
      c, ldc);
}

inline void CblasGemm(
    const CBLAS_ORDER Layout,
    const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
    const size_t m, const size_t n, const size_t k,
    const qlten::QLTEN_Float alpha,
    const qlten::QLTEN_Float *a, const size_t lda,
    const qlten::QLTEN_Float *b, const size_t ldb,
    const qlten::QLTEN_Float beta,
    qlten::QLTEN_Float *c, const size_t ldc) {
  cblas_sgemm(
      Layout,
      transa, transb,
      m, n, k,
      alpha,
      a, lda,
      b, ldb,
      beta,
      c, ldc);
}

inline void CblasGemm(
    const CBLAS_ORDER Layout,
    const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
    const size_t m, const size_t n, const size_t k,
    const qlten::QLTEN_ComplexFloat alpha,
    const qlten::QLTEN_ComplexFloat *a, const size_t lda,
    const qlten::QLTEN_ComplexFloat *b, const size_t ldb,
    const qlten::QLTEN_ComplexFloat beta,
    qlten::QLTEN_ComplexFloat *c, const size_t ldc) {
  cblas_cgemm(
      Layout,
      transa, transb,
      m, n, k,
      &alpha,
      a, lda,
      b, ldb,
      &beta,
      c, ldc);
}

template<typename TenElemT, typename QNT>
void RunTestTenCtrct1DCase(QLTensor<TenElemT, QNT> &t, const QNT &div) {
  t.Random(div);
  QLTensor<TenElemT, QNT> t_res;
  auto t_dag = Dag(t);
  Timer contract_timer("contract");
  size_t start_flops = flop;
  Contract(&t, &t_dag, {{0},
                        {0}}, &t_res);
  size_t end_flops = flop;
  double elapsed_time = contract_timer.Elapsed();
  std::cout << "flop = " << end_flops - start_flops << std::endl;
  double Gflops_s = (end_flops - start_flops) * 1.e-9 / elapsed_time;
  std::cout << "Gflops/s = " << Gflops_s << std::endl;
  TenElemT res = 0.0;
  for (auto &coors : GenAllCoors(t.GetShape())) {
    res += qlten::norm(t.GetElem(coors));
  }
  GtestExpectNear(t_res.GetElem({}), res, GetEpsilon<TenElemT>());
}

TEST_F(TestContraction, 1DCase) {
  RunTestTenCtrct1DCase(dten_1d_s, qn0);
  RunTestTenCtrct1DCase(dten_1d_s, qnp1);
  RunTestTenCtrct1DCase(dten_1d_s, qnm1);

  RunTestTenCtrct1DCase(zten_1d_s, qn0);
  RunTestTenCtrct1DCase(zten_1d_s, qnp1);
  RunTestTenCtrct1DCase(zten_1d_s, qnm1);

  RunTestTenCtrct1DCase(ften_1d_s, qn0);
  RunTestTenCtrct1DCase(ften_1d_s, qnp1);
  RunTestTenCtrct1DCase(ften_1d_s, qnm1);

  RunTestTenCtrct1DCase(cften_1d_s, qn0);
  RunTestTenCtrct1DCase(cften_1d_s, qnp1);
  RunTestTenCtrct1DCase(cften_1d_s, qnm1);
}

template<typename TenElemT, typename QNT>
void RunTestTenCtrct2DInOutInOutSectDegnDsCase1(
    const QLTensor<TenElemT, QNT> &ta,
    const QLTensor<TenElemT, QNT> &tb
) {
  auto m = ta.GetShape()[0];
  auto n = tb.GetShape()[1];
  auto k1 = ta.GetShape()[1];
  auto k2 = tb.GetShape()[0];
  auto ta_size = m * k1;
  auto tb_size = k2 * n;
  auto dense_ta = new TenElemT[ta_size];
  auto dense_tb = new TenElemT[tb_size];
  auto dense_res = new TenElemT[m * n];
  size_t idx = 0;
  for (auto &coors : GenAllCoors(ta.GetShape())) {
    dense_ta[idx] = ta.GetElem(coors);
    idx++;
  }
  idx = 0;
  for (auto &coors : GenAllCoors(ta.GetShape())) {
    dense_tb[idx] = tb.GetElem(coors);
    idx++;
  }
  CblasGemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      m, n, k1,
      1.0,
      dense_ta, k1,
      dense_tb, n,
      0.0,
      dense_res, n);
  QLTensor<TenElemT, QNT> res;
  Contract(&ta, &tb, {{1},
                      {0}}, &res);
  idx = 0;
  for (auto &coors : GenAllCoors(res.GetShape())) {
    GtestExpectNear(res.GetElem(coors), dense_res[idx], GetEpsilon<TenElemT>());
    idx++;
  }

  delete[] dense_ta;
  delete[] dense_tb;
  delete[] dense_res;
}

template<typename TenElemT, typename QNT>
void RunTestTenCtrct2DCase2(
    const QLTensor<TenElemT, QNT> &ta,
    const QLTensor<TenElemT, QNT> &tb
) {
  auto m = ta.GetShape()[0];
  auto n = tb.GetShape()[1];
  auto k1 = ta.GetShape()[1];
  auto k2 = tb.GetShape()[0];
  auto ta_size = m * k1;
  auto tb_size = k2 * n;
  auto dense_ta = new TenElemT[ta_size];
  auto dense_tb = new TenElemT[tb_size];
  size_t idx = 0;
  for (auto &coors : GenAllCoors(ta.GetShape())) {
    dense_ta[idx] = ta.GetElem(coors);
    idx++;
  }
  idx = 0;
  for (auto &coors : GenAllCoors(tb.GetShape())) {
    dense_tb[idx] = tb.GetElem({coors[1], coors[0]});
    idx++;
  }
  TenElemT res_scalar = 0.0;
  for (size_t i = 0; i < ta_size; ++i) {
    res_scalar += (dense_ta[i] * dense_tb[i]);
  }
  QLTensor<TenElemT, QNT> res;
  Contract(&ta, &tb, {{0, 1},
                      {1, 0}}, &res);
  GtestExpectNear(res.GetElem({}), res_scalar, GetEpsilon<TenElemT>());

  delete[] dense_ta;
  delete[] dense_tb;
}

TEST_F(TestContraction, 2DCase) {
  auto dten_2d_s2 = dten_2d_s;
  dten_2d_s.Random(qn0);
  dten_2d_s2.Random(qn0);
  RunTestTenCtrct2DInOutInOutSectDegnDsCase1(dten_2d_s, dten_2d_s2);
  RunTestTenCtrct2DCase2(dten_2d_s, dten_2d_s2);
  dten_2d_s.Random(qnp1);
  dten_2d_s2.Random(qn0);
  RunTestTenCtrct2DInOutInOutSectDegnDsCase1(dten_2d_s, dten_2d_s2);
  RunTestTenCtrct2DCase2(dten_2d_s, dten_2d_s2);
  dten_2d_s.Random(qnp1);
  dten_2d_s2.Random(qnm1);
  RunTestTenCtrct2DInOutInOutSectDegnDsCase1(dten_2d_s, dten_2d_s2);
  RunTestTenCtrct2DCase2(dten_2d_s, dten_2d_s2);

  auto zten_2d_s2 = zten_2d_s;
  zten_2d_s.Random(qn0);
  zten_2d_s2.Random(qn0);
  RunTestTenCtrct2DInOutInOutSectDegnDsCase1(zten_2d_s, zten_2d_s2);
  RunTestTenCtrct2DCase2(zten_2d_s, zten_2d_s2);
  zten_2d_s.Random(qnp1);
  zten_2d_s2.Random(qn0);
  RunTestTenCtrct2DInOutInOutSectDegnDsCase1(zten_2d_s, zten_2d_s2);
  RunTestTenCtrct2DCase2(zten_2d_s, zten_2d_s2);
  zten_2d_s.Random(qnp1);
  zten_2d_s2.Random(qnm1);
  RunTestTenCtrct2DInOutInOutSectDegnDsCase1(zten_2d_s, zten_2d_s2);
  RunTestTenCtrct2DCase2(zten_2d_s, zten_2d_s2);

  auto ften_2d_s2 = ften_2d_s;
  ften_2d_s.Random(qn0);
  ften_2d_s2.Random(qn0);
  RunTestTenCtrct2DInOutInOutSectDegnDsCase1(ften_2d_s, ften_2d_s2);
  RunTestTenCtrct2DCase2(ften_2d_s, ften_2d_s2);
  ften_2d_s.Random(qnp1);
  ften_2d_s2.Random(qn0);
  RunTestTenCtrct2DInOutInOutSectDegnDsCase1(ften_2d_s, ften_2d_s2);
  RunTestTenCtrct2DCase2(ften_2d_s, ften_2d_s2);
  ften_2d_s.Random(qnp1);
  ften_2d_s2.Random(qnm1);
  RunTestTenCtrct2DInOutInOutSectDegnDsCase1(ften_2d_s, ften_2d_s2);
  RunTestTenCtrct2DCase2(ften_2d_s, ften_2d_s2);

  auto cften_2d_s2 = cften_2d_s;
  cften_2d_s.Random(qn0);
  cften_2d_s2.Random(qn0);
  RunTestTenCtrct2DInOutInOutSectDegnDsCase1(cften_2d_s, cften_2d_s2);
  RunTestTenCtrct2DCase2(cften_2d_s, cften_2d_s2);
  cften_2d_s.Random(qnp1);
  cften_2d_s2.Random(qn0);
  RunTestTenCtrct2DInOutInOutSectDegnDsCase1(cften_2d_s, cften_2d_s2);
  RunTestTenCtrct2DCase2(cften_2d_s, cften_2d_s2);
  cften_2d_s.Random(qnp1);
  cften_2d_s2.Random(qnm1);
  RunTestTenCtrct2DInOutInOutSectDegnDsCase1(cften_2d_s, cften_2d_s2);
  RunTestTenCtrct2DCase2(cften_2d_s, cften_2d_s2);
}

template<typename TenElemT, typename QNT>
void RunTestTenCtrct3DCase1(
    const QLTensor<TenElemT, QNT> &ta,
    const QLTensor<TenElemT, QNT> &tb
) {
  auto m = ta.GetShape()[0] * ta.GetShape()[1];
  auto n = tb.GetShape()[1] * tb.GetShape()[2];
  auto k1 = ta.GetShape()[2];
  auto k2 = tb.GetShape()[0];
  auto ta_size = m * k1;
  auto tb_size = k2 * n;
  auto dense_ta = new TenElemT[ta_size];
  auto dense_tb = new TenElemT[tb_size];
  auto dense_res = new TenElemT[m * n];
  size_t idx = 0;
  for (auto &coors : GenAllCoors(ta.GetShape())) {
    dense_ta[idx] = ta.GetElem(coors);
    idx++;
  }
  idx = 0;
  for (auto &coors : GenAllCoors(tb.GetShape())) {
    dense_tb[idx] = tb.GetElem(coors);
    idx++;
  }
  CblasGemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      m, n, k1,
      1.0,
      dense_ta, k1,
      dense_tb, n,
      0.0,
      dense_res, n);
  QLTensor<TenElemT, QNT> res;
  Contract(&ta, &tb, {{2},
                      {0}}, &res);
  idx = 0;
  for (auto &coors : GenAllCoors(res.GetShape())) {
    const auto actual = res.GetElem(coors);
    const auto expected = dense_res[idx];
    GtestExpectNear(actual, expected, qlten::abs(expected) * GetEpsilon<TenElemT>());
    idx++;
  }

  delete[] dense_ta;
  delete[] dense_tb;
  delete[] dense_res;
}

template<typename TenElemT, typename QNT>
void RunTestTenCtrct3DCase2(
    const QLTensor<TenElemT, QNT> &ta,
    const QLTensor<TenElemT, QNT> &tb
) {
  auto m = ta.GetShape()[0];
  auto n = tb.GetShape()[2];
  auto k1 = ta.GetShape()[1] * ta.GetShape()[2];
  auto k2 = tb.GetShape()[0] * tb.GetShape()[1];
  auto ta_size = m * k1;
  auto tb_size = k2 * n;
  auto dense_ta = new TenElemT[ta_size];
  auto dense_tb = new TenElemT[tb_size];
  auto dense_res = new TenElemT[m * n];
  size_t idx = 0;
  for (auto &coors : GenAllCoors(ta.GetShape())) {
    dense_ta[idx] = ta.GetElem(coors);
    idx++;
  }
  idx = 0;
  for (auto &coors : GenAllCoors(tb.GetShape())) {
    dense_tb[idx] = tb.GetElem(coors);
    idx++;
  }
  CblasGemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      m, n, k1,
      1.0,
      dense_ta, k1,
      dense_tb, n,
      0.0,
      dense_res, n);
  QLTensor<TenElemT, QNT> res;
  Contract(&ta, &tb, {{1, 2},
                      {0, 1}}, &res);
  idx = 0;
  for (auto &coors : GenAllCoors(res.GetShape())) {
    GtestExpectNear(res.GetElem(coors), dense_res[idx], GetEpsilon<TenElemT>() * qlten::abs(res.GetElem(coors)));
    idx++;
  }

  delete[] dense_ta;
  delete[] dense_tb;
  delete[] dense_res;
}

template<typename TenElemT, typename QNT>
void RunTestTenCtrct3DCase3(
    const QLTensor<TenElemT, QNT> &ta,
    const QLTensor<TenElemT, QNT> &tb
) {
  auto m = ta.GetShape()[0];
  auto n = tb.GetShape()[2];
  auto k1 = ta.GetShape()[1] * ta.GetShape()[2];
  auto k2 = tb.GetShape()[0] * tb.GetShape()[1];
  auto ta_size = m * k1;
  auto tb_size = k2 * n;
  auto dense_ta = new TenElemT[ta_size];
  auto dense_tb = new TenElemT[tb_size];
  size_t idx = 0;
  for (auto &coors : GenAllCoors(ta.GetShape())) {
    dense_ta[idx] = ta.GetElem(coors);
    idx++;
  }
  idx = 0;
  for (auto &coors : GenAllCoors(tb.GetShape())) {
    dense_tb[idx] = tb.GetElem(coors);
    idx++;
  }
  TenElemT res_scalar = 0.0;
  for (size_t i = 0; i < ta_size; ++i) {
    res_scalar += (dense_ta[i] * dense_tb[i]);
  }
  QLTensor<TenElemT, QNT> res;
  Contract(&ta, &tb, {{0, 1, 2},
                      {0, 1, 2}}, &res);
  GtestExpectNear(res.GetElem({}), res_scalar, GetEpsilon<TenElemT>() * qlten::abs(res_scalar));

  delete[] dense_ta;
  delete[] dense_tb;
}

TEST_F(TestContraction, 3DCase) {
  auto dten_3d_s2 = dten_3d_s;
  dten_3d_s.Random(qn0);
  dten_3d_s2.Random(qn0);
  RunTestTenCtrct3DCase1(dten_3d_s, dten_3d_s2);
  dten_3d_s.Random(qnp1);
  dten_3d_s2.Random(qn0);
  RunTestTenCtrct3DCase1(dten_3d_s, dten_3d_s2);
  dten_3d_s.Random(qnp1);
  dten_3d_s2.Random(qnm1);
  RunTestTenCtrct3DCase1(dten_3d_s, dten_3d_s2);
  dten_3d_s.Random(qn0);
  dten_3d_s3.Random(qn0);
  RunTestTenCtrct3DCase2(dten_3d_s, dten_3d_s3);
  dten_3d_s.Random(qnp1);
  dten_3d_s3.Random(qn0);
  RunTestTenCtrct3DCase2(dten_3d_s, dten_3d_s3);
  dten_3d_s.Random(qnp1);
  dten_3d_s3.Random(qnm1);
  RunTestTenCtrct3DCase2(dten_3d_s, dten_3d_s3);
  dten_3d_s.Random(qn0);
  dten_3d_s4.Random(qn0);
  RunTestTenCtrct3DCase3(dten_3d_s, dten_3d_s4);
  dten_3d_s.Random(qnp1);
  dten_3d_s4.Random(qn0);
  RunTestTenCtrct3DCase3(dten_3d_s, dten_3d_s4);
  dten_3d_s.Random(qnp1);
  dten_3d_s4.Random(qnm1);
  RunTestTenCtrct3DCase3(dten_3d_s, dten_3d_s4);

  auto zten_3d_s2 = zten_3d_s;
  zten_3d_s.Random(qn0);
  zten_3d_s2.Random(qn0);
  RunTestTenCtrct3DCase1(zten_3d_s, zten_3d_s2);
  zten_3d_s.Random(qnp1);
  zten_3d_s2.Random(qn0);
  RunTestTenCtrct3DCase1(zten_3d_s, zten_3d_s2);
  zten_3d_s.Random(qnp1);
  zten_3d_s2.Random(qnm1);
  RunTestTenCtrct3DCase1(zten_3d_s, zten_3d_s2);
  zten_3d_s.Random(qn0);
  zten_3d_s3.Random(qn0);
  RunTestTenCtrct3DCase2(zten_3d_s, zten_3d_s3);
  zten_3d_s.Random(qnp1);
  zten_3d_s3.Random(qn0);
  RunTestTenCtrct3DCase2(zten_3d_s, zten_3d_s3);
  zten_3d_s.Random(qnp1);
  zten_3d_s3.Random(qnm1);
  RunTestTenCtrct3DCase2(zten_3d_s, zten_3d_s3);
  zten_3d_s.Random(qn0);
  zten_3d_s4.Random(qn0);
  RunTestTenCtrct3DCase3(zten_3d_s, zten_3d_s4);
  zten_3d_s.Random(qnp1);
  zten_3d_s4.Random(qn0);
  RunTestTenCtrct3DCase3(zten_3d_s, zten_3d_s4);
  zten_3d_s.Random(qnp1);
  zten_3d_s4.Random(qnm1);
  RunTestTenCtrct3DCase3(zten_3d_s, zten_3d_s4);

  auto ften_3d_s2 = ften_3d_s;
  ften_3d_s.Random(qn0);
  ften_3d_s2.Random(qn0);
  RunTestTenCtrct3DCase1(ften_3d_s, ften_3d_s2);
  ften_3d_s.Random(qnp1);
  ften_3d_s2.Random(qn0);
  RunTestTenCtrct3DCase1(ften_3d_s, ften_3d_s2);
  ften_3d_s.Random(qnp1);
  ften_3d_s2.Random(qnm1);
  RunTestTenCtrct3DCase1(ften_3d_s, ften_3d_s2);
  ften_3d_s.Random(qn0);
  ften_3d_s3.Random(qn0);
  RunTestTenCtrct3DCase2(ften_3d_s, ften_3d_s3);
  ften_3d_s.Random(qnp1);
  ften_3d_s3.Random(qn0);
  RunTestTenCtrct3DCase2(ften_3d_s, ften_3d_s3);
  ften_3d_s.Random(qnp1);
  ften_3d_s3.Random(qnm1);
  RunTestTenCtrct3DCase2(ften_3d_s, ften_3d_s3);
  ften_3d_s.Random(qn0);
  ften_3d_s4.Random(qn0);
  RunTestTenCtrct3DCase3(ften_3d_s, ften_3d_s4);
  ften_3d_s.Random(qnp1);
  ften_3d_s4.Random(qn0);
  RunTestTenCtrct3DCase3(ften_3d_s, ften_3d_s4);
  ften_3d_s.Random(qnp1);
  ften_3d_s4.Random(qnm1);
  RunTestTenCtrct3DCase3(ften_3d_s, ften_3d_s4);

  auto cften_3d_s2 = cften_3d_s;
  cften_3d_s.Random(qn0);
  cften_3d_s2.Random(qn0);
  RunTestTenCtrct3DCase1(cften_3d_s, cften_3d_s2);
  cften_3d_s.Random(qnp1);
  cften_3d_s2.Random(qn0);
  RunTestTenCtrct3DCase1(cften_3d_s, cften_3d_s2);
  cften_3d_s.Random(qnp1);
  cften_3d_s2.Random(qnm1);
  RunTestTenCtrct3DCase1(cften_3d_s, cften_3d_s2);
  cften_3d_s.Random(qn0);
  cften_3d_s3.Random(qn0);
  RunTestTenCtrct3DCase2(cften_3d_s, cften_3d_s3);
  cften_3d_s.Random(qnp1);
  cften_3d_s3.Random(qn0);
  RunTestTenCtrct3DCase2(cften_3d_s, cften_3d_s3);
  cften_3d_s.Random(qnp1);
  cften_3d_s3.Random(qnm1);
  RunTestTenCtrct3DCase2(cften_3d_s, cften_3d_s3);
  cften_3d_s.Random(qn0);
  cften_3d_s4.Random(qn0);
  RunTestTenCtrct3DCase3(cften_3d_s, cften_3d_s4);
  cften_3d_s.Random(qnp1);
  cften_3d_s4.Random(qn0);
  RunTestTenCtrct3DCase3(cften_3d_s, cften_3d_s4);
  cften_3d_s.Random(qnp1);
  cften_3d_s4.Random(qnm1);
  RunTestTenCtrct3DCase3(cften_3d_s, cften_3d_s4);
}

template<typename TenElemT, typename QNT>
bool BSDTCompareShell(
    const BlockSparseDataTensor<TenElemT, QNT> &bsdta,
    const BlockSparseDataTensor<TenElemT, QNT> &bsdtb
) {
  if (bsdta.IsScalar() != bsdtb.IsScalar()) {
    return false;
  } else if (bsdta.IsScalar()) {
    return true;
  }
  const auto &blk_idx_data_blk_map_a = bsdta.GetBlkIdxDataBlkMap();
  const auto &blk_idx_data_blk_map_b = bsdtb.GetBlkIdxDataBlkMap();
  if (blk_idx_data_blk_map_b.size() != blk_idx_data_blk_map_a.size()) { return false; }
  auto lhs_idx_blk_it = blk_idx_data_blk_map_a.begin();
  auto rhs_idx_blk_it = blk_idx_data_blk_map_b.begin();
  size_t data_blk_size = blk_idx_data_blk_map_a.size();
  for (size_t i = 0; i < data_blk_size; ++i) {
    if (lhs_idx_blk_it->first != rhs_idx_blk_it->first) { return false; }
    lhs_idx_blk_it++;
    rhs_idx_blk_it++;
  }
  return true;
}

template<typename TenElemT, typename QNT>
void RunTestTenMatBasedCtrct(
    const QLTensor<TenElemT, QNT> &ta,
    const std::vector<size_t> &ctrct_axes_a, //continuous number
    const QLTensor<TenElemT, QNT> &tb,
    const std::vector<size_t> &ctrct_axes_b
) {
  using TenT = QLTensor<TenElemT, QNT>;
  TenT res1, res2, res3, res4;
  Contract<TenElemT, QNT, true, true>(ta, tb,
                                      ctrct_axes_a[0], ctrct_axes_b[0],
                                      ctrct_axes_a.size(), res1);

  Contract<TenElemT, QNT, true, false>(ta, tb,
                                       ctrct_axes_a[0], ctrct_axes_b[0],
                                       ctrct_axes_a.size(), res2);

  Contract<TenElemT, QNT, false, true>(ta, tb,
                                       ctrct_axes_a[0], ctrct_axes_b[0],
                                       ctrct_axes_a.size(), res3);

  Contract<TenElemT, QNT, false, false>(ta, tb,
                                        ctrct_axes_a[0], ctrct_axes_b[0],
                                        ctrct_axes_a.size(), res4);
  TenT benchmark_res;
  const size_t rank_a = ta.Rank();
  const size_t rank_b = tb.Rank();
  if ((ctrct_axes_a[0] == 0 || ctrct_axes_a.back() == rank_a - 1) &&
      (ctrct_axes_b[0] == 0 || ctrct_axes_b.back() == rank_b - 1)) {
    Contract(&ta, &tb, {ctrct_axes_a, ctrct_axes_b}, &benchmark_res);
  } else {

    std::vector<size_t> trans_axes_a;
    trans_axes_a.reserve(rank_a);
    for (size_t i = ctrct_axes_a.back() + 1; i < rank_a; i++) {
      trans_axes_a.push_back(i);
    }
    for (size_t i = 0; i <= ctrct_axes_a.back(); i++) {
      trans_axes_a.push_back(i);
    }
    TenT ta_copy(ta), tb_copy(tb);
    ta_copy.Transpose(trans_axes_a);

    std::vector<size_t> trans_axes_b;
    trans_axes_b.reserve(rank_b);
    for (size_t i = ctrct_axes_b.back() + 1; i < rank_b; i++) {
      trans_axes_b.push_back(i);
    }
    for (size_t i = 0; i <= ctrct_axes_b.back(); i++) {
      trans_axes_b.push_back(i);
    }
    tb_copy.Transpose(trans_axes_b);

    std::vector<size_t> ctrct_axes_after_transpose_a, ctrct_axes_after_transpose_b;
    size_t ctrct_axes_size = ctrct_axes_a.size();
    for (size_t i = rank_a - ctrct_axes_size; i < rank_a; i++) {
      ctrct_axes_after_transpose_a.push_back(i);
    }
    for (size_t i = rank_b - ctrct_axes_size; i < rank_b; i++) {
      ctrct_axes_after_transpose_b.push_back(i);
    }

    Contract(&ta_copy, &tb_copy, {ctrct_axes_after_transpose_a, ctrct_axes_after_transpose_b}, &benchmark_res);
  }

  const BlockSparseDataTensor<TenElemT, QNT> &bsdt_res = benchmark_res.GetBlkSparDataTen();
  using RealT = typename RealTypeTrait<TenElemT>::type;
  EXPECT_EQ(benchmark_res.GetIndexes(), res1.GetIndexes());
  const BlockSparseDataTensor<TenElemT, QNT> &bsdt_res1 = res1.GetBlkSparDataTen();
  EXPECT_EQ(bsdt_res.GetActualRawDataSize(), bsdt_res1.GetActualRawDataSize());
  TenT diff1 = benchmark_res + (-res1);
  EXPECT_NEAR(diff1.Get2Norm() / std::max(res1.Get2Norm(), RealT(1e-5)), 0.0, GetEpsilon<TenElemT>());

  EXPECT_EQ(benchmark_res.GetIndexes(), res2.GetIndexes());
  TenT diff2 = benchmark_res + (-res2);
  EXPECT_NEAR(diff2.Normalize() / std::max(res2.Get2Norm(), RealT(1e-5)), 0.0, GetEpsilon<TenElemT>());

  EXPECT_EQ(benchmark_res.GetIndexes(), res3.GetIndexes());
  EXPECT_TRUE(BSDTCompareShell(bsdt_res, res3.GetBlkSparDataTen()));
  TenT diff3 = benchmark_res + (-res3);
  auto norm_diff3 = diff3.Get2Norm();
  auto norm_res3 = res3.Get2Norm();
  EXPECT_NEAR(norm_diff3 / std::max<RealT>(norm_res3, RealT(1e-5)), 0.0, GetEpsilon<TenElemT>() * 10);
  if (norm_diff3 / std::max<RealT>(norm_res3, RealT(1e-5)) > GetEpsilon<TenElemT>()) {
    std::cout << "norm_diff3 = " << std::scientific << norm_diff3 << std::endl;
    std::cout << "norm_res3 = " << norm_res3 << std::endl;
  }

  EXPECT_EQ(benchmark_res.GetIndexes(), res4.GetIndexes());
  TenT diff4 = benchmark_res + (-res4);
  EXPECT_NEAR(diff4.Get2Norm() / std::max(res4.Get2Norm(), RealT(1e-5)), 0.0, GetEpsilon<TenElemT>() * 10);
}

TEST_F(TestContraction, MatBasedContract) {
  auto dten_3d_s2 = dten_3d_s;
  dten_3d_s.Random(qn0);
  dten_3d_s2.Random(qn0);
  RunTestTenMatBasedCtrct(dten_3d_s, {2}, dten_3d_s2, {0});
  RunTestTenMatBasedCtrct(dten_3d_s, {1}, dten_3d_s2, {0});
  dten_3d_s.Random(qnp1);
  dten_3d_s2.Random(qn0);
  RunTestTenMatBasedCtrct(dten_3d_s, {2}, dten_3d_s2, {0});
  RunTestTenMatBasedCtrct(dten_3d_s, {1}, dten_3d_s2, {0});
  dten_3d_s.Random(qnp1);
  dten_3d_s2.Random(qnm1);
  RunTestTenMatBasedCtrct(dten_3d_s, {2}, dten_3d_s2, {0});
  RunTestTenMatBasedCtrct(dten_3d_s, {1}, dten_3d_s2, {0});
  dten_3d_s.Random(qn0);
  dten_3d_s3.Random(qn0);
  RunTestTenMatBasedCtrct(dten_3d_s, {1, 2}, dten_3d_s3, {0, 1});
  dten_3d_s.Random(qnp1);
  dten_3d_s3.Random(qn0);
  RunTestTenMatBasedCtrct(dten_3d_s, {1, 2}, dten_3d_s3, {0, 1});
  dten_3d_s.Random(qnp1);
  dten_3d_s3.Random(qnm1);
  RunTestTenMatBasedCtrct(dten_3d_s, {1, 2}, dten_3d_s3, {0, 1});
  dten_3d_s.Random(qn0);
  dten_3d_s4.Random(qn0);
  RunTestTenMatBasedCtrct(dten_3d_s, {0, 1, 2}, dten_3d_s4, {0, 1, 2});
  dten_3d_s.Random(qnp1);
  dten_3d_s4.Random(qn0);
  RunTestTenMatBasedCtrct(dten_3d_s, {0, 1, 2}, dten_3d_s4, {0, 1, 2});
  dten_3d_s.Random(qnp1);
  dten_3d_s4.Random(qnm1);
  RunTestTenMatBasedCtrct(dten_3d_s, {0, 1, 2}, dten_3d_s4, {0, 1, 2});

  auto zten_3d_s2 = zten_3d_s;
  zten_3d_s.Random(qn0);
  zten_3d_s2.Random(qn0);
  RunTestTenMatBasedCtrct(zten_3d_s, {2}, zten_3d_s2, {0});
  RunTestTenMatBasedCtrct(zten_3d_s, {1}, zten_3d_s2, {0});
  zten_3d_s.Random(qnp1);
  zten_3d_s2.Random(qn0);
  RunTestTenMatBasedCtrct(zten_3d_s, {2}, zten_3d_s2, {0});
  RunTestTenMatBasedCtrct(zten_3d_s, {1}, zten_3d_s2, {0});
  zten_3d_s.Random(qnp1);
  zten_3d_s2.Random(qnm1);
  RunTestTenMatBasedCtrct(zten_3d_s, {2}, zten_3d_s2, {0});
  RunTestTenMatBasedCtrct(zten_3d_s, {1}, zten_3d_s2, {0});
  zten_3d_s.Random(qn0);
  zten_3d_s3.Random(qn0);
  RunTestTenMatBasedCtrct(zten_3d_s, {1, 2}, zten_3d_s3, {0, 1});
  zten_3d_s.Random(qnp1);
  zten_3d_s3.Random(qn0);
  RunTestTenMatBasedCtrct(zten_3d_s, {1, 2}, zten_3d_s3, {0, 1});
  zten_3d_s.Random(qnp1);
  zten_3d_s3.Random(qnm1);
  RunTestTenMatBasedCtrct(zten_3d_s, {1, 2}, zten_3d_s3, {0, 1});
  zten_3d_s.Random(qn0);
  zten_3d_s4.Random(qn0);
  RunTestTenMatBasedCtrct(zten_3d_s, {0, 1, 2}, zten_3d_s4, {0, 1, 2});
  zten_3d_s.Random(qnp1);
  zten_3d_s4.Random(qn0);
  RunTestTenMatBasedCtrct(zten_3d_s, {0, 1, 2}, zten_3d_s4, {0, 1, 2});
  zten_3d_s.Random(qnp1);
  zten_3d_s4.Random(qnm1);
  RunTestTenMatBasedCtrct(zten_3d_s, {0, 1, 2}, zten_3d_s4, {0, 1, 2});

  auto ften_3d_s2 = ften_3d_s;
  ften_3d_s.Random(qn0);
  ften_3d_s2.Random(qn0);
  RunTestTenMatBasedCtrct(ften_3d_s, {2}, ften_3d_s2, {0});
  RunTestTenMatBasedCtrct(ften_3d_s, {1}, ften_3d_s2, {0});
  ften_3d_s.Random(qnp1);
  ften_3d_s2.Random(qn0);
  RunTestTenMatBasedCtrct(ften_3d_s, {2}, ften_3d_s2, {0});
  RunTestTenMatBasedCtrct(ften_3d_s, {1}, ften_3d_s2, {0});
  ften_3d_s.Random(qnp1);
  ften_3d_s2.Random(qnm1);
  RunTestTenMatBasedCtrct(ften_3d_s, {2}, ften_3d_s2, {0});
  RunTestTenMatBasedCtrct(ften_3d_s, {1}, ften_3d_s2, {0});
  ften_3d_s.Random(qn0);
  ften_3d_s3.Random(qn0);
  RunTestTenMatBasedCtrct(ften_3d_s, {1, 2}, ften_3d_s3, {0, 1});
  ften_3d_s.Random(qnp1);
  ften_3d_s3.Random(qn0);
  RunTestTenMatBasedCtrct(ften_3d_s, {1, 2}, ften_3d_s3, {0, 1});
  ften_3d_s.Random(qnp1);
  ften_3d_s3.Random(qnm1);
  RunTestTenMatBasedCtrct(ften_3d_s, {1, 2}, ften_3d_s3, {0, 1});
  ften_3d_s.Random(qn0);
  ften_3d_s4.Random(qn0);
  RunTestTenMatBasedCtrct(ften_3d_s, {0, 1, 2}, ften_3d_s4, {0, 1, 2});
  ften_3d_s.Random(qnp1);
  ften_3d_s4.Random(qn0);
  RunTestTenMatBasedCtrct(ften_3d_s, {0, 1, 2}, ften_3d_s4, {0, 1, 2});
  ften_3d_s.Random(qnp1);
  ften_3d_s4.Random(qnm1);
  RunTestTenMatBasedCtrct(ften_3d_s, {0, 1, 2}, ften_3d_s4, {0, 1, 2});

  auto cften_3d_s2 = cften_3d_s;
  cften_3d_s.Random(qn0);
  cften_3d_s2.Random(qn0);
  RunTestTenMatBasedCtrct(cften_3d_s, {2}, cften_3d_s2, {0});
  RunTestTenMatBasedCtrct(cften_3d_s, {1}, cften_3d_s2, {0});
  cften_3d_s.Random(qnp1);
  cften_3d_s2.Random(qn0);
  RunTestTenMatBasedCtrct(cften_3d_s, {2}, cften_3d_s2, {0});
  RunTestTenMatBasedCtrct(cften_3d_s, {1}, cften_3d_s2, {0});
  cften_3d_s.Random(qnp1);
  cften_3d_s2.Random(qnm1);
  RunTestTenMatBasedCtrct(cften_3d_s, {2}, cften_3d_s2, {0});
  RunTestTenMatBasedCtrct(cften_3d_s, {1}, cften_3d_s2, {0});
  cften_3d_s.Random(qn0);
  cften_3d_s3.Random(qn0);
  RunTestTenMatBasedCtrct(cften_3d_s, {1, 2}, cften_3d_s3, {0, 1});
  cften_3d_s.Random(qnp1);
  cften_3d_s3.Random(qn0);
  RunTestTenMatBasedCtrct(cften_3d_s, {1, 2}, cften_3d_s3, {0, 1});
  cften_3d_s.Random(qnp1);
  cften_3d_s3.Random(qnm1);
  RunTestTenMatBasedCtrct(cften_3d_s, {1, 2}, cften_3d_s3, {0, 1});
  cften_3d_s.Random(qn0);
  cften_3d_s4.Random(qn0);
  RunTestTenMatBasedCtrct(cften_3d_s, {0, 1, 2}, cften_3d_s4, {0, 1, 2});
  cften_3d_s.Random(qnp1);
  cften_3d_s4.Random(qn0);
  RunTestTenMatBasedCtrct(cften_3d_s, {0, 1, 2}, cften_3d_s4, {0, 1, 2});
  cften_3d_s.Random(qnp1);
  cften_3d_s4.Random(qnm1);
  RunTestTenMatBasedCtrct(cften_3d_s, {0, 1, 2}, cften_3d_s4, {0, 1, 2});
}

template<typename TenElemT, typename QNT>
void RunTestContractContiguousAxesDefault(
    const QLTensor<TenElemT, QNT> &ta,
    const std::vector<size_t> &ctrct_axes_a,
    const QLTensor<TenElemT, QNT> &tb,
    const std::vector<size_t> &ctrct_axes_b
) {
  using TenT = QLTensor<TenElemT, QNT>;
  TenT res_enum, res_bool;
  ContractContiguousAxes<TenElemT, QNT>(ta, tb,
                                        ctrct_axes_a[0], ctrct_axes_b[0],
                                        ctrct_axes_a.size(), res_enum);
  Contract<TenElemT, QNT, true, true>(ta, tb,
                                      ctrct_axes_a[0], ctrct_axes_b[0],
                                      ctrct_axes_a.size(), res_bool);
  using RealT = typename RealTypeTrait<TenElemT>::type;
  EXPECT_EQ(res_bool.GetIndexes(), res_enum.GetIndexes());
  TenT diff = res_bool + (-res_enum);
  EXPECT_NEAR(diff.Get2Norm() / std::max(res_enum.Get2Norm(), RealT(1e-5)),
              0.0, GetEpsilon<TenElemT>());
}

TEST_F(TestContraction, ContractContiguousAxesDefaultMatchesBoolTrueTrue) {
  auto dten_3d_s2 = dten_3d_s;
  dten_3d_s.Random(qn0);
  dten_3d_s2.Random(qn0);
  RunTestContractContiguousAxesDefault(dten_3d_s, {2}, dten_3d_s2, {0});
  // dten_3d_s3 is a fixture member; populate before use to mirror the
  // existing MatBasedContract test pattern at test_ten_ctrct.cc:745-747.
  dten_3d_s3.Random(qn0);
  RunTestContractContiguousAxesDefault(dten_3d_s, {1, 2}, dten_3d_s3, {0, 1});
}

TEST_F(TestContraction, ContractContiguousAxesMixedPrecisionMatchesBoolWrapper) {
  auto zten_3d_s2 = zten_3d_s;
  auto dten_3d_s2 = dten_3d_s;
  zten_3d_s.Random(qn0);
  zten_3d_s2.Random(qn0);
  dten_3d_s.Random(qn0);
  dten_3d_s2.Random(qn0);

  using ZTen = QLTensor<QLTEN_Complex, U1QN>;
  ZTen res_z_d_enum, res_z_d_bool;
  ContractContiguousAxes<QLTEN_Complex, U1QN>(zten_3d_s, dten_3d_s2,
                                              2, 0, 1, res_z_d_enum);
  Contract<QLTEN_Complex, U1QN, true, true>(zten_3d_s, dten_3d_s2,
                                            2, 0, 1, res_z_d_bool);
  ZTen diff_z_d = res_z_d_bool + (-res_z_d_enum);
  EXPECT_NEAR(diff_z_d.Get2Norm()
                  / std::max(res_z_d_enum.Get2Norm(), 1e-5),
              0.0, GetEpsilon<QLTEN_Complex>());

  ZTen res_d_z_enum, res_d_z_bool;
  ContractContiguousAxes<QLTEN_Complex, U1QN>(dten_3d_s, zten_3d_s2,
                                              2, 0, 1, res_d_z_enum);
  Contract<QLTEN_Complex, U1QN, true, true>(dten_3d_s, zten_3d_s2,
                                            2, 0, 1, res_d_z_bool);
  ZTen diff_d_z = res_d_z_bool + (-res_d_z_enum);
  EXPECT_NEAR(diff_d_z.Get2Norm()
                  / std::max(res_d_z_enum.Get2Norm(), 1e-5),
              0.0, GetEpsilon<QLTEN_Complex>());
}

template<typename TenElemT, typename QNT>
void RunTestContractContiguousAxesAllSides(
    const QLTensor<TenElemT, QNT> &ta,
    const std::vector<size_t> &ctrct_axes_a,
    const QLTensor<TenElemT, QNT> &tb,
    const std::vector<size_t> &ctrct_axes_b
) {
  using TenT = QLTensor<TenElemT, QNT>;
  using RealT = typename RealTypeTrait<TenElemT>::type;
  const auto eps = GetEpsilon<TenElemT>();

  TenT enum_TH, enum_HH, enum_TT, enum_HT;
  ContractContiguousAxes<TenElemT, QNT, CtrctSide::Tail, CtrctSide::Head>(
      ta, tb, ctrct_axes_a[0], ctrct_axes_b[0], ctrct_axes_a.size(), enum_TH);
  ContractContiguousAxes<TenElemT, QNT, CtrctSide::Head, CtrctSide::Head>(
      ta, tb, ctrct_axes_a[0], ctrct_axes_b[0], ctrct_axes_a.size(), enum_HH);
  ContractContiguousAxes<TenElemT, QNT, CtrctSide::Tail, CtrctSide::Tail>(
      ta, tb, ctrct_axes_a[0], ctrct_axes_b[0], ctrct_axes_a.size(), enum_TT);
  ContractContiguousAxes<TenElemT, QNT, CtrctSide::Head, CtrctSide::Tail>(
      ta, tb, ctrct_axes_a[0], ctrct_axes_b[0], ctrct_axes_a.size(), enum_HT);

  TenT bool_TT, bool_FT, bool_TF, bool_FF;
  Contract<TenElemT, QNT, true,  true >(ta, tb, ctrct_axes_a[0],
      ctrct_axes_b[0], ctrct_axes_a.size(), bool_TT);
  Contract<TenElemT, QNT, false, true >(ta, tb, ctrct_axes_a[0],
      ctrct_axes_b[0], ctrct_axes_a.size(), bool_FT);
  Contract<TenElemT, QNT, true,  false>(ta, tb, ctrct_axes_a[0],
      ctrct_axes_b[0], ctrct_axes_a.size(), bool_TF);
  Contract<TenElemT, QNT, false, false>(ta, tb, ctrct_axes_a[0],
      ctrct_axes_b[0], ctrct_axes_a.size(), bool_FF);

  auto expect_match = [&](TenT &lhs, TenT &rhs, const char *label) {
    EXPECT_EQ(lhs.GetIndexes(), rhs.GetIndexes()) << label;
    TenT diff = lhs + (-rhs);
    EXPECT_NEAR(diff.Get2Norm()
                    / std::max(rhs.Get2Norm(), RealT(1e-5)),
                0.0, eps * 10) << label;
  };
  expect_match(enum_TH, bool_TT, "<Tail,Head> vs <true,true>");
  expect_match(enum_HH, bool_FT, "<Head,Head> vs <false,true>");
  expect_match(enum_TT, bool_TF, "<Tail,Tail> vs <true,false>");
  expect_match(enum_HT, bool_FF, "<Head,Tail> vs <false,false>");
}

TEST_F(TestContraction, ContractContiguousAxesAllSidesMatchBoolWrapper) {
  auto dten_3d_s2 = dten_3d_s;
  dten_3d_s.Random(qn0);
  dten_3d_s2.Random(qn0);
  RunTestContractContiguousAxesAllSides(dten_3d_s, {2}, dten_3d_s2, {0});
  RunTestContractContiguousAxesAllSides(dten_3d_s, {1}, dten_3d_s2, {0});
  dten_3d_s.Random(qnp1);
  dten_3d_s2.Random(qnm1);
  // dten_3d_s3 is a fixture member; populate before use to mirror the
  // existing MatBasedContract test pattern at test_ten_ctrct.cc:745-747.
  dten_3d_s3.Random(qn0);
  RunTestContractContiguousAxesAllSides(dten_3d_s, {1, 2}, dten_3d_s3, {0, 1});
}

TEST_F(TestContraction, ContractTailHeadContiguousMatchesGenericContract) {
  dten_3d_s.Random(qn0);
  dten_3d_s3.Random(qn0);

  DQLTensor generic_result;
  Contract(&dten_3d_s, &dten_3d_s3, {{1, 2}, {0, 1}}, &generic_result);

  DQLTensor tail_head_result;
  ContiguousContractStats stats;
  ContractTailHeadContiguous(
      dten_3d_s, dten_3d_s3, 1, 0, 2, tail_head_result, &stats);

  EXPECT_EQ(tail_head_result.GetIndexes(), generic_result.GetIndexes());
  DQLTensor diff = tail_head_result + (-generic_result);
  EXPECT_NEAR(diff.Get2Norm() / std::max(generic_result.Get2Norm(), 1e-5),
              0.0, GetEpsilon<QLTEN_Double>() * 10);
  EXPECT_EQ(stats.transpose_prepare_calls, 0U);
  EXPECT_EQ(stats.transpose_prepare_bytes, 0U);
  EXPECT_GT(stats.raw_data_contract_tasks, 0U);
  EXPECT_GT(stats.gemm_calls, 0U);
}

TEST_F(TestContraction, ContractContiguousAxesReportsTransposePrepareStats) {
  dten_3d_s.Random(qn0);
  auto dten_3d_s2 = dten_3d_s;
  dten_3d_s2.Random(qn0);

  DQLTensor tail_head_result;
  ContiguousContractStats tail_head_stats;
  ContractContiguousAxes<QLTEN_Double, U1QN, CtrctSide::Tail, CtrctSide::Head>(
      dten_3d_s, dten_3d_s2, 2, 0, 1, tail_head_result, &tail_head_stats);

  DQLTensor head_head_result;
  ContiguousContractStats head_head_stats;
  ContractContiguousAxes<QLTEN_Double, U1QN, CtrctSide::Head, CtrctSide::Head>(
      dten_3d_s, dten_3d_s2, 1, 0, 1, head_head_result, &head_head_stats);

  EXPECT_EQ(tail_head_stats.transpose_prepare_calls, 0U);
  EXPECT_EQ(tail_head_stats.transpose_prepare_bytes, 0U);
  EXPECT_GT(tail_head_stats.raw_data_contract_tasks, 0U);
  EXPECT_GT(tail_head_stats.gemm_calls, 0U);
  EXPECT_GT(head_head_stats.transpose_prepare_calls, 0U);
  EXPECT_GT(head_head_stats.transpose_prepare_bytes, 0U);
  EXPECT_GT(head_head_stats.raw_data_contract_tasks, 0U);
  EXPECT_GT(head_head_stats.gemm_calls, 0U);
}

template<typename TenElemT, typename QNT>
void ExpectTensorNear(
    const QLTensor<TenElemT, QNT> &actual,
    const QLTensor<TenElemT, QNT> &expected,
    const typename RealTypeTrait<TenElemT>::type eps
) {
  using RealT = typename RealTypeTrait<TenElemT>::type;
  EXPECT_EQ(actual.GetIndexes(), expected.GetIndexes());
  QLTensor<TenElemT, QNT> diff = actual + (-expected);
  const auto denom = std::max(expected.Get2Norm(), RealT(1e-8));
  EXPECT_NEAR(diff.Get2Norm() / denom, 0.0, eps);
}

template<typename TenElemT, typename QNT>
void ExpectSameBlockTopology(
    const QLTensor<TenElemT, QNT> &lhs,
    const QLTensor<TenElemT, QNT> &rhs
) {
  const auto &lhs_blocks = lhs.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  const auto &rhs_blocks = rhs.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  ASSERT_EQ(lhs_blocks.size(), rhs_blocks.size());
  auto lhs_it = lhs_blocks.begin();
  auto rhs_it = rhs_blocks.begin();
  for (; lhs_it != lhs_blocks.end(); ++lhs_it, ++rhs_it) {
    EXPECT_EQ(lhs_it->first, rhs_it->first);
    EXPECT_EQ(lhs_it->second.blk_coors, rhs_it->second.blk_coors);
    EXPECT_EQ(lhs_it->second.shape, rhs_it->second.shape);
  }
}

inline void PrintContiguousAccumulateStats(
    const char *label,
    const ContiguousContractStats &stats
) {
  std::cout << label
            << " transpose_prepare_calls=" << stats.transpose_prepare_calls
            << " raw_data_contract_tasks=" << stats.raw_data_contract_tasks
            << " gemm_calls=" << stats.gemm_calls
            << " accumulate_calls=" << stats.accumulate_calls
            << " accumulate_gemm_calls=" << stats.accumulate_gemm_calls
            << " output_tensor_rebuilds=" << stats.output_tensor_rebuilds
            << " temporary_output_bytes_avoided="
            << stats.temporary_output_bytes_avoided
            << " output_topology_expansions="
            << stats.output_topology_expansions
            << " output_expand_copy_bytes="
            << stats.output_expand_copy_bytes
            << " output_expand_new_blocks="
            << stats.output_expand_new_blocks
            << " output_untouched_scale_bytes="
            << stats.output_untouched_scale_bytes
            << std::endl;
}

template<typename TenElemT, typename QNT>
void RunContractTailHeadContiguousAccumulateCase(
    const QLTensor<TenElemT, QNT> &lhs,
    const QLTensor<TenElemT, QNT> &rhs,
    const TenElemT alpha
) {
  using TenT = QLTensor<TenElemT, QNT>;
  const auto eps = GetEpsilon<TenElemT>() * 100;

  TenT tmp;
  ContractTailHeadContiguous(lhs, rhs, 2, 0, 2, tmp);

  TenT first_accum;
  ContiguousContractStats first_stats;
  ContractTailHeadContiguousAccumulate(
      lhs, rhs, 2, 0, 2, alpha, TenElemT(0), first_accum, &first_stats);
  PrintContiguousAccumulateStats("first_accumulate", first_stats);
  TenT first_expected = tmp * alpha;
  ExpectTensorNear(first_accum, first_expected, eps);
  ExpectSameBlockTopology(first_accum, tmp);
  EXPECT_EQ(first_stats.transpose_prepare_calls, 0U);
  EXPECT_EQ(first_stats.transpose_prepare_bytes, 0U);
  EXPECT_EQ(first_stats.accumulate_calls, 1U);
  EXPECT_EQ(first_stats.output_tensor_rebuilds, 1U);
  EXPECT_GT(first_stats.accumulate_gemm_calls, 0U);
  EXPECT_EQ(first_stats.accumulate_gemm_calls, first_stats.gemm_calls);
  EXPECT_GT(first_stats.temporary_output_bytes_avoided, 0U);

  TenT repeated_accum = first_accum;
  ContiguousContractStats repeated_stats;
  ContractTailHeadContiguousAccumulate(
      lhs, rhs, 2, 0, 2, alpha, TenElemT(1), repeated_accum, &repeated_stats);
  PrintContiguousAccumulateStats("repeated_accumulate", repeated_stats);
  TenT repeated_expected = first_expected + (tmp * alpha);
  ExpectTensorNear(repeated_accum, repeated_expected, eps);
  ExpectSameBlockTopology(repeated_accum, tmp);
  EXPECT_EQ(repeated_stats.transpose_prepare_calls, 0U);
  EXPECT_EQ(repeated_stats.transpose_prepare_bytes, 0U);
  EXPECT_EQ(repeated_stats.accumulate_calls, 1U);
  EXPECT_EQ(repeated_stats.output_tensor_rebuilds, 0U);
  EXPECT_GT(repeated_stats.accumulate_gemm_calls, 0U);
  EXPECT_EQ(repeated_stats.accumulate_gemm_calls, repeated_stats.gemm_calls);
  EXPECT_GT(repeated_stats.temporary_output_bytes_avoided, 0U);
}

template<typename TenElemT, typename QNT>
void RunContractTailHeadContiguousRepeatedAccumulateAfterContractInitCase(
    const QLTensor<TenElemT, QNT> &lhs,
    const QLTensor<TenElemT, QNT> &rhs,
    const TenElemT alpha
) {
  using TenT = QLTensor<TenElemT, QNT>;
  const auto eps = GetEpsilon<TenElemT>() * 100;

  TenT tmp;
  ContractTailHeadContiguous(lhs, rhs, 2, 0, 2, tmp);

  TenT out;
  ContractTailHeadContiguous(lhs, rhs, 2, 0, 2, out);
  TenT expected = tmp;
  for (size_t i = 0; i < 4; ++i) {
    ContiguousContractStats stats;
    ContractTailHeadContiguousAccumulate(
        lhs, rhs, 2, 0, 2, alpha, TenElemT(1), out, &stats);
    expected = expected + (tmp * alpha);
    ExpectTensorNear(out, expected, eps);
    ExpectSameBlockTopology(out, tmp);
    EXPECT_EQ(stats.transpose_prepare_calls, 0U);
    EXPECT_EQ(stats.transpose_prepare_bytes, 0U);
    EXPECT_EQ(stats.output_tensor_rebuilds, 0U);
    EXPECT_EQ(stats.accumulate_calls, 1U);
    EXPECT_EQ(stats.accumulate_gemm_calls, stats.gemm_calls);
  }
}

TEST_F(TestContraction, ContractTailHeadContiguousAccumulateLeftMaterialized) {
  DQLTensor rank4({
      idx_out_l,
      idx_out_s,
      idx_out_s,
      idx_out_l
  });
  DQLTensor state({
      idx_in_s,
      idx_in_l,
      idx_out_s,
      idx_out_l
  });
  rank4.Random(qn0);
  state.Random(qn0);

  RunContractTailHeadContiguousAccumulateCase(rank4, state, QLTEN_Double(1.75));
  RunContractTailHeadContiguousRepeatedAccumulateAfterContractInitCase(
      rank4, state, QLTEN_Double(0.625));
}

TEST_F(TestContraction, ContractTailHeadContiguousAccumulateRightMaterialized) {
  DQLTensor state({
      idx_in_l,
      idx_in_s,
      idx_out_s,
      idx_out_l
  });
  DQLTensor rank4({
      idx_in_s,
      idx_in_l,
      idx_out_s,
      idx_out_l
  });
  state.Random(qn0);
  rank4.Random(qn0);

  RunContractTailHeadContiguousAccumulateCase(state, rank4, QLTEN_Double(0.375));
  RunContractTailHeadContiguousRepeatedAccumulateAfterContractInitCase(
      state, rank4, QLTEN_Double(1.125));
}

TEST_F(TestContraction, TryContractTailHeadContiguousAccumulateMismatchIsNoOp) {
  DQLTensor state({
      idx_in_l,
      idx_in_s,
      idx_out_s,
      idx_out_l
  });
  DQLTensor rank4({
      idx_in_s,
      idx_in_l,
      idx_out_s,
      idx_out_l
  });
  state.Random(qn0);
  rank4.Random(qn0);

  DQLTensor out;
  ContractTailHeadContiguous(state, rank4, 2, 0, 2, out);
  DQLTensor before = out;

  DQLTensor mismatched_state(state.GetIndexes());
  DQLTensor mismatched_rank4(rank4.GetIndexes());
  mismatched_state.Random(qnp2);
  mismatched_rank4.Random(qnp2);
  ContiguousContractStats stats;
  const bool accumulated = TryContractTailHeadContiguousAccumulate(
      mismatched_state, mismatched_rank4, 2, 0, 2,
      QLTEN_Double(1), QLTEN_Double(1), out, &stats);

  EXPECT_FALSE(accumulated);
  ExpectTensorNear(out, before, GetEpsilon<QLTEN_Double>() * 10);
  EXPECT_EQ(stats.accumulate_calls, 0U);
  EXPECT_EQ(stats.accumulate_gemm_calls, 0U);
  EXPECT_EQ(stats.output_tensor_rebuilds, 0U);
}

TEST_F(TestContraction, ContractTailHeadContiguousAccumulateScalesScalarNoTaskOutput) {
  DQLTensor lhs({idx_out_s});
  DQLTensor rhs({idx_in_s});
  DQLTensor out(IndexVec<U1QN>{});
  out() = QLTEN_Double(7);

  ContiguousContractStats scale_stats;
  ContractTailHeadContiguousAccumulate(
      lhs, rhs, 0, 0, 1,
      QLTEN_Double(3), QLTEN_Double(2), out, &scale_stats);

  EXPECT_NEAR(out(), QLTEN_Double(14), GetEpsilon<QLTEN_Double>());
  EXPECT_EQ(scale_stats.raw_data_contract_tasks, 0U);
  EXPECT_EQ(scale_stats.accumulate_gemm_calls, 0U);
  EXPECT_EQ(scale_stats.output_untouched_scale_bytes, sizeof(QLTEN_Double));

  ContiguousContractStats zero_stats;
  ContractTailHeadContiguousAccumulate(
      lhs, rhs, 0, 0, 1,
      QLTEN_Double(3), QLTEN_Double(0), out, &zero_stats);

  EXPECT_NEAR(out(), QLTEN_Double(0), GetEpsilon<QLTEN_Double>());
  EXPECT_EQ(zero_stats.raw_data_contract_tasks, 0U);
  EXPECT_EQ(zero_stats.accumulate_gemm_calls, 0U);
  EXPECT_EQ(zero_stats.output_untouched_scale_bytes, sizeof(QLTEN_Double));
}

TEST_F(TestContraction, ContractTailHeadContiguousAccumulateUpdatesExistingScalarOutput) {
  DQLTensor lhs({idx_out_s});
  lhs.Random(qn0);
  const DQLTensor rhs = Dag(lhs);

  DQLTensor tmp;
  ContractTailHeadContiguous(lhs, rhs, 0, 0, 1, tmp);
  ASSERT_TRUE(tmp.IsScalar());

  DQLTensor first_out(IndexVec<U1QN>{});
  ContiguousContractStats first_stats;
  ContractTailHeadContiguousAccumulate(
      lhs, rhs, 0, 0, 1,
      QLTEN_Double(3), QLTEN_Double(0), first_out, &first_stats);
  EXPECT_NEAR(first_out(), QLTEN_Double(3) * tmp(),
              GetEpsilon<QLTEN_Double>() * 100);
  EXPECT_GT(first_stats.raw_data_contract_tasks, 0U);
  EXPECT_EQ(first_stats.accumulate_gemm_calls,
            first_stats.raw_data_contract_tasks);
  EXPECT_EQ(first_stats.output_tensor_rebuilds, 0U);
  EXPECT_EQ(first_stats.temporary_output_bytes_avoided,
            sizeof(QLTEN_Double));

  DQLTensor out(IndexVec<U1QN>{});
  out() = QLTEN_Double(5);

  ContiguousContractStats stats;
  ContractTailHeadContiguousAccumulate(
      lhs, rhs, 0, 0, 1,
      QLTEN_Double(3), QLTEN_Double(2), out, &stats);

  const QLTEN_Double expected = QLTEN_Double(2) * QLTEN_Double(5) +
                                QLTEN_Double(3) * tmp();
  EXPECT_NEAR(out(), expected, GetEpsilon<QLTEN_Double>() * 100);
  EXPECT_GT(stats.raw_data_contract_tasks, 0U);
  EXPECT_EQ(stats.accumulate_gemm_calls, stats.raw_data_contract_tasks);
  EXPECT_EQ(stats.output_tensor_rebuilds, 0U);
  EXPECT_EQ(stats.output_untouched_scale_bytes, 0U);
  EXPECT_EQ(stats.temporary_output_bytes_avoided, sizeof(QLTEN_Double));

  DQLTensor try_out(IndexVec<U1QN>{});
  try_out() = QLTEN_Double(5);
  ContiguousContractStats try_stats;
  const bool accumulated = TryContractTailHeadContiguousAccumulate(
      lhs, rhs, 0, 0, 1,
      QLTEN_Double(3), QLTEN_Double(2), try_out, &try_stats);
  EXPECT_TRUE(accumulated);
  EXPECT_NEAR(try_out(), expected, GetEpsilon<QLTEN_Double>() * 100);
  EXPECT_GT(try_stats.raw_data_contract_tasks, 0U);
  EXPECT_EQ(try_stats.accumulate_gemm_calls,
            try_stats.raw_data_contract_tasks);
  EXPECT_EQ(try_stats.output_tensor_rebuilds, 0U);
  EXPECT_EQ(try_stats.temporary_output_bytes_avoided, sizeof(QLTEN_Double));
}

TEST_F(TestContraction, ContractTailHeadContiguousAccumulateScalesSupersetBlocks) {
  DQLTensor state({
      idx_in_l,
      idx_in_s,
      idx_out_s,
      idx_out_l
  });
  DQLTensor rank4({
      idx_in_s,
      idx_in_l,
      idx_out_s,
      idx_out_l
  });
  state.Random(qn0);
  rank4.Random(qn0);

  DQLTensor contract_result;
  ContractTailHeadContiguous(state, rank4, 2, 0, 2, contract_result);
  DQLTensor extra(contract_result.GetIndexes());
  extra.Random(qnp1);
  DQLTensor out = contract_result + extra;
  ASSERT_GT(out.GetQNBlkNum(), contract_result.GetQNBlkNum());
  const DQLTensor before = out;

  ContiguousContractStats stats;
  ContractTailHeadContiguousAccumulate(
      state, rank4, 2, 0, 2,
      QLTEN_Double(0.5), QLTEN_Double(2.0), out, &stats);
  PrintContiguousAccumulateStats("superset_accumulate", stats);

  DQLTensor expected = (before * QLTEN_Double(2.0)) +
                       (contract_result * QLTEN_Double(0.5));
  ExpectTensorNear(out, expected, GetEpsilon<QLTEN_Double>() * 100);
  EXPECT_EQ(stats.transpose_prepare_calls, 0U);
  EXPECT_EQ(stats.transpose_prepare_bytes, 0U);
  EXPECT_EQ(stats.output_topology_expansions, 0U);
  EXPECT_EQ(stats.output_expand_copy_bytes, 0U);
  EXPECT_GT(stats.output_untouched_scale_bytes, 0U);
}

TEST_F(TestContraction, ContractTailHeadContiguousAccumulateExpandsOutputTopology) {
  DQLTensor state({
      idx_in_l,
      idx_in_s,
      idx_out_s,
      idx_out_l
  });
  DQLTensor rank4({
      idx_in_s,
      idx_in_l,
      idx_out_s,
      idx_out_l
  });
  state.Random(qn0);
  rank4.Random(qn0);

  DQLTensor contract_result;
  ContractTailHeadContiguous(state, rank4, 2, 0, 2, contract_result);
  DQLTensor out(contract_result.GetIndexes());
  out.Random(qnp1);
  ASSERT_LT(out.GetQNBlkNum(), (out + contract_result).GetQNBlkNum());
  const DQLTensor before = out;

  ContiguousContractStats try_stats;
  DQLTensor try_out = out;
  const bool try_accumulated = TryContractTailHeadContiguousAccumulate(
      state, rank4, 2, 0, 2,
      QLTEN_Double(0.5), QLTEN_Double(2.0), try_out, &try_stats);
  EXPECT_FALSE(try_accumulated);
  ExpectTensorNear(try_out, before, GetEpsilon<QLTEN_Double>() * 10);

  ContiguousContractStats stats;
  ContractTailHeadContiguousAccumulate(
      state, rank4, 2, 0, 2,
      QLTEN_Double(0.5), QLTEN_Double(2.0), out, &stats);
  PrintContiguousAccumulateStats("expanded_accumulate", stats);

  DQLTensor expected = (before * QLTEN_Double(2.0)) +
                       (contract_result * QLTEN_Double(0.5));
  ExpectTensorNear(out, expected, GetEpsilon<QLTEN_Double>() * 100);
  EXPECT_EQ(stats.transpose_prepare_calls, 0U);
  EXPECT_EQ(stats.transpose_prepare_bytes, 0U);
  EXPECT_GT(stats.output_topology_expansions, 0U);
  EXPECT_GT(stats.output_expand_copy_bytes, 0U);
  EXPECT_GT(stats.output_expand_new_blocks, 0U);
}

TEST_F(TestContraction, ContractTailHeadContiguousAccumulateComplexAlpha) {
  ZQLTensor rank4({
      idx_out_l,
      idx_out_s,
      idx_out_s,
      idx_out_l
  });
  ZQLTensor state({
      idx_in_s,
      idx_in_l,
      idx_out_s,
      idx_out_l
  });
  rank4.Random(qn0);
  state.Random(qn0);

  RunContractTailHeadContiguousAccumulateCase(
      rank4, state, QLTEN_Complex(0.5, -0.25));
}

TEST(TestContractTailHeadContiguousAccumulate, U1U1BlockSparseMultipleBlocks) {
  using U1U1QN = QN<U1QNVal, U1QNVal>;
  using U1U1Index = Index<U1U1QN>;
  using U1U1QNSct = QNSector<U1U1QN>;
  using U1U1Ten = QLTensor<QLTEN_Double, U1U1QN>;

  U1U1QN qn0({
      QNCard("n", U1QNVal(0)),
      QNCard("sz", U1QNVal(0))
  });
  U1U1QN qnp({
      QNCard("n", U1QNVal(1)),
      QNCard("sz", U1QNVal(1))
  });
  U1U1QN qnm({
      QNCard("n", U1QNVal(-1)),
      QNCard("sz", U1QNVal(-1))
  });
  U1U1Index idx_in({
      U1U1QNSct(qnm, 2),
      U1U1QNSct(qn0, 3),
      U1U1QNSct(qnp, 2)
  }, TenIndexDirType::IN);
  U1U1Index idx_out = InverseIndex(idx_in);
  U1U1Ten lhs({
      idx_out,
      idx_out,
      idx_out,
      idx_out
  });
  U1U1Ten rhs({
      idx_in,
      idx_in,
      idx_out,
      idx_out
  });
  lhs.Random(qn0);
  rhs.Random(qn0);

  RunContractTailHeadContiguousAccumulateCase(lhs, rhs, QLTEN_Double(1.25));
  RunContractTailHeadContiguousRepeatedAccumulateAfterContractInitCase(
      lhs, rhs, QLTEN_Double(0.75));
}
