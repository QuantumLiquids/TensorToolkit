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
#include "qlten/tensor_manipulation/ten_ctrct_based_mat_trans.h"      // Contract
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
  GtestExpectNear(t_res.GetElem({}), res, kEpsilon);
}

TEST_F(TestContraction, 1DCase) {
  RunTestTenCtrct1DCase(dten_1d_s, qn0);
  RunTestTenCtrct1DCase(dten_1d_s, qnp1);
  RunTestTenCtrct1DCase(dten_1d_s, qnm1);

  RunTestTenCtrct1DCase(zten_1d_s, qn0);
  RunTestTenCtrct1DCase(zten_1d_s, qnp1);
  RunTestTenCtrct1DCase(zten_1d_s, qnm1);
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
    GtestExpectNear(res.GetElem(coors), dense_res[idx], kEpsilon);
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
  GtestExpectNear(res.GetElem({}), res_scalar, kEpsilon);

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
    GtestExpectNear(res.GetElem(coors), dense_res[idx], kEpsilon);
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
    GtestExpectNear(res.GetElem(coors), dense_res[idx], kEpsilon);
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
  GtestExpectNear(res.GetElem({}), res_scalar, kEpsilon);

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

  EXPECT_EQ(benchmark_res.GetIndexes(), res1.GetIndexes());
  const BlockSparseDataTensor<TenElemT, QNT> &bsdt_res1 = res1.GetBlkSparDataTen();
  EXPECT_EQ(bsdt_res.GetActualRawDataSize(), bsdt_res1.GetActualRawDataSize());
  TenT diff1 = benchmark_res + (-res1);
  EXPECT_NEAR(diff1.Get2Norm() / std::max(res1.Get2Norm(), 1e-5), 0.0, kDoubleEpsilon);

  EXPECT_EQ(benchmark_res.GetIndexes(), res2.GetIndexes());
  TenT diff2 = benchmark_res + (-res2);
  EXPECT_NEAR(diff2.Normalize() / std::max(res2.Get2Norm(), 1e-5), 0.0, kDoubleEpsilon);

  EXPECT_EQ(benchmark_res.GetIndexes(), res3.GetIndexes());
  EXPECT_TRUE(BSDTCompareShell(bsdt_res, res3.GetBlkSparDataTen()));
  TenT diff3 = benchmark_res + (-res3);
  double norm_diff3 = diff3.Get2Norm();
  double norm_res3 = res3.Get2Norm();
  EXPECT_NEAR(norm_diff3 / std::max(norm_res3, 1e-5), 0.0, kDoubleEpsilon * 10);
  if (norm_diff3 / std::max(norm_res3, 1e-5) > kDoubleEpsilon) {
    std::cout << "norm_diff3 = " << std::scientific << norm_diff3 << std::endl;
    std::cout << "norm_res3 = " << norm_res3 << std::endl;
  }

  EXPECT_EQ(benchmark_res.GetIndexes(), res4.GetIndexes());
  TenT diff4 = benchmark_res + (-res4);
  EXPECT_NEAR(diff4.Get2Norm() / std::max(res4.Get2Norm(), 1e-5), 0.0, kDoubleEpsilon * 10);
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
}

