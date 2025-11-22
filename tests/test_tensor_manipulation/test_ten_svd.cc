// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-12-03 21:34
* 
* Description: QuantumLiquids/tensor project. Unittests for tensor SVD.
*/

#include "gtest/gtest.h"
#include "qlten/qltensor_all.h"
#include "qlten/tensor_manipulation/ten_decomp/ten_svd.h"   // SVD
#include "qlten/tensor_manipulation/ten_ctrct.h"            // Contract
#include "qlten/tensor_manipulation/basic_operations.h"     // Dag
#include "qlten/utility/utils_inl.h"
#include "qlten/utility/timer.h"
#include "../testing_utility.h"
#include "../test_utility/hp_numeric.h"                     //qltest::MatSVD
using namespace qlten;
using U1QN = special_qn::U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DQLTensor = QLTensor<QLTEN_Double, U1QN>;
using ZQLTensor = QLTensor<QLTEN_Complex, U1QN>;
using FQLTensor = QLTensor<QLTEN_Float, U1QN>;
using CQLTensor = QLTensor<QLTEN_ComplexFloat, U1QN>;

struct TestSvd : public testing::Test {
  std::string qn_nm = "qn";
  U1QN qn0 = U1QN({QNCard(qn_nm, U1QNVal(0))});
  U1QN qnp1 = U1QN({QNCard(qn_nm, U1QNVal(1))});
  U1QN qnp2 = U1QN({QNCard(qn_nm, U1QNVal(2))});
  U1QN qnm1 = U1QN({QNCard(qn_nm, U1QNVal(-1))});
  U1QN qnm2 = U1QN({QNCard(qn_nm, U1QNVal(-2))});

  size_t d_s = 3;
  QNSctT qnsct0_s = QNSctT(qn0, d_s);
  QNSctT qnsctp1_s = QNSctT(qnp1, d_s);
  QNSctT qnsctm1_s = QNSctT(qnm1, d_s);
  IndexT idx_in_s = IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, IN);
  IndexT idx_out_s = IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, OUT);

  DQLTensor dten_1d_s = DQLTensor({idx_out_s});
  DQLTensor dten_2d_s = DQLTensor({idx_in_s, idx_out_s});
  DQLTensor dten_3d_s = DQLTensor({idx_in_s, idx_out_s, idx_out_s});
  DQLTensor dten_4d_s = DQLTensor({idx_in_s, idx_out_s, idx_out_s, idx_out_s});

  ZQLTensor zten_1d_s = ZQLTensor({idx_out_s});
  ZQLTensor zten_2d_s = ZQLTensor({idx_in_s, idx_out_s});
  ZQLTensor zten_3d_s = ZQLTensor({idx_in_s, idx_out_s, idx_out_s});
  ZQLTensor zten_4d_s = ZQLTensor({idx_in_s, idx_out_s, idx_out_s, idx_out_s});

  FQLTensor ften_1d_s = FQLTensor({idx_out_s});
  FQLTensor ften_2d_s = FQLTensor({idx_in_s, idx_out_s});
  FQLTensor ften_3d_s = FQLTensor({idx_in_s, idx_out_s, idx_out_s});
  FQLTensor ften_4d_s = FQLTensor({idx_in_s, idx_out_s, idx_out_s, idx_out_s});

  CQLTensor cten_1d_s = CQLTensor({idx_out_s});
  CQLTensor cten_2d_s = CQLTensor({idx_in_s, idx_out_s});
  CQLTensor cten_3d_s = CQLTensor({idx_in_s, idx_out_s, idx_out_s});
  CQLTensor cten_4d_s = CQLTensor({idx_in_s, idx_out_s, idx_out_s, idx_out_s});
};

inline size_t IntDot(const size_t &size, const size_t *x, const size_t *y) {
  size_t res = 0;
  for (size_t i = 0; i < size; ++i) { res += x[i] * y[i]; }
  return res;
}

inline double ToDouble(const double d) {
  return d;
}

inline double ToDouble(const float d) {
  return static_cast<double>(d);
}

inline double ToDouble(const QLTEN_Complex z) {
  return z.real();
}

inline double ToDouble(const QLTEN_ComplexFloat z) {
  return z.real();
}

inline void SVDTensRestore(
    const DQLTensor *pu,
    const DQLTensor *ps,
    const DQLTensor *pvt,
    const size_t ldims,
    DQLTensor *pres) {
  DQLTensor t_restored_tmp;
  Contract(pu, ps, {{ldims}, {0}}, &t_restored_tmp);
  Contract(&t_restored_tmp, pvt, {{ldims}, {0}}, pres);
}

inline void SVDTensRestore(
    const FQLTensor *pu,
    const FQLTensor *ps,
    const FQLTensor *pvt,
    const size_t ldims,
    FQLTensor *pres) {
  FQLTensor t_restored_tmp;
  Contract(pu, ps, {{ldims}, {0}}, &t_restored_tmp);
  Contract(&t_restored_tmp, pvt, {{ldims}, {0}}, pres);
}

inline void SVDTensRestore(
    const ZQLTensor *pu,
    const DQLTensor *ps,
    const ZQLTensor *pvt,
    const size_t ldims,
    ZQLTensor *pres) {
  ZQLTensor t_restored_tmp;
  auto zs = ToComplex(*ps);
  Contract(pu, &zs, {{ldims}, {0}}, &t_restored_tmp);
  Contract(&t_restored_tmp, pvt, {{ldims}, {0}}, pres);
}

inline void SVDTensRestore(
    const CQLTensor *pu,
    const FQLTensor *ps,
    const CQLTensor *pvt,
    const size_t ldims,
    CQLTensor *pres) {
  CQLTensor t_restored_tmp;
  auto zs = ToComplex(*ps);
  Contract(pu, &zs, {{ldims}, {0}}, &t_restored_tmp);
  Contract(&t_restored_tmp, pvt, {{ldims}, {0}}, pres);
}

template<typename ElemT, typename QNT>
void CheckIsIdTen(const QLTensor<ElemT, QNT> &t) {
  double epsilon = kEpsilon;
  using TenT = QLTensor<ElemT, QNT>;
  if constexpr (std::is_same_v<ElemT, float> || std::is_same_v<ElemT, std::complex<float>> 
#ifdef USE_GPU
      || std::is_same_v<ElemT, cuda::std::complex<float>>
#endif
      ) {
    epsilon = 2.0e-5;
  }
  auto shape = t.GetShape();
  EXPECT_EQ(shape.size(), 2);
  EXPECT_EQ(shape[0], shape[1]);
  for (size_t i = 0; i < shape[0]; ++i) {
    for (size_t j = 0; j < shape[1]; ++j) {
      QLTEN_Complex elem = t.GetElem({i, j});
      if (i == j) {
        EXPECT_NEAR(elem.real(), 1.0, epsilon);
      } else {
        EXPECT_NEAR(elem.real(), 0.0, epsilon);
      }
      EXPECT_NEAR(elem.imag(), 0.0, epsilon);
    }
  }
}

template<typename TenElemT, typename QNT>
void RunTestSvdCase(
    QLTensor<TenElemT, QNT> &t,
    const size_t &ldims,
    const size_t &rdims,
    const double &cutoff,
    const size_t &dmin,
    const size_t &dmax,
    const QNT *random_div = nullptr) {
  if (random_div != nullptr) {
    qlten::SetRandomSeed(0);
    t.Random(*random_div);
  }
  double epsilon = kEpsilon;
  if constexpr (std::is_same_v<TenElemT, float> || std::is_same_v<TenElemT, std::complex<float>> 
#ifdef USE_GPU
      || std::is_same_v<TenElemT, cuda::std::complex<float>>
#endif
      ) {
    epsilon = 2.0e-5;
  }
  QLTensor<TenElemT, QNT> u, vt;
  QLTensor<typename RealTypeTrait<TenElemT>::type, QNT> s;
  double trunc_err;
  size_t D;
  std::string qn_nm = "qn";
  U1QN qn0 = U1QN({QNCard(qn_nm, U1QNVal(0))});
  SVD(
      &t,
      ldims,
      qn0,
      cutoff, dmin, dmax,
      &u, &s, &vt, &trunc_err, &D
  );

  // Canonical check
  QLTensor<TenElemT, QNT> temp1, temp2;
  auto u_dag = Dag(u);
  std::vector<size_t> cano_check_u_ctrct_axes;
  for (size_t i = 0; i < ldims; ++i) { cano_check_u_ctrct_axes.push_back(i); }
  Contract(&u, &u_dag, {cano_check_u_ctrct_axes, cano_check_u_ctrct_axes}, &temp1);
  std::cout << "Check unitary for U" << std::endl;
  CheckIsIdTen(temp1);
  auto vt_dag = Dag(vt);
  std::vector<size_t> cano_check_vt_ctrct_axes;
  for (size_t i = 1; i <= rdims; ++i) { cano_check_vt_ctrct_axes.push_back(i); }
  Contract(&vt, &vt_dag, {cano_check_vt_ctrct_axes, cano_check_vt_ctrct_axes}, &temp2);
  std::cout << "Check unitary for V" << std::endl;
  CheckIsIdTen(temp2);

  auto ndim = ldims + rdims;
  size_t rows = 1, cols = 1;
  for (size_t i = 0; i < ndim; ++i) {
    if (i < ldims) {
      rows *= t.GetIndexes()[i].dim();
    } else {
      cols *= t.GetIndexes()[i].dim();
    }
  }
  auto dense_mat = new TenElemT[rows * cols];
  auto offsets = CalcMultiDimDataOffsets(t.GetShape());
  for (auto &coors : GenAllCoors(t.GetShape())) {
    dense_mat[IntDot(ndim, coors.data(), offsets.data())] = t.GetElem(coors);
  }
  TenElemT *dense_u;
  TenElemT *dense_vt;
  typename RealTypeTrait<TenElemT>::type *dense_s;
  qlten::test::MatSVD(dense_mat, rows, cols, dense_u, dense_s, dense_vt);
  size_t dense_sdim;
  if (rows > cols) {
    dense_sdim = cols;
  } else {
    dense_sdim = rows;
  }

  std::vector<double> dense_svs;
  for (size_t i = 0; i < dense_sdim; ++i) {
    if (dense_s[i] > 1.0E-13) {
      dense_svs.push_back(dense_s[i]);
    }
  }
  std::sort(dense_svs.begin(), dense_svs.end());
  auto endit = dense_svs.cend();
  auto begit = endit - dmax;
  if (dmax > dense_svs.size()) { begit = dense_svs.cbegin(); }
  auto saved_dense_svs = std::vector<double>(begit, endit);
  std::vector<double> qn_svs;
  for (size_t i = 0; i < s.GetShape()[0]; i++) {
    qn_svs.push_back(ToDouble(s.GetElem({i, i})));
  }
  std::sort(qn_svs.begin(), qn_svs.end());
  EXPECT_EQ(qn_svs.size(), saved_dense_svs.size());
  for (size_t i = 0; i < qn_svs.size(); ++i) {
#ifndef  USE_GPU
    EXPECT_NEAR(qn_svs[i], saved_dense_svs[i], epsilon);
#else
    // more tolerance for CUDA, because of different algorithm used in CUDA and LAPACK.
    EXPECT_NEAR(qn_svs[i], saved_dense_svs[i], 1.5 * epsilon);
#endif
  }

  double total_square_sum = 0.0;
  for (auto &sv : dense_svs) {
    total_square_sum += sv * sv;
  }
  double saved_square_sum = 0.0;
  for (auto &ssv : saved_dense_svs) {
    saved_square_sum += ssv * ssv;
  }
  auto dense_trunc_err = 1 - saved_square_sum / total_square_sum;
  EXPECT_NEAR(trunc_err, dense_trunc_err, epsilon);

  if (trunc_err < 1.0E-10) {
    QLTensor<TenElemT, QNT> t_restored;
    SVDTensRestore(&u, &s, &vt, ldims, &t_restored);
    for (auto &coors : GenAllCoors(t.GetShape())) {
      GtestExpectNear(t_restored.GetElem(coors), t.GetElem(coors), epsilon);
    }
  } else {
    QLTensor<TenElemT, QNT> t_restored;
    SVDTensRestore(&u, &s, &vt, ldims, &t_restored);
    auto t_diff = t + (-t_restored);
    auto t_diff_norm = t_diff.Normalize();
    auto t_norm = t.Normalize();
    auto norm_ratio = (t_diff_norm / t_norm);
    GtestExpectNear(static_cast<double>(norm_ratio * norm_ratio), trunc_err, 1E-02);
  }

  delete[] dense_mat;
  free(dense_s);
  free(dense_u);
  free(dense_vt);
}

TEST_F(TestSvd, 2DCase) {
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s * 3,
      &qn0);
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s,
      &qn0);
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s - 1,
      &qn0);
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s + 1,
      &qn0);

  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s * 3,
      &qnp1);
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s,
      &qnp1);
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s - 1,
      &qnp1);
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s + 1,
      &qnp1);

  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s * 3,
      &qnp2);
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s,
      &qnp2);
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s * 3,
      &qnm1);
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s,
      &qnm1);
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s * 3,
      &qnm2);
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s,
      &qnm2);
  RunTestSvdCase(
      zten_2d_s,
      1, 1,
      0, 1, d_s * 3,
      &qn0);
  RunTestSvdCase(
      zten_2d_s,
      1, 1,
      0, 1, d_s,
      &qn0);
  RunTestSvdCase(
      zten_2d_s,
      1, 1,
      0, 1, d_s * 3,
      &qnp1);
  RunTestSvdCase(
      zten_2d_s,
      1, 1,
      0, 1, d_s,
      &qnp1);
  RunTestSvdCase(
      zten_2d_s,
      1, 1,
      0, 1, d_s * 3,
      &qnp2);
  RunTestSvdCase(
      zten_2d_s,
      1, 1,
      0, 1, d_s,
      &qnp2);
  RunTestSvdCase(
      zten_2d_s,
      1, 1,
      0, 1, d_s * 3,
      &qnm1);
  RunTestSvdCase(
      zten_2d_s,
      1, 1,
      0, 1, d_s,
      &qnm1);
  RunTestSvdCase(
      zten_2d_s,
      1, 1,
      0, 1, d_s * 3,
      &qnm2);
  RunTestSvdCase(
      zten_2d_s,
      1, 1,
      0, 1, d_s,
      &qnm2);
}

TEST_F(TestSvd, 3DCase) {
  RunTestSvdCase(
      dten_3d_s,
      1, 2,
      0, 1, d_s * 3,
      &qn0);
  RunTestSvdCase(
      dten_3d_s,
      1, 2,
      0, 1, d_s * 2,
      &qn0);
  RunTestSvdCase(
      dten_3d_s,
      1, 2,
      0, 1, d_s - 1,
      &qn0);
  RunTestSvdCase(
      dten_3d_s,
      1, 2,
      0, 1, d_s + 1,
      &qn0);

  RunTestSvdCase(
      dten_3d_s,
      1, 2,
      0, 1, d_s * 3,
      &qnp1);
  RunTestSvdCase(
      dten_3d_s,
      1, 2,
      0, 1, d_s * 2,
      &qnp1);
  RunTestSvdCase(
      dten_3d_s,
      1, 2,
      0, 1, d_s + 1,
      &qnp1);
  RunTestSvdCase(
      dten_3d_s,
      1, 2,
      0, 1, d_s - 1,
      &qnp1);

  RunTestSvdCase(
      dten_3d_s,
      2, 1,
      0, 1, d_s * 3,
      &qn0);
  RunTestSvdCase(
      dten_3d_s,
      2, 1,
      0, 1, d_s * 2,
      &qn0);
  RunTestSvdCase(
      dten_3d_s,
      2, 1,
      0, 1, d_s + 1,
      &qn0);
  RunTestSvdCase(
      dten_3d_s,
      2, 1,
      0, 1, d_s - 1,
      &qn0);

  RunTestSvdCase(
      dten_3d_s,
      2, 1,
      0, 1, d_s * 3,
      &qnp1);
  RunTestSvdCase(
      dten_3d_s,
      2, 1,
      0, 1, d_s * 2,
      &qnp1);

  RunTestSvdCase(
      zten_3d_s,
      1, 2,
      0, 1, d_s * 3,
      &qn0);
  RunTestSvdCase(
      zten_3d_s,
      1, 2,
      0, 1, d_s * 2,
      &qn0);
  RunTestSvdCase(
      zten_3d_s,
      1, 2,
      0, 1, d_s * 3,
      &qnp1);
  RunTestSvdCase(
      zten_3d_s,
      1, 2,
      0, 1, d_s * 2,
      &qnp1);
  RunTestSvdCase(
      zten_3d_s,
      2, 1,
      0, 1, d_s * 3,
      &qn0);
  RunTestSvdCase(
      zten_3d_s,
      2, 1,
      0, 1, d_s * 2,
      &qn0);
  RunTestSvdCase(
      zten_3d_s,
      2, 1,
      0, 1, d_s * 3,
      &qnp1);
  RunTestSvdCase(
      zten_3d_s,
      2, 1,
      0, 1, d_s * 2,
      &qnp1);
}

TEST_F(TestSvd, 4DCase) {
  RunTestSvdCase(
      dten_4d_s,
      2, 2,
      0, 1, (d_s * 3) * (d_s * 3),
      &qn0);
  RunTestSvdCase(
      dten_4d_s,
      2, 2,
      0, 1, (d_s * 3),
      &qn0);
  RunTestSvdCase(
      dten_4d_s,
      2, 2,
      0, 1, (d_s * 3) + 1,
      &qn0);
  RunTestSvdCase(
      dten_4d_s,
      2, 2,
      0, 1, (d_s * 3) - 1,
      &qn0);

  RunTestSvdCase(
      dten_4d_s,
      2, 2,
      0, 1, (d_s * 3) * (d_s * 3),
      &qnp1);
  RunTestSvdCase(
      dten_4d_s,
      2, 2,
      0, 1, (d_s * 3),
      &qnp1);
  RunTestSvdCase(
      dten_4d_s,
      2, 2,
      0, 1, (d_s * 3) + 1,
      &qnp1);
  RunTestSvdCase(
      dten_4d_s,
      2, 2,
      0, 1, (d_s * 3) - 1,
      &qnp1);

  RunTestSvdCase(
      dten_4d_s,
      1, 3,
      0, 1, d_s * 3,
      &qn0);
  RunTestSvdCase(
      dten_4d_s,
      1, 3,
      0, 1, d_s * 2,
      &qn0);
  RunTestSvdCase(
      dten_4d_s,
      1, 3,
      0, 1, d_s * 3,
      &qnp1);
  RunTestSvdCase(
      dten_4d_s,
      1, 3,
      0, 1, d_s * 2,
      &qnp1);

  RunTestSvdCase(
      zten_4d_s,
      2, 2,
      0, 1, (d_s * 3) * (d_s * 3),
      &qn0);
  RunTestSvdCase(
      zten_4d_s,
      2, 2,
      0, 1, (d_s * 3),
      &qn0);
  RunTestSvdCase(
      zten_4d_s,
      2, 2,
      0, 1, (d_s * 3) * (d_s * 3),
      &qnp1);
  RunTestSvdCase(
      zten_4d_s,
      2, 2,
      0, 1, (d_s * 3),
      &qnp1);
  RunTestSvdCase(
      zten_4d_s,
      1, 3,
      0, 1, d_s * 3,
      &qn0);
  RunTestSvdCase(
      zten_4d_s,
      1, 3,
      0, 1, d_s * 2,
      &qn0);
  RunTestSvdCase(
      zten_4d_s,
      1, 3,
      0, 1, d_s * 3,
      &qnp1);
  RunTestSvdCase(
      zten_4d_s,
      1, 3,
      0, 1, d_s * 2,
      &qnp1);
}

TEST_F(TestSvd, 2DCaseFloat) {
  RunTestSvdCase(
      ften_2d_s,
      1, 1,
      0, 1, d_s * 3,
      &qn0);
  RunTestSvdCase(
      ften_2d_s,
      1, 1,
      0, 1, d_s,
      &qn0);
  RunTestSvdCase(
      ften_2d_s,
      1, 1,
      0, 1, d_s * 3,
      &qnp1);

  RunTestSvdCase(
      cten_2d_s,
      1, 1,
      0, 1, d_s * 3,
      &qn0);
  RunTestSvdCase(
      cten_2d_s,
      1, 1,
      0, 1, d_s,
      &qn0);
  RunTestSvdCase(
      cten_2d_s,
      1, 1,
      0, 1, d_s * 3,
      &qnp1);
}

TEST_F(TestSvd, 3DCaseFloat) {
  RunTestSvdCase(
      ften_3d_s,
      1, 2,
      0, 1, d_s * 3,
      &qn0);
  RunTestSvdCase(
      ften_3d_s,
      2, 1,
      0, 1, d_s * 3,
      &qn0);

  RunTestSvdCase(
      cten_3d_s,
      1, 2,
      0, 1, d_s * 3,
      &qn0);
  RunTestSvdCase(
      cten_3d_s,
      2, 1,
      0, 1, d_s * 3,
      &qn0);
}

struct TestSvdOmpParallel : public testing::Test {
  std::string qn_nm = "qn";
};

IndexT RandIndex(const unsigned qn_sct_num,  //how many quantum number sectors?
                 const unsigned max_dim_in_one_qn_sct, // maximum dimension in every quantum number sector?
                 const TenIndexDirType dir) {
  QNSectorVec<U1QN> qnsv(qn_sct_num);
  for (size_t i = 0; i < qn_sct_num; i++) {
    auto qn = U1QN({QNCard("qn", U1QNVal(i))});
    qlten::SetRandomSeed(i * i / 3);
    unsigned degeneracy = static_cast<unsigned>(qlten::RandUint32() % max_dim_in_one_qn_sct) + 1;
    qnsv[i] = QNSector(qn, degeneracy);
  }
  return Index(qnsv, dir);
}
