// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-23 09:29
*
* Description: QuantumLiquids/tensor project. Raw data operation tasks in block sparse
* data tensor.
*/

/**
@file raw_data_operation_tasks.h
@brief Raw data operation tasks in block sparse data tensor.
*/
#ifndef QLTEN_QLTENSOR_BLK_SPAR_DATA_TEN_RAW_DATA_OPERATION_TASKS_H
#define QLTEN_QLTENSOR_BLK_SPAR_DATA_TEN_RAW_DATA_OPERATION_TASKS_H

#include "qlten/framework/value_t.h"    // ShapeT, CoorsT

#include <vector>       // vector
#include <map>          // map
#include <algorithm>    // sort

namespace qlten {

/**
Task for data transpose.
*/
struct RawDataTransposeTask {
  size_t ten_rank = 0;

  std::vector<size_t> transed_order;

  size_t original_blk_idx = 0;
  ShapeT original_shape;
  size_t original_data_offset = 0;

  size_t transed_blk_idx = 0;
  ShapeT transed_shape;
  size_t transed_data_offset = 0;

  double scale_factor =
      1; //for Bosonic tensor, tranpose scale factors are always 1; For Ferminoic tensor, it is sometimes -1.

  RawDataTransposeTask(
      const size_t ten_rank,
      const std::vector<size_t> &transed_order,
      const size_t original_blk_idx,
      const ShapeT &original_shape,
      const size_t original_data_offset,
      const size_t transed_blk_idx,
      const ShapeT &transed_shape,
      double scale_factor
  ) : ten_rank(ten_rank),
      transed_order(transed_order),
      original_blk_idx(original_blk_idx),
      original_shape(original_shape),
      original_data_offset(original_data_offset),
      transed_blk_idx(transed_blk_idx),
      transed_shape(transed_shape), scale_factor(scale_factor) {}

  static void SortTasksByOriginalBlkIdx(
      std::vector<RawDataTransposeTask> &tasks
  ) {
    std::sort(
        tasks.begin(),
        tasks.end(),
        [](
            const RawDataTransposeTask &task_a,
            const RawDataTransposeTask &task_b
        ) -> bool {
          return task_a.original_blk_idx < task_b.original_blk_idx;
        }
    );
  }

  static void SortTasksByTranspoedBlkIdx(
      std::vector<RawDataTransposeTask> &tasks
  ) {
    std::sort(
        tasks.begin(),
        tasks.end(),
        [](
            const RawDataTransposeTask &task_a,
            const RawDataTransposeTask &task_b
        ) -> bool {
          return task_a.transed_blk_idx < task_b.transed_blk_idx;
        }
    );
  }
};

/**
Task for data copy.

@note The constructor will **not** initialize the dest_data_offset member!
*/
struct RawDataCopyTask {
  CoorsT src_blk_coors;
  size_t src_data_offset;
  size_t src_data_size;

  size_t dest_data_offset;

  bool copy_and_add;

  // Optional members
  CoorsT dest_blk_coors;

  RawDataCopyTask(
      const CoorsT &src_blk_coors,
      const size_t src_data_offset,
      const size_t src_data_size,
      const bool copy_and_add = false
  ) : src_blk_coors(src_blk_coors),
      src_data_offset(src_data_offset),
      src_data_size(src_data_size),
      copy_and_add(copy_and_add) {}

  RawDataCopyTask(
      const CoorsT &src_blk_coors,
      const size_t src_data_offset,
      const size_t src_data_size,
      const size_t dest_data_offset,
      const bool copy_and_add = false
  ) : src_blk_coors(src_blk_coors),
      src_data_offset(src_data_offset),
      src_data_size(src_data_size),
      dest_data_offset(dest_data_offset),
      copy_and_add(copy_and_add) {}
};

/**
Task for tensor linear combination data copy.
*/
template<typename CoefT>
struct RawDataCopyAndScaleTask {
  size_t src_data_offset;
  size_t src_data_size;

  CoorsT dest_blk_coors;
  CoefT coef;
  bool copy_and_add;

  RawDataCopyAndScaleTask(
      const size_t src_data_offset,
      const size_t src_data_size,
      const CoorsT &dest_blk_coors,
      const CoefT coef,
      const bool copy_and_add
  ) : src_data_offset(src_data_offset),
      src_data_size(src_data_size),
      dest_blk_coors(dest_blk_coors),
      coef(coef),
      copy_and_add(copy_and_add) {}
};

struct RawDataInplaceReverseTask {
  size_t data_offset;
  size_t data_size;
  RawDataInplaceReverseTask(
      const size_t data_offset,
      const size_t data_size
  ) : data_offset(data_offset), data_size(data_size) {}
};

struct RawDataFermionNormTask {
  size_t data_offset;
  size_t data_size;
  int sign;
  RawDataFermionNormTask(
      const size_t data_offset,
      const size_t data_size,
      const int sign
  ) : data_offset(data_offset), data_size(data_size), sign(sign) {}
};
/**
Task to set a piece of raw data zeros.
*/
struct RawDataSetZerosTask {
  // Actual used members
  size_t data_offset;
  size_t data_size;
  // Optional members
  CoorsT blk_coors;
  size_t extra_data_offset;

  RawDataSetZerosTask(
      const CoorsT &blk_coors,
      const size_t data_size,
      const size_t extra_data_offset
  ) : data_size(data_size),
      blk_coors(blk_coors),
      extra_data_offset(extra_data_offset) {}
};

struct RawDataCtrctTask {
  size_t a_blk_idx;
  size_t a_data_offset;

  size_t b_blk_idx;
  size_t b_data_offset;

  size_t c_blk_idx = 0;     // initialize it for c is rank 0 (scalar) case
  size_t c_data_offset;

  size_t m;
  size_t k;
  size_t n;
  double f_ex_sign;     // fermion exchange sign, equals +1 or -1
  QLTEN_Double beta;

  RawDataCtrctTask(
      const size_t a_blk_idx,
      const size_t a_data_offset,
      const size_t b_blk_idx,
      const size_t b_data_offset,
      const size_t m,
      const size_t k,
      const size_t n,
      const double f_ex_sign,
      const QLTEN_Double beta
  ) : a_blk_idx(a_blk_idx),
      a_data_offset(a_data_offset),
      b_blk_idx(b_blk_idx),
      b_data_offset(b_data_offset),
      m(m), k(k), n(n),
      f_ex_sign(f_ex_sign),
      beta(beta) {}

  RawDataCtrctTask(
      const size_t a_blk_idx,
      const size_t a_data_offset,
      const size_t b_blk_idx,
      const size_t b_data_offset,
      const size_t c_blk_idx,
      const size_t m,
      const size_t k,
      const size_t n,
      const double f_ex_sign,
      const QLTEN_Double beta
  ) : a_blk_idx(a_blk_idx),
      a_data_offset(a_data_offset),
      b_blk_idx(b_blk_idx),
      b_data_offset(b_data_offset),
      c_blk_idx(c_blk_idx),
      m(m), k(k), n(n),
      f_ex_sign(f_ex_sign),
      beta(beta) {}

  static void SortTasksByCBlkIdx(
      std::vector<RawDataCtrctTask> &tasks
  ) {
    std::sort(
        tasks.begin(),
        tasks.end(),
        [](
            const RawDataCtrctTask &task_a,
            const RawDataCtrctTask &task_b
        ) -> bool {
          if (task_a.c_blk_idx < task_b.c_blk_idx) {
            return true;
          } else if (task_a.c_blk_idx == task_b.c_blk_idx) {
            return task_a.beta < task_b.beta;     // Keep beta == 0 task first
          } else {
            return false;
          }
        }
    );
  }
};

template<typename ElemT>
struct DataBlkMatSvdRes {
  size_t m = 0;
  size_t n = 0;
  size_t k = 0;
  ElemT *u = nullptr;
  QLTEN_Double *s = nullptr;
  ElemT *vt = nullptr;

  DataBlkMatSvdRes(void) = default;

  DataBlkMatSvdRes(
      const size_t m,
      const size_t n,
      const size_t k,
      ElemT *u,
      QLTEN_Double *s,
      ElemT *vt
  ) : m(m), n(n), k(k), u(u), s(s), vt(vt) {}
};

template<typename ElemT>
void DeleteDataBlkMatSvdResMap(
    std::map<size_t, DataBlkMatSvdRes<ElemT>>
    &idx_svd_res_map
) {
  for (
    auto &idx_svd_res: idx_svd_res_map) {
    auto svd_res = idx_svd_res.second;
    qlten::QLFree(svd_res.u);
    svd_res.u = nullptr;
    qlten::QLFree(svd_res.s);
    svd_res.s = nullptr;
    qlten::QLFree(svd_res.vt);
    svd_res.vt = nullptr;
  }
}

struct SymMatEVDTask {
  size_t data_offset;
  size_t mat_size; // linear size
  SymMatEVDTask(
      const size_t data_offset,
      const size_t mat_size
  ) : data_offset(data_offset), mat_size(mat_size) {}
};

template<typename ElemT>
struct DataBlkMatQrRes {
  size_t m = 0;
  size_t n = 0;
  size_t k = 0;
  ElemT *q = nullptr;
  ElemT *r = nullptr;

  DataBlkMatQrRes(void) = default;

  DataBlkMatQrRes(
      const size_t m,
      const size_t n,
      const size_t k,
      ElemT *q,
      ElemT *r
  ) : m(m), n(n), k(k), q(q), r(r) {}
};

template<typename ElemT>
void DeleteDataBlkMatQrResMap(
    std::map<size_t, DataBlkMatQrRes<ElemT>>
    &idx_qr_res_map
) {
  for (auto &idx_qr_res: idx_qr_res_map) {
    auto qr_res = idx_qr_res.second;
    qlten::QLFree(qr_res.q);
    qr_res.q = nullptr;
    qlten::QLFree(qr_res.r);
    qr_res.r = nullptr;
  }
}
} /* qlten */
#endif /* ifndef QLTEN_QLTENSOR_BLK_SPAR_DATA_TEN_RAW_DATA_OPERATION_TASKS_H */
