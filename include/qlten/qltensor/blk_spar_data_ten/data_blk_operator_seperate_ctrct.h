// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2021-7-29
*
* Description: QuantumLiquids/tensor project. Data block level operations for block
* sparse data tensor for seperated contract task.
*/

#pragma once

#include "qlten/qltensor/blk_spar_data_ten/blk_spar_data_ten.h"
#include "qlten/qltensor/blk_spar_data_ten/raw_data_operations.h"
#include "qlten/qltensor/blk_spar_data_ten/raw_data_operation_tasks.h"
#include "qlten/framework/hp_numeric/lapack.h"    // MatSVD, MatQR
#include "qlten/utility/timer.h"
#include "qlten/qltensor/blk_spar_data_ten/data_blk_operations.h"

namespace qlten {

/**
Generate data blocks for two tensor contraction.


@param ctrct_axes_set To-be contracted tensor axes indexes.
       For example, {{0, 1}, {3, 2}}.
*/
template<typename ElemT, typename QNT>
template<typename T>
typename std::enable_if<!Fermionicable<T>::IsFermionic(), std::vector<RawDataCtrctTask>>::type
BlockSparseDataTensor<ElemT, QNT>::DataBlkGenFor1SectTenCtrct(
    const std::map<size_t, qlten::DataBlk<QNT>> &a_blk_idx_data_blk_map_select,
    const std::map<size_t, qlten::DataBlk<QNT>> &b_blk_idx_data_blk_map,
    const std::vector<std::vector<size_t>> &ctrct_axes_set,
    const std::vector<std::vector<size_t>> &saved_axes_set,
    std::vector<size_t> &b_blk_idx_qnblk_info_part_hash_map
) {
  auto a_blk_idx_qnblk_info_part_hash_map = GenBlkIdxQNBlkCoorPartHashMap(
      a_blk_idx_data_blk_map_select,
      ctrct_axes_set[0]
  );
  std::vector<RawDataCtrctTask> raw_data_ctrct_tasks;
  std::unordered_map<size_t, size_t>
      a_blk_idx_m_map,
      a_blk_idx_k_map,
      b_blk_idx_n_map;
  bool c_is_scalar = IsScalar();
  bool scalar_c_first_task = true;
#ifndef NDEBUG
  if (c_is_scalar) {
    assert(saved_axes_set[0].empty() && saved_axes_set[1].empty());
  }
#endif /* ifndef NDEBUG */
  if (c_is_scalar) {
    for (size_t i = 0; i < a_blk_idx_qnblk_info_part_hash_map.size(); i += 2) {
      for (size_t j = 0; j < b_blk_idx_qnblk_info_part_hash_map.size(); j += 2) {
        if (a_blk_idx_qnblk_info_part_hash_map[i + 1] == b_blk_idx_qnblk_info_part_hash_map[j + 1]) {
          auto a_blk_idx = a_blk_idx_qnblk_info_part_hash_map[i];
          auto b_blk_idx = b_blk_idx_qnblk_info_part_hash_map[j];
          const auto &a_data_blk = a_blk_idx_data_blk_map_select.at(a_blk_idx);
          const auto &b_data_blk = b_blk_idx_data_blk_map.at(b_blk_idx);
          size_t m(1), n(1);
          size_t k = VecMultiSelectElemts(a_data_blk.shape, ctrct_axes_set[0]);
          QLTEN_Double beta;
          if (scalar_c_first_task) {
            beta = 0.0;
            raw_data_size_ = 1;     // Set raw data size at first task scheduling
            scalar_c_first_task = false;
          } else {
            beta = 1.0;
          }
          double f_sign = 1.0;
          raw_data_ctrct_tasks.push_back(
              RawDataCtrctTask(
                  a_blk_idx,
                  a_data_blk.data_offset,
                  b_blk_idx,
                  b_data_blk.data_offset,
                  m, k, n,
                  f_sign,
                  beta
              )
          );
        }

      }
    }
  } else {
    for (size_t i = 0; i < a_blk_idx_qnblk_info_part_hash_map.size(); i += 2) {
      size_t m(0), k(0);
      for (size_t j = 0; j < b_blk_idx_qnblk_info_part_hash_map.size(); j += 2) {
        if (a_blk_idx_qnblk_info_part_hash_map[i + 1] == b_blk_idx_qnblk_info_part_hash_map[j + 1]) {
          auto a_blk_idx = a_blk_idx_qnblk_info_part_hash_map[i];
          auto b_blk_idx = b_blk_idx_qnblk_info_part_hash_map[j];
          const auto &a_data_blk = a_blk_idx_data_blk_map_select.at(a_blk_idx);
          const auto &b_data_blk = b_blk_idx_data_blk_map.at(b_blk_idx);
          // Calculate m, k, n
          size_t n;
          if (m == 0) {
            m = VecMultiSelectElemts(a_data_blk.shape, saved_axes_set[0]);
            k = VecMultiSelectElemts(a_data_blk.shape, ctrct_axes_set[0]);
          }
          if (b_blk_idx_n_map.find(b_blk_idx) != b_blk_idx_n_map.end()) {
            n = b_blk_idx_n_map.at(b_blk_idx);
          } else {
            n = VecMultiSelectElemts(b_data_blk.shape, saved_axes_set[1]);
            b_blk_idx_n_map[b_blk_idx] = n;
          }


          // Create raw data contraction task
          auto c_blk_coors = GenTenCtrctDataBlkCoors(
              a_data_blk.blk_coors,
              b_data_blk.blk_coors,
              saved_axes_set
          );
          auto c_blk_idx = BlkCoorsToBlkIdx(c_blk_coors);
          QLTEN_Double beta;
          if (blk_idx_data_blk_map_.find(c_blk_idx) !=
              blk_idx_data_blk_map_.end()
              ) {
            beta = 1.0;
          } else {
            auto c_blk_shape = GenTenCtrctDataBlkCoors(
                a_data_blk.shape,
                b_data_blk.shape,
                saved_axes_set
            );
            blk_idx_data_blk_map_[c_blk_idx] =
                DataBlk<QNT>(std::move(c_blk_coors), std::move(c_blk_shape));
            beta = 0.0;
          }
          double f_sign = 1.0;
          raw_data_ctrct_tasks.push_back(
              RawDataCtrctTask(
                  a_blk_idx,
                  a_data_blk.data_offset,
                  b_blk_idx,
                  b_data_blk.data_offset,
                  c_blk_idx,
                  m, k, n,
                  f_sign,
                  beta
              )
          );
        }
      }
    }
    DataBlksOffsetRefresh();
  }
  for (auto &task: raw_data_ctrct_tasks) {
    if (!c_is_scalar) {
      task.c_data_offset = blk_idx_data_blk_map_[task.c_blk_idx].data_offset;
    } else {
      task.c_data_offset = 0;
    }
  }

  return raw_data_ctrct_tasks;
}

///< Fermion case
template<typename ElemT, typename QNT>
template<typename T>
typename std::enable_if<Fermionicable<T>::IsFermionic(), std::vector<RawDataCtrctTask>>::type
BlockSparseDataTensor<ElemT, QNT>::DataBlkGenFor1SectTenCtrct(
    const std::map<size_t, qlten::DataBlk<QNT>> &a_blk_idx_data_blk_map_select,
    const std::map<size_t, qlten::DataBlk<QNT>> &b_blk_idx_data_blk_map,
    const std::vector<std::vector<size_t>> &ctrct_axes_set,
    const std::vector<std::vector<size_t>> &saved_axes_set,
    std::vector<size_t> &b_blk_idx_qnblk_info_part_hash_map,
    const std::vector<TenIndexDirType> &a_ctrct_idx_dir
) {
  auto a_blk_idx_qnblk_info_part_hash_map = GenBlkIdxQNBlkCoorPartHashMap(
      a_blk_idx_data_blk_map_select,
      ctrct_axes_set[0]
  );

  std::vector<RawDataCtrctTask> raw_data_ctrct_tasks;
  std::unordered_map<size_t, size_t>
      a_blk_idx_m_map,
      a_blk_idx_k_map,
      b_blk_idx_n_map;
  bool c_is_scalar = IsScalar();
  bool scalar_c_first_task = true;
#ifndef NDEBUG
  if (c_is_scalar) {
    assert(saved_axes_set[0].empty() && saved_axes_set[1].empty());
  }
#endif /* ifndef NDEBUG */
  if (c_is_scalar) {
    for (size_t i = 0; i < a_blk_idx_qnblk_info_part_hash_map.size(); i += 2) {
      for (size_t j = 0; j < b_blk_idx_qnblk_info_part_hash_map.size(); j += 2) {
        if (a_blk_idx_qnblk_info_part_hash_map[i + 1] == b_blk_idx_qnblk_info_part_hash_map[j + 1]) {
          auto a_blk_idx = a_blk_idx_qnblk_info_part_hash_map[i];
          auto b_blk_idx = b_blk_idx_qnblk_info_part_hash_map[j];
          const auto &a_data_blk = a_blk_idx_data_blk_map_select.at(a_blk_idx);
          const auto &b_data_blk = b_blk_idx_data_blk_map.at(b_blk_idx);
          size_t m(1), n(1);
          size_t k = VecMultiSelectElemts(a_data_blk.shape, ctrct_axes_set[0]);
          QLTEN_Double beta;
          if (scalar_c_first_task) {
            beta = 0.0;
            raw_data_size_ = 1;     // Set raw data size at first task scheduling
            scalar_c_first_task = false;
          } else {
            beta = 1.0;
          }
          double f_sign = 1.0;
          f_sign = FermionExchangeSignForCtrct(a_data_blk.GetBlkFermionParities(),
                                               b_data_blk.GetBlkFermionParities(),
                                               ctrct_axes_set[0], ctrct_axes_set[1],
                                               a_ctrct_idx_dir);
          raw_data_ctrct_tasks.push_back(
              RawDataCtrctTask(
                  a_blk_idx,
                  a_data_blk.data_offset,
                  b_blk_idx,
                  b_data_blk.data_offset,
                  m, k, n,
                  f_sign,
                  beta
              )
          );
        }

      }
    }
  } else {
    for (size_t i = 0; i < a_blk_idx_qnblk_info_part_hash_map.size(); i += 2) {
      size_t m(0), k(0);
      for (size_t j = 0; j < b_blk_idx_qnblk_info_part_hash_map.size(); j += 2) {
        if (a_blk_idx_qnblk_info_part_hash_map[i + 1] == b_blk_idx_qnblk_info_part_hash_map[j + 1]) {
          auto a_blk_idx = a_blk_idx_qnblk_info_part_hash_map[i];
          auto b_blk_idx = b_blk_idx_qnblk_info_part_hash_map[j];
          const auto &a_data_blk = a_blk_idx_data_blk_map_select.at(a_blk_idx);
          const auto &b_data_blk = b_blk_idx_data_blk_map.at(b_blk_idx);
          // Calculate m, k, n
          size_t n;
          if (m == 0) {
            m = VecMultiSelectElemts(a_data_blk.shape, saved_axes_set[0]);
            k = VecMultiSelectElemts(a_data_blk.shape, ctrct_axes_set[0]);
          }
          if (b_blk_idx_n_map.find(b_blk_idx) != b_blk_idx_n_map.end()) {
            n = b_blk_idx_n_map.at(b_blk_idx);
          } else {
            n = VecMultiSelectElemts(b_data_blk.shape, saved_axes_set[1]);
            b_blk_idx_n_map[b_blk_idx] = n;
          }


          // Create raw data contraction task
          auto c_blk_coors = GenTenCtrctDataBlkCoors(
              a_data_blk.blk_coors,
              b_data_blk.blk_coors,
              saved_axes_set
          );
          auto c_blk_idx = BlkCoorsToBlkIdx(c_blk_coors);
          QLTEN_Double beta;
          if (blk_idx_data_blk_map_.find(c_blk_idx) !=
              blk_idx_data_blk_map_.end()
              ) {
            beta = 1.0;
          } else {
            auto c_blk_shape = GenTenCtrctDataBlkCoors(
                a_data_blk.shape,
                b_data_blk.shape,
                saved_axes_set
            );
            blk_idx_data_blk_map_[c_blk_idx] =
                DataBlk<QNT>(std::move(c_blk_coors), std::move(c_blk_shape));
            beta = 0.0;
          }
          double f_sign = 1.0;

          f_sign = FermionExchangeSignForCtrct(a_data_blk.GetBlkFermionParities(),
                                               b_data_blk.GetBlkFermionParities(),
                                               ctrct_axes_set[0], ctrct_axes_set[1],
                                               a_ctrct_idx_dir);

          raw_data_ctrct_tasks.push_back(
              RawDataCtrctTask(
                  a_blk_idx,
                  a_data_blk.data_offset,
                  b_blk_idx,
                  b_data_blk.data_offset,
                  c_blk_idx,
                  m, k, n,
                  f_sign,
                  beta
              )
          );
        }
      }
    }
    DataBlksOffsetRefresh();
  }
  for (auto &task: raw_data_ctrct_tasks) {
    if (!c_is_scalar) {
      task.c_data_offset = blk_idx_data_blk_map_[task.c_blk_idx].data_offset;
    } else {
      task.c_data_offset = 0;
    }
  }

  return raw_data_ctrct_tasks;
}

}//qlten