// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-26 21:11
*
* Description: QuantumLiquids/tensor project. Raw data operation member functions in
* BlockSparseDataTensor.
*/

/**
@file raw_data_operations.h
@brief Raw data operation member functions in BlockSparseDataTensor.
*/
#ifndef QLTEN_QLTENSOR_BLK_SPAR_DATA_TEN_RAW_DATA_OPERATIONS_H
#define QLTEN_QLTENSOR_BLK_SPAR_DATA_TEN_RAW_DATA_OPERATIONS_H
#include <iostream>     // endl, istream, ostream
#include <cmath>        // sqrt
#include <cstdlib>      // malloc, free, calloc
#include <cstring>      // memcpy, memset
#include <omp.h>
#include <cassert>     // assert

#include "qlten/qltensor/blk_spar_data_ten/blk_spar_data_ten.h"
#include "qlten/framework/value_t.h"                                      // CoorsT, ShapeT
#include "qlten/qltensor/blk_spar_data_ten/raw_data_operation_tasks.h"    // RawDataTransposeTask
#include "qlten/framework/hp_numeric/ten_trans.h"                         // TensorTranspose
#include "qlten/framework/hp_numeric/blas_level1.h"                       // VectorAddTo
#include "qlten/framework/hp_numeric/blas_level3.h"                       // MatMultiply
#include "qlten/framework/hp_numeric/lapack.h"                            // MatSVD
#include "qlten/framework/hp_numeric/omp_set.h"
#include "qlten/utility/utils_inl.h"                                      // Rand, CalcScalarNorm2, CalcConj, SubMatMemCpy

#ifdef Release
#define NDEBUG
#endif

namespace qlten {

/**
Release the raw data, set the pointer to null, set the size to 0.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataFree_(void) {
  free(pactual_raw_data_);
  pactual_raw_data_ = nullptr;
  actual_raw_data_size_ = 0;
}

/**
Directly set raw data point to nullptr and set actual raw data size to 0.

@note The memory may leak!!
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataDiscard_(void) {
  pactual_raw_data_ = nullptr;
  actual_raw_data_size_ = 0;
}

/**
Allocate memoery using a size.

@param init Whether initialize the memory to 0.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataAlloc_(
    const size_t size,
    const bool init
) {
  free(pactual_raw_data_);
  ///< @question: if need to justify size!=0
  if (!init) {
    pactual_raw_data_ = (ElemT *) malloc(size * sizeof(ElemT));
  } else {
    pactual_raw_data_ = (ElemT *) calloc(size, sizeof(ElemT));
  }
  actual_raw_data_size_ = size;
}

/**
Insert a subarray to the raw data array and decide whether initialize the memory
of the subarray.

@param offset Start data offset for inserting subarray.
@param size   The size of the subarray.
@param init   Whether initialize the inserted subarray to 0.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataInsert_(
    const size_t offset,
    const size_t size,
    const bool init
) {
  if (actual_raw_data_size_ == 0) {
    assert(offset == 0);
    assert(pactual_raw_data_ == nullptr);
    if (!init) {
      pactual_raw_data_ = (ElemT *) malloc(size * sizeof(ElemT));
    } else {
      pactual_raw_data_ = (ElemT *) calloc(size, sizeof(ElemT));
    }
    actual_raw_data_size_ = size;
  } else {
    size_t new_data_size = actual_raw_data_size_ + size;
    ElemT *new_pdata = (ElemT *) malloc(new_data_size * sizeof(ElemT));
    hp_numeric::VectorCopy(pactual_raw_data_, offset, new_pdata);
    if (init) {
      std::fill(pactual_raw_data_ + offset, pactual_raw_data_ + offset + size, ElemT(0));
    }
    hp_numeric::VectorCopy(
        pactual_raw_data_ + offset,
        actual_raw_data_size_ - offset,
        new_pdata + (offset + size)
    );
    free(pactual_raw_data_);
    pactual_raw_data_ = new_pdata;
    actual_raw_data_size_ = new_data_size;
  }
  return;
}

/**
Random set all the actual raw data to [0, 1].
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataRand_(void) {
  for (size_t i = 0; i < actual_raw_data_size_; ++i) {
    Rand(pactual_raw_data_[i]);
  }
}

/**
Tensor transpose for the 1D raw data array.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataTranspose_(
    const std::vector<RawDataTransposeTask> &raw_data_trans_tasks) {
  ElemT *ptransed_actual_raw_data = (ElemT *) malloc(actual_raw_data_size_ * sizeof(ElemT));
  for (auto &trans_task : raw_data_trans_tasks) {
    hp_numeric::TensorTranspose(
        trans_task.transed_order,
        trans_task.ten_rank,
        pactual_raw_data_ + trans_task.original_data_offset,
        trans_task.original_shape,
        ptransed_actual_raw_data + trans_task.transed_data_offset,
        trans_task.transed_shape
    );
  }
  free(pactual_raw_data_);
  pactual_raw_data_ = ptransed_actual_raw_data;
}

/**
calculate the 2-norm (square root of elements' square summation) of the raw data array.

@return The 2-norm
*/
template<typename ElemT, typename QNT>
QLTEN_Double BlockSparseDataTensor<ElemT, QNT>::RawDataNorm_(void) {
  return hp_numeric::Vector2Norm(pactual_raw_data_, actual_raw_data_size_);
}

/**
Normalize the raw data array.

@return The norm before normalization.
*/
template<typename ElemT, typename QNT>
QLTEN_Double BlockSparseDataTensor<ElemT, QNT>::RawDataNormalize_(void) {
  double norm = hp_numeric::Vector2Norm(pactual_raw_data_, actual_raw_data_size_);
  double inv_norm = 1.0 / norm;
  hp_numeric::VectorScale(pactual_raw_data_, actual_raw_data_size_, inv_norm);
  return norm;
}

/**
Complex conjugate for raw data.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataConj_(void) {
  if (std::is_same<ElemT, QLTEN_Double>::value) {
    // Do nothing
  } else {
    for (size_t i = 0; i < actual_raw_data_size_; ++i) {
      pactual_raw_data_[i] = CalcConj(pactual_raw_data_[i]);
    }
  }
}

/**
Copy a piece of raw data from another place. You can decided whether add this
piece on the original one.

@param raw_data_copy_tasks Raw data copy task list.
@param psrc_raw_data The pointer to source data.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataCopy_(
    const std::vector<RawDataCopyTask> &raw_data_copy_tasks,
    const ElemT *psrc_raw_data
) {
  for (auto &task : raw_data_copy_tasks) {
    if (task.copy_and_add) {
      hp_numeric::VectorAddTo(
          psrc_raw_data + task.src_data_offset,
          task.src_data_size,
          pactual_raw_data_ + task.dest_data_offset
      );
    } else {
      hp_numeric::VectorCopy(
          psrc_raw_data + task.src_data_offset,
          task.src_data_size,
          pactual_raw_data_ + task.dest_data_offset
      );
    }
  }
}

template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataCopy_(
    const std::vector<ElemT *> &src_pointers,
    const std::vector<ElemT *> &dest_pointers,
    const std::vector<size_t> &copy_sizes
) {
  size_t task_size = src_pointers.size();
  int ompth = hp_numeric::tensor_manipulation_num_threads;
#pragma omp parallel for default(none) \
                shared(task_size, dest_pointers, src_pointers, copy_sizes)\
                num_threads(ompth)\
                schedule(dynamic)
  for (size_t i = 0; i < task_size; i++) {
    memcpy(
        dest_pointers[i],
        src_pointers[i],
        copy_sizes[i] * sizeof(ElemT)
    );
  }
}

/**
Copy a piece of raw data from another place. 
The destination must be different and there is no addition 

@param raw_data_copy_tasks Raw data copy task list.
@param psrc_raw_data The pointer to source data.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataCopyNoAdd_(
    const std::vector<RawDataCopyTask> &raw_data_copy_tasks,
    const ElemT *psrc_raw_data
) {
  size_t task_size = raw_data_copy_tasks.size();
  size_t ompth = hp_numeric::tensor_manipulation_num_threads;

#pragma omp parallel for default(none) \
                shared(task_size, raw_data_copy_tasks, pactual_raw_data_, psrc_raw_data)\
                num_threads(ompth)\
                schedule(dynamic)
  for (size_t i = 0; i < task_size; i++) {
    RawDataCopyTask task = raw_data_copy_tasks[i];
    memcpy(
        pactual_raw_data_ + task.dest_data_offset,
        psrc_raw_data + task.src_data_offset,
        task.src_data_size * sizeof(ElemT)
    );
  }
}

/**
Copy and scale a piece of raw data from another place. You can decided whether
add this piece on the original one.

@param raw_data_copy_tasks Raw data copy task list.
@param psrc_raw_data The pointer to source data.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataCopyAndScale_(
    const RawDataCopyAndScaleTask<ElemT> &raw_data_copy_and_scale_task,
    const ElemT *psrc_raw_data
) {
  auto dest_data_offset = blk_idx_data_blk_map_[
      BlkCoorsToBlkIdx(
          raw_data_copy_and_scale_task.dest_blk_coors
      )
  ].data_offset;
  if (raw_data_copy_and_scale_task.copy_and_add) {
    hp_numeric::VectorAddTo(
        psrc_raw_data + raw_data_copy_and_scale_task.src_data_offset,
        raw_data_copy_and_scale_task.src_data_size,
        pactual_raw_data_ + dest_data_offset,
        raw_data_copy_and_scale_task.coef
    );
  } else {
    hp_numeric::VectorScaleCopy(
        psrc_raw_data + raw_data_copy_and_scale_task.src_data_offset,
        raw_data_copy_and_scale_task.src_data_size,
        pactual_raw_data_ + dest_data_offset,
        raw_data_copy_and_scale_task.coef
    );
  }
}

/**
Set a piece of data to zeros.
@param offset  starting point of the piece of data
@param size    the size the piece of data
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataSetZeros_(
    const size_t offset,
    const size_t size
) {
  memset(pactual_raw_data_ + offset, 0, size * sizeof(ElemT));
}

template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataSetZeros_(
    const std::vector<size_t> &offsets,
    const std::vector<size_t> &sizes
) {
  assert(offsets.size() == sizes.size());
  for (size_t i = 0; i < offsets.size(); i++) {
    RawDataSetZeros_(offsets[i], sizes[i]);
  }
}

template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataSetZeros_(
    const std::vector<RawDataSetZerosTask> &set_zeros_tasks
) {
  for (auto &task : set_zeros_tasks) {
    RawDataSetZeros_(task.data_offset, task.data_size);
  }
}

/**
Duplicate a whole same size real raw data array from another place.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataDuplicateFromReal_(
    const QLTEN_Double *preal_raw_data_, const size_t size) {
  if (std::is_same<ElemT, QLTEN_Complex>::value) {
    hp_numeric::VectorRealToCplx(preal_raw_data_, size, pactual_raw_data_);
  } else {
    assert(false);    // TODO: To-be implemented!
  }
}

/**
Multiply the raw data by a scalar.

@param s A scalar.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataMultiplyByScalar_(
    const ElemT s
) {
  if (actual_raw_data_size_ != 0) {
    hp_numeric::VectorScale(pactual_raw_data_, actual_raw_data_size_, s);
  }
}

/**
Multiply two matrices and assign in.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataTwoMatMultiplyAndAssignIn_(
    const ElemT *a,
    const ElemT *b,
    const size_t c_data_offset,
    const size_t m, const size_t k, const size_t n,
    const ElemT beta
) {
  assert(actual_raw_data_size_ != 0);
  hp_numeric::MatMultiply(
      a,
      b,
      m, k, n,
      beta,
      pactual_raw_data_ + c_data_offset
  );
}

template<typename ElemT, typename QNT>
ElemT *BlockSparseDataTensor<ElemT, QNT>::RawDataGenDenseDataBlkMat_(
    const TenDecompDataBlkMat<QNT> &data_blk_mat
) const {
  auto rows = data_blk_mat.rows;
  auto cols = data_blk_mat.cols;
  ElemT *mat = (ElemT *) calloc(rows * cols, sizeof(ElemT));
  for (auto &elem : data_blk_mat.elems) {
    auto i = elem.first[0];
    auto j = elem.first[1];
    auto row_offset = std::get<1>(data_blk_mat.row_scts[i]);
    auto col_offset = std::get<1>(data_blk_mat.col_scts[j]);
    auto m = std::get<2>(data_blk_mat.row_scts[i]);
    auto n = std::get<2>(data_blk_mat.col_scts[j]);
    auto blk_idx_in_bsdt = elem.second;
    auto sub_mem_begin = pactual_raw_data_ +
        blk_idx_data_blk_map_.at(blk_idx_in_bsdt).data_offset;
    SubMatMemCpy(
        rows, cols,
        row_offset, col_offset,
        m, n, sub_mem_begin,
        mat
    );
  }
  return mat;
}

/**
Read raw data from a stream.

@param is Input stream.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataRead_(std::istream &is) {
  is.seekg(1, std::ios::cur);    // Skip the line break.
  is.read((char *) pactual_raw_data_, actual_raw_data_size_ * sizeof(ElemT));
}

/**
Write raw data to a stream.

@param os Output stream.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataWrite_(std::ostream &os) const {
  os.write((char *) pactual_raw_data_, actual_raw_data_size_ * sizeof(ElemT));
  os << std::endl;
}

template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::ElementWiseInv(void) {
  int ompth = hp_numeric::tensor_manipulation_num_threads;
#pragma omp parallel for default(none) \
                shared(pactual_raw_data_, actual_raw_data_size_)\
                num_threads(ompth)\
                schedule(static)
  for (size_t i = 0; i < actual_raw_data_size_; i++) {
    *(pactual_raw_data_ + i) = 1.0 / (*(pactual_raw_data_ + i));
  }
}

template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::ElementWiseInv(double tolerance) {
  int ompth = hp_numeric::tensor_manipulation_num_threads;
#pragma omp parallel for default(none) \
                shared(pactual_raw_data_, tolerance)\
                num_threads(ompth)\
                schedule(static)
  for (size_t i = 0; i < actual_raw_data_size_; i++) {
    ElemT &elem = *(pactual_raw_data_ + i);
    elem = (std::abs(elem) < tolerance) ? ElemT(0) : 1.0 / elem;
  }
}

template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::ElementWiseSqrt(void) {
  int ompth = hp_numeric::tensor_manipulation_num_threads;
#pragma omp parallel for default(none) \
                shared(pactual_raw_data_, actual_raw_data_size_)\
                num_threads(ompth)\
                schedule(static)
  for (size_t i = 0; i < actual_raw_data_size_; i++) {
    *(pactual_raw_data_ + i) = std::sqrt(*(pactual_raw_data_ + i));
  }
}

template<typename T>
int sign(T val) {
  return (T(0) < val) - (val < T(0));
}

template<typename ElemT, typename QNT>
template<typename TenElemT, typename std::enable_if<std::is_same<TenElemT, QLTEN_Double>::value>::type *>
void BlockSparseDataTensor<ElemT, QNT>::ElementWiseSign() {
  int ompth = hp_numeric::tensor_manipulation_num_threads;
#pragma omp parallel for default(none) \
                shared(pactual_raw_data_, actual_raw_data_size_)\
                num_threads(ompth)\
                schedule(static)
  for (size_t i = 0; i < actual_raw_data_size_; i++) {
    *(pactual_raw_data_ + i) = sign(*(pactual_raw_data_ + i));
  }
}

template<typename ElemT, typename QNT>
template<typename TenElemT, typename std::enable_if<std::is_same<TenElemT, QLTEN_Complex>::value>::type *>
void BlockSparseDataTensor<ElemT, QNT>::ElementWiseSign(void) {
  int ompth = hp_numeric::tensor_manipulation_num_threads;
#pragma omp parallel for default(none) \
                shared(pactual_raw_data_, actual_raw_data_size_)\
                num_threads(ompth)\
                schedule(static)
  for (size_t i = 0; i < actual_raw_data_size_; i++) {
    *(pactual_raw_data_ + i) = sign((pactual_raw_data_ + i)->real());
  }
}

template<typename RandGenerator>
inline void RandSign(QLTEN_Double *number,
                     std::uniform_real_distribution<double> &dist,
                     RandGenerator &g) {
  if (*number > 1e-5) {
    *number = dist(g);
  } else if (*number < -1e-5) {
    *number = -dist(g);
  }
}

template<typename RandGenerator>
inline void RandSign(QLTEN_Complex *number,
                     std::uniform_real_distribution<double> &dist,
                     RandGenerator &g) {
  if (number->real() > 1e-5) {
    number->real(dist(g));
  } else if (number->real() < -1e-5) {
    number->real(-dist(g));
  }

  if (number->imag() > 1e-5) {
    number->imag(dist(g));
  } else if (number->imag() < -1e-5) {
    number->imag(-dist(g));
  }
}

template<typename ElemT, typename QNT>
template<typename RandGenerator>
void BlockSparseDataTensor<ElemT, QNT>::ElementWiseRandSign(std::uniform_real_distribution<double> &dist,
                                                            RandGenerator &g) {
  for (size_t i = 0; i < actual_raw_data_size_; i++) {
    RandSign(pactual_raw_data_ + i, dist, g);
  }
}

inline QLTEN_Double BoundNumber(QLTEN_Double number, double bound) {
  return sign(number) * bound;
}

inline QLTEN_Complex BoundNumber(QLTEN_Complex number, double bound) {
  return number * bound / std::abs(number);
}

template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::ElementWiseBoundTo(double bound) {
  for (size_t i = 0; i < actual_raw_data_size_; i++) {
    ElemT *elem = pactual_raw_data_ + i;
    if (std::abs(*elem) > bound) {
      *elem *= BoundNumber(*elem, bound);
    }
  }
}

} /* qlten */
#endif /* ifndef QLTEN_QLTENSOR_BLK_SPAR_DATA_TEN_RAW_DATA_OPERATIONS_H */
