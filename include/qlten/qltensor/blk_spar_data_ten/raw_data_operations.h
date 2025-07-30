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
#include <cmath>        // sqrt
#include <cstdlib>      // malloc, free, calloc
#include <cstring>      // memcpy, memset
#include <omp.h>
#include <cassert>     // assert

#ifdef USE_GPU
#include <thrust/execution_policy.h>   //thrust::device
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#endif
#include "qlten/framework/mem_ops.h"
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
  qlten::QLFree(pactual_raw_data_);
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
Allocate memory using a size.

@param init Whether initialize the memory to 0.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataAlloc_(
    const size_t size,
    const bool init
) {
  qlten::QLFree(pactual_raw_data_);
  ///< @question: if need to justify size!=0
  if (!init) {
    pactual_raw_data_ = (ElemT *) qlten::QLMalloc(size * sizeof(ElemT));
  } else {
    pactual_raw_data_ = (ElemT *) qlten::QLCalloc(size, sizeof(ElemT));
  }
  actual_raw_data_size_ = size;
}

/**
Insert a subarray to the raw data array and decide whether initialize the memory
of the subarray.

For cuda code, we assume the the raw data in host have
higher validation than those in device. And the insertion
is performed in host and make the raw data in host valid.

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
      pactual_raw_data_ = (ElemT *) qlten::QLMalloc(size * sizeof(ElemT));
    } else {
      pactual_raw_data_ = (ElemT *) qlten::QLCalloc(size, sizeof(ElemT));
    }
    actual_raw_data_size_ = size;
  } else {
    size_t new_data_size = actual_raw_data_size_ + size;
    ElemT *new_pdata = (ElemT *) qlten::QLMalloc(new_data_size * sizeof(ElemT));
#ifndef USE_GPU
    hp_numeric::VectorCopy(pactual_raw_data_, offset, new_pdata);
    if (init) {
      std::fill(new_pdata + offset, new_pdata + offset + size, ElemT(0));
    }
    hp_numeric::VectorCopy(
        pactual_raw_data_ + offset,
        actual_raw_data_size_ - offset,
        new_pdata + (offset + size)
    );
#else
    // Copy old data to new locations.
    auto cuda_err = cudaMemcpy(new_pdata, pactual_raw_data_, offset * sizeof(ElemT), cudaMemcpyDeviceToDevice);
    if (cuda_err != cudaSuccess) {
      std::cerr << "cudaMemcpy error (1): " << cuda_err << std::endl;
    }

    if (init) {
      cuda_err = cudaMemset(new_pdata + offset, 0, size * sizeof(ElemT));
      if (cuda_err != cudaSuccess) {
        std::cerr << "cudaMemset error: " << cuda_err << std::endl;
      }
    }

    cuda_err = cudaMemcpy(new_pdata + (offset + size),
                          pactual_raw_data_ + offset,
                          (actual_raw_data_size_ - offset) * sizeof(ElemT),
                          cudaMemcpyDeviceToDevice);
    if (cuda_err != cudaSuccess) {
      std::cerr << "cudaMemcpy error (2): " << cuda_err << std::endl;
    }
#endif
    // Free old data and update pointer.
    qlten::QLFree(pactual_raw_data_);
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
#ifndef USE_GPU
  // CPU implementation
  for (size_t i = 0; i < actual_raw_data_size_; ++i) {
    Rand(pactual_raw_data_[i]);
  }
#else
  // GPU implementation
  dim3 blockDim(1024);
  dim3 gridDim((actual_raw_data_size_ + blockDim.x - 1) / blockDim.x);

  // Launch CUDA kernel to populate random data
  RandomKernel<<<gridDim, blockDim>>>(pactual_raw_data_, actual_raw_data_size_);
  cudaDeviceSynchronize();  // Synchronize for error checking

  auto cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess) {
    std::cerr << "CUDA kernel error: " << cuda_err << std::endl;
  }
#endif
}

/**
Fill all the actual raw data with given value.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataFill_(const ElemT &value) {
#ifndef USE_GPU
  // CPU implementation
  for (size_t i = 0; i < actual_raw_data_size_; ++i) {
    pactual_raw_data_[i] = value;
  }
#else
  // GPU implementation
  dim3 blockDim(1024);
  dim3 gridDim((actual_raw_data_size_ + blockDim.x - 1) / blockDim.x);

  // Launch CUDA kernel to fill data with given value
  FillKernel<<<gridDim, blockDim>>>(pactual_raw_data_, actual_raw_data_size_, value);
  cudaDeviceSynchronize();  // Synchronize for error checking

  auto cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess) {
    std::cerr << "CUDA kernel error: " << cuda_err << std::endl;
  }
#endif
}

/**
Tensor transpose for the 1D raw data array.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataTranspose_(
    const std::vector<RawDataTransposeTask> &raw_data_trans_tasks) {
  ElemT *ptransed_actual_raw_data = (ElemT *) qlten::QLMalloc(actual_raw_data_size_ * sizeof(ElemT));
  for (auto &trans_task : raw_data_trans_tasks) {
    hp_numeric::TensorTranspose(
        trans_task.transed_order,
        trans_task.ten_rank,
        pactual_raw_data_ + trans_task.original_data_offset,
        trans_task.original_shape,
        ptransed_actual_raw_data + trans_task.transed_data_offset,
        trans_task.transed_shape,
        trans_task.scale_factor
    );
  }
  qlten::QLFree(pactual_raw_data_);
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

template<typename ElemT, typename QNT>
QLTEN_Double BlockSparseDataTensor<ElemT, QNT>::RawDataFermionNorm_(
    const std::vector<RawDataFermionNormTask> &tasks) {
  double sum_square = 0.0;
  for (auto &task : tasks) {
    sum_square += task.sign * hp_numeric::VectorSumSquares(pactual_raw_data_ + task.data_offset, task.data_size);
  }
  if (sum_square < 0) {
    std::cerr << "warning : Norm^2 < 0." << std::endl;
  }
  return std::sqrt(sum_square);
}

/**
Normalize the raw data array.

@return The norm before normalization.
*/
template<typename ElemT, typename QNT>
QLTEN_Double BlockSparseDataTensor<ElemT, QNT>::RawDataNormalize_(double norm) {
  double inv_norm = 1.0 / norm;
  hp_numeric::VectorScale(pactual_raw_data_, actual_raw_data_size_, inv_norm);
  return norm;
}

/**
Complex conjugate for raw data.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataConj_(void) {
  if constexpr (std::is_same<ElemT, QLTEN_Double>::value) {
    // Do nothing
  } else {
#ifndef  USE_GPU
    for (size_t i = 0; i < actual_raw_data_size_; ++i) {
      pactual_raw_data_[i] = CalcConj(pactual_raw_data_[i]);
    }
#else
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (actual_raw_data_size_ + threadsPerBlock - 1) / threadsPerBlock;
    ConjugateKernel<<<blocksPerGrid, threadsPerBlock>>>(pactual_raw_data_, actual_raw_data_size_);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
      assert(false);
    }
#endif
  }
}

/**
Complex conjugate for raw data in fermionic tensor network
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::FermionicRawDataConj_(const std::vector<RawDataInplaceReverseTask> &tasks) {
  RawDataConj_();
  for (auto &task : tasks) {
    hp_numeric::VectorScale(pactual_raw_data_ + task.data_offset, task.data_size, -1.0);
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

#ifdef USE_GPU
  // CUDA-specific memory copy (for GPU)
  for (size_t i = 0; i < task_size; i++) {
    cudaMemcpy(
        dest_pointers[i],     // destination on the device
        src_pointers[i],      // source on the device
        copy_sizes[i] * sizeof(ElemT), // size to copy
        cudaMemcpyDeviceToDevice  // direction of copy
    );
  }
#else
  // CPU memory copy (without CUDA)
  int ompth = hp_numeric::tensor_manipulation_num_threads;
#pragma omp parallel for default(none) \
               shared(task_size, dest_pointers, src_pointers, copy_sizes) \
               num_threads(ompth) \
               schedule(dynamic)
  for (size_t i = 0; i < task_size; i++) {
    memcpy(
        dest_pointers[i],
        src_pointers[i],
        copy_sizes[i] * sizeof(ElemT)
    );
  }
#endif
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

#ifdef USE_GPU
  // GPU memory copy
  for (size_t i = 0; i < task_size; i++) {
    RawDataCopyTask task = raw_data_copy_tasks[i];
    cudaMemcpy(
        pactual_raw_data_ + task.dest_data_offset,
        psrc_raw_data + task.src_data_offset,
        task.src_data_size * sizeof(ElemT),
        cudaMemcpyDeviceToDevice
    );
  }
#else
  // CPU memory copy with OpenMP parallelization for large datasets
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
#endif
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
  qlten::QLMemset(pactual_raw_data_ + offset, 0, size * sizeof(ElemT));
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
  if constexpr (std::is_same<ElemT, QLTEN_Complex>::value) {
    hp_numeric::VectorRealToCplx(preal_raw_data_, size, pactual_raw_data_);
  } else {
    assert(false);
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
    const double alpha, //for fermion sign
    const ElemT beta
) {
  assert(actual_raw_data_size_ != 0);
  hp_numeric::MatMultiply(
      alpha,
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
  ElemT *mat = (ElemT *) qlten::QLCalloc(rows * cols, sizeof(ElemT));
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
#ifndef  USE_GPU
  is.seekg(1, std::ios::cur);    // Skip the line break.
  is.read((char *) pactual_raw_data_, actual_raw_data_size_ * sizeof(ElemT));
#else
  // Skip the line break in the input stream
  is.seekg(1, std::ios::cur);

  // Allocate temporary host buffer
  ElemT *h_buffer = new ElemT[actual_raw_data_size_];

  // Read data from the input stream into host memory
  is.read(reinterpret_cast<char *>(h_buffer), actual_raw_data_size_ * sizeof(ElemT));

  // Copy data from host to device
  cudaMemcpy(pactual_raw_data_, h_buffer,
             actual_raw_data_size_ * sizeof(ElemT),
             cudaMemcpyHostToDevice);

  // Free the host memory
  delete[] h_buffer;
#endif
}

/**
Write raw data to a stream.

@param os Output stream.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataWrite_(std::ostream &os) const {
#ifndef USE_GPU
  os.write((char *) pactual_raw_data_, actual_raw_data_size_ * sizeof(ElemT));
  os << std::endl;
#else
  // Allocate temporary host buffer
    ElemT *h_buffer = new ElemT[actual_raw_data_size_];

    // Copy data from device to host
    cudaMemcpy(h_buffer, pactual_raw_data_,
               actual_raw_data_size_ * sizeof(ElemT),
               cudaMemcpyDeviceToHost);

    // Write the data to output stream
    os.write(reinterpret_cast<char *>(h_buffer), actual_raw_data_size_ * sizeof(ElemT));
    os << std::endl;

    // Free host memory
    delete[] h_buffer;
#endif
}

template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataMPISend(const MPI_Comm &mpi_comm,
                                                       const int dest,
                                                       const int tag) const {
  if (IsScalar() && actual_raw_data_size_ == 0) {
    ElemT zero = ElemT(0);
    hp_numeric::MPI_Send(&zero, 1, dest, tag, mpi_comm);
  } else {
    hp_numeric::MPI_Send(pactual_raw_data_, actual_raw_data_size_, dest, tag, mpi_comm);
  }
}

template<typename ElemT, typename QNT>
MPI_Status BlockSparseDataTensor<ElemT, QNT>::RawDataMPIRecv(
    const MPI_Comm &mpi_comm,
    const int source,
    const int tag_data
) {
  assert(source != MPI_ANY_SOURCE);
  auto status = hp_numeric::MPI_Recv(pactual_raw_data_, actual_raw_data_size_, source, tag_data, mpi_comm);
  return status;
}

template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataMPIBcast(const MPI_Comm &comm, const int root) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == root && IsScalar() && actual_raw_data_size_ == 0) {
    ElemT zero = ElemT(0);
    hp_numeric::MPI_Bcast(&zero, 1, root, comm);
  } else {
    hp_numeric::MPI_Bcast(pactual_raw_data_, actual_raw_data_size_, root, comm);
  }
}

#ifndef  USE_GPU
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

template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::ElementWiseSquare(void) {
  int ompth = hp_numeric::tensor_manipulation_num_threads;
#pragma omp parallel for default(none) \
                shared(pactual_raw_data_, actual_raw_data_size_)\
                num_threads(ompth)\
                schedule(static)
  for (size_t i = 0; i < actual_raw_data_size_; i++) {
    *(pactual_raw_data_ + i) = *(pactual_raw_data_ + i) * *(pactual_raw_data_ + i);
  }
}

/**
Element-wise multiplication of raw data with another tensor's raw data.
This function multiplies elements from the current tensor with elements from rhs_data.

@param data_offset Starting offset in the current tensor's raw data.
@param rhs_data Pointer to the right-hand side tensor's raw data.
@param size Number of elements to multiply.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataElementWiseMultiply_(
    const size_t data_offset,
    const ElemT *rhs_data,
    const size_t size
) {
  int ompth = hp_numeric::tensor_manipulation_num_threads;
#pragma omp parallel for default(none) \
                shared(pactual_raw_data_, rhs_data, data_offset, size)\
                num_threads(ompth)\
                schedule(static)
  for (size_t i = 0; i < size; i++) {
    pactual_raw_data_[data_offset + i] *= rhs_data[i];
  }
}
#endif

#ifndef  USE_GPU
inline double sign(double val) {
  return double(int(0 < val) - int(val < 0));
}

// used in PEPS variational update
inline std::complex<double> sign(std::complex<double> val) {
  return std::complex<double>(sign(val.real()), sign(val.imag()));
}

template<typename ElemT, typename QNT>
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

template<typename RandGenerator>
inline void RandSign(QLTEN_Double *number,
                     std::uniform_real_distribution<double> &dist,
                     RandGenerator &g,
                     double tolerance) {
  if (*number > tolerance) {
    *number = dist(g);
  } else if (*number < -tolerance) {
    *number = -dist(g);
  }
}

template<typename RandGenerator>
inline void RandSign(QLTEN_Complex *number,
                     std::uniform_real_distribution<double> &dist,
                     RandGenerator &g,
                     double tolerance) {
  if (number->real() > tolerance) {
    number->real(dist(g));
  } else if (number->real() < -tolerance) {
    number->real(-dist(g));
  }

  if (number->imag() > tolerance) {
    number->imag(dist(g));
  } else if (number->imag() < -tolerance) {
    number->imag(-dist(g));
  }
}

template<typename ElemT, typename QNT>
template<typename RandGenerator>
void BlockSparseDataTensor<ElemT, QNT>::ElementWiseRandSign(std::uniform_real_distribution<double> &dist,
                                                            RandGenerator &g) {
  const ElemT *max_ele = std::max_element(pactual_raw_data_, pactual_raw_data_ + actual_raw_data_size_,
                                          [](const ElemT &a, const ElemT &b) {
                                            return std::abs(a) < std::abs(b);
                                          });
  for (size_t i = 0; i < actual_raw_data_size_; i++) {
    RandSign(pactual_raw_data_ + i, dist, g, std::abs(*max_ele) * 1e-3);
  }
}

inline QLTEN_Double BoundNumber(QLTEN_Double number, double bound) {
  return sign(number) * bound;
}

inline QLTEN_Complex BoundNumber(QLTEN_Complex number, double bound) {
  double abs_val = std::abs(number);
  if (abs_val > bound) {
    return number * bound / abs_val;
  }
  return number;
}

template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::ElementWiseBoundTo(double bound) {
  for (size_t i = 0; i < actual_raw_data_size_; i++) {
    ElemT *elem = pactual_raw_data_ + i;
    if (std::abs(*elem) > bound) {
      *elem = BoundNumber(*elem, bound);
    }
  }
}

template<typename ElemT, typename QNT>
double BlockSparseDataTensor<ElemT, QNT>::GetMaxAbs() const {
  auto max_abs_value_iter = std::max_element(pactual_raw_data_,
                                             pactual_raw_data_ + actual_raw_data_size_,
                                             [](ElemT a, ElemT b) {
                                               return std::abs(a) < std::abs(b);
                                             }
  );
  return std::abs(*max_abs_value_iter);
}

template<typename ElemT, typename QNT>
ElemT BlockSparseDataTensor<ElemT, QNT>::GetFirstNonZeroElement(void) const {
  for (size_t i = 0; i < actual_raw_data_size_; i++) {
    if (pactual_raw_data_[i] != ElemT(0)) {
      return pactual_raw_data_[i];
    }
  }
  return ElemT(0);
}
#else //GPU code
// CUDA Kernel for element-wise inverse (no tolerance)
template<typename ElemT>
__global__
inline void ElementWiseInvKernel(ElemT *data, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] = 1.0 / data[idx];
  }
}

// CUDA Kernel for element-wise inverse (with tolerance)
template<typename ElemT>
__global__ 
inline void ElementWiseInvKernelWithTolerance(ElemT *data, size_t size, double tolerance) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    ElemT val = data[idx];
    data[idx] = (qlten::abs(val) < tolerance) ? ElemT(0) : 1.0 / val;
  }
}

template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::ElementWiseInv() {
  const int threadsPerBlock = 256;
  const int blocks = (actual_raw_data_size_ + threadsPerBlock - 1) / threadsPerBlock;

  // Launch kernel
  ElementWiseInvKernel<<<blocks, threadsPerBlock>>>(pactual_raw_data_, actual_raw_data_size_);

  // Synchronize device
  cudaDeviceSynchronize();
}

template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::ElementWiseInv(double tolerance) {
  const int threadsPerBlock = 256;
  const int blocks = (actual_raw_data_size_ + threadsPerBlock - 1) / threadsPerBlock;

  // Launch kernel with tolerance
  ElementWiseInvKernelWithTolerance<<<blocks, threadsPerBlock>>>(pactual_raw_data_, actual_raw_data_size_, tolerance);

  // Synchronize device
  cudaDeviceSynchronize();
}

template<typename ElemT>
__global__ 
inline void ElementWiseSqrtKernel(ElemT *data, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] = qlten::sqrt(data[idx]);
  }
}

template<typename ElemT>
__global__ 
inline void ElementWiseSquareKernel(ElemT *data, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] = data[idx] * data[idx];
  }
}

template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::ElementWiseSqrt() {
  const int threadsPerBlock = 256;
  const int blocks = (actual_raw_data_size_ + threadsPerBlock - 1) / threadsPerBlock;

  // Launch kernel
  ElementWiseSqrtKernel<<<blocks, threadsPerBlock>>>(pactual_raw_data_, actual_raw_data_size_);

  // Synchronize device
  cudaDeviceSynchronize();
}

template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::ElementWiseSquare() {
  const int threadsPerBlock = 256;
  const int blocks = (actual_raw_data_size_ + threadsPerBlock - 1) / threadsPerBlock;

  // Launch kernel
  ElementWiseSquareKernel<<<blocks, threadsPerBlock>>>(pactual_raw_data_, actual_raw_data_size_);

  // Synchronize device
  cudaDeviceSynchronize();
}

template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataElementWiseMultiply_(
    const size_t data_offset,
    const ElemT *rhs_data,
    const size_t size
) {
  const int threadsPerBlock = 256;
  const int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

  // Launch kernel
  ElementWiseMultiplyKernel<<<blocks, threadsPerBlock>>>(
      pactual_raw_data_ + data_offset, 
      rhs_data, 
      size
  );

  // Synchronize device
  cudaDeviceSynchronize();
}

__global__
inline void ElementWiseSignKernel(double *data, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] = double(int(0 < data[idx]) - int(data[idx] < 0));
  }
}

__global__
inline void ElementWiseSignKernel(cuda::std::complex<double> *data, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx].real((int(0 < data[idx].real()) - int(data[idx].real() < 0)));
    data[idx].imag((int(0 < data[idx].imag()) - int(data[idx].imag() < 0)));
  }
}

template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::ElementWiseSign() {
  const int threadsPerBlock = 256;
  const int blocks = (actual_raw_data_size_ + threadsPerBlock - 1) / threadsPerBlock;

  // Launch kernel
  ElementWiseSignKernel<<<blocks, threadsPerBlock>>>(pactual_raw_data_, actual_raw_data_size_);

  // Synchronize device
  cudaDeviceSynchronize();
}

__global__
inline void ElementWiseBoundKernel(double *data, size_t size, double bound) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    double sign = (int(0 < data[idx]) - int(data[idx] < 0));
    data[idx] = sign * bound;
  }
}

__global__
inline void ElementWiseBoundKernel(cuda::std::complex<double> *data, size_t size, double bound) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    double abs_val = cuda::std::abs(data[idx]);
    if (abs_val > bound) {
      data[idx] = data[idx] * bound / abs_val;
    }
  }
}

template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::ElementWiseBoundTo(double bound) {
  const int threadsPerBlock = 256;
  const int blocks = (actual_raw_data_size_ + threadsPerBlock - 1) / threadsPerBlock;

  // Launch kernel
  ElementWiseBoundKernel<<<blocks, threadsPerBlock>>>(pactual_raw_data_, actual_raw_data_size_, bound);

  // Synchronize device
  cudaDeviceSynchronize();
}

// Kernel to compute the maximum absolute value
template<typename ElemT>
__global__
inline void maxAbsKernel(const ElemT *data, double *result, int size) {
  extern __shared__ double shared_data[]; // Shared memory for reduction

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;

  // Initialize shared memory
  shared_data[tid] = 0.0;
  if (idx < size) {
    shared_data[tid] = qlten::abs(data[idx]);
  }
  __syncthreads();

  // Perform parallel reduction in shared memory
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_data[tid] = std::max(shared_data[tid], shared_data[tid + stride]);
    }
    __syncthreads();
  }

  // Write the result of this block to global memory
  if (tid == 0) {
    result[blockIdx.x] = shared_data[0];
  }
}

// Define a functor to compute the absolute value for complex numbers
struct AbsFunctor {
  __host__ __device__
  double operator()(const cuda::std::complex<double> &z) const {
    return hypot(z.real(), z.imag());
  }

  __host__ __device__
  double operator()(const double &x) const {
    return std::abs(x); // For real numbers
  }
};

template<typename ElemT, typename QNT>
double BlockSparseDataTensor<ElemT, QNT>::GetMaxAbs() const {
  // Wrap raw device pointer into a Thrust device pointer
  thrust::device_ptr<const ElemT> dev_ptr(pactual_raw_data_);

  // Use transform_reduce to compute the maximum absolute value
  return thrust::transform_reduce(
      thrust::device, dev_ptr, dev_ptr + actual_raw_data_size_,
      AbsFunctor(),                      // Transformation (absolute value)
      0.0,                               // Initial value
      thrust::maximum<double>()          // Reduction operation (max)
  );
}

template<typename ElemT, typename QNT>
ElemT BlockSparseDataTensor<ElemT, QNT>::GetFirstNonZeroElement(void) const {
  for (size_t i = 0; i < actual_raw_data_size_; i++) {
    ElemT single_data;
    // cuda mem copy pactual_raw_data_[i]
    cudaMemcpy(&single_data, pactual_raw_data_ + i, sizeof(ElemT), cudaMemcpyDeviceToHost);
    if ( single_data!= ElemT(0)) {
      return single_data;
    }
  }
  return ElemT(0);
}

#endif //USE_GPU
} /* qlten */
#endif /* ifndef QLTEN_QLTENSOR_BLK_SPAR_DATA_TEN_RAW_DATA_OPERATIONS_H */
