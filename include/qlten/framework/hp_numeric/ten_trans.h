// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-21 15:27
*
* Description: High performance tensor transpose function based on HPTT library.
*/

/**
@file ten_trans.h
@brief High performance tensor transpose function based on HPTT library.
*/
#ifndef QLTEN_FRAMEWORK_HP_NUMERIC_TEN_TRANS_H
#define QLTEN_FRAMEWORK_HP_NUMERIC_TEN_TRANS_H

#include <vector>     // vector

#ifndef USE_GPU
#ifndef PLAIN_TRANSPOSE
#include "hptt.h"
#endif
#else
#ifndef PLAIN_TRANSPOSE
#include "gpu_set.h"  // CutensorHandleManager
#endif
#endif
namespace qlten {

/// High performance numerical functions.
namespace hp_numeric {

#ifndef USE_GPU //USE CPU TRANPOSE
inline int tensor_transpose_num_threads = kTensorOperationDefaultNumThreads;

#ifndef PLAIN_TRANSPOSE

template<typename ElemT>
void TensorTranspose(
    const std::vector<size_t> &transed_order,
    const size_t ten_rank,
    ElemT *original_data,
    const std::vector<size_t> &original_shape,
    ElemT *transed_data,
    const std::vector<size_t> &transed_shape,
    const double alpha  //scale factor, for boson tensor alpha = 1; for Fermion tensor additional minus sign may require
) {
  const int dim = ten_rank;
  int perm[dim];
  for (int i = 0; i < dim; ++i) { perm[i] = transed_order[i]; }
  int sizeA[dim];
  for (int i = 0; i < dim; ++i) { sizeA[i] = original_shape[i]; }
  int outerSizeB[dim];
  for (int i = 0; i < dim; ++i) { outerSizeB[i] = transed_shape[i]; }
  auto tentrans_plan = hptt::create_plan(perm, dim,
                                         alpha, original_data, sizeA, sizeA,
                                         0.0, transed_data, outerSizeB,
                                         hptt::ESTIMATE,
                                         tensor_transpose_num_threads, {},
                                         true
  );
  tentrans_plan->execute();
}

template<typename ElemT>
void TensorTranspose(
    const std::vector<int> &transed_order,
    const size_t ten_rank,
    ElemT *original_data,
    const std::vector<size_t> &original_shape,
    ElemT *transed_data,
    const std::vector<int> &transed_shape,
    const double alpha //scale factor, for boson tensor alpha = 1; for Fermion tensor additional minus sign may require
) {
  const int dim = ten_rank;
  int sizeA[dim];
  for (int i = 0; i < dim; ++i) { sizeA[i] = original_shape[i]; }
  auto tentrans_plan = hptt::create_plan(transed_order.data(), dim,
                                         alpha, original_data, sizeA, sizeA,
                                         0.0, transed_data, transed_shape.data(),
                                         hptt::ESTIMATE,
                                         tensor_transpose_num_threads, {},
                                         true
  );
  tentrans_plan->execute();
}

#else
template<typename ElemT, typename IntType>
void TensorTranspose(
    const std::vector<IntType> &transed_order,
    const size_t ten_rank,
    ElemT *original_data,
    const std::vector<size_t> &original_shape,
    ElemT *transed_data,
    const std::vector<IntType> &transed_shape,
    const double scale
) {
  // Calculate the strides for each dimension in the original tensor
  std::vector<size_t> original_strides(ten_rank, 1);
  size_t stride = 1;
  for (int i = ten_rank - 1; i >= 0; --i) {
    original_strides[i] = stride;
    stride *= original_shape[i];
  }

  // Calculate the total number of elements in the original tensor
  size_t total_elements = stride;

  // Calculate the strides for each dimension in the transposed tensor
  std::vector<size_t> transed_strides(ten_rank, 1);
  stride = 1;
  for (int i = ten_rank - 1; i >= 0; --i) {
    transed_strides[i] = stride;
    stride *= transed_shape[i];
  }

  // Loop over each element in the original tensor
  std::vector<size_t> original_coords(ten_rank, 0);
  std::vector<size_t> transed_coords(ten_rank);
  for (size_t index = 0; index < total_elements; ++index) {
    // Calculate the coordinates of the current element in the original tensor

    size_t remaining_index = index;

    for (int i = 0; i < ten_rank; ++i) {
      original_coords[i] = remaining_index / original_strides[i];
      remaining_index %= original_strides[i];
    }

    // Permute the coordinates according to the transposition order

    for (size_t i = 0; i < ten_rank; ++i) {
      transed_coords[i] = original_coords[transed_order[i]];
    }

    // Calculate the index of the corresponding element in the transposed tensor
    size_t transed_index = 0;
    for (size_t i = 0; i < ten_rank; ++i) {
      transed_index += transed_coords[i] * transed_strides[i];
    }

    // Copy the data to the transposed position
    transed_data[transed_index] = scale * original_data[index];
  }
}

#endif

#else // USE GPU case

// Utility Functions
template<typename ElemT>
cutensorDataType_t GetCuTensorDataType();

template<>
cutensorDataType_t GetCuTensorDataType<QLTEN_Double>() { return CUTENSOR_R_64F; }
template<>
cutensorDataType_t GetCuTensorDataType<QLTEN_Complex>() { return CUTENSOR_C_64F; }

template<typename ElemT>
ElemT TypedAlpha(double);
template<>
QLTEN_Double TypedAlpha<QLTEN_Double>(double alpha) { return alpha; }
template<>
QLTEN_Complex TypedAlpha<QLTEN_Complex>(double alpha) { return {alpha, 0}; }

template<typename ElemT, typename IntType>
void TensorTranspose(
    const std::vector<IntType> &transed_order,
    const size_t ten_rank,
    const ElemT *original_data,
    const std::vector<size_t> &original_shape,
    ElemT *transed_data,
    const std::vector<IntType> &transed_shape,
    const double alpha //scale factor, for boson tensor alpha = 1; for Fermion tensor additional minus sign may require
) {
  // Convert size_t to int64_t for cuTENSOR
  std::vector<int64_t> shapeA(ten_rank);
  std::vector<int64_t> strideA(ten_rank);
  std::vector<int64_t> shapeB(ten_rank);
  std::vector<int64_t> strideB(ten_rank);

  // Compute strides for input and output tensors
  int64_t stride = 1;
  for (int i = int(ten_rank) - 1; i >= 0; --i) {
    shapeA[i] = original_shape[i];
    strideA[i] = stride;
    stride *= original_shape[i];
  }

  stride = 1;
  for (int i = int(ten_rank) - 1; i >= 0; --i) {
    shapeB[i] = transed_shape[i];
    strideB[i] = stride;
    stride *= transed_shape[i];
  }

  // Initialize tensor descriptors
  cutensorTensorDescriptor_t descA, descB;
  auto handle = CutensorHandleManager::GetHandle();
  int32_t perm[ten_rank];
  int32_t no_perm[ten_rank];
  for (size_t i = 0; i < ten_rank; ++i) {
    perm[i] = transed_order[i];
    no_perm[i] = i;
  }

  /**********************
   * Create Tensor Descriptors
   **********************/
  cutensorCreateTensorDescriptor(handle, &descA,
                                 ten_rank, shapeA.data(), strideA.data(),
                                 GetCuTensorDataType<ElemT>(), sizeof(ElemT));
  cutensorCreateTensorDescriptor(handle, &descB,
                                 ten_rank, shapeB.data(), strideB.data(),
                                 GetCuTensorDataType<ElemT>(), sizeof(ElemT));

  /*******************************
   * Create Permutation Descriptor
   *******************************/

  // Create permutation operation descriptor
  cutensorOperationDescriptor_t op_desc;

  cutensorStatus_t status = cutensorCreatePermutation(
      handle, &op_desc,
      descA, no_perm, CUTENSOR_OP_IDENTITY, // Identity operation
      descB, perm, // Permuted order for output tensor
      CUTENSOR_COMPUTE_DESC_64F);
  HANDLE_CUTENSOR_ERROR(status);

  /*****************************
   * Optional (but recommended): ensure that the scalar type is correct.
   *****************************/

  cutensorDataType_t scalarType;
  HANDLE_CUTENSOR_ERROR(cutensorOperationDescriptorGetAttribute(handle, op_desc,
                                                                CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                                                                (void *) &scalarType,
                                                                sizeof(scalarType)));

  assert(scalarType == GetCuTensorDataType<ElemT>());
  /**************************
   * Set the algorithm to use
   ***************************/
  const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT_PATIENT; //CUTENSOR_ALGO_DEFAULT;
  cutensorPlanPreference_t planPref;
  HANDLE_CUTENSOR_ERROR(cutensorCreatePlanPreference(handle,
                                                     &planPref,
                                                     algo,
                                                     CUTENSOR_JIT_MODE_DEFAULT));
  // Create the execution plan
  cutensorPlan_t perm_plan;
  status = cutensorCreatePlan(
      handle, &perm_plan, op_desc, planPref, 0); // No preference and no workspace limit
  HANDLE_CUTENSOR_ERROR(status);

  // Perform the permutation operation
  ElemT typed_alpha = TypedAlpha<ElemT>(alpha);
  status = cutensorPermute(
      handle, perm_plan,
      &typed_alpha, original_data, transed_data,
      nullptr); // Using default CUDA stream
  HANDLE_CUTENSOR_ERROR(status);

  // Clean up resources
  cutensorDestroyPlan(perm_plan);
  cutensorDestroyOperationDescriptor(op_desc);
  cutensorDestroyPlanPreference(planPref);
  cutensorDestroyTensorDescriptor(descA);
  cutensorDestroyTensorDescriptor(descB);
}

#endif//USE_GPU

} /* hp_numeric */
} /* qlten */
#endif /* ifndef QLTEN_FRAMEWORK_HP_NUMERIC_TEN_TRANS_H */
