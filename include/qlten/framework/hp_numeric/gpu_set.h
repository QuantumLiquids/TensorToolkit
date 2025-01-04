// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author:   Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2024-11-25
*
* Description: API for setting CUDA handles and Error handles
*/

#ifndef QLTEN_FRAMEWORK_HP_NUMERIC_GPU_SET_H
#define QLTEN_FRAMEWORK_HP_NUMERIC_GPU_SET_H

#ifdef USE_GPU
#include <iostream>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cutensor.h>
#include <cuda_runtime.h>

/** A Singleton Class to manage the cublasHandle_t lifecycle
 *
 * usage:
 * cublasHandle_t handle = CublasHandleManager::getHandle();
 *
 */

namespace qlten {
namespace hp_numeric {
class CublasHandleManager {
 public:
  // Get the singleton instance of the manager
  static cublasHandle_t &GetHandle() {
    static CublasHandleManager instance;  // Ensures handle is initialized once
    return instance.handle;
  }

  // Delete copy constructor and assignment operator to prevent copying
  CublasHandleManager(const CublasHandleManager &) = delete;
  CublasHandleManager &operator=(const CublasHandleManager &) = delete;

 private:
  cublasHandle_t handle;

  // Private constructor to initialize the handle
  CublasHandleManager() {
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "Failed to create cuBLAS handle!" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // Destructor to clean up the handle
  ~CublasHandleManager() {
    if (cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "Failed to destroy cuBLAS handle!" << std::endl;
    }
  }
};

class CusolverHandleManager {
 public:
  // Get the singleton instance of the manager
  static cusolverDnHandle_t &GetHandle() {
    static CusolverHandleManager instance;  // Ensures handle is initialized once
    return instance.handle;
  }

  // Delete copy constructor and assignment operator to prevent copying
  CusolverHandleManager(const CusolverHandleManager &) = delete;
  CusolverHandleManager &operator=(const CusolverHandleManager &) = delete;

 private:
  cusolverDnHandle_t handle;

  // Private constructor to initialize the handle
  CusolverHandleManager() {
    if (cusolverDnCreate(&handle) != CUSOLVER_STATUS_SUCCESS) {
      std::cerr << "Failed to create cuSOLVER handle!" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // Destructor to clean up the handle
  ~CusolverHandleManager() {
    if (cusolverDnDestroy(handle) != CUSOLVER_STATUS_SUCCESS) {
      std::cerr << "Failed to destroy cuSOLVER handle!" << std::endl;
    }
  }
};

class CutensorHandleManager {
 public:
  // Get the singleton instance of the manager
  static cutensorHandle_t &GetHandle() {
    static CutensorHandleManager instance;  // Ensures handle is initialized once
    return instance.handle;
  }

  // Delete copy constructor and assignment operator to prevent copying
  CutensorHandleManager(const CutensorHandleManager &) = delete;
  CutensorHandleManager &operator=(const CutensorHandleManager &) = delete;

 private:
  cutensorHandle_t handle;

  // Private constructor to initialize the handle
  CutensorHandleManager() {
    if (cutensorCreate(&handle) != CUTENSOR_STATUS_SUCCESS) {
      std::cerr << "Failed to create cuTENSOR handle!" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // Destructor to clean up the handle
  ~CutensorHandleManager() {
    if (cutensorDestroy(handle) != CUTENSOR_STATUS_SUCCESS) {
      std::cerr << "Failed to destroy cuTENSOR handle!" << std::endl;
    }
  }
};
}//hp_numeric

// Handles cuTENSOR errors
inline void HandleCutensorError(cutensorStatus_t status, int line) {
  if (status != CUTENSOR_STATUS_SUCCESS) {
    std::cerr << "Error at line " << line << ": "
              << cutensorGetErrorString(status) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

inline const char *cusolverGetErrorString(cusolverStatus_t status) {
  switch (status) {
    case CUSOLVER_STATUS_SUCCESS: return "CUSOLVER_STATUS_SUCCESS";
    case CUSOLVER_STATUS_NOT_INITIALIZED: return "CUSOLVER_STATUS_NOT_INITIALIZED";
    case CUSOLVER_STATUS_ALLOC_FAILED: return "CUSOLVER_STATUS_ALLOC_FAILED";
    case CUSOLVER_STATUS_INVALID_VALUE: return "CUSOLVER_STATUS_INVALID_VALUE";
    case CUSOLVER_STATUS_ARCH_MISMATCH: return "CUSOLVER_STATUS_ARCH_MISMATCH";
    case CUSOLVER_STATUS_EXECUTION_FAILED: return "CUSOLVER_STATUS_EXECUTION_FAILED";
    case CUSOLVER_STATUS_INTERNAL_ERROR: return "CUSOLVER_STATUS_INTERNAL_ERROR";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case CUSOLVER_STATUS_NOT_SUPPORTED: return "CUSOLVER_STATUS_NOT_SUPPORTED";
    case CUSOLVER_STATUS_ZERO_PIVOT: return "CUSOLVER_STATUS_ZERO_PIVOT";
    case CUSOLVER_STATUS_INVALID_LICENSE: return "CUSOLVER_STATUS_INVALID_LICENSE";
    default: return "Unknown cuSOLVER error";
  }
}

inline void HandleCuSolverError(cusolverStatus_t status, int line) {
  if (status != CUSOLVER_STATUS_SUCCESS) {
    std::cerr << "Error at line " << line << ": "
              << cusolverGetErrorString(status) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

template<typename T>
const char *cublasGetErrorString(T status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    default: return "Unknown cuBLAS error";
  }
}

inline void HandleCuBlasError(cublasStatus_t status, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBLAS Error at line " << line << ": "
              << cublasGetErrorString(status) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

// Utility to simplify calls with automatic line number capture
#define HANDLE_CUTENSOR_ERROR(x) HandleCutensorError(x, __LINE__)
#define HANDLE_CUSOLVER_ERROR(x) HandleCuSolverError(x, __LINE__)
#define HANDLE_CUBLAS_ERROR(x) HandleCuBlasError(x, __LINE__)
}//qlten

#endif

#endif //QLTEN_FRAMEWORK_HP_NUMERIC_GPU_SET_H
