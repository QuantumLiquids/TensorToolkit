// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-09-11 16:20
*
* Description: QuantumLiquids/tensor project. Inline utility functions used by template headers.
*/
#ifndef QLTEN_UTILITY_UTILS_INL_H
#define QLTEN_UTILITY_UTILS_INL_H

#include <vector>
#include <numeric>
#include <complex>
#include <cmath>        // abs
#include <algorithm>    // swap
#include <cassert>      // assert

#ifdef USE_GPU
#include <cuda.h>
#include <curand_kernel.h>               // cuRAND
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>     // thrust::device
#include <thrust/equal.h>                // thrust::equal
#include <thrust/functional.h>
#endif

#include "qlten/framework/value_t.h"    // CoorsT, ShapeT
#include "qlten/framework/consts.h"
#include "qlten/framework/mem_ops.h"    // Memcpy
#include "qlten/utility/random.h"
#ifdef Release
#define NDEBUG
#endif

namespace qlten {

//// Algorithms
// Inplace reorder a vector.
template<typename T>
void InplaceReorder(std::vector<T> &v, const std::vector<size_t> &order) {
  std::vector<size_t> indices(order);
  for (size_t i = 0; i < indices.size(); ++i) {
    auto current = i;
    while (i != indices[current]) {
      auto next = indices[current];
      std::swap(v[current], v[next]);
      indices[current] = current;
      current = next;
    }
    indices[current] = current;
  }
}

// parity = false = 0 means even sector; parity =true = 1 means odd sector
template<typename T>
int FermionicInplaceReorder(std::vector<T> &v, const std::vector<size_t> &order,
                            std::vector<bool> &parities) {
  int exchange_sign = 0;
  std::vector<size_t> indices(order);
  for (size_t i = 0; i < indices.size(); ++i) {
    auto current = i;
    while (i != indices[current]) {
      auto next = indices[current];
      std::swap(v[current], v[next]);
      int sign_between = 0;
      for (size_t j = std::min(current, next) + 1; j < std::max(current, next); j++) {
        sign_between += parities[j];
      }
      exchange_sign += sign_between * (parities[current] + parities[next]) + (parities[current] * parities[next]);

      std::vector<bool>::swap(parities[current], parities[next]);
      indices[current] = current;
      current = next;
    }
    indices[current] = current;
  }
  return (exchange_sign & 1) ? -1 : 1;
}

inline std::vector<int> Reorder(const std::vector<size_t> &v1, const std::vector<int> &order) {
  size_t data_size = v1.size();
  std::vector<int> v2;
  v2.reserve(data_size);
  for (size_t i = 0; i < data_size; i++) {
    v2.push_back(v1[order[i]]);
  }
  return v2;
}

// Calculate Cartesian product.
template<typename T>
T CalcCartProd(T v) {
  T s = {{}};
  for (const auto &u : v) {
    T r;
    r.reserve(s.size() * u.size());
    for (const auto &x : s) {
      for (const auto y : u) {
        r.push_back(x);
        r.back().push_back(y);
      }
    }
    s = std::move(r);
  }
  return s;
}

/*
 * Generate all the coordinates in the following order:
 *    (0, 0, 0,....., 0),
 *    (0, 0, 0,....., 1),
 *    (0, 0, 0,....., 2),
 *    ......
 *    (shape[0], shape[1],.....,shape[n-1]-1),
 *    (shape[0], shape[1],.....,shape[n-1])
 */
inline std::vector<CoorsT> GenAllCoors(const ShapeT &shape) {
  std::vector<CoorsT> each_coors(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    each_coors[i].reserve(shape[i]);
    for (size_t j = 0; j < shape[i]; ++j) {
      each_coors[i].push_back(j);
    }
  }
  return CalcCartProd(each_coors);
}

inline std::vector<size_t> CalcMultiDimDataOffsets(const ShapeT &shape) {
  auto ndim = shape.size();
  if (ndim == 0) { return {}; }
  std::vector<size_t> offsets(ndim);
  offsets[ndim - 1] = 1;
  for (int i = ndim - 2; i >= 0; --i) {
    offsets[i] = offsets[i + 1] * shape[i + 1];
  }
  return offsets;
}

// Calculate offset for the effective one dimension array.
inline size_t CalcEffOneDimArrayOffset(
    const CoorsT &coors,
    const std::vector<size_t> &data_offsets
) {
  assert(coors.size() == data_offsets.size());
  size_t ndim = coors.size();
  size_t offset = 0;
  for (size_t i = 0; i < ndim; ++i) {
    offset += coors[i] * data_offsets[i];
  }
  return offset;
}

// Multiply selected elements in a vector
template<typename T>
inline T VecMultiSelectElemts(
    const std::vector<T> &v,
    const std::vector<size_t> elem_idxes
) {
  auto selected_elem_num = elem_idxes.size();
  if (selected_elem_num == 0) {
    return T(1);
  }
  T res;
  if (selected_elem_num == 1) {
    return v[elem_idxes[0]];
  } else {
    res = v[elem_idxes[0]];
  }
  for (size_t i = 1; i < selected_elem_num; ++i) {
    res *= v[elem_idxes[i]];
  }
  return res;
}

// Add two coordinates together
inline CoorsT CoorsAdd(const CoorsT &coors1, const CoorsT &coors2) {
  assert(coors1.size() == coors2.size());
  CoorsT res;
  res.reserve(coors1.size());
  for (size_t i = 0; i < coors1.size(); ++i) {
    res.push_back(coors1[i] + coors2[i]);
  }
  return res;
}

#ifndef USE_GPU
//// Equivalence check
inline bool DoubleEq(const QLTEN_Double a, const QLTEN_Double b) {
  if (qlten::abs(a - b) < kDoubleEpsilon) {
    return true;
  } else {
    return false;
  }
}

inline bool ComplexEq(const QLTEN_Complex a, const QLTEN_Complex b) {
  return qlten::abs(a - b) < kDoubleEpsilon;
}

inline bool ArrayEq(
    const QLTEN_Double *parray1, const size_t size1,
    const QLTEN_Double *parray2, const size_t size2) {
  if (size1 != size2) {
    return false;
  }
  for (size_t i = 0; i < size1; ++i) {
    if (!DoubleEq(parray1[i], parray2[i])) {
      return false;
    }
  }
  return true;
}

inline bool ArrayEq(
    const QLTEN_Complex *parray1, const size_t size1,
    const QLTEN_Complex *parray2, const size_t size2) {
  if (size1 != size2) {
    return false;
  }
  for (size_t i = 0; i < size1; ++i) {
    if (!ComplexEq(parray1[i], parray2[i])) {
      return false;
    }
  }
  return true;
}
#else

// Define custom comparator for floating-point equality
struct DoubleEqual {
  const double epsilon;
  DoubleEqual(double eps) : epsilon(eps) {}

  __device__
  bool operator()(const double &a, const double &b) const {
    return qlten::abs(a - b) < epsilon;  // Floating-point comparison
  }
};

struct ComplexEqual {
  const double epsilon;
  ComplexEqual(double eps) : epsilon(eps) {}

  __device__
  bool operator()(const QLTEN_Complex &a, const QLTEN_Complex &b) const {
    return qlten::abs(a - b) < epsilon;  // Floating-point comparison
  }
};

inline bool ArrayEq(const double *d_array1, const double *d_array2, size_t size) {
  thrust::device_ptr<const double> dev_ptr1(d_array1);
  thrust::device_ptr<const double> dev_ptr2(d_array2);
  return thrust::equal(thrust::device, dev_ptr1, dev_ptr1 + size, dev_ptr2, DoubleEqual(kDoubleEpsilon));
}

inline bool ArrayEq(const QLTEN_Complex *d_array1, const QLTEN_Complex *d_array2, size_t size) {
  return thrust::equal(thrust::device, d_array1, d_array1 + size, d_array2, ComplexEqual(kDoubleEpsilon));
}

#endif//USE_GPU

//// Random

inline QLTEN_Double drand(void) {
  return static_cast<QLTEN_Double>(qlten::Uniform01());
}

inline QLTEN_Complex zrand(void) {
  return QLTEN_Complex(drand(), drand());
}

inline void Rand(QLTEN_Double &d) {
  d = drand();
}

inline void Rand(QLTEN_Complex &z) {
  z = zrand();
}
#ifdef USE_GPU
__global__
inline void RandomKernel(QLTEN_Double *data, size_t size, unsigned long long seed) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    curandState state;
    curand_init(seed, idx, 0, &state);  // Seed, sequence number, offset
    data[idx] = curand_uniform(&state);  // Generate [0, 1] uniform random numbers
  }
}

__global__
inline void RandomKernel(QLTEN_Complex *data, size_t size, unsigned long long seed) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    curandState state;
    curand_init(seed, idx, 0, &state);  // Seed, sequence number, offset
    data[idx].real(curand_uniform(&state));
    data[idx].imag(curand_uniform(&state));
  }
}

__global__
inline void FillKernel(QLTEN_Double *data, size_t size, QLTEN_Double value) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] = value;
  }
}

__global__
inline void FillKernel(QLTEN_Complex *data, size_t size, QLTEN_Complex value) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] = value;
  }
}

__global__
inline void ElementWiseMultiplyKernel(QLTEN_Double *data, const QLTEN_Double *rhs_data, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] *= rhs_data[idx];
  }
}

__global__
inline void ElementWiseMultiplyKernel(QLTEN_Complex *data, const QLTEN_Complex *rhs_data, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] *= rhs_data[idx];
  }
}
#endif

template<typename ElemType>
inline ElemType RandT() {
  ElemType val;
  Rand(val);
  return val;
}

//// Math
inline QLTEN_Double CalcScalarNorm2(QLTEN_Double d) {
  return d * d;
}

inline QLTEN_Double CalcScalarNorm2(QLTEN_Complex z) {
#ifndef USE_GPU
  return std::norm(z);
#else
  return cuda::std::norm(z);
#endif
}

inline QLTEN_Double CalcScalarNorm(QLTEN_Double d) {
  return std::abs(d);
}

inline QLTEN_Double CalcScalarNorm(QLTEN_Complex z) {
  return std::sqrt(CalcScalarNorm2(z));
}

inline QLTEN_Double CalcConj(QLTEN_Double d) {
  return d;
}

inline QLTEN_Complex CalcConj(QLTEN_Complex z) {
#ifndef USE_GPU
  return std::conj(z);
#else
  return cuda::std::conj(z);
#endif
}

#ifdef USE_GPU
__global__
inline void ConjugateKernel(QLTEN_Complex *data, size_t size) {
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    data[idx] = cuda::std::conj(data[idx]); // Call your conjugate function
  }
}

#endif
template<typename TenElemType>
inline std::vector<TenElemType> SquareVec(const std::vector<TenElemType> &v) {
  std::vector<TenElemType> res(v.size());
  for (size_t i = 0; i < v.size(); ++i) { res[i] = std::pow(v[i], 2.0); }
  return res;
}

template<typename TenElemType>
inline std::vector<TenElemType> NormVec(const std::vector<TenElemType> &v) {
  TenElemType sum = std::accumulate(v.begin(), v.end(), 0.0);
  std::vector<TenElemType> res(v.size());
  for (size_t i = 0; i < v.size(); ++i) { res[i] = v[i] / sum; }
  return res;
}

//// Matrix operation
template<typename T>
inline std::vector<T> SliceFromBegin(const std::vector<T> &v, size_t to) {
  auto first = v.cbegin();
  return std::vector<T>(first, first + to);
}

template<typename T>
inline std::vector<T> SliceFromEnd(const std::vector<T> &v, size_t to) {
  auto last = v.cend();
  return std::vector<T>(last - to, last);
}

template<typename ElemT>
void SubMatMemCpy(
    const size_t m, const size_t n,
    const size_t row_offset, const size_t col_offset,
    const size_t sub_m, const size_t sub_n,
    const ElemT *sub_mem_begin,
    ElemT *mem_begin
) {
  size_t offset = row_offset * n + col_offset;
#ifndef USE_GPU
  size_t sub_offset = 0;
  for (size_t row_idx = row_offset; row_idx < row_offset + sub_m; ++row_idx) {
    qlten::QLMemcpy(
        mem_begin + offset,
        sub_mem_begin + sub_offset,
        sub_n * sizeof(ElemT)
    );
    offset += n;
    sub_offset += sub_n;
  }
#else //USE_GPU
  cudaMemcpy2D(
        mem_begin + offset, 
	n * sizeof(ElemT),                        // Destination pointer and pitch
        sub_mem_begin, sub_n * sizeof(ElemT),     // Source pointer and pitch
        sub_n * sizeof(ElemT), 
	sub_m, 
	cudaMemcpyDeviceToDevice);      // Width (bytes) and height (rows)
#endif
}


//template <typename MatElemType>
//inline MatElemType *MatGetRows(
//const MatElemType *mat, const long &rows, const long &cols,
//const long &from, const long &num_rows) {
//auto new_size = num_rows*cols;
//auto new_mat = new MatElemType [new_size];
//std::memcpy(new_mat, mat+(from*cols), new_size*sizeof(MatElemType));
//return new_mat;
//}


//template <typename MatElemType>
//inline void MatGetRows(
//const MatElemType *mat, const long &rows, const long &cols,
//const long &from, const long &num_rows,
//MatElemType *new_mat) {
//auto new_size = num_rows*cols;
//std::memcpy(new_mat, mat+(from*cols), new_size*sizeof(MatElemType));
//}


//template <typename MatElemType>
//inline void MatGetCols(
//const MatElemType *mat, const long rows, const long cols,
//const long from, const long num_cols,
//MatElemType *new_mat) {
//long offset = from;
//long new_offset = 0;
//for (long i = 0; i < rows; ++i) {
//std::memcpy(new_mat+new_offset, mat+offset, num_cols*sizeof(MatElemType));
//offset += cols;
//new_offset += num_cols;
//}
//}


//template <typename MatElemType>
//inline MatElemType *MatGetCols(
//const MatElemType *mat, const long rows, const long cols,
//const long from, const long num_cols) {
//auto new_size = num_cols * rows;
//auto new_mat = new MatElemType [new_size];
//MatGetCols(mat, rows, cols, from, num_cols, new_mat);
//return new_mat;
//}


//inline void GenDiagMat(
//const double *diag_v, const long &diag_v_dim, double *full_mat) {
//for (long i = 0; i < diag_v_dim; ++i) {
//*(full_mat + (i*diag_v_dim + i)) = diag_v[i];
//}
//}


//// Free the resources of a QLTensor.
//template <typename TenElemType>
//inline void QLTenFree(QLTensor<TenElemType> *pt) {
//for (auto &pblk : pt->blocks()) { delete pblk; }
//}


} /* qlten */
#endif /* ifndef QLTEN_UTILITY_UTILS_INL_H */
