// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2026-05-01
*
* Description: QuantumLiquids/tensor project. Diagonal tensor product helpers.
*/

/**
@file ten_diagonal_tensor_product.h
@brief Accumulate the diagonal of a tensor product into an existing tensor.
*/
#ifndef QLTEN_TENSOR_MANIPULATION_TEN_DIAGONAL_TENSOR_PRODUCT_H
#define QLTEN_TENSOR_MANIPULATION_TEN_DIAGONAL_TENSOR_PRODUCT_H

#include <cassert>      // assert
#include <cstddef>      // size_t
#include <sstream>      // ostringstream
#include <stdexcept>    // invalid_argument, runtime_error
#include <string>       // string
#include <utility>      // pair
#include <vector>       // vector

#include "qlten/framework/value_t.h"      // QLTEN_Double, QLTEN_Complex
#ifndef USE_GPU
#include "qlten/framework/hp_numeric/backend_selector.h"    // cblas_*gemm
#endif
#include "qlten/qltensor/qltensor.h"      // QLTensor
#include "qlten/utility/utils_inl.h"      // CalcEffOneDimArrayOffset, CalcMultiDimDataOffsets

#ifdef Release
#define NDEBUG
#endif

namespace qlten {
namespace detail {

constexpr size_t kDiagonalTensorProductBlasThreshold = 64;

inline std::string CoorsToString(const CoorsT &coors) {
  std::ostringstream oss;
  oss << "{";
  for (size_t i = 0; i < coors.size(); ++i) {
    if (i != 0) {
      oss << ", ";
    }
    oss << coors[i];
  }
  oss << "}";
  return oss.str();
}

template<typename QNT>
bool IndexSpaceEqualIgnoreDirection(
    const Index<QNT> &lhs,
    const Index<QNT> &rhs
) {
  if (lhs.dim() != rhs.dim() ||
      lhs.GetQNSctNum() != rhs.GetQNSctNum()) {
    return false;
  }
  for (size_t i = 0; i < lhs.GetQNSctNum(); ++i) {
    if (lhs.GetQNSct(i) != rhs.GetQNSct(i)) {
      return false;
    }
  }
  return true;
}

template<typename QNT>
void RequireSameIndexSpace(
    const Index<QNT> &lhs,
    const Index<QNT> &rhs,
    const char *message
) {
  if (!IndexSpaceEqualIgnoreDirection(lhs, rhs)) {
    throw std::invalid_argument(
        std::string("DiagonalTensorProductAccumulate: ") + message
    );
  }
}

template<typename QNT>
void RequireSameIndexSpaceFor(
    const Index<QNT> &lhs,
    const Index<QNT> &rhs,
    const char *context,
    const std::string &message
) {
  if (!IndexSpaceEqualIgnoreDirection(lhs, rhs)) {
    throw std::invalid_argument(std::string(context) + ": " + message);
  }
}

#ifndef USE_GPU
inline size_t Rank4BlkCoorsToBlkIdx(
    const ShapeT &blk_shape,
    const size_t coor0,
    const size_t coor1,
    const size_t coor2,
    const size_t coor3
) {
  return ((coor0 * blk_shape[1] + coor1) * blk_shape[2] + coor2) *
         blk_shape[3] + coor3;
}

inline void BlasScale(
    QLTEN_Double *data,
    const size_t size,
    const QLTEN_Double beta
) {
  cblas_dscal(size, beta, data, 1);
}

inline void BlasScale(
    QLTEN_Float *data,
    const size_t size,
    const QLTEN_Float beta
) {
  cblas_sscal(size, beta, data, 1);
}

inline void BlasScale(
    QLTEN_Complex *data,
    const size_t size,
    const QLTEN_Complex beta
) {
  cblas_zscal(size, &beta, data, 1);
}

inline void BlasScale(
    QLTEN_ComplexFloat *data,
    const size_t size,
    const QLTEN_ComplexFloat beta
) {
  cblas_cscal(size, &beta, data, 1);
}

inline void GerUpdate(
    const QLTEN_Double alpha,
    const QLTEN_Double *left_diag,
    const size_t left_inc,
    const QLTEN_Double *right_diag,
    const size_t right_inc,
    const size_t rows,
    const size_t cols,
    QLTEN_Double *out_data
) {
  cblas_dger(
      CblasRowMajor, rows, cols,
      alpha,
      left_diag, left_inc,
      right_diag, right_inc,
      out_data, cols
  );
}

inline void GerUpdate(
    const QLTEN_Float alpha,
    const QLTEN_Float *left_diag,
    const size_t left_inc,
    const QLTEN_Float *right_diag,
    const size_t right_inc,
    const size_t rows,
    const size_t cols,
    QLTEN_Float *out_data
) {
  cblas_sger(
      CblasRowMajor, rows, cols,
      alpha,
      left_diag, left_inc,
      right_diag, right_inc,
      out_data, cols
  );
}

inline void GerUpdate(
    const QLTEN_Complex alpha,
    const QLTEN_Complex *left_diag,
    const size_t left_inc,
    const QLTEN_Complex *right_diag,
    const size_t right_inc,
    const size_t rows,
    const size_t cols,
    QLTEN_Complex *out_data
) {
  cblas_zgeru(
      CblasRowMajor, rows, cols,
      &alpha,
      left_diag, left_inc,
      right_diag, right_inc,
      out_data, cols
  );
}

inline void GerUpdate(
    const QLTEN_ComplexFloat alpha,
    const QLTEN_ComplexFloat *left_diag,
    const size_t left_inc,
    const QLTEN_ComplexFloat *right_diag,
    const size_t right_inc,
    const size_t rows,
    const size_t cols,
    QLTEN_ComplexFloat *out_data
) {
  cblas_cgeru(
      CblasRowMajor, rows, cols,
      &alpha,
      left_diag, left_inc,
      right_diag, right_inc,
      out_data, cols
  );
}
#endif

template<typename ElemT>
void ScaleDataBlock(
    ElemT *data,
    const size_t size,
    const ElemT beta
) {
  if (beta == ElemT(1)) {
    return;
  }
  if (beta == ElemT(0)) {
    for (size_t i = 0; i < size; ++i) {
      data[i] = ElemT(0);
    }
    return;
  }
#ifndef USE_GPU
  BlasScale(data, size, beta);
#else
  for (size_t i = 0; i < size; ++i) {
    data[i] *= beta;
  }
#endif
}

template<typename ElemT>
ElemT DiagonalElementFromRank4Block(
    const ElemT *data,
    const ShapeT &shape,
    const size_t flat_diag_idx
) {
  const size_t coor0 = flat_diag_idx / shape[1];
  const size_t coor1 = flat_diag_idx % shape[1];
  return data[((coor0 * shape[1] + coor1) * shape[2] + coor0) *
              shape[3] + coor1];
}

inline std::string AxisPairToString(
    const std::pair<size_t, size_t> &axis_pair
) {
  std::ostringstream oss;
  oss << "(" << axis_pair.first << ", " << axis_pair.second << ")";
  return oss.str();
}

template<typename ElemT, typename QNT>
void ValidateDiagonalAxisPairs(
    const QLTensor<ElemT, QNT> &tensor,
    const std::vector<std::pair<size_t, size_t>> &axis_pairs,
    const char *context
) {
  if (tensor.IsDefault()) {
    throw std::invalid_argument(
        std::string(context) + " requires a non-default tensor."
    );
  }
  if (axis_pairs.empty() || tensor.Rank() != 2 * axis_pairs.size()) {
    throw std::invalid_argument(
        std::string(context) +
        " requires disjoint axis pairs covering every tensor axis."
    );
  }

  std::vector<bool> axis_used(tensor.Rank(), false);
  for (const auto &axis_pair : axis_pairs) {
    const size_t kept_axis = axis_pair.first;
    const size_t diagonal_axis = axis_pair.second;
    if (kept_axis >= tensor.Rank() || diagonal_axis >= tensor.Rank()) {
      throw std::invalid_argument(
          std::string(context) + ": axis pair " +
          AxisPairToString(axis_pair) + " is out of range."
      );
    }
    if (kept_axis == diagonal_axis) {
      throw std::invalid_argument(
          std::string(context) + ": axis pair " +
          AxisPairToString(axis_pair) + " repeats the same axis."
      );
    }
    if (axis_used[kept_axis] || axis_used[diagonal_axis]) {
      throw std::invalid_argument(
          std::string(context) + ": axis pair " +
          AxisPairToString(axis_pair) + " overlaps a previous axis pair."
      );
    }
    axis_used[kept_axis] = true;
    axis_used[diagonal_axis] = true;
    RequireSameIndexSpaceFor(
        tensor.GetIndex(kept_axis),
        tensor.GetIndex(diagonal_axis),
        context,
        "axis pair " + AxisPairToString(axis_pair) +
        " must describe the same diagonal space."
    );
  }
}

template<typename ElemT, typename QNT>
IndexVec<QNT> MakeDiagonalIndexes(
    const QLTensor<ElemT, QNT> &tensor,
    const std::vector<std::pair<size_t, size_t>> &axis_pairs
) {
  IndexVec<QNT> indexes;
  indexes.reserve(axis_pairs.size());
  for (const auto &axis_pair : axis_pairs) {
    indexes.push_back(tensor.GetIndex(axis_pair.first));
  }
  return indexes;
}

template<typename QNT>
bool IsDiagonalBlock(
    const DataBlk<QNT> &data_blk,
    const std::vector<std::pair<size_t, size_t>> &axis_pairs
) {
  for (const auto &axis_pair : axis_pairs) {
    if (data_blk.blk_coors[axis_pair.first] !=
        data_blk.blk_coors[axis_pair.second]) {
      return false;
    }
  }
  return true;
}

template<typename QNT>
CoorsT MakeDiagonalBlockCoors(
    const DataBlk<QNT> &data_blk,
    const std::vector<std::pair<size_t, size_t>> &axis_pairs
) {
  CoorsT blk_coors;
  blk_coors.reserve(axis_pairs.size());
  for (const auto &axis_pair : axis_pairs) {
    blk_coors.push_back(data_blk.blk_coors[axis_pair.first]);
  }
  return blk_coors;
}

template<typename QNT>
void ValidateExtractedDiagonalBlockShape(
    const DataBlk<QNT> &src_blk,
    const DataBlk<QNT> &out_blk,
    const std::vector<std::pair<size_t, size_t>> &axis_pairs,
    const char *context
) {
  if (out_blk.shape.size() != axis_pairs.size()) {
    throw std::runtime_error(
        std::string(context) + ": output block " +
        CoorsToString(out_blk.blk_coors) + " has incompatible rank."
    );
  }
  for (size_t i = 0; i < axis_pairs.size(); ++i) {
    const size_t kept_axis = axis_pairs[i].first;
    const size_t diagonal_axis = axis_pairs[i].second;
    if (src_blk.shape[kept_axis] != src_blk.shape[diagonal_axis] ||
        src_blk.shape[kept_axis] != out_blk.shape[i]) {
      throw std::runtime_error(
          std::string(context) + ": source block " +
          CoorsToString(src_blk.blk_coors) +
          " is not shape-compatible with output block " +
          CoorsToString(out_blk.blk_coors) + "."
      );
    }
  }
}

inline void FillFlatIdxCoors(
    size_t flat_idx,
    const ShapeT &shape,
    const std::vector<size_t> &offsets,
    CoorsT &coors
) {
  assert(coors.size() == shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    coors[i] = flat_idx / offsets[i];
    flat_idx %= offsets[i];
  }
}

template<typename ElemT, typename QNT>
void CopyExtractedDiagonalBlock(
    const ElemT *src_block_data,
    const DataBlk<QNT> &src_blk,
    ElemT *out_block_data,
    const DataBlk<QNT> &out_blk,
    const std::vector<std::pair<size_t, size_t>> &axis_pairs,
    const char *context
) {
  ValidateExtractedDiagonalBlockShape(src_blk, out_blk, axis_pairs, context);

  const auto src_offsets = CalcMultiDimDataOffsets(src_blk.shape);
  const auto out_offsets = CalcMultiDimDataOffsets(out_blk.shape);
  CoorsT src_data_coors(src_blk.shape.size(), 0);
  CoorsT out_data_coors(out_blk.shape.size(), 0);
  for (size_t out_flat_idx = 0; out_flat_idx < out_blk.size; ++out_flat_idx) {
    FillFlatIdxCoors(out_flat_idx, out_blk.shape, out_offsets, out_data_coors);
    for (size_t i = 0; i < axis_pairs.size(); ++i) {
      src_data_coors[axis_pairs[i].first] = out_data_coors[i];
      src_data_coors[axis_pairs[i].second] = out_data_coors[i];
    }
    out_block_data[out_flat_idx] =
        src_block_data[CalcEffOneDimArrayOffset(src_data_coors, src_offsets)];
  }
}

template<typename QNT>
void ValidateLeftBlockShape(
    const DataBlk<QNT> &left_blk,
    const DataBlk<QNT> &out_blk
) {
  if (left_blk.shape.size() != 4 || out_blk.shape.size() != 4 ||
      left_blk.shape[0] != out_blk.shape[0] ||
      left_blk.shape[1] != out_blk.shape[1] ||
      left_blk.shape[2] != out_blk.shape[0] ||
      left_blk.shape[3] != out_blk.shape[1]) {
    throw std::runtime_error(
        "DiagonalTensorProductAccumulate: left_op block " +
        CoorsToString(left_blk.blk_coors) +
        " is not shape-compatible with out block " +
        CoorsToString(out_blk.blk_coors) + "."
    );
  }
}

template<typename QNT>
void ValidateRightBlockShape(
    const DataBlk<QNT> &right_blk,
    const DataBlk<QNT> &out_blk
) {
  if (right_blk.shape.size() != 4 || out_blk.shape.size() != 4 ||
      right_blk.shape[0] != out_blk.shape[2] ||
      right_blk.shape[1] != out_blk.shape[3] ||
      right_blk.shape[2] != out_blk.shape[2] ||
      right_blk.shape[3] != out_blk.shape[3]) {
    throw std::runtime_error(
        "DiagonalTensorProductAccumulate: right_op block " +
        CoorsToString(right_blk.blk_coors) +
        " is not shape-compatible with out block " +
        CoorsToString(out_blk.blk_coors) + "."
    );
  }
}

#ifndef USE_GPU
template<typename ElemT, typename QNT>
void ApplyDiagonalTensorProductBlock(
    const ElemT *left_data,
    const DataBlk<QNT> &left_blk,
    const ElemT *right_data,
    const DataBlk<QNT> &right_blk,
    ElemT *out_data,
    const DataBlk<QNT> &out_blk,
    const ElemT alpha,
    const ElemT beta
) {
  ValidateLeftBlockShape(left_blk, out_blk);
  ValidateRightBlockShape(right_blk, out_blk);

  const size_t rows = out_blk.shape[0] * out_blk.shape[1];
  const size_t cols = out_blk.shape[2] * out_blk.shape[3];
  ElemT *out_block_data = out_data + out_blk.data_offset;
  const ElemT *left_block_data = left_data + left_blk.data_offset;
  const ElemT *right_block_data = right_data + right_blk.data_offset;

  if (rows == 0 || cols == 0) {
    return;
  }

  if (rows * cols < kDiagonalTensorProductBlasThreshold) {
    for (size_t row = 0; row < rows; ++row) {
      const ElemT left_elem = DiagonalElementFromRank4Block(
          left_block_data, left_blk.shape, row);
      for (size_t col = 0; col < cols; ++col) {
        const ElemT right_elem = DiagonalElementFromRank4Block(
            right_block_data, right_blk.shape, col);
        const size_t out_offset = row * cols + col;
        out_block_data[out_offset] =
            beta * out_block_data[out_offset] +
            alpha * left_elem * right_elem;
      }
    }
    return;
  }

  ScaleDataBlock(out_block_data, out_blk.size, beta);
  GerUpdate(
      alpha,
      left_block_data, rows + 1,
      right_block_data, cols + 1,
      rows, cols,
      out_block_data
  );
}
#endif

template<typename ElemT, typename QNT>
void ValidateDiagonalOuterProductInputs(
    const QLTensor<ElemT, QNT> &left_diag,
    const QLTensor<ElemT, QNT> &right_diag,
    const QLTensor<ElemT, QNT> &out
) {
  if (left_diag.IsDefault() || right_diag.IsDefault() || out.IsDefault() ||
      left_diag.Rank() == 0 || right_diag.Rank() == 0 ||
      out.Rank() != left_diag.Rank() + right_diag.Rank()) {
    throw std::invalid_argument(
        "DiagonalOuterProductAccumulate requires non-scalar left/right "
        "diagonal tensors and an output tensor whose rank is their rank sum."
    );
  }

  for (size_t i = 0; i < left_diag.Rank(); ++i) {
    RequireSameIndexSpaceFor(
        out.GetIndex(i),
        left_diag.GetIndex(i),
        "DiagonalOuterProductAccumulate",
        "out index " + std::to_string(i) +
        " must match left_diag index " + std::to_string(i) + "."
    );
  }
  for (size_t i = 0; i < right_diag.Rank(); ++i) {
    RequireSameIndexSpaceFor(
        out.GetIndex(left_diag.Rank() + i),
        right_diag.GetIndex(i),
        "DiagonalOuterProductAccumulate",
        "out index " + std::to_string(left_diag.Rank() + i) +
        " must match right_diag index " + std::to_string(i) + "."
    );
  }
}

inline CoorsT SliceCoors(
    const CoorsT &coors,
    const size_t offset,
    const size_t count
) {
  CoorsT sliced_coors;
  sliced_coors.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    sliced_coors.push_back(coors[offset + i]);
  }
  return sliced_coors;
}

template<typename QNT>
void ValidateDiagonalOuterProductBlockShape(
    const DataBlk<QNT> &left_blk,
    const DataBlk<QNT> &right_blk,
    const DataBlk<QNT> &out_blk
) {
  if (out_blk.shape.size() != left_blk.shape.size() + right_blk.shape.size()) {
    throw std::runtime_error(
        "DiagonalOuterProductAccumulate: output block " +
        CoorsToString(out_blk.blk_coors) + " has incompatible rank."
    );
  }
  for (size_t i = 0; i < left_blk.shape.size(); ++i) {
    if (out_blk.shape[i] != left_blk.shape[i]) {
      throw std::runtime_error(
          "DiagonalOuterProductAccumulate: left_diag block " +
          CoorsToString(left_blk.blk_coors) +
          " is not shape-compatible with out block " +
          CoorsToString(out_blk.blk_coors) + "."
      );
    }
  }
  for (size_t i = 0; i < right_blk.shape.size(); ++i) {
    if (out_blk.shape[left_blk.shape.size() + i] != right_blk.shape[i]) {
      throw std::runtime_error(
          "DiagonalOuterProductAccumulate: right_diag block " +
          CoorsToString(right_blk.blk_coors) +
          " is not shape-compatible with out block " +
          CoorsToString(out_blk.blk_coors) + "."
      );
    }
  }
}

#ifndef USE_GPU
template<typename ElemT, typename QNT>
void ApplyDiagonalOuterProductBlock(
    const ElemT *left_data,
    const DataBlk<QNT> &left_blk,
    const ElemT *right_data,
    const DataBlk<QNT> &right_blk,
    ElemT *out_data,
    const DataBlk<QNT> &out_blk,
    const ElemT alpha,
    const ElemT beta
) {
  ValidateDiagonalOuterProductBlockShape(left_blk, right_blk, out_blk);

  const size_t rows = left_blk.size;
  const size_t cols = right_blk.size;
  ElemT *out_block_data = out_data + out_blk.data_offset;
  const ElemT *left_block_data = left_data + left_blk.data_offset;
  const ElemT *right_block_data = right_data + right_blk.data_offset;

  if (rows == 0 || cols == 0) {
    return;
  }

  if (rows * cols < kDiagonalTensorProductBlasThreshold) {
    for (size_t row = 0; row < rows; ++row) {
      for (size_t col = 0; col < cols; ++col) {
        const size_t out_offset = row * cols + col;
        out_block_data[out_offset] =
            beta * out_block_data[out_offset] +
            alpha * left_block_data[row] * right_block_data[col];
      }
    }
    return;
  }

  ScaleDataBlock(out_block_data, out_blk.size, beta);
  GerUpdate(
      alpha,
      left_block_data, 1,
      right_block_data, 1,
      rows, cols,
      out_block_data
  );
}
#endif

template<typename ElemT, typename QNT>
void ValidateDiagonalTensorProductInputs(
    const QLTensor<ElemT, QNT> &left_op,
    const QLTensor<ElemT, QNT> &right_op,
    const QLTensor<ElemT, QNT> &out
) {
  if (left_op.IsDefault() || right_op.IsDefault() || out.IsDefault() ||
      left_op.Rank() != 4 || right_op.Rank() != 4 || out.Rank() != 4) {
    throw std::invalid_argument(
        "DiagonalTensorProductAccumulate requires rank-4 left_op, "
        "right_op, and out tensors."
    );
  }

  RequireSameIndexSpace(
      left_op.GetIndex(0), left_op.GetIndex(2),
      "left_op index pair (0, 2) must describe the same diagonal space."
  );
  RequireSameIndexSpace(
      left_op.GetIndex(1), left_op.GetIndex(3),
      "left_op index pair (1, 3) must describe the same diagonal space."
  );
  RequireSameIndexSpace(
      right_op.GetIndex(0), right_op.GetIndex(2),
      "right_op index pair (0, 2) must describe the same diagonal space."
  );
  RequireSameIndexSpace(
      right_op.GetIndex(1), right_op.GetIndex(3),
      "right_op index pair (1, 3) must describe the same diagonal space."
  );
  RequireSameIndexSpace(
      out.GetIndex(0), left_op.GetIndex(0),
      "out index 0 must match left_op diagonal index 0."
  );
  RequireSameIndexSpace(
      out.GetIndex(1), left_op.GetIndex(1),
      "out index 1 must match left_op diagonal index 1."
  );
  RequireSameIndexSpace(
      out.GetIndex(2), right_op.GetIndex(0),
      "out index 2 must match right_op diagonal index 0."
  );
  RequireSameIndexSpace(
      out.GetIndex(3), right_op.GetIndex(1),
      "out index 3 must match right_op diagonal index 1."
  );
}

}  // namespace detail

/**
Extract a block-sparse tensor diagonal using explicit axis pairs.

Each axis pair `(kept_axis, diagonal_axis)` must describe the same index space,
and the set of pairs must cover every input axis exactly once. The returned
tensor keeps the indexes from `kept_axis` in the order provided by
`axis_pairs`. Missing source diagonal blocks contribute no output block.
*/
template<typename ElemT, typename QNT>
QLTensor<ElemT, QNT> ExtractDiagonal(
    const QLTensor<ElemT, QNT> &tensor,
    const std::vector<std::pair<size_t, size_t>> &axis_pairs
) {
#ifdef USE_GPU
  throw std::runtime_error("ExtractDiagonal does not support GPU tensors yet.");
#else
  constexpr const char *context = "ExtractDiagonal";
  detail::ValidateDiagonalAxisPairs(tensor, axis_pairs, context);

  QLTensor<ElemT, QNT> diagonal(
      detail::MakeDiagonalIndexes(tensor, axis_pairs));
  const auto &src_bsdt = tensor.GetBlkSparDataTen();
  const auto &src_blocks = src_bsdt.GetBlkIdxDataBlkMap();
  if (src_blocks.empty()) {
    return diagonal;
  }

  const ElemT *src_data = src_bsdt.GetActualRawDataPtr();
  if (src_data == nullptr) {
    throw std::runtime_error(
        "ExtractDiagonal: source tensor has blocks but no raw data."
    );
  }

  std::vector<CoorsT> diagonal_blk_coors_s;
  diagonal_blk_coors_s.reserve(src_blocks.size());
  for (const auto &src_entry : src_blocks) {
    const auto &src_blk = src_entry.second;
    if (detail::IsDiagonalBlock(src_blk, axis_pairs)) {
      diagonal_blk_coors_s.push_back(
          detail::MakeDiagonalBlockCoors(src_blk, axis_pairs));
    }
  }
  if (diagonal_blk_coors_s.empty()) {
    return diagonal;
  }

  auto &diagonal_bsdt = diagonal.GetBlkSparDataTen();
  diagonal_bsdt.DataBlksInsert(diagonal_blk_coors_s, true, true);
  const auto &diagonal_blocks = diagonal_bsdt.GetBlkIdxDataBlkMap();
  ElemT *diagonal_data = diagonal_bsdt.GetActualRawDataPtr();
  if (diagonal_data == nullptr) {
    throw std::runtime_error(
        "ExtractDiagonal: output tensor has blocks but no raw data."
    );
  }

  for (const auto &src_entry : src_blocks) {
    const auto &src_blk = src_entry.second;
    if (!detail::IsDiagonalBlock(src_blk, axis_pairs)) {
      continue;
    }
    const CoorsT diagonal_blk_coors =
        detail::MakeDiagonalBlockCoors(src_blk, axis_pairs);
    const size_t diagonal_blk_idx =
        diagonal_bsdt.BlkCoorsToBlkIdx(diagonal_blk_coors);
    const auto &diagonal_blk = diagonal_blocks.at(diagonal_blk_idx);
    detail::CopyExtractedDiagonalBlock(
        src_data + src_blk.data_offset,
        src_blk,
        diagonal_data + diagonal_blk.data_offset,
        diagonal_blk,
        axis_pairs,
        context
    );
  }
  return diagonal;
#endif
}

/**
Accumulate an outer product of two already extracted diagonal tensors.

For every stored output block, this computes
\f[
out(l, r) = beta \cdot out(l, r) +
            alpha \cdot left\_diag(l) \cdot right\_diag(r).
\f]

The output indexes must be the concatenation of the left and right diagonal
index spaces. Missing left or right diagonal blocks contribute zero, and output
blocks are never inserted or removed.
*/
template<typename ElemT, typename QNT>
void DiagonalOuterProductAccumulate(
    const QLTensor<ElemT, QNT> &left_diag,
    const QLTensor<ElemT, QNT> &right_diag,
    QLTensor<ElemT, QNT> &out,
    ElemT alpha = ElemT(1),
    ElemT beta = ElemT(1)
) {
#ifdef USE_GPU
  throw std::runtime_error(
      "DiagonalOuterProductAccumulate does not support GPU tensors yet."
  );
#else
  detail::ValidateDiagonalOuterProductInputs(left_diag, right_diag, out);

  const auto &left_bsdt = left_diag.GetBlkSparDataTen();
  const auto &right_bsdt = right_diag.GetBlkSparDataTen();
  auto &out_bsdt = out.GetBlkSparDataTen();

  const auto &left_blocks = left_bsdt.GetBlkIdxDataBlkMap();
  const auto &right_blocks = right_bsdt.GetBlkIdxDataBlkMap();
  const auto &out_blocks = out_bsdt.GetBlkIdxDataBlkMap();

  const ElemT *left_data = left_bsdt.GetActualRawDataPtr();
  const ElemT *right_data = right_bsdt.GetActualRawDataPtr();
  ElemT *out_data = out_bsdt.GetActualRawDataPtr();

  if (!left_blocks.empty() && left_data == nullptr) {
    throw std::runtime_error(
        "DiagonalOuterProductAccumulate: left_diag has blocks but no raw data."
    );
  }
  if (!right_blocks.empty() && right_data == nullptr) {
    throw std::runtime_error(
        "DiagonalOuterProductAccumulate: right_diag has blocks but no raw data."
    );
  }
  if (!out_blocks.empty() && out_data == nullptr) {
    throw std::runtime_error(
        "DiagonalOuterProductAccumulate: out has blocks but no raw data."
    );
  }

  const size_t left_rank = left_diag.Rank();
  const size_t right_rank = right_diag.Rank();
  for (const auto &out_entry : out_blocks) {
    const auto &out_blk = out_entry.second;
    const CoorsT left_blk_coors =
        detail::SliceCoors(out_blk.blk_coors, 0, left_rank);
    const CoorsT right_blk_coors =
        detail::SliceCoors(out_blk.blk_coors, left_rank, right_rank);
    const auto left_it = left_blocks.find(
        left_bsdt.BlkCoorsToBlkIdx(left_blk_coors));
    const auto right_it = right_blocks.find(
        right_bsdt.BlkCoorsToBlkIdx(right_blk_coors));

    if (left_it == left_blocks.end() || right_it == right_blocks.end() ||
        alpha == ElemT(0)) {
      detail::ScaleDataBlock(out_data + out_blk.data_offset, out_blk.size, beta);
      continue;
    }

    detail::ApplyDiagonalOuterProductBlock(
        left_data, left_it->second,
        right_data, right_it->second,
        out_data, out_blk,
        alpha, beta
    );
  }
#endif
}

/**
Accumulate the diagonal of a tensor product into an existing rank-4 tensor.

The operator tensors are interpreted as rank-4 square operators with paired
diagonal spaces:

@code
left_op(a0, a1, a0, a1)
right_op(b0, b1, b0, b1)
out(a0, a1, b0, b1)
@endcode

For every stored output block, this computes
\f[
out(a_0,a_1,b_0,b_1) =
  beta \cdot out(a_0,a_1,b_0,b_1) +
  alpha \cdot left\_op(a_0,a_1,a_0,a_1)
        \cdot right\_op(b_0,b_1,b_0,b_1).
\f]

Index directions are ignored for compatibility checks, but quantum-number
sector order and degeneracies must match. Missing operator diagonal blocks
contribute zero, and output blocks are never inserted or removed.
*/
template<typename ElemT, typename QNT>
void DiagonalTensorProductAccumulate(
    const QLTensor<ElemT, QNT> &left_op,
    const QLTensor<ElemT, QNT> &right_op,
    QLTensor<ElemT, QNT> &out,
    ElemT alpha = ElemT(1),
    ElemT beta = ElemT(1)
) {
#ifdef USE_GPU
  throw std::runtime_error(
      "DiagonalTensorProductAccumulate does not support GPU tensors yet."
  );
#else
  detail::ValidateDiagonalTensorProductInputs(left_op, right_op, out);

  const auto &left_bsdt = left_op.GetBlkSparDataTen();
  const auto &right_bsdt = right_op.GetBlkSparDataTen();
  auto &out_bsdt = out.GetBlkSparDataTen();

  const auto &left_blocks = left_bsdt.GetBlkIdxDataBlkMap();
  const auto &right_blocks = right_bsdt.GetBlkIdxDataBlkMap();
  const auto &out_blocks = out_bsdt.GetBlkIdxDataBlkMap();

  const ElemT *left_data = left_bsdt.GetActualRawDataPtr();
  const ElemT *right_data = right_bsdt.GetActualRawDataPtr();
  ElemT *out_data = out_bsdt.GetActualRawDataPtr();

  if (!left_blocks.empty() && left_data == nullptr) {
    throw std::runtime_error(
        "DiagonalTensorProductAccumulate: left_op has blocks but no raw data."
    );
  }
  if (!right_blocks.empty() && right_data == nullptr) {
    throw std::runtime_error(
        "DiagonalTensorProductAccumulate: right_op has blocks but no raw data."
    );
  }
  if (!out_blocks.empty() && out_data == nullptr) {
    throw std::runtime_error(
        "DiagonalTensorProductAccumulate: out has blocks but no raw data."
    );
  }

  for (const auto &out_entry : out_blocks) {
    const auto &out_blk = out_entry.second;

    const auto left_it = left_blocks.find(
        detail::Rank4BlkCoorsToBlkIdx(
            left_bsdt.blk_shape,
            out_blk.blk_coors[0], out_blk.blk_coors[1],
            out_blk.blk_coors[0], out_blk.blk_coors[1]));
    const auto right_it = right_blocks.find(
        detail::Rank4BlkCoorsToBlkIdx(
            right_bsdt.blk_shape,
            out_blk.blk_coors[2], out_blk.blk_coors[3],
            out_blk.blk_coors[2], out_blk.blk_coors[3]));

    if (left_it == left_blocks.end() || right_it == right_blocks.end() ||
        alpha == ElemT(0)) {
      detail::ScaleDataBlock(out_data + out_blk.data_offset, out_blk.size, beta);
      continue;
    }

    detail::ApplyDiagonalTensorProductBlock(
        left_data, left_it->second,
        right_data, right_it->second,
        out_data, out_blk,
        alpha, beta
    );
  }
#endif
}

} /* qlten */
#endif /* ifndef QLTEN_TENSOR_MANIPULATION_TEN_DIAGONAL_TENSOR_PRODUCT_H */
