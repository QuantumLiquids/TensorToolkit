// SPDX-License-Identifier: LGPL-3.0-only
/*
* Description: DMRG axis-local bosonic tensor operations.
*/

/**
@file dmrg/axis_ops.h
@brief Axis-local tensor operations for bosonic DMRG workflows.
*/
#ifndef QLTEN_TENSOR_MANIPULATION_DMRG_AXIS_OPS_H
#define QLTEN_TENSOR_MANIPULATION_DMRG_AXIS_OPS_H

#include <algorithm>    // fill
#include <cstddef>      // size_t
#include <functional>   // multiplies
#include <numeric>      // accumulate
#include <set>          // set
#include <stdexcept>    // invalid_argument, runtime_error
#include <string>       // string
#include <type_traits>  // is_same
#include <utility>      // move
#include <vector>       // vector

#include "qlten/framework/flops_count.h"  // flop
#include "qlten/qltensor_all.h"
#include "qlten/utility/utils_inl.h"    // CalcMultiDimDataOffsets, CalcEffOneDimArrayOffset

namespace qlten::dmrg {

/**
 * @brief Diagonal operator acting on one tensor axis.
 *
 * The operator is represented in the degeneracy basis of `index`.  For sector
 * `s` and degeneracy offset `d`, an input slice on the selected axis is
 * multiplied by `values[s][d]`.
 *
 * @note These DMRG axis primitives are bosonic-only.  Downstream fermionic
 *       DMRG code should use Jordan-Wigner transformed bosonic operators before
 *       calling this API.
 */
template<typename ElemT, typename QNT>
struct AxisDiagonalOp {
  /// Axis index whose sector order and degeneracies define `values`.
  Index<QNT> index;
  /// Diagonal values laid out as `[qn_sector][degeneracy_offset]`.
  std::vector<std::vector<ElemT>> values;
};

/**
 * @brief Compact projector acting on one tensor axis.
 *
 * `kept_degeneracies[s]` lists the old degeneracy offsets retained from input
 * sector `s`.  Empty sectors are deleted from the output index.  Non-empty
 * sectors keep input sector order, while degeneracies inside each sector follow
 * the exact order listed in `kept_degeneracies[s]`.
 */
template<typename QNT>
struct AxisProjector {
  /// Axis index expected on the input tensor.
  Index<QNT> input_index;
  /// Retained degeneracy offsets laid out as `[input_sector][kept_offset]`.
  std::vector<std::vector<size_t>> kept_degeneracies;
};

/**
 * @brief One nonzero entry of an axis-local monomial operator.
 *
 * This maps one degeneracy basis vector from `(in_sector, in_offset)` to
 * `(out_sector, out_offset)` with coefficient `coef`.
 */
template<typename ElemT, typename QNT>
struct AxisMonomialOpEntry {
  /// Input quantum-number sector coordinate.
  size_t in_sector;
  /// Input degeneracy coordinate inside `in_sector`.
  size_t in_offset;
  /// Output quantum-number sector coordinate.
  size_t out_sector;
  /// Output degeneracy coordinate inside `out_sector`.
  size_t out_offset;
  /// Multiplicative coefficient for this basis-vector map.
  ElemT coef;
};

/**
 * @brief Sparse one-hot/shift operator acting on one tensor axis.
 *
 * Entries are interpreted as a raw-data scatter from the input axis basis to
 * the output axis basis.  Multiple entries may target the same output slice;
 * their contributions are accumulated.
 *
 * @note `flux` is validated entry-by-entry as
 *       `output_qn == input_qn + flux`.
 */
template<typename ElemT, typename QNT>
struct AxisMonomialOp {
  /// Axis index expected on the input tensor.
  Index<QNT> input_index;
  /// Axis index used on the output tensor.
  Index<QNT> output_index;
  /// Quantum-number shift carried by every entry.
  QNT flux;
  /// Nonzero monomial entries.
  std::vector<AxisMonomialOpEntry<ElemT, QNT>> entries;
};

/**
 * @brief Per-sector scalar applied in-place to one tensor axis.
 *
 * Every stored block whose selected-axis sector is `s` is multiplied as a whole
 * by `values_by_sector[s]`.  This is the fast path for parity/sign and other
 * sector-scalar operations.
 */
template<typename ElemT, typename QNT>
struct AxisSectorScalar {
  /// Tensor axis to scale.
  size_t axis;
  /// Multiplicative value for each sector of `axis`.
  std::vector<ElemT> values_by_sector;
};

/**
 * @brief Layout work counters for rank-2 axis application.
 *
 * These counters track full-state layout movement in the DMRG hot path.  The
 * preserve-order primitive below writes directly into the requested output
 * layout, so successful CPU calls currently leave all fields at zero.
 */
struct Rank2AxisApplyStats {
  /// Number of full-state layout copy calls.
  size_t layout_copy_calls = 0;
  /// Number of full-state layout transpose calls.
  size_t layout_transpose_calls = 0;
  /// Full-state layout traffic in GiB.
  double layout_gb = 0.0;
};

/**
 * @brief Layout work counters for positional axis movement.
 */
struct AxisLayoutMoveStats {
  /// Number of generic full-tensor layout transpose calls.
  size_t layout_transpose_calls = 0;
  /// Number of dense block matrix transpose calls.
  size_t block_matrix_transpose_calls = 0;
  /// Raw-data traffic in GB, counting reads plus writes.
  double layout_gb = 0.0;
};

namespace detail {

inline std::string AxisOpsPrefix(const char *func) {
  return std::string(func) + ": ";
}

template<typename ElemT>
constexpr bool IsComplexElem() {
  using BareElemT = typename std::remove_cv<ElemT>::type;
  return std::is_same<BareElemT, QLTEN_Complex>::value ||
         std::is_same<BareElemT, QLTEN_ComplexFloat>::value;
}

template<typename ElemT>
constexpr size_t ScaleFlopCost() {
  return IsComplexElem<ElemT>() ? 6 : 1;
}

template<typename ElemT>
constexpr size_t AxpyFlopCost() {
  return IsComplexElem<ElemT>() ? 8 : 2;
}

template<typename ElemT>
constexpr size_t ProjectFlopCost() {
  return IsComplexElem<ElemT>() ? 2 : 1;
}

template<typename ElemT>
void CountScaleFlops(const size_t size) {
#ifdef QLTEN_COUNT_FLOPS
  flop += ScaleFlopCost<ElemT>() * size;
#else
  (void) size;
#endif
}

template<typename ElemT>
void CountAxpyFlops(const size_t size) {
#ifdef QLTEN_COUNT_FLOPS
  flop += AxpyFlopCost<ElemT>() * size;
#else
  (void) size;
#endif
}

template<typename ElemT>
void CountProjectFlops(const size_t size) {
#ifdef QLTEN_COUNT_FLOPS
  flop += ProjectFlopCost<ElemT>() * size;
#else
  (void) size;
#endif
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

template<typename ElemT, typename QNT>
void RequireUsableAxis(
    const QLTensor<ElemT, QNT> &tensor,
    const size_t axis,
    const char *func
) {
  if (tensor.IsDefault()) {
    throw std::invalid_argument(
        AxisOpsPrefix(func) + "input tensor must not be default."
    );
  }
  if (axis >= tensor.Rank()) {
    throw std::invalid_argument(
        AxisOpsPrefix(func) + "axis is out of range."
    );
  }
}

template<typename ElemT, typename QNT>
void RequireRawDataIfBlocksPresent(
    const QLTensor<ElemT, QNT> &tensor,
    const char *func,
    const char *name
) {
  const auto &bsdt = tensor.GetBlkSparDataTen();
  if (!bsdt.GetBlkIdxDataBlkMap().empty() &&
      bsdt.GetActualRawDataPtr() == nullptr) {
    throw std::runtime_error(
        AxisOpsPrefix(func) + name + " has blocks but no raw data."
    );
  }
}

template<typename ElemT, typename QNT>
void RequireDiagonalOpMatchesAxis(
    const Index<QNT> &axis_index,
    const AxisDiagonalOp<ElemT, QNT> &op,
    const char *func
) {
  if (!IndexSpaceEqualIgnoreDirection(axis_index, op.index)) {
    throw std::invalid_argument(
        AxisOpsPrefix(func) + "operator index does not match tensor axis."
    );
  }
  if (op.values.size() != axis_index.GetQNSctNum()) {
    throw std::invalid_argument(
        AxisOpsPrefix(func) + "values size must match the number of sectors."
    );
  }
  for (size_t sct = 0; sct < axis_index.GetQNSctNum(); ++sct) {
    if (op.values[sct].size() != axis_index.GetQNSct(sct).GetDegeneracy()) {
      throw std::invalid_argument(
          AxisOpsPrefix(func) +
          "values[" + std::to_string(sct) +
          "] size must match sector degeneracy."
      );
    }
  }
}

inline size_t DeletedSectorMarker() {
  return static_cast<size_t>(-1);
}

template<typename QNT>
std::vector<size_t> ValidateProjectorAndBuildSectorMap(
    const Index<QNT> &axis_index,
    const AxisProjector<QNT> &projector,
    const char *func
) {
  if (!IndexSpaceEqualIgnoreDirection(axis_index, projector.input_index)) {
    throw std::invalid_argument(
        AxisOpsPrefix(func) + "projector input index does not match tensor axis."
    );
  }
  if (projector.kept_degeneracies.size() != axis_index.GetQNSctNum()) {
    throw std::invalid_argument(
        AxisOpsPrefix(func) +
        "kept_degeneracies size must match the number of input sectors."
    );
  }

  std::vector<size_t> input_sector_to_output_sector(
      axis_index.GetQNSctNum(), DeletedSectorMarker());
  size_t output_sector = 0;
  bool has_kept_degeneracy = false;
  for (size_t sct = 0; sct < axis_index.GetQNSctNum(); ++sct) {
    const auto &kept = projector.kept_degeneracies[sct];
    if (kept.empty()) {
      continue;
    }
    has_kept_degeneracy = true;
    const size_t input_degeneracy = axis_index.GetQNSct(sct).GetDegeneracy();
    std::set<size_t> seen_offsets;
    for (const size_t offset : kept) {
      if (offset >= input_degeneracy) {
        throw std::invalid_argument(
            AxisOpsPrefix(func) + "kept degeneracy offset is out of range."
        );
      }
      if (!seen_offsets.insert(offset).second) {
        throw std::invalid_argument(
            AxisOpsPrefix(func) + "kept degeneracy offsets must be unique."
        );
      }
    }
    input_sector_to_output_sector[sct] = output_sector;
    ++output_sector;
  }
  if (!has_kept_degeneracy) {
    throw std::invalid_argument(
        AxisOpsPrefix(func) + "projector must keep at least one degeneracy."
    );
  }
  return input_sector_to_output_sector;
}

template<typename QNT>
Index<QNT> MakeProjectedIndex(
    const Index<QNT> &axis_index,
    const AxisProjector<QNT> &projector
) {
  QNSectorVec<QNT> output_qnscts;
  for (size_t sct = 0; sct < axis_index.GetQNSctNum(); ++sct) {
    const auto &kept = projector.kept_degeneracies[sct];
    if (!kept.empty()) {
      output_qnscts.emplace_back(axis_index.GetQNSct(sct).GetQn(), kept.size());
    }
  }
  return Index<QNT>(output_qnscts, axis_index.GetDir());
}

template<typename ElemT, typename QNT>
IndexVec<QNT> ProjectedOutputIndexes(
    const QLTensor<ElemT, QNT> &input,
    const size_t axis,
    const AxisProjector<QNT> &projector
) {
  IndexVec<QNT> output_indexes = input.GetIndexes();
  output_indexes[axis] = MakeProjectedIndex(input.GetIndex(axis), projector);
  return output_indexes;
}

template<typename QNT>
size_t BlkCoorsToBlkIdx(
    const CoorsT &blk_coors,
    const IndexVec<QNT> &indexes
);

template<typename ElemT, typename QNT>
std::vector<CoorsT> GenerateProjectedOutputBlockCoors(
    const QLTensor<ElemT, QNT> &input,
    const size_t axis,
    const std::vector<size_t> &input_sector_to_output_sector,
    const IndexVec<QNT> &output_indexes
) {
  std::set<size_t> seen_blk_idxs;
  std::vector<CoorsT> output_blk_coors_s;
  const auto &input_blocks = input.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  for (const auto &entry : input_blocks) {
    const auto &input_blk = entry.second;
    const size_t output_sector =
        input_sector_to_output_sector[input_blk.blk_coors[axis]];
    if (output_sector == DeletedSectorMarker()) {
      continue;
    }
    CoorsT output_blk_coors = input_blk.blk_coors;
    output_blk_coors[axis] = output_sector;
    const size_t output_blk_idx = BlkCoorsToBlkIdx(output_blk_coors, output_indexes);
    if (seen_blk_idxs.insert(output_blk_idx).second) {
      output_blk_coors_s.push_back(std::move(output_blk_coors));
    }
  }
  return output_blk_coors_s;
}

template<typename ElemT, typename QNT>
bool BlockTopologyEqual(
    const QLTensor<ElemT, QNT> &lhs,
    const QLTensor<ElemT, QNT> &rhs
) {
  const auto &lhs_blocks = lhs.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  const auto &rhs_blocks = rhs.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  if (lhs_blocks.size() != rhs_blocks.size()) {
    return false;
  }
  auto lhs_it = lhs_blocks.begin();
  auto rhs_it = rhs_blocks.begin();
  for (; lhs_it != lhs_blocks.end(); ++lhs_it, ++rhs_it) {
    if (lhs_it->first != rhs_it->first ||
        lhs_it->second.blk_coors != rhs_it->second.blk_coors ||
        lhs_it->second.shape != rhs_it->second.shape) {
      return false;
    }
  }
  return true;
}

template<typename ElemT, typename QNT>
void RequireSameLayout(
    const QLTensor<ElemT, QNT> &input,
    const QLTensor<ElemT, QNT> &output,
    const char *func
) {
  if (output.IsDefault()) {
    throw std::invalid_argument(
        AxisOpsPrefix(func) + "output must not be default when beta is nonzero."
    );
  }
  if (input.GetIndexes() != output.GetIndexes() ||
      !BlockTopologyEqual(input, output)) {
    throw std::invalid_argument(
        AxisOpsPrefix(func) +
        "output must have the same index and block topology as input when "
        "beta is nonzero."
    );
  }
}

template<typename ElemT, typename QNT>
void InitializeOutputLikeInput(
    const QLTensor<ElemT, QNT> &input,
    QLTensor<ElemT, QNT> &output
) {
  output = QLTensor<ElemT, QNT>(input.GetIndexes());
  std::vector<CoorsT> blk_coors_s;
  const auto &input_blocks = input.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  blk_coors_s.reserve(input_blocks.size());
  for (const auto &entry : input_blocks) {
    blk_coors_s.push_back(entry.second.blk_coors);
  }
  if (!blk_coors_s.empty()) {
    output.GetBlkSparDataTen().DataBlksInsert(blk_coors_s, true, true);
  }
}

template<typename ElemT>
void ScaleRawData(ElemT *data, const size_t size, const ElemT scale) {
  if (size == 0 || scale == ElemT(1)) {
    return;
  }
  CountScaleFlops<ElemT>(size);
  if (scale == ElemT(0)) {
    std::fill(data, data + size, ElemT(0));
    return;
  }
  for (size_t i = 0; i < size; ++i) {
    data[i] *= scale;
  }
}

template<typename ElemT, typename QNT>
void ScaleTensorRawData(QLTensor<ElemT, QNT> &tensor, const ElemT scale) {
  auto &bsdt = tensor.GetBlkSparDataTen();
  ScaleRawData(bsdt.GetActualRawDataPtr(), bsdt.GetActualRawDataSize(), scale);
}

template<typename ElemT>
bool AllValuesEqual(const std::vector<ElemT> &values, const ElemT value) {
  for (const auto &entry : values) {
    if (entry != value) {
      return false;
    }
  }
  return true;
}

template<typename ElemT>
bool AllValuesSame(const std::vector<ElemT> &values) {
  if (values.empty()) {
    return true;
  }
  return AllValuesEqual(values, values[0]);
}

inline size_t AxisInnerSize(const ShapeT &shape, const size_t axis) {
  size_t inner_size = 1;
  for (size_t i = axis + 1; i < shape.size(); ++i) {
    inner_size *= shape[i];
  }
  return inner_size;
}

template<typename ElemT>
void AddAxisDiagonalBlock(
    const ElemT *input_data,
    ElemT *output_data,
    const ShapeT &shape,
    const size_t axis,
    const std::vector<ElemT> &values,
    const ElemT alpha
) {
  if (alpha == ElemT(0) || values.empty()) {
    return;
  }
  const size_t axis_dim = shape[axis];
  const size_t inner_size = AxisInnerSize(shape, axis);
  const size_t outer_size = (axis_dim == 0 || inner_size == 0)
                            ? 0
                            : std::accumulate(shape.begin(), shape.begin() + axis,
                                              size_t(1), std::multiplies<size_t>());

  if (AllValuesEqual(values, ElemT(0))) {
    return;
  }
  if (AllValuesSame(values)) {
    const ElemT coef = alpha * values[0];
    const size_t block_size = outer_size * axis_dim * inner_size;
    CountAxpyFlops<ElemT>(block_size);
    for (size_t i = 0; i < block_size; ++i) {
      output_data[i] += coef * input_data[i];
    }
    return;
  }

  for (size_t outer = 0; outer < outer_size; ++outer) {
    const size_t outer_base = outer * axis_dim * inner_size;
    for (size_t axis_coor = 0; axis_coor < axis_dim; ++axis_coor) {
      const ElemT coef = alpha * values[axis_coor];
      if (coef == ElemT(0)) {
        continue;
      }
      const size_t base = outer_base + axis_coor * inner_size;
      CountAxpyFlops<ElemT>(inner_size);
      for (size_t inner = 0; inner < inner_size; ++inner) {
        output_data[base + inner] += coef * input_data[base + inner];
      }
    }
  }
}

template<typename ElemT>
void ScaleAxisDiagonalBlockInPlace(
    ElemT *data,
    const ShapeT &shape,
    const size_t axis,
    const std::vector<ElemT> &values
) {
  const size_t axis_dim = shape[axis];
  const size_t inner_size = AxisInnerSize(shape, axis);
  const size_t outer_size = (axis_dim == 0 || inner_size == 0)
                            ? 0
                            : std::accumulate(shape.begin(), shape.begin() + axis,
                                              size_t(1), std::multiplies<size_t>());
  const size_t block_size = outer_size * axis_dim * inner_size;

  if (AllValuesEqual(values, ElemT(1))) {
    return;
  }
  if (AllValuesEqual(values, ElemT(0))) {
    CountScaleFlops<ElemT>(block_size);
    std::fill(data, data + block_size, ElemT(0));
    return;
  }
  if (AllValuesSame(values)) {
    ScaleRawData(data, block_size, values[0]);
    return;
  }

  for (size_t outer = 0; outer < outer_size; ++outer) {
    const size_t outer_base = outer * axis_dim * inner_size;
    for (size_t axis_coor = 0; axis_coor < axis_dim; ++axis_coor) {
      const ElemT coef = values[axis_coor];
      if (coef == ElemT(1)) {
        continue;
      }
      const size_t base = outer_base + axis_coor * inner_size;
      if (coef == ElemT(0)) {
        CountScaleFlops<ElemT>(inner_size);
        std::fill(data + base, data + base + inner_size, ElemT(0));
      } else {
        CountScaleFlops<ElemT>(inner_size);
        for (size_t inner = 0; inner < inner_size; ++inner) {
          data[base + inner] *= coef;
        }
      }
    }
  }
}

template<typename QNT>
size_t BlkCoorsToBlkIdx(
    const CoorsT &blk_coors,
    const IndexVec<QNT> &indexes
) {
  const ShapeT blk_shape = CalcQNSctNumOfIdxs(indexes);
  const auto offsets = CalcMultiDimDataOffsets(blk_shape);
  return CalcEffOneDimArrayOffset(blk_coors, offsets);
}

template<typename ElemT, typename QNT>
IndexVec<QNT> MonomialOutputIndexes(
    const QLTensor<ElemT, QNT> &input,
    const size_t axis,
    const AxisMonomialOp<ElemT, QNT> &op
) {
  IndexVec<QNT> output_indexes = input.GetIndexes();
  output_indexes[axis] = op.output_index;
  return output_indexes;
}

template<typename ElemT, typename QNT>
void RequireMonomialOpMatchesAxis(
    const Index<QNT> &axis_index,
    const AxisMonomialOp<ElemT, QNT> &op,
    const char *func
) {
  if (!IndexSpaceEqualIgnoreDirection(axis_index, op.input_index)) {
    throw std::invalid_argument(
        AxisOpsPrefix(func) + "operator input index does not match tensor axis."
    );
  }
  if (op.output_index.GetDir() != axis_index.GetDir()) {
    throw std::invalid_argument(
        AxisOpsPrefix(func) +
        "operator output index direction must match tensor axis direction."
    );
  }
  for (size_t entry_idx = 0; entry_idx < op.entries.size(); ++entry_idx) {
    const auto &entry = op.entries[entry_idx];
    if (entry.in_sector >= op.input_index.GetQNSctNum() ||
        entry.out_sector >= op.output_index.GetQNSctNum()) {
      throw std::invalid_argument(
          AxisOpsPrefix(func) + "entry sector is out of range."
      );
    }
    const auto &in_sct = op.input_index.GetQNSct(entry.in_sector);
    const auto &out_sct = op.output_index.GetQNSct(entry.out_sector);
    if (entry.in_offset >= in_sct.GetDegeneracy() ||
        entry.out_offset >= out_sct.GetDegeneracy()) {
      throw std::invalid_argument(
          AxisOpsPrefix(func) + "entry degeneracy offset is out of range."
      );
    }
    if (out_sct.GetQn() != in_sct.GetQn() + op.flux) {
      throw std::invalid_argument(
          AxisOpsPrefix(func) +
          "entry quantum numbers are inconsistent with operator flux."
      );
    }
  }
}

template<typename ElemT, typename QNT>
IndexVec<QNT> Rank2OutputIndexes(
    const QLTensor<ElemT, QNT> &input,
    const size_t axis,
    const QLTensor<ElemT, QNT> &rank2_op
) {
  IndexVec<QNT> output_indexes = input.GetIndexes();
  output_indexes[axis] = rank2_op.GetIndex(1);
  return output_indexes;
}

template<typename ElemT, typename QNT>
void RequireRank2OpMatchesAxis(
    const Index<QNT> &axis_index,
    const QLTensor<ElemT, QNT> &rank2_op,
    const char *func
) {
  if (rank2_op.IsDefault()) {
    throw std::invalid_argument(
        AxisOpsPrefix(func) + "rank2_op must not be default."
    );
  }
  if (rank2_op.Rank() != 2) {
    throw std::invalid_argument(
        AxisOpsPrefix(func) + "rank2_op must have rank 2."
    );
  }
  if (rank2_op.GetIndex(0) != InverseIndex(axis_index)) {
    throw std::invalid_argument(
        AxisOpsPrefix(func) +
        "rank2_op input index must be the inverse of the tensor axis."
    );
  }
}

template<typename ElemT, typename QNT>
std::vector<CoorsT> GenerateRank2OutputBlockCoors(
    const QLTensor<ElemT, QNT> &input,
    const size_t axis,
    const QLTensor<ElemT, QNT> &rank2_op,
    const IndexVec<QNT> &output_indexes
) {
  std::set<size_t> seen_blk_idxs;
  std::vector<CoorsT> output_blk_coors_s;
  const auto &input_blocks = input.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  const auto &op_blocks = rank2_op.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  for (const auto &input_entry : input_blocks) {
    const auto &input_blk = input_entry.second;
    for (const auto &op_entry : op_blocks) {
      const auto &op_blk = op_entry.second;
      if (op_blk.blk_coors[0] != input_blk.blk_coors[axis]) {
        continue;
      }
      CoorsT output_blk_coors = input_blk.blk_coors;
      output_blk_coors[axis] = op_blk.blk_coors[1];
      const size_t output_blk_idx = BlkCoorsToBlkIdx(output_blk_coors, output_indexes);
      if (seen_blk_idxs.insert(output_blk_idx).second) {
        output_blk_coors_s.push_back(std::move(output_blk_coors));
      }
    }
  }
  return output_blk_coors_s;
}

template<typename QNT>
IndexVec<QNT> MoveHeadAxisToTailIndexes(
    const IndexVec<QNT> &input_indexes
) {
  IndexVec<QNT> output_indexes;
  output_indexes.reserve(input_indexes.size());
  output_indexes.insert(output_indexes.end(),
                        input_indexes.begin() + 1,
                        input_indexes.end());
  output_indexes.push_back(input_indexes.front());
  return output_indexes;
}

inline CoorsT MoveHeadAxisToTailCoors(const CoorsT &input_coors) {
  CoorsT output_coors;
  output_coors.reserve(input_coors.size());
  output_coors.insert(output_coors.end(),
                      input_coors.begin() + 1,
                      input_coors.end());
  output_coors.push_back(input_coors.front());
  return output_coors;
}

template<typename ElemT>
double RawReadWriteGb(const size_t elem_count) {
  return 2.0 * static_cast<double>(elem_count) *
         static_cast<double>(sizeof(ElemT)) / 1.0e9;
}

template<typename ElemT>
void MoveHeadAxisBlockToTail(
    const ElemT *input_data,
    ElemT *output_data,
    const ShapeT &input_shape
) {
  const size_t head_dim = input_shape.front();
  const size_t tail_size = std::accumulate(input_shape.begin() + 1,
                                           input_shape.end(),
                                           size_t(1),
                                           std::multiplies<size_t>());
  for (size_t head = 0; head < head_dim; ++head) {
    const size_t input_base = head * tail_size;
    for (size_t tail = 0; tail < tail_size; ++tail) {
      output_data[tail * head_dim + head] = input_data[input_base + tail];
    }
  }
}

template<typename ElemT>
void AddRank2AxisBlock(
    const ElemT *input_data,
    const ElemT *op_data,
    ElemT *output_data,
    const ShapeT &input_shape,
    const ShapeT &op_shape,
    const ShapeT &output_shape,
    const size_t axis
) {
  const size_t input_axis_dim = input_shape[axis];
  const size_t output_axis_dim = output_shape[axis];
  const size_t inner_size = AxisInnerSize(input_shape, axis);
  const size_t outer_size = std::accumulate(input_shape.begin(), input_shape.begin() + axis,
                                            size_t(1), std::multiplies<size_t>());
  for (size_t outer = 0; outer < outer_size; ++outer) {
    const size_t input_outer_base = outer * input_axis_dim * inner_size;
    const size_t output_outer_base = outer * output_axis_dim * inner_size;
    for (size_t in_offset = 0; in_offset < input_axis_dim; ++in_offset) {
      const size_t input_base = input_outer_base + in_offset * inner_size;
      const size_t op_base = in_offset * op_shape[1];
      for (size_t out_offset = 0; out_offset < output_axis_dim; ++out_offset) {
        const ElemT coef = op_data[op_base + out_offset];
        if (coef == ElemT(0)) {
          continue;
        }
        const size_t output_base = output_outer_base + out_offset * inner_size;
        CountAxpyFlops<ElemT>(inner_size);
        for (size_t inner = 0; inner < inner_size; ++inner) {
          output_data[output_base + inner] +=
              coef * input_data[input_base + inner];
        }
      }
    }
  }
}

template<typename ElemT, typename QNT>
std::vector<CoorsT> GenerateMonomialOutputBlockCoors(
    const QLTensor<ElemT, QNT> &input,
    const size_t axis,
    const AxisMonomialOp<ElemT, QNT> &op,
    const IndexVec<QNT> &output_indexes
) {
  std::set<size_t> seen_blk_idxs;
  std::vector<CoorsT> output_blk_coors_s;
  const auto &input_blocks = input.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  for (const auto &input_entry : input_blocks) {
    const auto &input_blk = input_entry.second;
    for (const auto &op_entry : op.entries) {
      if (op_entry.in_sector != input_blk.blk_coors[axis]) {
        continue;
      }
      CoorsT output_blk_coors = input_blk.blk_coors;
      output_blk_coors[axis] = op_entry.out_sector;
      const size_t output_blk_idx = BlkCoorsToBlkIdx(output_blk_coors, output_indexes);
      if (seen_blk_idxs.insert(output_blk_idx).second) {
        output_blk_coors_s.push_back(std::move(output_blk_coors));
      }
    }
  }
  return output_blk_coors_s;
}

template<typename ElemT, typename QNT>
bool BlockTopologyMatchesCoors(
    const QLTensor<ElemT, QNT> &tensor,
    const std::vector<CoorsT> &blk_coors_s
) {
  std::set<size_t> expected;
  for (const auto &blk_coors : blk_coors_s) {
    expected.insert(tensor.GetBlkSparDataTen().BlkCoorsToBlkIdx(blk_coors));
  }
  const auto &blocks = tensor.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  if (blocks.size() != expected.size()) {
    return false;
  }
  auto expected_it = expected.begin();
  for (const auto &block_entry : blocks) {
    if (block_entry.first != *expected_it) {
      return false;
    }
    ++expected_it;
  }
  return true;
}

template<typename ElemT>
void AddAxisMonomialEntryBlock(
    const ElemT *input_data,
    ElemT *output_data,
    const ShapeT &input_shape,
    const ShapeT &output_shape,
    const size_t axis,
    const size_t in_offset,
    const size_t out_offset,
    const ElemT coef
) {
  if (coef == ElemT(0)) {
    return;
  }
  const size_t input_axis_dim = input_shape[axis];
  const size_t output_axis_dim = output_shape[axis];
  const size_t inner_size = AxisInnerSize(input_shape, axis);
  const size_t outer_size = std::accumulate(input_shape.begin(), input_shape.begin() + axis,
                                            size_t(1), std::multiplies<size_t>());
  CountAxpyFlops<ElemT>(outer_size * inner_size);
  for (size_t outer = 0; outer < outer_size; ++outer) {
    const size_t input_base =
        outer * input_axis_dim * inner_size + in_offset * inner_size;
    const size_t output_base =
        outer * output_axis_dim * inner_size + out_offset * inner_size;
    for (size_t inner = 0; inner < inner_size; ++inner) {
      output_data[output_base + inner] += coef * input_data[input_base + inner];
    }
  }
}

template<typename ElemT>
void CopyProjectedAxisBlock(
    const ElemT *input_data,
    ElemT *output_data,
    const ShapeT &input_shape,
    const ShapeT &output_shape,
    const size_t axis,
    const std::vector<size_t> &kept_degeneracies
) {
  const size_t input_axis_dim = input_shape[axis];
  const size_t output_axis_dim = output_shape[axis];
  const size_t inner_size = AxisInnerSize(input_shape, axis);
  const size_t outer_size = std::accumulate(input_shape.begin(), input_shape.begin() + axis,
                                            size_t(1), std::multiplies<size_t>());
  CountProjectFlops<ElemT>(outer_size * kept_degeneracies.size() * inner_size);
  for (size_t outer = 0; outer < outer_size; ++outer) {
    const size_t input_outer_base = outer * input_axis_dim * inner_size;
    const size_t output_outer_base = outer * output_axis_dim * inner_size;
    for (size_t out_d = 0; out_d < kept_degeneracies.size(); ++out_d) {
      const size_t input_base =
          input_outer_base + kept_degeneracies[out_d] * inner_size;
      const size_t output_base = output_outer_base + out_d * inner_size;
      std::copy(input_data + input_base,
                input_data + input_base + inner_size,
                output_data + output_base);
    }
  }
}

}  // namespace detail

/**
 * @brief Apply a diagonal axis operator out-of-place.
 *
 * Computes
 * \f[
 *   output = beta \cdot output + alpha \cdot D_{axis}(input)
 * \f]
 * where `D_axis` multiplies each selected-axis degeneracy slice by the
 * corresponding `AxisDiagonalOp::values` entry.
 *
 * If `beta == 0`, `output` is overwritten and initialized with the same index
 * and block topology as `input`.  If `beta != 0`, `output` must already have the
 * same index and block topology as `input`.
 *
 * @pre `input` is non-default and `axis < input.Rank()`.
 * @pre `op.index` has the same sector order and degeneracies as
 *      `input.GetIndex(axis)`; index direction is ignored.
 * @pre `input` and `output` do not alias.  Use ApplyAxisDiagonalInPlace() for
 *      in-place scaling.
 * @throws std::invalid_argument on invalid axes, index/layout mismatch, or
 *         input/output aliasing.
 * @throws std::runtime_error when used with GPU tensors, or when a tensor has
 *         stored blocks but no allocated raw data.
 */
template<typename ElemT, typename QNT>
void ApplyAxisDiagonal(
    const QLTensor<ElemT, QNT> &input,
    size_t axis,
    const AxisDiagonalOp<ElemT, QNT> &op,
    QLTensor<ElemT, QNT> &output,
    ElemT alpha = ElemT(1),
    ElemT beta = ElemT(0)
) {
#ifdef USE_GPU
  throw std::runtime_error("ApplyAxisDiagonal does not support GPU tensors yet.");
#else
  static_assert(!Fermionicable<QNT>::IsFermionic(),
                "ApplyAxisDiagonal is bosonic-only.");
  constexpr const char *kFunc = "ApplyAxisDiagonal";
  if (&input == &output) {
    throw std::invalid_argument(
        detail::AxisOpsPrefix(kFunc) +
        "input/output aliasing is not allowed; use ApplyAxisDiagonalInPlace."
    );
  }
  detail::RequireUsableAxis(input, axis, kFunc);
  detail::RequireDiagonalOpMatchesAxis(input.GetIndex(axis), op, kFunc);
  detail::RequireRawDataIfBlocksPresent(input, kFunc, "input");

  if (beta == ElemT(0)) {
    detail::InitializeOutputLikeInput(input, output);
  } else {
    detail::RequireSameLayout(input, output, kFunc);
    detail::RequireRawDataIfBlocksPresent(output, kFunc, "output");
    detail::ScaleTensorRawData(output, beta);
  }
  detail::RequireRawDataIfBlocksPresent(output, kFunc, "output");

  if (alpha == ElemT(0)) {
    return;
  }

  const auto &input_bsdt = input.GetBlkSparDataTen();
  auto &output_bsdt = output.GetBlkSparDataTen();
  const ElemT *input_data = input_bsdt.GetActualRawDataPtr();
  ElemT *output_data = output_bsdt.GetActualRawDataPtr();
  for (const auto &input_entry : input_bsdt.GetBlkIdxDataBlkMap()) {
    const auto &input_blk = input_entry.second;
    const auto &output_blk = output_bsdt.GetBlkIdxDataBlkMap().at(input_entry.first);
    const auto &values = op.values[input_blk.blk_coors[axis]];
    detail::AddAxisDiagonalBlock(
        input_data + input_blk.data_offset,
        output_data + output_blk.data_offset,
        input_blk.shape,
        axis,
        values,
        alpha
    );
  }
#endif
}

/**
 * @brief Project one tensor axis into a compact subspace.
 *
 * This topology-changing projection copies only selected degeneracy slices from
 * `input` into `output`.  Empty input sectors are removed from the output index,
 * non-empty sectors keep input sector order, and degeneracies follow the exact
 * order supplied in `projector.kept_degeneracies`.
 *
 * `output` is always overwritten and rebuilt with the compact index and block
 * topology.  This is the compact projector path that can reduce later generic
 * Contract() traversal work.
 *
 * @pre `input` is non-default and `axis < input.Rank()`.
 * @pre `projector.input_index` matches `input.GetIndex(axis)` ignoring direction.
 * @pre `projector.kept_degeneracies.size()` equals the number of input sectors.
 * @pre At least one degeneracy is kept globally.
 * @pre Every kept offset is in range and unique within its sector.
 * @pre `input` and `output` do not alias.
 * @throws std::invalid_argument on invalid axes, projector shape/offsets, or
 *         input/output aliasing.
 * @throws std::runtime_error when used with GPU tensors, or when `input` has
 *         stored blocks but no allocated raw data.
 */
template<typename ElemT, typename QNT>
void ProjectAxis(
    const QLTensor<ElemT, QNT> &input,
    size_t axis,
    const AxisProjector<QNT> &projector,
    QLTensor<ElemT, QNT> &output
) {
#ifdef USE_GPU
  throw std::runtime_error("ProjectAxis does not support GPU tensors yet.");
#else
  static_assert(!Fermionicable<QNT>::IsFermionic(),
                "ProjectAxis is bosonic-only.");
  constexpr const char *kFunc = "ProjectAxis";
  if (&input == &output) {
    throw std::invalid_argument(
        detail::AxisOpsPrefix(kFunc) + "input/output aliasing is not allowed."
    );
  }
  detail::RequireUsableAxis(input, axis, kFunc);
  detail::RequireRawDataIfBlocksPresent(input, kFunc, "input");
  const auto input_sector_to_output_sector =
      detail::ValidateProjectorAndBuildSectorMap(input.GetIndex(axis),
                                                 projector,
                                                 kFunc);
  const auto output_indexes = detail::ProjectedOutputIndexes(input, axis, projector);
  const auto output_blk_coors_s =
      detail::GenerateProjectedOutputBlockCoors(input,
                                                axis,
                                                input_sector_to_output_sector,
                                                output_indexes);

  output = QLTensor<ElemT, QNT>(output_indexes);
  if (!output_blk_coors_s.empty()) {
    output.GetBlkSparDataTen().DataBlksInsert(output_blk_coors_s, true, true);
  }
  detail::RequireRawDataIfBlocksPresent(output, kFunc, "output");

  const auto &input_bsdt = input.GetBlkSparDataTen();
  auto &output_bsdt = output.GetBlkSparDataTen();
  const ElemT *input_data = input_bsdt.GetActualRawDataPtr();
  ElemT *output_data = output_bsdt.GetActualRawDataPtr();
  const auto &output_blocks = output_bsdt.GetBlkIdxDataBlkMap();

  for (const auto &entry : input_bsdt.GetBlkIdxDataBlkMap()) {
    const auto &input_blk = entry.second;
    const size_t output_sector =
        input_sector_to_output_sector[input_blk.blk_coors[axis]];
    if (output_sector == detail::DeletedSectorMarker()) {
      continue;
    }
    CoorsT output_blk_coors = input_blk.blk_coors;
    output_blk_coors[axis] = output_sector;
    const auto output_blk_idx = output_bsdt.BlkCoorsToBlkIdx(output_blk_coors);
    const auto &output_blk = output_blocks.at(output_blk_idx);
    detail::CopyProjectedAxisBlock(
        input_data + input_blk.data_offset,
        output_data + output_blk.data_offset,
        input_blk.shape,
        output_blk.shape,
        axis,
        projector.kept_degeneracies[input_blk.blk_coors[axis]]
    );
  }
#endif
}

/**
 * @brief Apply a diagonal axis operator in-place.
 *
 * This function scales or zeroes stored raw-data slices without changing the
 * tensor's index set or block topology.  Projector-like 0/1 operators therefore
 * leave zeroed slices inside existing dense blocks; a later generic Contract()
 * will still traverse those dense blocks.
 *
 * @pre `tensor` is non-default and `axis < tensor.Rank()`.
 * @pre `op.index` has the same sector order and degeneracies as
 *      `tensor.GetIndex(axis)`; index direction is ignored.
 * @throws std::invalid_argument on invalid axes or index mismatch.
 * @throws std::runtime_error when used with GPU tensors, or when `tensor` has
 *         stored blocks but no allocated raw data.
 */
template<typename ElemT, typename QNT>
void ApplyAxisDiagonalInPlace(
    QLTensor<ElemT, QNT> &tensor,
    size_t axis,
    const AxisDiagonalOp<ElemT, QNT> &op
) {
#ifdef USE_GPU
  throw std::runtime_error("ApplyAxisDiagonalInPlace does not support GPU tensors yet.");
#else
  static_assert(!Fermionicable<QNT>::IsFermionic(),
                "ApplyAxisDiagonalInPlace is bosonic-only.");
  constexpr const char *kFunc = "ApplyAxisDiagonalInPlace";
  detail::RequireUsableAxis(tensor, axis, kFunc);
  detail::RequireDiagonalOpMatchesAxis(tensor.GetIndex(axis), op, kFunc);
  detail::RequireRawDataIfBlocksPresent(tensor, kFunc, "tensor");

  auto &bsdt = tensor.GetBlkSparDataTen();
  ElemT *data = bsdt.GetActualRawDataPtr();
  for (const auto &entry : bsdt.GetBlkIdxDataBlkMap()) {
    const auto &block = entry.second;
    const auto &values = op.values[block.blk_coors[axis]];
    detail::ScaleAxisDiagonalBlockInPlace(
        data + block.data_offset,
        block.shape,
        axis,
        values
    );
  }
#endif
}

/**
 * @brief Scale each stored block according to the sector on one axis.
 *
 * This is a whole-block operation: for each stored data block, only the block
 * coordinate on `scale.axis` is inspected, and the entire block is multiplied by
 * `scale.values_by_sector[sector]`.
 *
 * @pre `tensor` is non-default and `scale.axis < tensor.Rank()`.
 * @pre `scale.values_by_sector.size()` equals the number of sectors of the
 *      selected axis.
 * @throws std::invalid_argument on invalid axes or sector-count mismatch.
 * @throws std::runtime_error when used with GPU tensors, or when `tensor` has
 *         stored blocks but no allocated raw data.
 */
template<typename ElemT, typename QNT>
void ScaleAxisSectorsInPlace(
    QLTensor<ElemT, QNT> &tensor,
    const AxisSectorScalar<ElemT, QNT> &scale
) {
#ifdef USE_GPU
  throw std::runtime_error("ScaleAxisSectorsInPlace does not support GPU tensors yet.");
#else
  static_assert(!Fermionicable<QNT>::IsFermionic(),
                "ScaleAxisSectorsInPlace is bosonic-only.");
  constexpr const char *kFunc = "ScaleAxisSectorsInPlace";
  detail::RequireUsableAxis(tensor, scale.axis, kFunc);
  const auto &axis_index = tensor.GetIndex(scale.axis);
  if (scale.values_by_sector.size() != axis_index.GetQNSctNum()) {
    throw std::invalid_argument(
        detail::AxisOpsPrefix(kFunc) +
        "values_by_sector size must match the number of sectors."
    );
  }
  detail::RequireRawDataIfBlocksPresent(tensor, kFunc, "tensor");

  auto &bsdt = tensor.GetBlkSparDataTen();
  ElemT *data = bsdt.GetActualRawDataPtr();
  for (const auto &entry : bsdt.GetBlkIdxDataBlkMap()) {
    const auto &block = entry.second;
    detail::ScaleRawData(
        data + block.data_offset,
        block.size,
        scale.values_by_sector[block.blk_coors[scale.axis]]
    );
  }
#endif
}

/**
 * @brief Apply a monomial axis operator out-of-place.
 *
 * Computes
 * \f[
 *   output = beta \cdot output + alpha \cdot M_{axis}(input)
 * \f]
 * where `M_axis` scatters each listed input degeneracy slice to the listed
 * output degeneracy slice.  Multiple entries may write the same output slice
 * and are accumulated.
 *
 * If `beta == 0`, `output` is overwritten and initialized with the generated
 * output index and block topology.  If `beta != 0`, `output` must already have
 * exactly that generated topology.
 *
 * @pre `input` is non-default and `axis < input.Rank()`.
 * @pre `op.input_index` matches `input.GetIndex(axis)` ignoring direction.
 * @pre `op.output_index.GetDir() == input.GetIndex(axis).GetDir()`.
 * @pre Each entry is in range and satisfies
 *      `output_qn == input_qn + op.flux`.
 * @pre `input` and `output` do not alias.
 * @throws std::invalid_argument on invalid axes, entries, topology mismatch, or
 *         input/output aliasing.
 * @throws std::runtime_error when used with GPU tensors, or when a tensor has
 *         stored blocks but no allocated raw data.
 */
template<typename ElemT, typename QNT>
void ApplyAxisMonomial(
    const QLTensor<ElemT, QNT> &input,
    size_t axis,
    const AxisMonomialOp<ElemT, QNT> &op,
    QLTensor<ElemT, QNT> &output,
    ElemT alpha = ElemT(1),
    ElemT beta = ElemT(0)
) {
#ifdef USE_GPU
  throw std::runtime_error("ApplyAxisMonomial does not support GPU tensors yet.");
#else
  static_assert(!Fermionicable<QNT>::IsFermionic(),
                "ApplyAxisMonomial is bosonic-only.");
  constexpr const char *kFunc = "ApplyAxisMonomial";
  if (&input == &output) {
    throw std::invalid_argument(
        detail::AxisOpsPrefix(kFunc) +
        "input/output aliasing is not allowed."
    );
  }
  detail::RequireUsableAxis(input, axis, kFunc);
  detail::RequireMonomialOpMatchesAxis(input.GetIndex(axis), op, kFunc);
  detail::RequireRawDataIfBlocksPresent(input, kFunc, "input");

  const auto output_indexes = detail::MonomialOutputIndexes(input, axis, op);
  const auto output_blk_coors_s =
      detail::GenerateMonomialOutputBlockCoors(input, axis, op, output_indexes);

  if (beta == ElemT(0)) {
    output = QLTensor<ElemT, QNT>(output_indexes);
    if (!output_blk_coors_s.empty()) {
      output.GetBlkSparDataTen().DataBlksInsert(output_blk_coors_s, true, true);
    }
  } else {
    if (output.IsDefault()) {
      throw std::invalid_argument(
          detail::AxisOpsPrefix(kFunc) +
          "output must not be default when beta is nonzero."
      );
    }
    if (output.GetIndexes() != output_indexes ||
        !detail::BlockTopologyMatchesCoors(output, output_blk_coors_s)) {
      throw std::invalid_argument(
          detail::AxisOpsPrefix(kFunc) +
          "output must have the generated index and block topology when beta "
          "is nonzero."
      );
    }
    detail::RequireRawDataIfBlocksPresent(output, kFunc, "output");
    detail::ScaleTensorRawData(output, beta);
  }

  if (alpha == ElemT(0) || op.entries.empty() || output_blk_coors_s.empty()) {
    return;
  }
  detail::RequireRawDataIfBlocksPresent(output, kFunc, "output");

  const auto &input_bsdt = input.GetBlkSparDataTen();
  auto &output_bsdt = output.GetBlkSparDataTen();
  const ElemT *input_data = input_bsdt.GetActualRawDataPtr();
  ElemT *output_data = output_bsdt.GetActualRawDataPtr();
  const auto &output_blocks = output_bsdt.GetBlkIdxDataBlkMap();

  for (const auto &input_entry : input_bsdt.GetBlkIdxDataBlkMap()) {
    const auto &input_blk = input_entry.second;
    for (const auto &op_entry : op.entries) {
      if (op_entry.in_sector != input_blk.blk_coors[axis]) {
        continue;
      }
      CoorsT output_blk_coors = input_blk.blk_coors;
      output_blk_coors[axis] = op_entry.out_sector;
      const size_t output_blk_idx = output_bsdt.BlkCoorsToBlkIdx(output_blk_coors);
      const auto &output_blk = output_blocks.at(output_blk_idx);
      detail::AddAxisMonomialEntryBlock(
          input_data + input_blk.data_offset,
          output_data + output_blk.data_offset,
          input_blk.shape,
          output_blk.shape,
          axis,
          op_entry.in_offset,
          op_entry.out_offset,
          alpha * op_entry.coef
      );
    }
  }
#endif
}

/**
 * @brief Apply a rank-2 operator to one tensor axis without reordering axes.
 *
 * The operator is interpreted in the standard TensorToolkit matrix layout
 * `{input_index, output_index}`.  `rank2_op.GetIndex(0)` is contracted with
 * `input.GetIndex(target_axis)`, and the output tensor keeps the original
 * state-axis order with only `target_axis` replaced by
 * `rank2_op.GetIndex(1)`.
 *
 * `out` is always overwritten.  The implementation writes output blocks
 * directly in the final axis order and does not perform full-state layout
 * copies or transposes.
 *
 * @pre `input` is non-default and `target_axis < input.Rank()`.
 * @pre `rank2_op` is non-default, has rank 2, and
 *      `rank2_op.GetIndex(0) == InverseIndex(input.GetIndex(target_axis))`.
 * @pre `out` does not alias `input` or `rank2_op`.
 * @throws std::invalid_argument on invalid axes, rank/index mismatch, or
 *         output aliasing.
 * @throws std::runtime_error when used with GPU tensors, or when a tensor has
 *         stored blocks but no allocated raw data.
 */
template<typename ElemT, typename QNT>
void ApplyRank2ToAxisPreserveOrder(
    const QLTensor<ElemT, QNT> &input,
    const QLTensor<ElemT, QNT> &rank2_op,
    size_t target_axis,
    QLTensor<ElemT, QNT> &out,
    Rank2AxisApplyStats *stats = nullptr
) {
#ifdef USE_GPU
  throw std::runtime_error(
      "ApplyRank2ToAxisPreserveOrder does not support GPU tensors yet.");
#else
  static_assert(!Fermionicable<QNT>::IsFermionic(),
                "ApplyRank2ToAxisPreserveOrder is bosonic-only.");
  constexpr const char *kFunc = "ApplyRank2ToAxisPreserveOrder";
  if (stats != nullptr) {
    *stats = Rank2AxisApplyStats{};
  }
  if (&input == &out || &rank2_op == &out) {
    throw std::invalid_argument(
        detail::AxisOpsPrefix(kFunc) + "output aliasing is not allowed."
    );
  }
  detail::RequireUsableAxis(input, target_axis, kFunc);
  detail::RequireRank2OpMatchesAxis(input.GetIndex(target_axis),
                                    rank2_op,
                                    kFunc);
  detail::RequireRawDataIfBlocksPresent(input, kFunc, "input");
  detail::RequireRawDataIfBlocksPresent(rank2_op, kFunc, "rank2_op");

  const auto output_indexes =
      detail::Rank2OutputIndexes(input, target_axis, rank2_op);
  const auto output_blk_coors_s =
      detail::GenerateRank2OutputBlockCoors(input,
                                            target_axis,
                                            rank2_op,
                                            output_indexes);
  out = QLTensor<ElemT, QNT>(output_indexes);
  if (!output_blk_coors_s.empty()) {
    out.GetBlkSparDataTen().DataBlksInsert(output_blk_coors_s, true, true);
  }
  if (output_blk_coors_s.empty()) {
    return;
  }
  detail::RequireRawDataIfBlocksPresent(out, kFunc, "out");

  const auto &input_bsdt = input.GetBlkSparDataTen();
  const auto &op_bsdt = rank2_op.GetBlkSparDataTen();
  auto &output_bsdt = out.GetBlkSparDataTen();
  const ElemT *input_data = input_bsdt.GetActualRawDataPtr();
  const ElemT *op_data = op_bsdt.GetActualRawDataPtr();
  ElemT *output_data = output_bsdt.GetActualRawDataPtr();
  const auto &output_blocks = output_bsdt.GetBlkIdxDataBlkMap();

  for (const auto &input_entry : input_bsdt.GetBlkIdxDataBlkMap()) {
    const auto &input_blk = input_entry.second;
    for (const auto &op_entry : op_bsdt.GetBlkIdxDataBlkMap()) {
      const auto &op_blk = op_entry.second;
      if (op_blk.blk_coors[0] != input_blk.blk_coors[target_axis]) {
        continue;
      }
      CoorsT output_blk_coors = input_blk.blk_coors;
      output_blk_coors[target_axis] = op_blk.blk_coors[1];
      const size_t output_blk_idx = output_bsdt.BlkCoorsToBlkIdx(output_blk_coors);
      const auto &output_blk = output_blocks.at(output_blk_idx);
      detail::AddRank2AxisBlock(
          input_data + input_blk.data_offset,
          op_data + op_blk.data_offset,
          output_data + output_blk.data_offset,
          input_blk.shape,
          op_blk.shape,
          output_blk.shape,
          target_axis
      );
    }
  }
#endif
}

/**
 * @brief Move tensor axis 0 to the last axis by positional axis order.
 *
 * The output layout is
 * `{input.GetIndex(1), ..., input.GetIndex(input.Rank() - 1),
 * input.GetIndex(0)}`.  This is equivalent to copying `input` into `out` and
 * calling `out.Transpose({1, 2, ..., rank - 1, 0})`, but this implementation
 * writes each dense block through a `d0 x (d1 * ... * dn)` matrix transpose
 * instead of invoking the generic tensor transpose path.
 *
 * The mapping is positional, not index-name based; duplicate indexes are valid.
 * This is a layout/data-movement operation and does not increment FLOP counters.
 *
 * @pre `input` is non-default and `input.Rank() > 0`.
 * @pre `out` does not alias `input`.
 * @post `out` is overwritten.
 * @throws std::invalid_argument on default input, rank-0 input, or output
 *         aliasing.
 * @throws std::runtime_error when used with GPU tensors, or when a tensor has
 *         stored blocks but no allocated raw data.
 */
template<typename ElemT, typename QNT>
void MoveHeadAxisToTail(
    const QLTensor<ElemT, QNT> &input,
    QLTensor<ElemT, QNT> &out,
    AxisLayoutMoveStats *stats = nullptr
) {
#ifdef USE_GPU
  throw std::runtime_error("MoveHeadAxisToTail does not support GPU tensors yet.");
#else
  static_assert(!Fermionicable<QNT>::IsFermionic(),
                "MoveHeadAxisToTail is bosonic-only.");
  constexpr const char *kFunc = "MoveHeadAxisToTail";
  if (stats != nullptr) {
    *stats = AxisLayoutMoveStats{};
  }
  if (input.IsDefault()) {
    throw std::invalid_argument(
        detail::AxisOpsPrefix(kFunc) + "input tensor must not be default."
    );
  }
  if (input.Rank() == 0) {
    throw std::invalid_argument(
        detail::AxisOpsPrefix(kFunc) + "input rank must be positive."
    );
  }
  if (&input == &out) {
    throw std::invalid_argument(
        detail::AxisOpsPrefix(kFunc) + "input/output aliasing is not allowed."
    );
  }
  detail::RequireRawDataIfBlocksPresent(input, kFunc, "input");

  if (input.Rank() == 1) {
    out = input;
    if (stats != nullptr) {
      stats->layout_gb += detail::RawReadWriteGb<ElemT>(
          input.GetBlkSparDataTen().GetActualRawDataSize());
    }
    return;
  }

  const auto output_indexes =
      detail::MoveHeadAxisToTailIndexes(input.GetIndexes());
  const auto &input_blocks = input.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  std::vector<CoorsT> output_blk_coors_s;
  output_blk_coors_s.reserve(input_blocks.size());
  for (const auto &entry : input_blocks) {
    output_blk_coors_s.push_back(
        detail::MoveHeadAxisToTailCoors(entry.second.blk_coors));
  }

  out = QLTensor<ElemT, QNT>(output_indexes);
  if (!output_blk_coors_s.empty()) {
    out.GetBlkSparDataTen().DataBlksInsert(output_blk_coors_s, true, true);
  }
  if (output_blk_coors_s.empty()) {
    return;
  }
  detail::RequireRawDataIfBlocksPresent(out, kFunc, "out");

  const auto &input_bsdt = input.GetBlkSparDataTen();
  auto &output_bsdt = out.GetBlkSparDataTen();
  const ElemT *input_data = input_bsdt.GetActualRawDataPtr();
  ElemT *output_data = output_bsdt.GetActualRawDataPtr();
  const auto &output_blocks = output_bsdt.GetBlkIdxDataBlkMap();

  for (const auto &entry : input_blocks) {
    const auto &input_blk = entry.second;
    const auto output_blk_coors =
        detail::MoveHeadAxisToTailCoors(input_blk.blk_coors);
    const size_t output_blk_idx = output_bsdt.BlkCoorsToBlkIdx(output_blk_coors);
    const auto &output_blk = output_blocks.at(output_blk_idx);
    detail::MoveHeadAxisBlockToTail(
        input_data + input_blk.data_offset,
        output_data + output_blk.data_offset,
        input_blk.shape
    );
    if (stats != nullptr) {
      stats->block_matrix_transpose_calls++;
      stats->layout_gb += detail::RawReadWriteGb<ElemT>(input_blk.size);
    }
  }
#endif
}

}  // namespace qlten::dmrg

#endif /* ifndef QLTEN_TENSOR_MANIPULATION_DMRG_AXIS_OPS_H */
