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
#include "qlten/framework/mem_ops.h"      // QLMemset
#include "qlten/framework/hp_numeric/blas_level1.h"      // VectorAddTo, VectorCopy, VectorScale
#include "qlten/framework/hp_numeric/blas_level3.h"      // MatMultiply
#include "qlten/framework/hp_numeric/blas_extensions.h"  // MatrixTransposeBatch
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
 * @brief Lightweight work counters for DMRG axis primitives.
 *
 * These counters are caller-owned and optional.  They report high-level work
 * that matters to preserve-order DMRG paths without changing the hot path when
 * the pointer is null.
 */
struct AxisOpStats {
  /// Number of output tensor topology rebuilds.
  size_t output_tensor_rebuilds = 0;
  /// Raw-data bytes copied by explicit copy helpers.
  size_t raw_data_copy_bytes = 0;
  /// Number of contraction-style transpose prepare calls.
  size_t transpose_prepare_calls = 0;
  /// Raw-data bytes moved by contraction-style transpose prepare calls.
  size_t transpose_prepare_bytes = 0;
  /// Number of full-state layout copy calls.
  size_t layout_copy_calls = 0;
  /// Number of full-state layout transpose calls.
  size_t layout_transpose_calls = 0;
  /// Full-state layout traffic in GiB.
  double layout_gb = 0.0;
  /// Number of BLAS GEMM calls issued by rank-2 axis application.
  size_t gemm_calls = 0;
  /// Sum of GEMM m dimensions.
  size_t gemm_m_total = 0;
  /// Sum of GEMM n dimensions.
  size_t gemm_n_total = 0;
  /// Sum of GEMM k dimensions.
  size_t gemm_k_total = 0;
  /// Number of GEMM calls using a head/tail boundary-axis fast path.
  size_t boundary_axis_fast_path_hits = 0;
  /// Number of batch-GEMM dispatch attempts for middle-axis rank-2 apply.
  size_t batch_gemm_calls = 0;
  /// Number of logical GEMM items covered by batch-GEMM dispatch attempts.
  size_t batched_gemm_items = 0;
  /// Number of batch-GEMM dispatch attempts that fell back to GEMM loops.
  size_t batch_gemm_fallback_calls = 0;
  /// Temporary numeric workspace bytes used to prepare batch GEMM operands.
  size_t batch_gemm_workspace_bytes = 0;
  /// Number of rank-2 topology lookups that used the op-sector index.
  size_t rank2_sector_index_hits = 0;
  /// Number of rank-2 input/op block pairs visited after sector filtering.
  size_t rank2_topology_block_pair_visits = 0;
  /// Number of direct scaled-copy calls.
  size_t direct_scaled_copy_calls = 0;
  /// Number of output element updates in the fused two-rank2 kernel.
  size_t fused_element_updates = 0;
  /// Number of calls that scaled an owned tensor in place.
  size_t in_place_scale_hits = 0;
  /// Number of truly fused two-rank2 calls.
  size_t two_rank2_fused_hits = 0;
  /// Number of two-rank2 calls executed through block-local GEMM.
  size_t two_rank2_block_gemm_hits = 0;
  /// Temporary intermediate block workspace bytes used by two-rank2 GEMM.
  size_t two_rank2_block_workspace_bytes = 0;
  /// Number of intermediate tensor rebuilds in two-rank2 calls.
  size_t two_rank2_intermediate_rebuilds = 0;
  /// Number of rank2+monomial calls using the 1D-sector direct write path.
  size_t rank2_monomial_direct_hits = 0;
  /// Temporary block workspace bytes used by the general rank2+monomial path.
  size_t rank2_monomial_block_workspace_bytes = 0;
};

using Rank2AxisApplyStats = AxisOpStats;

/**
 * @brief GEMM dispatch strategy for middle-axis rank-2 application.
 *
 * Boundary axes already use a single contiguous GEMM.  `kBatch` affects only
 * non-boundary axes where the contraction naturally decomposes into independent
 * GEMMs over the outer slices.  Backends without a safe batch-GEMM primitive
 * keep the old GEMM-loop kernel and report the fallback in `AxisOpStats`.
 */
enum class Rank2AxisApplyGemmMode {
  /// Use the legacy loop over independent GEMMs.
  kLoop,
  /// Try backend batch GEMM for independent middle-axis GEMMs.
  kBatch
};

/**
 * @brief Storage ownership mode for preserve-order axis scaling helpers.
 */
enum class AxisApplyStorageMode {
  /// Borrowed const input: rebuild output and apply out of place.
  kOutOfPlace,
  /// Owned mutable input: require input/output aliasing and scale in place.
  kConsumeOwnedInPlace
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
constexpr bool IsHpNumericElem() {
  using BareElemT = typename std::remove_cv<ElemT>::type;
  return std::is_same<BareElemT, QLTEN_Double>::value ||
         std::is_same<BareElemT, QLTEN_Float>::value ||
         std::is_same<BareElemT, QLTEN_Complex>::value ||
         std::is_same<BareElemT, QLTEN_ComplexFloat>::value;
}

inline auto BlasNoTrans() {
#ifdef USE_GPU
  return CUBLAS_OP_N;
#else
  return CblasNoTrans;
#endif
}

inline auto BlasTrans() {
#ifdef USE_GPU
  return CUBLAS_OP_T;
#else
  return CblasTrans;
#endif
}

template<typename ElemT>
constexpr size_t ScaleFlopCost() {
  return IsComplexElem<ElemT>() ? 6 : 1;
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
void CountProjectFlops(const size_t size) {
#ifdef QLTEN_COUNT_FLOPS
  flop += ProjectFlopCost<ElemT>() * size;
#else
  (void) size;
#endif
}

template<typename ElemT>
void MatrixTransposeBlocks(
    const std::vector<const ElemT *> &input_ptrs,
    const std::vector<ElemT *> &output_ptrs,
    const std::vector<size_t> &rows,
    const std::vector<size_t> &cols
) {
  static_assert(IsHpNumericElem<ElemT>(),
                "DMRG axis operations require a hp_numeric-supported ElemT.");
  if (input_ptrs.empty()) {
    return;
  }
#ifdef USE_GPU
  for (size_t i = 0; i < input_ptrs.size(); ++i) {
    hp_numeric::MatrixTranspose(input_ptrs[i], rows[i], cols[i], output_ptrs[i]);
  }
#else
  auto input_ptr_array = input_ptrs;
  auto output_ptr_array = output_ptrs;
  hp_numeric::MatrixTransposeBatch(input_ptr_array.data(),
                                   output_ptr_array.data(),
                                   rows.data(),
                                   cols.data(),
                                   input_ptr_array.size());
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

template<typename QNT>
size_t OutputBlockTopologyCapacity(const IndexVec<QNT> &indexes) {
  const ShapeT blk_shape = CalcQNSctNumOfIdxs(indexes);
  return std::accumulate(blk_shape.begin(),
                         blk_shape.end(),
                         size_t(1),
                         std::multiplies<size_t>());
}

struct OutputBlockTopology {
  std::vector<size_t> blk_idxs;
  std::vector<CoorsT> blk_coors_s;
};

template<typename QNT>
void ReserveOutputBlockTopology(
    OutputBlockTopology &topology,
    const IndexVec<QNT> &output_indexes,
    const size_t candidate_count
) {
  const size_t capacity = OutputBlockTopologyCapacity(output_indexes);
  const size_t reserve_size =
      candidate_count < capacity ? candidate_count : capacity;
  topology.blk_idxs.reserve(reserve_size);
  topology.blk_coors_s.reserve(reserve_size);
}

template<typename ElemT, typename QNT>
OutputBlockTopology InputLikeOutputBlockTopology(
    const QLTensor<ElemT, QNT> &input
) {
  OutputBlockTopology topology;
  const auto &input_blocks = input.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  topology.blk_idxs.reserve(input_blocks.size());
  topology.blk_coors_s.reserve(input_blocks.size());
  for (const auto &entry : input_blocks) {
    topology.blk_idxs.push_back(entry.first);
    topology.blk_coors_s.push_back(entry.second.blk_coors);
  }
  return topology;
}

template<typename ElemT, typename QNT>
OutputBlockTopology GenerateProjectedOutputBlockTopology(
    const QLTensor<ElemT, QNT> &input,
    const size_t axis,
    const std::vector<size_t> &input_sector_to_output_sector,
    const IndexVec<QNT> &output_indexes
) {
  std::set<size_t> seen_blk_idxs;
  OutputBlockTopology topology;
  const auto &input_blocks = input.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  ReserveOutputBlockTopology(topology, output_indexes, input_blocks.size());
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
      topology.blk_idxs.push_back(output_blk_idx);
      topology.blk_coors_s.push_back(std::move(output_blk_coors));
    }
  }
  return topology;
}

template<typename ElemT, typename QNT>
std::vector<CoorsT> GenerateProjectedOutputBlockCoors(
    const QLTensor<ElemT, QNT> &input,
    const size_t axis,
    const std::vector<size_t> &input_sector_to_output_sector,
    const IndexVec<QNT> &output_indexes
) {
  return GenerateProjectedOutputBlockTopology(
      input, axis, input_sector_to_output_sector, output_indexes).blk_coors_s;
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
  const auto topology = InputLikeOutputBlockTopology(input);
  if (!topology.blk_coors_s.empty()) {
    output.GetBlkSparDataTen().DataBlksInsert(topology.blk_idxs,
                                              topology.blk_coors_s,
                                              true,
                                              true);
  }
}

template<typename ElemT>
void ScaleRawData(ElemT *data, const size_t size, const ElemT scale) {
  static_assert(IsHpNumericElem<ElemT>(),
                "DMRG axis operations require a hp_numeric-supported ElemT.");
  if (size == 0 || scale == ElemT(1)) {
    return;
  }
  if (scale == ElemT(0)) {
    CountScaleFlops<ElemT>(size);
    QLMemset(data, 0, size * sizeof(ElemT));
    return;
  }
  hp_numeric::VectorScale(data, size, scale);
}

template<typename ElemT, typename QNT>
void ScaleTensorRawData(QLTensor<ElemT, QNT> &tensor, const ElemT scale) {
  auto &bsdt = tensor.GetBlkSparDataTen();
  ScaleRawData(bsdt.GetActualRawDataPtr(), bsdt.GetActualRawDataSize(), scale);
}

template<typename ElemT>
void AddRawDataTo(
    const ElemT *input_data,
    const size_t size,
    ElemT *output_data,
    const ElemT coef
) {
  static_assert(IsHpNumericElem<ElemT>(),
                "DMRG axis operations require a hp_numeric-supported ElemT.");
  if (size == 0 || coef == ElemT(0)) {
    return;
  }
  hp_numeric::VectorAddTo(input_data, size, output_data, coef);
}

template<typename ElemT>
void CopyRawData(
    const ElemT *input_data,
    const size_t size,
    ElemT *output_data
) {
  static_assert(IsHpNumericElem<ElemT>(),
                "DMRG axis operations require a hp_numeric-supported ElemT.");
  if (size == 0) {
    return;
  }
  hp_numeric::VectorCopy(input_data, size, output_data);
}

template<typename ElemT>
void CopyScaleRawData(
    const ElemT *input_data,
    const size_t size,
    ElemT *output_data,
    const ElemT scale
) {
  static_assert(IsHpNumericElem<ElemT>(),
                "DMRG axis operations require a hp_numeric-supported ElemT.");
  if (size == 0) {
    return;
  }
  if (scale == ElemT(0)) {
    CountScaleFlops<ElemT>(size);
    QLMemset(output_data, 0, size * sizeof(ElemT));
    return;
  }
  if (scale == ElemT(1)) {
    CopyRawData(input_data, size, output_data);
    return;
  }
  hp_numeric::VectorScaleCopy(input_data, size, output_data, scale);
}

template<typename ElemT, typename QNT>
void CopyTensorRawData(
    const QLTensor<ElemT, QNT> &input,
    QLTensor<ElemT, QNT> &output
) {
  const auto &input_bsdt = input.GetBlkSparDataTen();
  auto &output_bsdt = output.GetBlkSparDataTen();
  CopyRawData(input_bsdt.GetActualRawDataPtr(),
              input_bsdt.GetActualRawDataSize(),
              output_bsdt.GetActualRawDataPtr());
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
    AddRawDataTo(input_data, block_size, output_data, coef);
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
      AddRawDataTo(input_data + base, inner_size, output_data + base, coef);
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
    QLMemset(data, 0, block_size * sizeof(ElemT));
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
        QLMemset(data + base, 0, inner_size * sizeof(ElemT));
      } else {
        ScaleRawData(data + base, inner_size, coef);
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
void RequireAxisSectorScalarMatchesOutput(
    const IndexVec<QNT> &output_indexes,
    const AxisSectorScalar<ElemT, QNT> &scale,
    const char *func
) {
  if (scale.axis >= output_indexes.size()) {
    throw std::invalid_argument(
        AxisOpsPrefix(func) + "scale axis is out of range."
    );
  }
  if (scale.values_by_sector.size() !=
      output_indexes[scale.axis].GetQNSctNum()) {
    throw std::invalid_argument(
        AxisOpsPrefix(func) +
        "values_by_sector size must match the number of output sectors."
    );
  }
}

template<typename ElemT, typename QNT>
std::vector<std::vector<size_t>> Rank2OpBlockIndicesByInputSector(
    const QLTensor<ElemT, QNT> &rank2_op
) {
  const size_t input_sector_count = rank2_op.GetIndex(0).GetQNSctNum();
  std::vector<std::vector<size_t>> op_blk_idxs_by_input_sector(
      input_sector_count);
  const auto &op_blocks = rank2_op.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  std::vector<size_t> op_blk_counts_by_input_sector(input_sector_count, 0);
  for (const auto &op_entry : op_blocks) {
    const auto &op_blk = op_entry.second;
    ++op_blk_counts_by_input_sector[op_blk.blk_coors[0]];
  }
  for (size_t sector = 0; sector < input_sector_count; ++sector) {
    op_blk_idxs_by_input_sector[sector].reserve(
        op_blk_counts_by_input_sector[sector]);
  }
  for (const auto &op_entry : op_blocks) {
    const auto &op_blk = op_entry.second;
    op_blk_idxs_by_input_sector[op_blk.blk_coors[0]].push_back(op_entry.first);
  }
  return op_blk_idxs_by_input_sector;
}

template<typename ElemT, typename QNT>
OutputBlockTopology GenerateRank2OutputBlockTopology(
    const QLTensor<ElemT, QNT> &input,
    const size_t axis,
    const QLTensor<ElemT, QNT> &rank2_op,
    const IndexVec<QNT> &output_indexes,
    AxisOpStats *stats = nullptr
) {
  std::set<size_t> seen_blk_idxs;
  OutputBlockTopology topology;
  const auto &input_blocks = input.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  const auto &op_blocks = rank2_op.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  const auto op_blk_idxs_by_input_sector =
      Rank2OpBlockIndicesByInputSector(rank2_op);
  size_t candidate_count = 0;
  for (const auto &input_entry : input_blocks) {
    const auto &input_blk = input_entry.second;
    candidate_count +=
        op_blk_idxs_by_input_sector[input_blk.blk_coors[axis]].size();
  }
  ReserveOutputBlockTopology(topology, output_indexes, candidate_count);
  for (const auto &input_entry : input_blocks) {
    const auto &input_blk = input_entry.second;
    const auto &matching_op_blk_idxs =
        op_blk_idxs_by_input_sector[input_blk.blk_coors[axis]];
    if (stats != nullptr && !matching_op_blk_idxs.empty()) {
      ++stats->rank2_sector_index_hits;
    }
    for (const size_t op_blk_idx : matching_op_blk_idxs) {
      const auto &op_blk = op_blocks.at(op_blk_idx);
      if (stats != nullptr) {
        ++stats->rank2_topology_block_pair_visits;
      }
      CoorsT output_blk_coors = input_blk.blk_coors;
      output_blk_coors[axis] = op_blk.blk_coors[1];
      const size_t output_blk_idx = BlkCoorsToBlkIdx(output_blk_coors, output_indexes);
      if (seen_blk_idxs.insert(output_blk_idx).second) {
        topology.blk_idxs.push_back(output_blk_idx);
        topology.blk_coors_s.push_back(std::move(output_blk_coors));
      }
    }
  }
  return topology;
}

template<typename ElemT, typename QNT>
std::vector<CoorsT> GenerateRank2OutputBlockCoors(
    const QLTensor<ElemT, QNT> &input,
    const size_t axis,
    const QLTensor<ElemT, QNT> &rank2_op,
    const IndexVec<QNT> &output_indexes,
    AxisOpStats *stats = nullptr
) {
  return GenerateRank2OutputBlockTopology(
      input, axis, rank2_op, output_indexes, stats).blk_coors_s;
}

template<typename ElemT, typename QNT>
bool AxisMonomialNonzeroEntriesAreOneDimensionalSectorMaps(
    const AxisMonomialOp<ElemT, QNT> &monomial
) {
  for (const auto &entry : monomial.entries) {
    if (entry.coef == ElemT(0)) {
      continue;
    }
    if (entry.in_offset != 0 || entry.out_offset != 0) {
      return false;
    }
    if (monomial.input_index.GetQNSct(entry.in_sector).GetDegeneracy() != 1 ||
        monomial.output_index.GetQNSct(entry.out_sector).GetDegeneracy() != 1) {
      return false;
    }
  }
  return true;
}

template<typename ElemT, typename QNT>
IndexVec<QNT> Rank2ThenMonomialOutputIndexes(
    const QLTensor<ElemT, QNT> &input,
    const size_t rank2_axis,
    const QLTensor<ElemT, QNT> &rank2_op,
    const size_t monomial_axis,
    const AxisMonomialOp<ElemT, QNT> &monomial
) {
  IndexVec<QNT> output_indexes =
      Rank2OutputIndexes(input, rank2_axis, rank2_op);
  output_indexes[monomial_axis] = monomial.output_index;
  return output_indexes;
}

template<typename ElemT, typename QNT>
OutputBlockTopology GenerateRank2ThenMonomialOutputBlockTopology(
    const QLTensor<ElemT, QNT> &input,
    const size_t rank2_axis,
    const QLTensor<ElemT, QNT> &rank2_op,
    const size_t monomial_axis,
    const AxisMonomialOp<ElemT, QNT> &monomial,
    const IndexVec<QNT> &output_indexes,
    AxisOpStats *stats = nullptr
) {
  std::set<size_t> seen_blk_idxs;
  OutputBlockTopology topology;
  const auto &input_blocks = input.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  const auto &op_blocks = rank2_op.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  const auto op_blk_idxs_by_input_sector =
      Rank2OpBlockIndicesByInputSector(rank2_op);
  size_t candidate_count = 0;
  for (const auto &input_entry : input_blocks) {
    const auto &input_blk = input_entry.second;
    candidate_count +=
        op_blk_idxs_by_input_sector[input_blk.blk_coors[rank2_axis]].size() *
        monomial.entries.size();
  }
  ReserveOutputBlockTopology(topology, output_indexes, candidate_count);
  for (const auto &input_entry : input_blocks) {
    const auto &input_blk = input_entry.second;
    const auto &matching_op_blk_idxs =
        op_blk_idxs_by_input_sector[input_blk.blk_coors[rank2_axis]];
    if (stats != nullptr && !matching_op_blk_idxs.empty()) {
      ++stats->rank2_sector_index_hits;
    }
    for (const size_t op_blk_idx : matching_op_blk_idxs) {
      const auto &op_blk = op_blocks.at(op_blk_idx);
      if (stats != nullptr) {
        ++stats->rank2_topology_block_pair_visits;
      }
      CoorsT rank2_output_blk_coors = input_blk.blk_coors;
      rank2_output_blk_coors[rank2_axis] = op_blk.blk_coors[1];
      for (const auto &monomial_entry : monomial.entries) {
        if (monomial_entry.in_sector !=
            rank2_output_blk_coors[monomial_axis]) {
          continue;
        }
        CoorsT output_blk_coors = rank2_output_blk_coors;
        output_blk_coors[monomial_axis] = monomial_entry.out_sector;
        const size_t output_blk_idx =
            BlkCoorsToBlkIdx(output_blk_coors, output_indexes);
        if (seen_blk_idxs.insert(output_blk_idx).second) {
          topology.blk_idxs.push_back(output_blk_idx);
          topology.blk_coors_s.push_back(std::move(output_blk_coors));
        }
      }
    }
  }
  return topology;
}

template<typename ElemT, typename QNT>
std::vector<CoorsT> GenerateRank2ThenMonomialOutputBlockCoors(
    const QLTensor<ElemT, QNT> &input,
    const size_t rank2_axis,
    const QLTensor<ElemT, QNT> &rank2_op,
    const size_t monomial_axis,
    const AxisMonomialOp<ElemT, QNT> &monomial,
    const IndexVec<QNT> &output_indexes,
    AxisOpStats *stats = nullptr
) {
  return GenerateRank2ThenMonomialOutputBlockTopology(
      input,
      rank2_axis,
      rank2_op,
      monomial_axis,
      monomial,
      output_indexes,
      stats).blk_coors_s;
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

template<typename ElemT, typename QNT>
OutputBlockTopology GenerateMoveHeadAxisToTailOutputBlockTopology(
    const QLTensor<ElemT, QNT> &input,
    const IndexVec<QNT> &output_indexes
) {
  OutputBlockTopology topology;
  const auto &input_blocks = input.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  ReserveOutputBlockTopology(topology, output_indexes, input_blocks.size());
  for (const auto &entry : input_blocks) {
    CoorsT output_blk_coors = MoveHeadAxisToTailCoors(entry.second.blk_coors);
    topology.blk_idxs.push_back(
        BlkCoorsToBlkIdx(output_blk_coors, output_indexes));
    topology.blk_coors_s.push_back(std::move(output_blk_coors));
  }
  return topology;
}

template<typename ElemT>
double RawReadWriteGb(const size_t elem_count) {
  return 2.0 * static_cast<double>(elem_count) *
         static_cast<double>(sizeof(ElemT)) / 1.0e9;
}

inline void AddGemmStats(
    AxisOpStats *stats,
    const size_t m,
    const size_t k,
    const size_t n,
    const bool boundary_fast_path
) {
  if (stats == nullptr) {
    return;
  }
  ++stats->gemm_calls;
  stats->gemm_m_total += m;
  stats->gemm_k_total += k;
  stats->gemm_n_total += n;
  if (boundary_fast_path) {
    ++stats->boundary_axis_fast_path_hits;
  }
}

inline void AddRepeatedGemmStats(
    AxisOpStats *stats,
    const size_t m,
    const size_t k,
    const size_t n,
    const size_t gemm_count,
    const bool boundary_fast_path
) {
  if (stats == nullptr) {
    return;
  }
  stats->gemm_calls += gemm_count;
  stats->gemm_m_total += gemm_count * m;
  stats->gemm_k_total += gemm_count * k;
  stats->gemm_n_total += gemm_count * n;
  if (boundary_fast_path) {
    stats->boundary_axis_fast_path_hits += gemm_count;
  }
}

inline void AddBatchGemmStats(
    AxisOpStats *stats,
    const size_t gemm_count,
    const bool backend_batch
) {
  if (stats == nullptr) {
    return;
  }
  ++stats->batch_gemm_calls;
  stats->batched_gemm_items += gemm_count;
  if (!backend_batch) {
    ++stats->batch_gemm_fallback_calls;
  }
}

inline void AddAxisOpStats(AxisOpStats *stats, const AxisOpStats &other) {
  if (stats == nullptr) {
    return;
  }
  stats->output_tensor_rebuilds += other.output_tensor_rebuilds;
  stats->raw_data_copy_bytes += other.raw_data_copy_bytes;
  stats->transpose_prepare_calls += other.transpose_prepare_calls;
  stats->transpose_prepare_bytes += other.transpose_prepare_bytes;
  stats->layout_copy_calls += other.layout_copy_calls;
  stats->layout_transpose_calls += other.layout_transpose_calls;
  stats->layout_gb += other.layout_gb;
  stats->gemm_calls += other.gemm_calls;
  stats->gemm_m_total += other.gemm_m_total;
  stats->gemm_n_total += other.gemm_n_total;
  stats->gemm_k_total += other.gemm_k_total;
  stats->boundary_axis_fast_path_hits += other.boundary_axis_fast_path_hits;
  stats->batch_gemm_calls += other.batch_gemm_calls;
  stats->batched_gemm_items += other.batched_gemm_items;
  stats->batch_gemm_fallback_calls += other.batch_gemm_fallback_calls;
  stats->batch_gemm_workspace_bytes += other.batch_gemm_workspace_bytes;
  stats->rank2_sector_index_hits += other.rank2_sector_index_hits;
  stats->rank2_topology_block_pair_visits +=
      other.rank2_topology_block_pair_visits;
  stats->direct_scaled_copy_calls += other.direct_scaled_copy_calls;
  stats->fused_element_updates += other.fused_element_updates;
  stats->in_place_scale_hits += other.in_place_scale_hits;
  stats->two_rank2_fused_hits += other.two_rank2_fused_hits;
  stats->two_rank2_block_gemm_hits += other.two_rank2_block_gemm_hits;
  stats->two_rank2_block_workspace_bytes +=
      other.two_rank2_block_workspace_bytes;
  stats->two_rank2_intermediate_rebuilds +=
      other.two_rank2_intermediate_rebuilds;
  stats->rank2_monomial_direct_hits += other.rank2_monomial_direct_hits;
  stats->rank2_monomial_block_workspace_bytes +=
      other.rank2_monomial_block_workspace_bytes;
}

template<typename ElemT>
void AddRank2AxisMiddleLoop(
    const ElemT *input_data,
    const ElemT *op_data,
    ElemT *output_data,
    const ShapeT &input_shape,
    const ShapeT &op_shape,
    const ShapeT &output_shape,
    const size_t axis,
    AxisOpStats *stats = nullptr,
    const ElemT alpha = ElemT(1)
) {
  const size_t input_axis_dim = input_shape[axis];
  const size_t output_axis_dim = output_shape[axis];
  const size_t inner_size = AxisInnerSize(input_shape, axis);
  const size_t outer_size = std::accumulate(input_shape.begin(), input_shape.begin() + axis,
                                            size_t(1), std::multiplies<size_t>());

  for (size_t outer = 0; outer < outer_size; ++outer) {
    const size_t input_outer_base = outer * input_axis_dim * inner_size;
    const size_t output_outer_base = outer * output_axis_dim * inner_size;
    AddGemmStats(stats, output_axis_dim, input_axis_dim, inner_size, false);
    hp_numeric::MatMultiply(
        alpha,
        op_data,
        BlasTrans(),
        input_data + input_outer_base,
        BlasNoTrans(),
        output_axis_dim,
        input_axis_dim,
        inner_size,
        op_shape[1],
        inner_size,
        ElemT(1),
        output_data + output_outer_base
    );
  }
}

template<typename ElemT>
void AddRank2AxisMiddleBatch(
    const ElemT *input_data,
    const ElemT *op_data,
    ElemT *output_data,
    const ShapeT &input_shape,
    const ShapeT &op_shape,
    const ShapeT &output_shape,
    const size_t axis,
    AxisOpStats *stats = nullptr,
    const ElemT alpha = ElemT(1)
) {
  const size_t input_axis_dim = input_shape[axis];
  const size_t output_axis_dim = output_shape[axis];
  const size_t inner_size = AxisInnerSize(input_shape, axis);
  const size_t outer_size = std::accumulate(input_shape.begin(), input_shape.begin() + axis,
                                            size_t(1), std::multiplies<size_t>());
  (void) op_shape;
  if (outer_size == 0) {
    return;
  }
  if (alpha != ElemT(1)) {
    AddBatchGemmStats(stats, outer_size, false);
    AddRank2AxisMiddleLoop(input_data,
                           op_data,
                           output_data,
                           input_shape,
                           op_shape,
                           output_shape,
                           axis,
                           stats,
                           alpha);
    return;
  }

#if defined(USE_GPU)
  using AlphaT = typename RealTypeTrait<ElemT>::type;
  AddBatchGemmStats(stats, outer_size, true);
  AddRepeatedGemmStats(
      stats, output_axis_dim, input_axis_dim, inner_size, outer_size, false);
  hp_numeric::MatMultiplyStridedBatch(
      AlphaT(1),
      op_data,
      BlasTrans(),
      input_data,
      BlasNoTrans(),
      output_axis_dim,
      input_axis_dim,
      inner_size,
      op_shape[1],
      0,
      inner_size,
      static_cast<long long int>(input_axis_dim * inner_size),
      ElemT(1),
      output_data,
      static_cast<long long int>(output_axis_dim * inner_size),
      outer_size
  );
#elif defined(HP_NUMERIC_BACKEND_MKL)
#if defined(QLTEN_USE_MKL_GEMM_BATCH)
  constexpr bool kUsesBackendBatch = true;
#else
  constexpr bool kUsesBackendBatch = false;
#endif
  AddBatchGemmStats(stats, outer_size, kUsesBackendBatch);
  AddRepeatedGemmStats(
      stats, output_axis_dim, input_axis_dim, inner_size, outer_size, false);

  std::vector<ElemT> op_transposed(output_axis_dim * input_axis_dim);
  for (size_t input_coor = 0; input_coor < input_axis_dim; ++input_coor) {
    for (size_t output_coor = 0;
         output_coor < output_axis_dim;
         ++output_coor) {
      op_transposed[output_coor * input_axis_dim + input_coor] =
          op_data[input_coor * output_axis_dim + output_coor];
    }
  }
  if (stats != nullptr) {
    stats->batch_gemm_workspace_bytes +=
        op_transposed.size() * sizeof(ElemT);
  }

  std::vector<const ElemT *> a_array(outer_size, op_transposed.data());
  std::vector<const ElemT *> b_array(outer_size);
  std::vector<ElemT *> c_array(outer_size);
  std::vector<MKL_INT> m_array(outer_size,
                               static_cast<MKL_INT>(output_axis_dim));
  std::vector<MKL_INT> k_array(outer_size,
                               static_cast<MKL_INT>(input_axis_dim));
  std::vector<MKL_INT> n_array(outer_size, static_cast<MKL_INT>(inner_size));
  std::vector<ElemT> beta_array(outer_size, ElemT(1));
  for (size_t outer = 0; outer < outer_size; ++outer) {
    const size_t input_outer_base = outer * input_axis_dim * inner_size;
    const size_t output_outer_base = outer * output_axis_dim * inner_size;
    b_array[outer] = input_data + input_outer_base;
    c_array[outer] = output_data + output_outer_base;
  }

  hp_numeric::MatMultiplyBatch(a_array.data(),
                               b_array.data(),
                               m_array.data(),
                               k_array.data(),
                               n_array.data(),
                               beta_array.data(),
                               c_array.data(),
                               static_cast<MKL_INT>(outer_size));
#else
  AddBatchGemmStats(stats, outer_size, false);
  AddRank2AxisMiddleLoop(input_data,
                         op_data,
                         output_data,
                         input_shape,
                         op_shape,
                         output_shape,
                         axis,
                         stats);
#endif
}

template<Rank2AxisApplyGemmMode GemmMode = Rank2AxisApplyGemmMode::kLoop,
         typename ElemT>
void AddRank2AxisBlock(
    const ElemT *input_data,
    const ElemT *op_data,
    ElemT *output_data,
    const ShapeT &input_shape,
    const ShapeT &op_shape,
    const ShapeT &output_shape,
    const size_t axis,
    AxisOpStats *stats = nullptr,
    const ElemT alpha = ElemT(1)
) {
  static_assert(IsHpNumericElem<ElemT>(),
                "DMRG axis operations require a hp_numeric-supported ElemT.");
  const size_t input_axis_dim = input_shape[axis];
  const size_t output_axis_dim = output_shape[axis];
  const size_t inner_size = AxisInnerSize(input_shape, axis);
  const size_t outer_size = std::accumulate(input_shape.begin(), input_shape.begin() + axis,
                                            size_t(1), std::multiplies<size_t>());

  if (axis == 0) {
    AddGemmStats(stats, output_axis_dim, input_axis_dim, inner_size, true);
    hp_numeric::MatMultiply(
        alpha,
        op_data,
        BlasTrans(),
        input_data,
        BlasNoTrans(),
        output_axis_dim,
        input_axis_dim,
        inner_size,
        op_shape[1],
        inner_size,
        ElemT(1),
        output_data
    );
    return;
  }

  if (axis + 1 == input_shape.size()) {
    AddGemmStats(stats, outer_size, input_axis_dim, output_axis_dim, true);
    hp_numeric::MatMultiply(
        alpha,
        input_data,
        BlasNoTrans(),
        op_data,
        BlasNoTrans(),
        outer_size,
        input_axis_dim,
        output_axis_dim,
        input_axis_dim,
        op_shape[1],
        ElemT(1),
        output_data
    );
    return;
  }

  if constexpr (GemmMode == Rank2AxisApplyGemmMode::kBatch) {
    AddRank2AxisMiddleBatch(input_data,
                            op_data,
                            output_data,
                            input_shape,
                            op_shape,
                            output_shape,
                            axis,
                            stats,
                            alpha);
  } else {
    AddRank2AxisMiddleLoop(input_data,
                           op_data,
                           output_data,
                           input_shape,
                           op_shape,
                           output_shape,
                           axis,
                           stats,
                           alpha);
  }
}

template<typename ElemT, typename QNT>
IndexVec<QNT> TwoRank2OutputIndexes(
    const QLTensor<ElemT, QNT> &input,
    const size_t axis1,
    const QLTensor<ElemT, QNT> &op1,
    const size_t axis2,
    const QLTensor<ElemT, QNT> &op2
) {
  IndexVec<QNT> output_indexes = input.GetIndexes();
  output_indexes[axis1] = op1.GetIndex(1);
  output_indexes[axis2] = op2.GetIndex(1);
  return output_indexes;
}

template<typename ElemT, typename QNT>
OutputBlockTopology GenerateTwoRank2OutputBlockTopology(
    const QLTensor<ElemT, QNT> &input,
    const QLTensor<ElemT, QNT> &op1,
    const size_t axis1,
    const QLTensor<ElemT, QNT> &op2,
    const size_t axis2,
    const IndexVec<QNT> &output_indexes
) {
  std::set<size_t> seen_blk_idxs;
  OutputBlockTopology topology;
  const auto &input_blocks = input.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  const auto &op1_blocks = op1.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  const auto &op2_blocks = op2.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  const auto op1_blk_idxs_by_input_sector =
      Rank2OpBlockIndicesByInputSector(op1);
  const auto op2_blk_idxs_by_input_sector =
      Rank2OpBlockIndicesByInputSector(op2);
  size_t candidate_count = 0;
  for (const auto &input_entry : input_blocks) {
    const auto &input_blk = input_entry.second;
    candidate_count +=
        op1_blk_idxs_by_input_sector[input_blk.blk_coors[axis1]].size() *
        op2_blk_idxs_by_input_sector[input_blk.blk_coors[axis2]].size();
  }
  ReserveOutputBlockTopology(topology, output_indexes, candidate_count);

  for (const auto &input_entry : input_blocks) {
    const auto &input_blk = input_entry.second;
    const auto &op1_blk_idxs =
        op1_blk_idxs_by_input_sector[input_blk.blk_coors[axis1]];
    const auto &op2_blk_idxs =
        op2_blk_idxs_by_input_sector[input_blk.blk_coors[axis2]];
    for (const size_t op1_blk_idx : op1_blk_idxs) {
      const auto &op1_blk = op1_blocks.at(op1_blk_idx);
      for (const size_t op2_blk_idx : op2_blk_idxs) {
        const auto &op2_blk = op2_blocks.at(op2_blk_idx);
        CoorsT output_blk_coors = input_blk.blk_coors;
        output_blk_coors[axis1] = op1_blk.blk_coors[1];
        output_blk_coors[axis2] = op2_blk.blk_coors[1];
        const size_t output_blk_idx =
            BlkCoorsToBlkIdx(output_blk_coors, output_indexes);
        if (seen_blk_idxs.insert(output_blk_idx).second) {
          topology.blk_idxs.push_back(output_blk_idx);
          topology.blk_coors_s.push_back(std::move(output_blk_coors));
        }
      }
    }
  }
  return topology;
}

template<typename ElemT, typename QNT>
std::vector<CoorsT> GenerateTwoRank2OutputBlockCoors(
    const QLTensor<ElemT, QNT> &input,
    const QLTensor<ElemT, QNT> &op1,
    const size_t axis1,
    const QLTensor<ElemT, QNT> &op2,
    const size_t axis2,
    const IndexVec<QNT> &output_indexes
) {
  return GenerateTwoRank2OutputBlockTopology(
      input, op1, axis1, op2, axis2, output_indexes).blk_coors_s;
}

#ifdef QLTEN_DMRG_ENABLE_SCALAR_TWO_RANK2_FALLBACK
inline void DecodeFlatOffset(
    size_t flat_offset,
    const std::vector<size_t> &offsets,
    CoorsT &coors
) {
  for (size_t axis = 0; axis < offsets.size(); ++axis) {
    coors[axis] = flat_offset / offsets[axis];
    flat_offset %= offsets[axis];
  }
}

template<typename ElemT>
void AddTwoRank2AxesBlock(
    const ElemT *input_data,
    const ElemT *op1_data,
    const ElemT *op2_data,
    ElemT *output_data,
    const ShapeT &input_shape,
    const ShapeT &op1_shape,
    const ShapeT &op2_shape,
    const ShapeT &output_shape,
    const size_t axis1,
    const size_t axis2,
    AxisOpStats *stats = nullptr,
    const ElemT alpha = ElemT(1)
) {
  static_assert(IsHpNumericElem<ElemT>(),
                "DMRG axis operations require a hp_numeric-supported ElemT.");
  const size_t input_size = std::accumulate(input_shape.begin(),
                                            input_shape.end(),
                                            size_t(1),
                                            std::multiplies<size_t>());
  const size_t op1_output_dim = op1_shape[1];
  const size_t op2_output_dim = op2_shape[1];
  const auto input_offsets = CalcMultiDimDataOffsets(input_shape);
  const auto output_offsets = CalcMultiDimDataOffsets(output_shape);
  CoorsT input_coors(input_shape.size());
  CoorsT output_coors(input_shape.size());

  for (size_t input_offset = 0; input_offset < input_size; ++input_offset) {
    DecodeFlatOffset(input_offset, input_offsets, input_coors);
    output_coors = input_coors;
    const size_t input_axis1_coor = input_coors[axis1];
    const size_t input_axis2_coor = input_coors[axis2];
    for (size_t out_axis1_coor = 0;
         out_axis1_coor < op1_output_dim;
         ++out_axis1_coor) {
      const ElemT coef1 =
          op1_data[input_axis1_coor * op1_output_dim + out_axis1_coor];
      if (coef1 == ElemT(0)) {
        continue;
      }
      output_coors[axis1] = out_axis1_coor;
      for (size_t out_axis2_coor = 0;
           out_axis2_coor < op2_output_dim;
           ++out_axis2_coor) {
        const ElemT coef =
            coef1 * op2_data[input_axis2_coor * op2_output_dim +
                             out_axis2_coor];
        if (coef == ElemT(0)) {
          continue;
        }
        output_coors[axis2] = out_axis2_coor;
        const size_t output_offset =
            CalcEffOneDimArrayOffset(output_coors, output_offsets);
        output_data[output_offset] += alpha * coef * input_data[input_offset];
        if (stats != nullptr) {
          ++stats->fused_element_updates;
        }
      }
    }
  }
}
#endif  // QLTEN_DMRG_ENABLE_SCALAR_TWO_RANK2_FALLBACK

template<typename ElemT>
void AddTwoRank2AxesBlockGemm(
    const ElemT *input_data,
    const ElemT *op1_data,
    const ElemT *op2_data,
    ElemT *output_data,
    const ShapeT &input_shape,
    const ShapeT &op1_shape,
    const ShapeT &op2_shape,
    const ShapeT &output_shape,
    const size_t axis1,
    const size_t axis2,
    AxisOpStats *stats = nullptr,
    const ElemT alpha = ElemT(1)
) {
  static_assert(IsHpNumericElem<ElemT>(),
                "DMRG axis operations require a hp_numeric-supported ElemT.");
  ShapeT intermediate_shape = input_shape;
  intermediate_shape[axis1] = op1_shape[1];
  const size_t intermediate_size =
      std::accumulate(intermediate_shape.begin(),
                      intermediate_shape.end(),
                      size_t(1),
                      std::multiplies<size_t>());
  const size_t workspace_bytes = intermediate_size * sizeof(ElemT);
  ElemT *intermediate_data =
      static_cast<ElemT *>(qlten::QLMalloc(workspace_bytes));
  qlten::QLMemset(intermediate_data, 0, workspace_bytes);
  if (stats != nullptr) {
    stats->two_rank2_block_workspace_bytes += workspace_bytes;
  }

  AddRank2AxisBlock<Rank2AxisApplyGemmMode::kBatch>(
      input_data,
      op1_data,
      intermediate_data,
      input_shape,
      op1_shape,
      intermediate_shape,
      axis1,
      stats
  );
  AddRank2AxisBlock<Rank2AxisApplyGemmMode::kBatch>(
      intermediate_data,
      op2_data,
      output_data,
      intermediate_shape,
      op2_shape,
      output_shape,
      axis2,
      stats,
      alpha
  );
  qlten::QLFree(intermediate_data);
}

template<typename ElemT, typename QNT>
OutputBlockTopology GenerateMonomialOutputBlockTopology(
    const QLTensor<ElemT, QNT> &input,
    const size_t axis,
    const AxisMonomialOp<ElemT, QNT> &op,
    const IndexVec<QNT> &output_indexes
) {
  std::set<size_t> seen_blk_idxs;
  OutputBlockTopology topology;
  const auto &input_blocks = input.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  ReserveOutputBlockTopology(topology,
                             output_indexes,
                             input_blocks.size() * op.entries.size());
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
        topology.blk_idxs.push_back(output_blk_idx);
        topology.blk_coors_s.push_back(std::move(output_blk_coors));
      }
    }
  }
  return topology;
}

template<typename ElemT, typename QNT>
std::vector<CoorsT> GenerateMonomialOutputBlockCoors(
    const QLTensor<ElemT, QNT> &input,
    const size_t axis,
    const AxisMonomialOp<ElemT, QNT> &op,
    const IndexVec<QNT> &output_indexes
) {
  return GenerateMonomialOutputBlockTopology(
      input, axis, op, output_indexes).blk_coors_s;
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
  for (size_t outer = 0; outer < outer_size; ++outer) {
    const size_t input_base =
        outer * input_axis_dim * inner_size + in_offset * inner_size;
    const size_t output_base =
        outer * output_axis_dim * inner_size + out_offset * inner_size;
    AddRawDataTo(input_data + input_base,
                 inner_size,
                 output_data + output_base,
                 coef);
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
      CopyRawData(input_data + input_base,
                  inner_size,
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
 * @throws std::runtime_error when a tensor has stored blocks but no allocated
 *         raw data.
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
 * @throws std::runtime_error when a tensor has stored blocks but no allocated
 *         raw data.
 */
template<typename ElemT, typename QNT>
void ProjectAxis(
    const QLTensor<ElemT, QNT> &input,
    size_t axis,
    const AxisProjector<QNT> &projector,
    QLTensor<ElemT, QNT> &output
) {
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
  const auto output_topology =
      detail::GenerateProjectedOutputBlockTopology(input,
                                                   axis,
                                                   input_sector_to_output_sector,
                                                   output_indexes);

  output = QLTensor<ElemT, QNT>(output_indexes);
  if (!output_topology.blk_coors_s.empty()) {
    output.GetBlkSparDataTen().DataBlksInsert(output_topology.blk_idxs,
                                              output_topology.blk_coors_s,
                                              true,
                                              true);
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
 * @throws std::runtime_error when `tensor` has stored blocks but no allocated
 *         raw data.
 */
template<typename ElemT, typename QNT>
void ApplyAxisDiagonalInPlace(
    QLTensor<ElemT, QNT> &tensor,
    size_t axis,
    const AxisDiagonalOp<ElemT, QNT> &op
) {
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
}

/**
 * @brief Apply a diagonal axis operator with explicit storage ownership.
 *
 * `kOutOfPlace` treats `input` as borrowed const data and rebuilds `output`.
 * `kConsumeOwnedInPlace` requires `input` and `output` to be the same owned
 * tensor object and scales it in place.  Axis order and block topology are
 * preserved in both modes.
 */
template<typename ElemT, typename QNT>
void ApplyAxisDiagonalPreserveOrder(
    const QLTensor<ElemT, QNT> &input,
    size_t axis,
    const AxisDiagonalOp<ElemT, QNT> &op,
    QLTensor<ElemT, QNT> &output,
    AxisApplyStorageMode storage_mode,
    AxisOpStats *stats = nullptr
) {
  static_assert(!Fermionicable<QNT>::IsFermionic(),
                "ApplyAxisDiagonalPreserveOrder is bosonic-only.");
  constexpr const char *kFunc = "ApplyAxisDiagonalPreserveOrder";
  if (stats != nullptr) {
    *stats = AxisOpStats{};
  }
  if (storage_mode == AxisApplyStorageMode::kOutOfPlace) {
    if (&input == &output) {
      throw std::invalid_argument(
          detail::AxisOpsPrefix(kFunc) +
          "input/output aliasing is not allowed in out-of-place mode."
      );
    }
    ApplyAxisDiagonal(input, axis, op, output);
    if (stats != nullptr) {
      ++stats->output_tensor_rebuilds;
    }
    return;
  }

  if (&input != &output) {
    throw std::invalid_argument(
        detail::AxisOpsPrefix(kFunc) +
        "consume-owned mode requires input/output aliasing."
    );
  }
  ApplyAxisDiagonalInPlace(output, axis, op);
  if (stats != nullptr) {
    ++stats->in_place_scale_hits;
  }
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
 * @throws std::runtime_error when `tensor` has stored blocks but no allocated
 *         raw data.
 */
template<typename ElemT, typename QNT>
void ScaleAxisSectorsInPlace(
    QLTensor<ElemT, QNT> &tensor,
    const AxisSectorScalar<ElemT, QNT> &scale
) {
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
}

/**
 * @brief Apply per-sector axis scalars with explicit storage ownership.
 *
 * `kOutOfPlace` rebuilds `output` and directly writes scaled copies of input
 * blocks.  `kConsumeOwnedInPlace` requires `input` and `output` to alias and
 * scales that owned tensor directly.  Sector-scalar scaling itself never
 * changes block topology.
 */
template<typename ElemT, typename QNT>
void ApplyAxisSectorScalarsPreserveOrder(
    const QLTensor<ElemT, QNT> &input,
    const AxisSectorScalar<ElemT, QNT> &scale,
    QLTensor<ElemT, QNT> &output,
    AxisApplyStorageMode storage_mode,
    AxisOpStats *stats = nullptr
) {
  static_assert(!Fermionicable<QNT>::IsFermionic(),
                "ApplyAxisSectorScalarsPreserveOrder is bosonic-only.");
  constexpr const char *kFunc = "ApplyAxisSectorScalarsPreserveOrder";
  if (stats != nullptr) {
    *stats = AxisOpStats{};
  }
  if (storage_mode == AxisApplyStorageMode::kOutOfPlace) {
    if (&input == &output) {
      throw std::invalid_argument(
          detail::AxisOpsPrefix(kFunc) +
          "input/output aliasing is not allowed in out-of-place mode."
      );
    }
    detail::RequireUsableAxis(input, scale.axis, kFunc);
    const auto &axis_index = input.GetIndex(scale.axis);
    if (scale.values_by_sector.size() != axis_index.GetQNSctNum()) {
      throw std::invalid_argument(
          detail::AxisOpsPrefix(kFunc) +
          "values_by_sector size must match the number of sectors."
      );
    }
    detail::RequireRawDataIfBlocksPresent(input, kFunc, "input");
    detail::InitializeOutputLikeInput(input, output);
    if (stats != nullptr) {
      ++stats->output_tensor_rebuilds;
      stats->raw_data_copy_bytes +=
          input.GetBlkSparDataTen().GetActualRawDataSize() * sizeof(ElemT);
    }
    detail::RequireRawDataIfBlocksPresent(output, kFunc, "output");
    const auto &input_bsdt = input.GetBlkSparDataTen();
    auto &output_bsdt = output.GetBlkSparDataTen();
    const ElemT *input_data = input_bsdt.GetActualRawDataPtr();
    ElemT *output_data = output_bsdt.GetActualRawDataPtr();
    const auto &output_blocks = output_bsdt.GetBlkIdxDataBlkMap();
    for (const auto &input_entry : input_bsdt.GetBlkIdxDataBlkMap()) {
      const auto &input_blk = input_entry.second;
      const auto &output_blk = output_blocks.at(input_entry.first);
      detail::CopyScaleRawData(
          input_data + input_blk.data_offset,
          input_blk.size,
          output_data + output_blk.data_offset,
          scale.values_by_sector[input_blk.blk_coors[scale.axis]]
      );
      if (stats != nullptr) {
        ++stats->direct_scaled_copy_calls;
      }
    }
    return;
  }

  if (&input != &output) {
    throw std::invalid_argument(
        detail::AxisOpsPrefix(kFunc) +
        "consume-owned mode requires input/output aliasing."
    );
  }
  ScaleAxisSectorsInPlace(output, scale);
  if (stats != nullptr) {
    ++stats->in_place_scale_hits;
  }
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
 * @throws std::runtime_error when a tensor has stored blocks but no allocated
 *         raw data.
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
  const auto output_topology =
      detail::GenerateMonomialOutputBlockTopology(input,
                                                  axis,
                                                  op,
                                                  output_indexes);

  if (beta == ElemT(0)) {
    output = QLTensor<ElemT, QNT>(output_indexes);
    if (!output_topology.blk_coors_s.empty()) {
      output.GetBlkSparDataTen().DataBlksInsert(output_topology.blk_idxs,
                                                output_topology.blk_coors_s,
                                                true,
                                                true);
    }
  } else {
    if (output.IsDefault()) {
      throw std::invalid_argument(
          detail::AxisOpsPrefix(kFunc) +
          "output must not be default when beta is nonzero."
      );
    }
    if (output.GetIndexes() != output_indexes ||
        !detail::BlockTopologyMatchesCoors(output, output_topology.blk_coors_s)) {
      throw std::invalid_argument(
          detail::AxisOpsPrefix(kFunc) +
          "output must have the generated index and block topology when beta "
          "is nonzero."
      );
    }
    detail::RequireRawDataIfBlocksPresent(output, kFunc, "output");
    detail::ScaleTensorRawData(output, beta);
  }

  if (alpha == ElemT(0) || op.entries.empty() ||
      output_topology.blk_coors_s.empty()) {
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
 * @tparam GemmMode middle-axis GEMM dispatch strategy.  Boundary axes ignore
 *         this parameter because they already use one contiguous GEMM.
 *
 * @pre `input` is non-default and `target_axis < input.Rank()`.
 * @pre `rank2_op` is non-default, has rank 2, and
 *      `rank2_op.GetIndex(0) == InverseIndex(input.GetIndex(target_axis))`.
 * @pre `out` does not alias `input` or `rank2_op`.
 * @throws std::invalid_argument on invalid axes, rank/index mismatch, or
 *         output aliasing.
 * @throws std::runtime_error when a tensor has stored blocks but no allocated
 *         raw data.
 */
template<Rank2AxisApplyGemmMode GemmMode = Rank2AxisApplyGemmMode::kLoop,
         typename ElemT, typename QNT>
void ApplyRank2ToAxisPreserveOrder(
    const QLTensor<ElemT, QNT> &input,
    const QLTensor<ElemT, QNT> &rank2_op,
    size_t target_axis,
    QLTensor<ElemT, QNT> &out,
    Rank2AxisApplyStats *stats = nullptr
) {
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
  const auto output_topology =
      detail::GenerateRank2OutputBlockTopology(input,
                                               target_axis,
                                               rank2_op,
                                               output_indexes,
                                               stats);
  out = QLTensor<ElemT, QNT>(output_indexes);
  if (stats != nullptr) {
    ++stats->output_tensor_rebuilds;
  }
  if (!output_topology.blk_coors_s.empty()) {
    out.GetBlkSparDataTen().DataBlksInsert(output_topology.blk_idxs,
                                           output_topology.blk_coors_s,
                                           true,
                                           true);
  }
  if (output_topology.blk_coors_s.empty()) {
    return;
  }
  detail::RequireRawDataIfBlocksPresent(out, kFunc, "out");

  const auto &input_bsdt = input.GetBlkSparDataTen();
  const auto &op_bsdt = rank2_op.GetBlkSparDataTen();
  auto &output_bsdt = out.GetBlkSparDataTen();
  const ElemT *input_data = input_bsdt.GetActualRawDataPtr();
  const ElemT *op_data = op_bsdt.GetActualRawDataPtr();
  ElemT *output_data = output_bsdt.GetActualRawDataPtr();
  const auto &op_blocks = op_bsdt.GetBlkIdxDataBlkMap();
  const auto &output_blocks = output_bsdt.GetBlkIdxDataBlkMap();
  const auto op_blk_idxs_by_input_sector =
      detail::Rank2OpBlockIndicesByInputSector(rank2_op);

  for (const auto &input_entry : input_bsdt.GetBlkIdxDataBlkMap()) {
    const auto &input_blk = input_entry.second;
    const auto &matching_op_blk_idxs =
        op_blk_idxs_by_input_sector[input_blk.blk_coors[target_axis]];
    for (const size_t op_blk_idx : matching_op_blk_idxs) {
      const auto &op_blk = op_blocks.at(op_blk_idx);
      CoorsT output_blk_coors = input_blk.blk_coors;
      output_blk_coors[target_axis] = op_blk.blk_coors[1];
      const size_t output_blk_idx =
          output_bsdt.BlkCoorsToBlkIdx(output_blk_coors);
      const auto &output_blk = output_blocks.at(output_blk_idx);
      detail::AddRank2AxisBlock<GemmMode>(
          input_data + input_blk.data_offset,
          op_data + op_blk.data_offset,
          output_data + output_blk.data_offset,
          input_blk.shape,
          op_blk.shape,
          output_blk.shape,
          target_axis,
          stats
      );
    }
  }
}

/**
 * @brief Apply two rank-2 operators to two tensor axes without reordering axes.
 *
 * The operators are applied in the exact order supplied by the caller:
 * `op1` on `axis1`, then `op2` on `axis2`.  The final tensor keeps the original
 * positional axis order with those axes replaced by each operator's output
 * index.
 *
 * The implementation writes into the final output tensor through block-local
 * GEMM workspace. It does not construct the single-axis intermediate QLTensor
 * used by two sequential `ApplyRank2ToAxisPreserveOrder()` calls.
 *
 * @note Defining `QLTEN_DMRG_ENABLE_SCALAR_TWO_RANK2_FALLBACK` opts into the
 *       old CPU scalar reference kernel. Default builds always dispatch through
 *       GEMM/cuBLAS-backed block kernels.
 *
 * @pre `axis1 != axis2`.
 * @pre `out` does not alias `input`, `op1`, or `op2`.
 */
template<typename ElemT, typename QNT>
void ApplyTwoRank2ToAxesPreserveOrder(
    const QLTensor<ElemT, QNT> &input,
    const QLTensor<ElemT, QNT> &op1,
    size_t axis1,
    const QLTensor<ElemT, QNT> &op2,
    size_t axis2,
    QLTensor<ElemT, QNT> &out,
    AxisOpStats *stats = nullptr
) {
  static_assert(!Fermionicable<QNT>::IsFermionic(),
                "ApplyTwoRank2ToAxesPreserveOrder is bosonic-only.");
  constexpr const char *kFunc = "ApplyTwoRank2ToAxesPreserveOrder";
  if (stats != nullptr) {
    *stats = AxisOpStats{};
  }
  if (&input == &out || &op1 == &out || &op2 == &out) {
    throw std::invalid_argument(
        detail::AxisOpsPrefix(kFunc) + "output aliasing is not allowed."
    );
  }
  if (axis1 == axis2) {
    throw std::invalid_argument(
        detail::AxisOpsPrefix(kFunc) + "axes must be distinct."
    );
  }
  detail::RequireUsableAxis(input, axis1, kFunc);
  detail::RequireUsableAxis(input, axis2, kFunc);
  detail::RequireRank2OpMatchesAxis(input.GetIndex(axis1), op1, kFunc);
  detail::RequireRank2OpMatchesAxis(input.GetIndex(axis2), op2, kFunc);
  detail::RequireRawDataIfBlocksPresent(input, kFunc, "input");
  detail::RequireRawDataIfBlocksPresent(op1, kFunc, "op1");
  detail::RequireRawDataIfBlocksPresent(op2, kFunc, "op2");

  const auto output_indexes =
      detail::TwoRank2OutputIndexes(input, axis1, op1, axis2, op2);
  const auto output_topology =
      detail::GenerateTwoRank2OutputBlockTopology(input,
                                                  op1,
                                                  axis1,
                                                  op2,
                                                  axis2,
                                                  output_indexes);
  out = QLTensor<ElemT, QNT>(output_indexes);
  if (stats != nullptr) {
    ++stats->output_tensor_rebuilds;
#ifdef QLTEN_DMRG_ENABLE_SCALAR_TWO_RANK2_FALLBACK
    ++stats->two_rank2_fused_hits;
#else
    ++stats->two_rank2_block_gemm_hits;
#endif
  }
  if (!output_topology.blk_coors_s.empty()) {
    out.GetBlkSparDataTen().DataBlksInsert(output_topology.blk_idxs,
                                           output_topology.blk_coors_s,
                                           true,
                                           true);
  }
  if (output_topology.blk_coors_s.empty()) {
    return;
  }
  detail::RequireRawDataIfBlocksPresent(out, kFunc, "out");

  const auto &input_bsdt = input.GetBlkSparDataTen();
  const auto &op1_bsdt = op1.GetBlkSparDataTen();
  const auto &op2_bsdt = op2.GetBlkSparDataTen();
  auto &output_bsdt = out.GetBlkSparDataTen();
  const ElemT *input_data = input_bsdt.GetActualRawDataPtr();
  const ElemT *op1_data = op1_bsdt.GetActualRawDataPtr();
  const ElemT *op2_data = op2_bsdt.GetActualRawDataPtr();
  ElemT *output_data = output_bsdt.GetActualRawDataPtr();
  const auto &op1_blocks = op1_bsdt.GetBlkIdxDataBlkMap();
  const auto &op2_blocks = op2_bsdt.GetBlkIdxDataBlkMap();
  const auto &output_blocks = output_bsdt.GetBlkIdxDataBlkMap();
  const auto op1_blk_idxs_by_input_sector =
      detail::Rank2OpBlockIndicesByInputSector(op1);
  const auto op2_blk_idxs_by_input_sector =
      detail::Rank2OpBlockIndicesByInputSector(op2);

  for (const auto &input_entry : input_bsdt.GetBlkIdxDataBlkMap()) {
    const auto &input_blk = input_entry.second;
    const auto &op1_blk_idxs =
        op1_blk_idxs_by_input_sector[input_blk.blk_coors[axis1]];
    const auto &op2_blk_idxs =
        op2_blk_idxs_by_input_sector[input_blk.blk_coors[axis2]];
    for (const size_t op1_blk_idx : op1_blk_idxs) {
      const auto &op1_blk = op1_blocks.at(op1_blk_idx);
      for (const size_t op2_blk_idx : op2_blk_idxs) {
        const auto &op2_blk = op2_blocks.at(op2_blk_idx);
        CoorsT output_blk_coors = input_blk.blk_coors;
        output_blk_coors[axis1] = op1_blk.blk_coors[1];
        output_blk_coors[axis2] = op2_blk.blk_coors[1];
        const size_t output_blk_idx =
            output_bsdt.BlkCoorsToBlkIdx(output_blk_coors);
        const auto &output_blk = output_blocks.at(output_blk_idx);
#ifdef QLTEN_DMRG_ENABLE_SCALAR_TWO_RANK2_FALLBACK
        detail::AddTwoRank2AxesBlock(
            input_data + input_blk.data_offset,
            op1_data + op1_blk.data_offset,
            op2_data + op2_blk.data_offset,
            output_data + output_blk.data_offset,
            input_blk.shape,
            op1_blk.shape,
            op2_blk.shape,
            output_blk.shape,
            axis1,
            axis2,
            stats
        );
#else
        detail::AddTwoRank2AxesBlockGemm(
            input_data + input_blk.data_offset,
            op1_data + op1_blk.data_offset,
            op2_data + op2_blk.data_offset,
            output_data + output_blk.data_offset,
            input_blk.shape,
            op1_blk.shape,
            op2_blk.shape,
            output_blk.shape,
            axis1,
            axis2,
            stats
        );
#endif
      }
    }
  }
}

/**
 * @brief Apply one rank-2 operator, then consume the owned output in-place with
 *        an axis diagonal operator.
 */
template<typename ElemT, typename QNT>
void ApplyRank2ThenAxisDiagonalPreserveOrder(
    const QLTensor<ElemT, QNT> &input,
    const QLTensor<ElemT, QNT> &rank2_op,
    size_t rank2_axis,
    size_t diagonal_axis,
    const AxisDiagonalOp<ElemT, QNT> &diagonal,
    QLTensor<ElemT, QNT> &out,
    AxisOpStats *stats = nullptr
) {
  ApplyRank2ToAxisPreserveOrder(input,
                                rank2_op,
                                rank2_axis,
                                out,
                                stats);
  ApplyAxisDiagonalInPlace(out, diagonal_axis, diagonal);
  if (stats != nullptr) {
    ++stats->in_place_scale_hits;
  }
}

/**
 * @brief Apply one rank-2 operator, then consume the owned output in-place with
 *        sector scalars.
 */
template<typename ElemT, typename QNT>
void ApplyRank2ThenAxisSectorScalarsPreserveOrder(
    const QLTensor<ElemT, QNT> &input,
    const QLTensor<ElemT, QNT> &rank2_op,
    size_t rank2_axis,
    const AxisSectorScalar<ElemT, QNT> &scale,
    QLTensor<ElemT, QNT> &out,
    AxisOpStats *stats = nullptr
) {
  ApplyRank2ToAxisPreserveOrder(input,
                                rank2_op,
                                rank2_axis,
                                out,
                                stats);
  ScaleAxisSectorsInPlace(out, scale);
  if (stats != nullptr) {
    ++stats->in_place_scale_hits;
  }
}

/**
 * @brief Apply one rank-2 operator and fold final sector scalars into output
 *        writes.
 *
 * This is equivalent to ApplyRank2ToAxisPreserveOrder() followed by
 * ScaleAxisSectorsInPlace(), but it avoids the second full raw-data pass.
 * `scale.axis` refers to the final preserve-order output layout.
 */
template<typename ElemT, typename QNT>
void ApplyRank2ThenAxisSectorScalarsFusedPreserveOrder(
    const QLTensor<ElemT, QNT> &input,
    const QLTensor<ElemT, QNT> &rank2_op,
    size_t rank2_axis,
    const AxisSectorScalar<ElemT, QNT> &scale,
    QLTensor<ElemT, QNT> &out,
    AxisOpStats *stats = nullptr
) {
  static_assert(!Fermionicable<QNT>::IsFermionic(),
                "ApplyRank2ThenAxisSectorScalarsFusedPreserveOrder is "
                "bosonic-only.");
  constexpr const char *kFunc =
      "ApplyRank2ThenAxisSectorScalarsFusedPreserveOrder";
  if (stats != nullptr) {
    *stats = AxisOpStats{};
  }
  if (&input == &out || &rank2_op == &out) {
    throw std::invalid_argument(
        detail::AxisOpsPrefix(kFunc) + "output aliasing is not allowed."
    );
  }
  detail::RequireUsableAxis(input, rank2_axis, kFunc);
  detail::RequireRank2OpMatchesAxis(input.GetIndex(rank2_axis),
                                    rank2_op,
                                    kFunc);
  detail::RequireRawDataIfBlocksPresent(input, kFunc, "input");
  detail::RequireRawDataIfBlocksPresent(rank2_op, kFunc, "rank2_op");

  const auto output_indexes =
      detail::Rank2OutputIndexes(input, rank2_axis, rank2_op);
  detail::RequireAxisSectorScalarMatchesOutput(output_indexes, scale, kFunc);
  const auto output_topology =
      detail::GenerateRank2OutputBlockTopology(input,
                                               rank2_axis,
                                               rank2_op,
                                               output_indexes,
                                               stats);
  out = QLTensor<ElemT, QNT>(output_indexes);
  if (stats != nullptr) {
    ++stats->output_tensor_rebuilds;
  }
  if (!output_topology.blk_coors_s.empty()) {
    out.GetBlkSparDataTen().DataBlksInsert(output_topology.blk_idxs,
                                           output_topology.blk_coors_s,
                                           true,
                                           true);
  }
  if (output_topology.blk_coors_s.empty()) {
    return;
  }
  detail::RequireRawDataIfBlocksPresent(out, kFunc, "out");

  const auto &input_bsdt = input.GetBlkSparDataTen();
  const auto &op_bsdt = rank2_op.GetBlkSparDataTen();
  auto &output_bsdt = out.GetBlkSparDataTen();
  const ElemT *input_data = input_bsdt.GetActualRawDataPtr();
  const ElemT *op_data = op_bsdt.GetActualRawDataPtr();
  ElemT *output_data = output_bsdt.GetActualRawDataPtr();
  const auto &op_blocks = op_bsdt.GetBlkIdxDataBlkMap();
  const auto &output_blocks = output_bsdt.GetBlkIdxDataBlkMap();
  const auto op_blk_idxs_by_input_sector =
      detail::Rank2OpBlockIndicesByInputSector(rank2_op);

  for (const auto &input_entry : input_bsdt.GetBlkIdxDataBlkMap()) {
    const auto &input_blk = input_entry.second;
    const auto &matching_op_blk_idxs =
        op_blk_idxs_by_input_sector[input_blk.blk_coors[rank2_axis]];
    for (const size_t op_blk_idx : matching_op_blk_idxs) {
      const auto &op_blk = op_blocks.at(op_blk_idx);
      CoorsT output_blk_coors = input_blk.blk_coors;
      output_blk_coors[rank2_axis] = op_blk.blk_coors[1];
      const ElemT alpha =
          scale.values_by_sector[output_blk_coors[scale.axis]];
      if (alpha == ElemT(0)) {
        continue;
      }
      const size_t output_blk_idx =
          output_bsdt.BlkCoorsToBlkIdx(output_blk_coors);
      const auto &output_blk = output_blocks.at(output_blk_idx);
      detail::AddRank2AxisBlock(
          input_data + input_blk.data_offset,
          op_data + op_blk.data_offset,
          output_data + output_blk.data_offset,
          input_blk.shape,
          op_blk.shape,
          output_blk.shape,
          rank2_axis,
          stats,
          alpha
      );
    }
  }
}

/**
 * @brief Apply one rank-2 operator and one monomial axis operator without a
 *        rank-2 tensor intermediate.
 *
 * The final tensor keeps the original positional axis order with
 * `rank2_axis` replaced by `rank2_op.GetIndex(1)` and `monomial_axis` replaced
 * by `monomial.output_index`.  Monomial scatter entries accumulate exactly like
 * ApplyAxisMonomial().
 *
 * When every nonzero monomial entry maps a one-dimensional input sector to a
 * one-dimensional output sector with degeneracy offset `0 -> 0`, the kernel
 * writes rank-2 GEMM results directly into final output blocks with the
 * monomial coefficient folded into GEMM alpha.  General monomial operators
 * still require block-local rank-2 workspace before scattering; downstream
 * performance tests should cover both paths when relying on this routine.
 */
template<typename ElemT, typename QNT>
void ApplyRank2ThenAxisMonomialFusedPreserveOrder(
    const QLTensor<ElemT, QNT> &input,
    const QLTensor<ElemT, QNT> &rank2_op,
    size_t rank2_axis,
    size_t monomial_axis,
    const AxisMonomialOp<ElemT, QNT> &monomial,
    QLTensor<ElemT, QNT> &out,
    AxisOpStats *stats = nullptr
) {
  static_assert(!Fermionicable<QNT>::IsFermionic(),
                "ApplyRank2ThenAxisMonomialFusedPreserveOrder is "
                "bosonic-only.");
  constexpr const char *kFunc =
      "ApplyRank2ThenAxisMonomialFusedPreserveOrder";
  if (stats != nullptr) {
    *stats = AxisOpStats{};
  }
  if (&input == &out || &rank2_op == &out) {
    throw std::invalid_argument(
        detail::AxisOpsPrefix(kFunc) + "output aliasing is not allowed."
    );
  }
  detail::RequireUsableAxis(input, rank2_axis, kFunc);
  detail::RequireUsableAxis(input, monomial_axis, kFunc);
  detail::RequireRank2OpMatchesAxis(input.GetIndex(rank2_axis),
                                    rank2_op,
                                    kFunc);
  detail::RequireRawDataIfBlocksPresent(input, kFunc, "input");
  detail::RequireRawDataIfBlocksPresent(rank2_op, kFunc, "rank2_op");

  auto rank2_output_indexes =
      detail::Rank2OutputIndexes(input, rank2_axis, rank2_op);
  detail::RequireMonomialOpMatchesAxis(rank2_output_indexes[monomial_axis],
                                       monomial,
                                       kFunc);
  const auto output_indexes =
      detail::Rank2ThenMonomialOutputIndexes(input,
                                             rank2_axis,
                                             rank2_op,
                                             monomial_axis,
                                             monomial);
  const auto output_topology =
      detail::GenerateRank2ThenMonomialOutputBlockTopology(input,
                                                           rank2_axis,
                                                           rank2_op,
                                                           monomial_axis,
                                                           monomial,
                                                           output_indexes,
                                                           stats);
  out = QLTensor<ElemT, QNT>(output_indexes);
  if (stats != nullptr) {
    ++stats->output_tensor_rebuilds;
  }
  if (!output_topology.blk_coors_s.empty()) {
    out.GetBlkSparDataTen().DataBlksInsert(output_topology.blk_idxs,
                                           output_topology.blk_coors_s,
                                           true,
                                           true);
  }
  if (output_topology.blk_coors_s.empty()) {
    return;
  }
  detail::RequireRawDataIfBlocksPresent(out, kFunc, "out");

  const auto &input_bsdt = input.GetBlkSparDataTen();
  const auto &op_bsdt = rank2_op.GetBlkSparDataTen();
  auto &output_bsdt = out.GetBlkSparDataTen();
  const ElemT *input_data = input_bsdt.GetActualRawDataPtr();
  const ElemT *op_data = op_bsdt.GetActualRawDataPtr();
  ElemT *output_data = output_bsdt.GetActualRawDataPtr();
  const auto &op_blocks = op_bsdt.GetBlkIdxDataBlkMap();
  const auto &output_blocks = output_bsdt.GetBlkIdxDataBlkMap();
  const auto op_blk_idxs_by_input_sector =
      detail::Rank2OpBlockIndicesByInputSector(rank2_op);
  const bool use_direct_monomial_path =
      detail::AxisMonomialNonzeroEntriesAreOneDimensionalSectorMaps(monomial);
  if (stats != nullptr && use_direct_monomial_path) {
    ++stats->rank2_monomial_direct_hits;
  }

  for (const auto &input_entry : input_bsdt.GetBlkIdxDataBlkMap()) {
    const auto &input_blk = input_entry.second;
    const auto &matching_op_blk_idxs =
        op_blk_idxs_by_input_sector[input_blk.blk_coors[rank2_axis]];
    for (const size_t op_blk_idx : matching_op_blk_idxs) {
      const auto &op_blk = op_blocks.at(op_blk_idx);
      CoorsT rank2_output_blk_coors = input_blk.blk_coors;
      rank2_output_blk_coors[rank2_axis] = op_blk.blk_coors[1];

      if (use_direct_monomial_path) {
        for (const auto &monomial_entry : monomial.entries) {
          if (monomial_entry.coef == ElemT(0) ||
              monomial_entry.in_sector !=
                  rank2_output_blk_coors[monomial_axis]) {
            continue;
          }
          CoorsT output_blk_coors = rank2_output_blk_coors;
          output_blk_coors[monomial_axis] = monomial_entry.out_sector;
          const size_t output_blk_idx =
              output_bsdt.BlkCoorsToBlkIdx(output_blk_coors);
          const auto &output_blk = output_blocks.at(output_blk_idx);
          detail::AddRank2AxisBlock(
              input_data + input_blk.data_offset,
              op_data + op_blk.data_offset,
              output_data + output_blk.data_offset,
              input_blk.shape,
              op_blk.shape,
              output_blk.shape,
              rank2_axis,
              stats,
              monomial_entry.coef
          );
        }
        continue;
      }

      bool has_matching_nonzero_monomial_entry = false;
      for (const auto &monomial_entry : monomial.entries) {
        if (monomial_entry.coef != ElemT(0) &&
            monomial_entry.in_sector ==
                rank2_output_blk_coors[monomial_axis]) {
          has_matching_nonzero_monomial_entry = true;
          break;
        }
      }
      if (!has_matching_nonzero_monomial_entry) {
        continue;
      }

      ShapeT rank2_output_shape = input_blk.shape;
      rank2_output_shape[rank2_axis] = op_blk.shape[1];
      const size_t rank2_output_size =
          std::accumulate(rank2_output_shape.begin(),
                          rank2_output_shape.end(),
                          size_t(1),
                          std::multiplies<size_t>());
      const size_t workspace_bytes = rank2_output_size * sizeof(ElemT);
      ElemT *rank2_output_data =
          static_cast<ElemT *>(qlten::QLMalloc(workspace_bytes));
      qlten::QLMemset(rank2_output_data, 0, workspace_bytes);
      if (stats != nullptr) {
        stats->rank2_monomial_block_workspace_bytes += workspace_bytes;
      }
      detail::AddRank2AxisBlock(
          input_data + input_blk.data_offset,
          op_data + op_blk.data_offset,
          rank2_output_data,
          input_blk.shape,
          op_blk.shape,
          rank2_output_shape,
          rank2_axis,
          stats
      );

      for (const auto &monomial_entry : monomial.entries) {
        if (monomial_entry.in_sector !=
            rank2_output_blk_coors[monomial_axis]) {
          continue;
        }
        CoorsT output_blk_coors = rank2_output_blk_coors;
        output_blk_coors[monomial_axis] = monomial_entry.out_sector;
        const size_t output_blk_idx =
            output_bsdt.BlkCoorsToBlkIdx(output_blk_coors);
        const auto &output_blk = output_blocks.at(output_blk_idx);
        detail::AddAxisMonomialEntryBlock(
            rank2_output_data,
            output_data + output_blk.data_offset,
            rank2_output_shape,
            output_blk.shape,
            monomial_axis,
            monomial_entry.in_offset,
            monomial_entry.out_offset,
            monomial_entry.coef
        );
      }
      qlten::QLFree(rank2_output_data);
    }
  }
}

/**
 * @brief Apply two fused rank-2 operators, then consume the owned output
 *        in-place with an axis diagonal operator.
 */
template<typename ElemT, typename QNT>
void ApplyTwoRank2ThenAxisDiagonalPreserveOrder(
    const QLTensor<ElemT, QNT> &input,
    const QLTensor<ElemT, QNT> &op1,
    size_t axis1,
    const QLTensor<ElemT, QNT> &op2,
    size_t axis2,
    size_t diagonal_axis,
    const AxisDiagonalOp<ElemT, QNT> &diagonal,
    QLTensor<ElemT, QNT> &out,
    AxisOpStats *stats = nullptr
) {
  ApplyTwoRank2ToAxesPreserveOrder(input,
                                   op1,
                                   axis1,
                                   op2,
                                   axis2,
                                   out,
                                   stats);
  ApplyAxisDiagonalInPlace(out, diagonal_axis, diagonal);
  if (stats != nullptr) {
    ++stats->in_place_scale_hits;
  }
}

/**
 * @brief Apply two fused rank-2 operators, then consume the owned output
 *        in-place with sector scalars.
 */
template<typename ElemT, typename QNT>
void ApplyTwoRank2ThenAxisSectorScalarsPreserveOrder(
    const QLTensor<ElemT, QNT> &input,
    const QLTensor<ElemT, QNT> &op1,
    size_t axis1,
    const QLTensor<ElemT, QNT> &op2,
    size_t axis2,
    const AxisSectorScalar<ElemT, QNT> &scale,
    QLTensor<ElemT, QNT> &out,
    AxisOpStats *stats = nullptr
) {
  ApplyTwoRank2ToAxesPreserveOrder(input,
                                   op1,
                                   axis1,
                                   op2,
                                   axis2,
                                   out,
                                   stats);
  ScaleAxisSectorsInPlace(out, scale);
  if (stats != nullptr) {
    ++stats->in_place_scale_hits;
  }
}

/**
 * @brief Apply two rank-2 operators and fold final sector scalars into output
 *        writes.
 *
 * This is equivalent to ApplyTwoRank2ToAxesPreserveOrder() followed by
 * ScaleAxisSectorsInPlace(), but it avoids the second full raw-data scaling
 * pass. `scale.axis` refers to the final preserve-order output layout.
 */
template<typename ElemT, typename QNT>
void ApplyTwoRank2ThenAxisSectorScalarsFusedPreserveOrder(
    const QLTensor<ElemT, QNT> &input,
    const QLTensor<ElemT, QNT> &op1,
    size_t axis1,
    const QLTensor<ElemT, QNT> &op2,
    size_t axis2,
    const AxisSectorScalar<ElemT, QNT> &scale,
    QLTensor<ElemT, QNT> &out,
    AxisOpStats *stats = nullptr
) {
  static_assert(!Fermionicable<QNT>::IsFermionic(),
                "ApplyTwoRank2ThenAxisSectorScalarsFusedPreserveOrder is "
                "bosonic-only.");
  constexpr const char *kFunc =
      "ApplyTwoRank2ThenAxisSectorScalarsFusedPreserveOrder";
  if (stats != nullptr) {
    *stats = AxisOpStats{};
  }
  if (&input == &out || &op1 == &out || &op2 == &out) {
    throw std::invalid_argument(
        detail::AxisOpsPrefix(kFunc) + "output aliasing is not allowed."
    );
  }
  if (axis1 == axis2) {
    throw std::invalid_argument(
        detail::AxisOpsPrefix(kFunc) + "axes must be distinct."
    );
  }
  detail::RequireUsableAxis(input, axis1, kFunc);
  detail::RequireUsableAxis(input, axis2, kFunc);
  detail::RequireRank2OpMatchesAxis(input.GetIndex(axis1), op1, kFunc);
  detail::RequireRank2OpMatchesAxis(input.GetIndex(axis2), op2, kFunc);
  detail::RequireRawDataIfBlocksPresent(input, kFunc, "input");
  detail::RequireRawDataIfBlocksPresent(op1, kFunc, "op1");
  detail::RequireRawDataIfBlocksPresent(op2, kFunc, "op2");

  const auto output_indexes =
      detail::TwoRank2OutputIndexes(input, axis1, op1, axis2, op2);
  detail::RequireAxisSectorScalarMatchesOutput(output_indexes, scale, kFunc);
  const auto output_topology =
      detail::GenerateTwoRank2OutputBlockTopology(input,
                                                  op1,
                                                  axis1,
                                                  op2,
                                                  axis2,
                                                  output_indexes);
  out = QLTensor<ElemT, QNT>(output_indexes);
  if (stats != nullptr) {
    ++stats->output_tensor_rebuilds;
#ifdef QLTEN_DMRG_ENABLE_SCALAR_TWO_RANK2_FALLBACK
    ++stats->two_rank2_fused_hits;
#else
    ++stats->two_rank2_block_gemm_hits;
#endif
  }
  if (!output_topology.blk_coors_s.empty()) {
    out.GetBlkSparDataTen().DataBlksInsert(output_topology.blk_idxs,
                                           output_topology.blk_coors_s,
                                           true,
                                           true);
  }
  if (output_topology.blk_coors_s.empty()) {
    return;
  }
  detail::RequireRawDataIfBlocksPresent(out, kFunc, "out");

  const auto &input_bsdt = input.GetBlkSparDataTen();
  const auto &op1_bsdt = op1.GetBlkSparDataTen();
  const auto &op2_bsdt = op2.GetBlkSparDataTen();
  auto &output_bsdt = out.GetBlkSparDataTen();
  const ElemT *input_data = input_bsdt.GetActualRawDataPtr();
  const ElemT *op1_data = op1_bsdt.GetActualRawDataPtr();
  const ElemT *op2_data = op2_bsdt.GetActualRawDataPtr();
  ElemT *output_data = output_bsdt.GetActualRawDataPtr();
  const auto &op1_blocks = op1_bsdt.GetBlkIdxDataBlkMap();
  const auto &op2_blocks = op2_bsdt.GetBlkIdxDataBlkMap();
  const auto &output_blocks = output_bsdt.GetBlkIdxDataBlkMap();
  const auto op1_blk_idxs_by_input_sector =
      detail::Rank2OpBlockIndicesByInputSector(op1);
  const auto op2_blk_idxs_by_input_sector =
      detail::Rank2OpBlockIndicesByInputSector(op2);

  for (const auto &input_entry : input_bsdt.GetBlkIdxDataBlkMap()) {
    const auto &input_blk = input_entry.second;
    const auto &op1_blk_idxs =
        op1_blk_idxs_by_input_sector[input_blk.blk_coors[axis1]];
    const auto &op2_blk_idxs =
        op2_blk_idxs_by_input_sector[input_blk.blk_coors[axis2]];
    for (const size_t op1_blk_idx : op1_blk_idxs) {
      const auto &op1_blk = op1_blocks.at(op1_blk_idx);
      for (const size_t op2_blk_idx : op2_blk_idxs) {
        const auto &op2_blk = op2_blocks.at(op2_blk_idx);
        CoorsT output_blk_coors = input_blk.blk_coors;
        output_blk_coors[axis1] = op1_blk.blk_coors[1];
        output_blk_coors[axis2] = op2_blk.blk_coors[1];
        const ElemT alpha =
            scale.values_by_sector[output_blk_coors[scale.axis]];
        if (alpha == ElemT(0)) {
          continue;
        }
        const size_t output_blk_idx =
            output_bsdt.BlkCoorsToBlkIdx(output_blk_coors);
        const auto &output_blk = output_blocks.at(output_blk_idx);
#ifdef QLTEN_DMRG_ENABLE_SCALAR_TWO_RANK2_FALLBACK
        detail::AddTwoRank2AxesBlock(
            input_data + input_blk.data_offset,
            op1_data + op1_blk.data_offset,
            op2_data + op2_blk.data_offset,
            output_data + output_blk.data_offset,
            input_blk.shape,
            op1_blk.shape,
            op2_blk.shape,
            output_blk.shape,
            axis1,
            axis2,
            stats,
            alpha
        );
#else
        detail::AddTwoRank2AxesBlockGemm(
            input_data + input_blk.data_offset,
            op1_data + op1_blk.data_offset,
            op2_data + op2_blk.data_offset,
            output_data + output_blk.data_offset,
            input_blk.shape,
            op1_blk.shape,
            op2_blk.shape,
            output_blk.shape,
            axis1,
            axis2,
            stats,
            alpha
        );
#endif
      }
    }
  }
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
 * @throws std::runtime_error when a tensor has stored blocks but no allocated
 *         raw data.
 */
template<typename ElemT, typename QNT>
void MoveHeadAxisToTail(
    const QLTensor<ElemT, QNT> &input,
    QLTensor<ElemT, QNT> &out,
    AxisLayoutMoveStats *stats = nullptr
) {
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
    detail::InitializeOutputLikeInput(input, out);
    detail::RequireRawDataIfBlocksPresent(out, kFunc, "out");
    detail::CopyTensorRawData(input, out);
    if (stats != nullptr) {
      stats->layout_gb += detail::RawReadWriteGb<ElemT>(
          input.GetBlkSparDataTen().GetActualRawDataSize());
    }
    return;
  }

  const auto output_indexes =
      detail::MoveHeadAxisToTailIndexes(input.GetIndexes());
  const auto output_topology =
      detail::GenerateMoveHeadAxisToTailOutputBlockTopology(input,
                                                            output_indexes);

  out = QLTensor<ElemT, QNT>(output_indexes);
  if (!output_topology.blk_coors_s.empty()) {
    out.GetBlkSparDataTen().DataBlksInsert(output_topology.blk_idxs,
                                           output_topology.blk_coors_s,
                                           true,
                                           true);
  }
  if (output_topology.blk_coors_s.empty()) {
    return;
  }
  detail::RequireRawDataIfBlocksPresent(out, kFunc, "out");

  const auto &input_bsdt = input.GetBlkSparDataTen();
  auto &output_bsdt = out.GetBlkSparDataTen();
  const ElemT *input_data = input_bsdt.GetActualRawDataPtr();
  ElemT *output_data = output_bsdt.GetActualRawDataPtr();
  const auto &input_blocks = input_bsdt.GetBlkIdxDataBlkMap();
  const auto &output_blocks = output_bsdt.GetBlkIdxDataBlkMap();

  static_assert(detail::IsHpNumericElem<ElemT>(),
                "MoveHeadAxisToTail requires a hp_numeric-supported ElemT.");
  std::vector<const ElemT *> input_ptrs;
  std::vector<ElemT *> output_ptrs;
  std::vector<size_t> rows;
  std::vector<size_t> cols;
  input_ptrs.reserve(input_blocks.size());
  output_ptrs.reserve(input_blocks.size());
  rows.reserve(input_blocks.size());
  cols.reserve(input_blocks.size());
  size_t output_block_pos = 0;
  for (const auto &entry : input_blocks) {
    const auto &input_blk = entry.second;
    const size_t output_blk_idx = output_topology.blk_idxs[output_block_pos];
    ++output_block_pos;
    const auto &output_blk = output_blocks.at(output_blk_idx);
    input_ptrs.push_back(input_data + input_blk.data_offset);
    output_ptrs.push_back(output_data + output_blk.data_offset);
    rows.push_back(input_blk.shape.front());
    cols.push_back(std::accumulate(input_blk.shape.begin() + 1,
                                   input_blk.shape.end(),
                                   size_t(1),
                                   std::multiplies<size_t>()));
  }

  detail::MatrixTransposeBlocks(input_ptrs, output_ptrs, rows, cols);
  if (stats != nullptr) {
    stats->block_matrix_transpose_calls += input_ptrs.size();
    stats->layout_gb += detail::RawReadWriteGb<ElemT>(
        input_bsdt.GetActualRawDataSize());
  }
}

}  // namespace qlten::dmrg

#endif /* ifndef QLTEN_TENSOR_MANIPULATION_DMRG_AXIS_OPS_H */
