// SPDX-License-Identifier: LGPL-3.0-only
/*
 * Description: Cost estimators for tensor primitive operations.
 */

/**
@file tensor_op_cost.h
@brief Metadata-only cost estimators for tensor primitive operations.
*/
#ifndef QLTEN_TENSOR_MANIPULATION_TENSOR_OP_COST_H
#define QLTEN_TENSOR_MANIPULATION_TENSOR_OP_COST_H

#include <algorithm>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "qlten/qltensor_all.h"
#include "qlten/tensor_manipulation/dmrg/axis_ops.h"
#include "qlten/tensor_manipulation/ten_ctrct.h"

namespace qlten {

/**
 * @brief Metadata for one stored block in a block-sparse tensor layout.
 */
struct BlockSparseTensorBlock {
  size_t block_index = 0;
  CoorsT block_coords;
  ShapeT shape;
  size_t data_offset = 0;
  size_t element_count = 0;
};

inline bool operator==(
    const BlockSparseTensorBlock &lhs,
    const BlockSparseTensorBlock &rhs
) {
  return lhs.block_index == rhs.block_index &&
         lhs.block_coords == rhs.block_coords &&
         lhs.shape == rhs.shape &&
         lhs.data_offset == rhs.data_offset &&
         lhs.element_count == rhs.element_count;
}

/**
 * @brief Metadata-only block-sparse tensor topology.
 *
 * This type intentionally carries indexes, block coordinates, shapes, and raw
 * offsets only.  It does not own numeric raw data and should not be treated as
 * an executable tensor.
 */
template<typename QNT>
struct BlockSparseTensorLayout {
  bool is_default = true;
  IndexVec<QNT> indexes;
  std::map<size_t, BlockSparseTensorBlock> blocks;
  size_t scalar_raw_elem_count = 0;

  bool IsDefault() const { return is_default; }
  bool IsScalar() const { return !is_default && indexes.empty(); }
  size_t Rank() const { return indexes.size(); }
  size_t BlockCount() const { return blocks.size(); }

  size_t RawElementCount() const {
    if (IsScalar()) {
      return scalar_raw_elem_count == 0 ? 1 : scalar_raw_elem_count;
    }
    size_t total = 0;
    for (const auto &entry : blocks) {
      total += entry.second.element_count;
    }
    return total;
  }
};

template<typename QNT>
using TensorLayout = BlockSparseTensorLayout<QNT>;

template<typename QNT>
bool operator==(
    const BlockSparseTensorLayout<QNT> &lhs,
    const BlockSparseTensorLayout<QNT> &rhs
) {
  return lhs.is_default == rhs.is_default &&
         lhs.indexes == rhs.indexes &&
         lhs.blocks == rhs.blocks &&
         lhs.scalar_raw_elem_count == rhs.scalar_raw_elem_count;
}

/**
 * @brief Estimated work for a tensor primitive.
 *
 * `read_bytes` and `write_bytes` are logical touched bytes implied by the
 * primitive schedule, not measured cache or memory-system traffic.
 * `temp_peak_bytes` is estimated peak temporary workspace, not cumulative
 * allocation volume.
 */
template<typename QNT>
struct TensorOpCost {
  double flops = 0.0;
  size_t gemm_count = 0;
  size_t executable_block_pair_count = 0;
  size_t candidate_block_pair_count = 0;
  size_t output_block_count = 0;
  size_t output_raw_elem_count = 0;
  size_t read_bytes = 0;
  size_t write_bytes = 0;
  size_t temp_peak_bytes = 0;
  BlockSparseTensorLayout<QNT> output_layout;
};

/**
 * @brief Extract metadata-only layout/topology from a QLTensor.
 */
template<typename ElemT, typename QNT>
BlockSparseTensorLayout<QNT> MakeBlockSparseTensorLayout(
    const QLTensor<ElemT, QNT> &tensor
) {
  BlockSparseTensorLayout<QNT> layout;
  layout.is_default = tensor.IsDefault();
  if (layout.is_default) {
    return layout;
  }
  layout.indexes = tensor.GetIndexes();
  if (tensor.IsScalar()) {
    layout.scalar_raw_elem_count = 1;
    return layout;
  }
  const auto &blocks = tensor.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  for (const auto &entry : blocks) {
    const auto &data_blk = entry.second;
    BlockSparseTensorBlock block;
    block.block_index = entry.first;
    block.block_coords = data_blk.blk_coors;
    block.shape = data_blk.shape;
    block.data_offset = data_blk.data_offset;
    block.element_count = data_blk.size;
    layout.blocks.emplace(entry.first, std::move(block));
  }
  return layout;
}

/**
 * @brief Build a QLTensor shell from metadata-only layout/topology.
 *
 * The returned tensor owns index/block metadata but does not allocate raw numeric
 * data.  It is intended only for estimator composition and other dry-run paths.
 */
template<typename ElemT, typename QNT>
QLTensor<ElemT, QNT> MakeTensorShellFromLayout(
    const BlockSparseTensorLayout<QNT> &layout
) {
  if (layout.IsDefault()) {
    return QLTensor<ElemT, QNT>();
  }

  QLTensor<ElemT, QNT> shell(layout.indexes);
  if (layout.IsScalar() || layout.blocks.empty()) {
    return shell;
  }

  std::vector<size_t> block_indices;
  std::vector<CoorsT> block_coords;
  block_indices.reserve(layout.blocks.size());
  block_coords.reserve(layout.blocks.size());
  for (const auto &entry : layout.blocks) {
    block_indices.push_back(entry.first);
    block_coords.push_back(entry.second.block_coords);
  }
  shell.GetBlkSparDataTen().DataBlksInsert(
      block_indices, block_coords, false, false);
  return shell;
}

namespace tensor_cost_detail {

template<typename ElemT>
constexpr bool IsComplexElement() {
  using BareElemT = typename std::remove_cv<ElemT>::type;
  using RealT = typename RealTypeTrait<BareElemT>::type;
  return !std::is_same<BareElemT, RealT>::value;
}

template<typename ElemT>
constexpr double DenseMultiplyAddFlops() {
  return IsComplexElement<ElemT>() ? 8.0 : 2.0;
}

template<typename ElemT>
constexpr double ScaleElementFlops() {
  return IsComplexElement<ElemT>() ? 6.0 : 1.0;
}

template<typename ElemT>
constexpr double LinearCombineElementFlops() {
  return IsComplexElement<ElemT>() ? 8.0 : 2.0;
}

template<typename ElemT>
constexpr double FusedTwoRank2ElementFlops() {
  return IsComplexElement<ElemT>() ? 14.0 : 3.0;
}

inline size_t ShapeProductBeforeAxis(
    const ShapeT &shape,
    const size_t axis
) {
  return std::accumulate(shape.begin(), shape.begin() + axis,
                         size_t(1), std::multiplies<size_t>());
}

inline void RequireAxesSet(
    const size_t lhs_rank,
    const size_t rhs_rank,
    const std::vector<std::vector<size_t>> &axes_set
) {
  if (axes_set.size() != 2 || axes_set[0].size() != axes_set[1].size()) {
    throw std::invalid_argument(
        "EstimateContractCost: axes_set must contain two equally sized axes lists.");
  }
  std::vector<bool> lhs_seen(lhs_rank, false);
  std::vector<bool> rhs_seen(rhs_rank, false);
  for (size_t i = 0; i < axes_set[0].size(); ++i) {
    const size_t lhs_axis = axes_set[0][i];
    const size_t rhs_axis = axes_set[1][i];
    if (lhs_axis >= lhs_rank || rhs_axis >= rhs_rank) {
      throw std::invalid_argument(
          "EstimateContractCost: contraction axis is out of range.");
    }
    if (lhs_seen[lhs_axis] || rhs_seen[rhs_axis]) {
      throw std::invalid_argument(
          "EstimateContractCost: repeated contraction axis.");
    }
    lhs_seen[lhs_axis] = true;
    rhs_seen[rhs_axis] = true;
  }
}

template<typename ElemT, typename QNT>
void RequireContractInputs(
    const QLTensor<ElemT, QNT> &lhs,
    const QLTensor<ElemT, QNT> &rhs,
    const std::vector<std::vector<size_t>> &axes_set
) {
  if (lhs.IsDefault() || rhs.IsDefault()) {
    throw std::invalid_argument(
        "EstimateContractCost: input tensors must not be default.");
  }
  if (lhs.IsScalar() || rhs.IsScalar()) {
    throw std::invalid_argument(
        "EstimateContractCost: scalar input contraction is not supported.");
  }
  RequireAxesSet(lhs.Rank(), rhs.Rank(), axes_set);
  for (size_t i = 0; i < axes_set[0].size(); ++i) {
    if (lhs.GetIndex(axes_set[0][i]) !=
        InverseIndex(rhs.GetIndex(axes_set[1][i]))) {
      throw std::invalid_argument(
          "EstimateContractCost: contraction indexes do not match.");
    }
  }
}

inline std::vector<std::vector<size_t>> AxesSetFromPairs(
    const std::vector<std::pair<size_t, size_t>> &contract_axes
) {
  std::vector<std::vector<size_t>> axes_set(2);
  axes_set[0].reserve(contract_axes.size());
  axes_set[1].reserve(contract_axes.size());
  for (const auto &axes : contract_axes) {
    axes_set[0].push_back(axes.first);
    axes_set[1].push_back(axes.second);
  }
  return axes_set;
}

template<typename QNT>
void PopulateOutputStats(TensorOpCost<QNT> &cost) {
  cost.output_block_count = cost.output_layout.BlockCount();
  cost.output_raw_elem_count = cost.output_layout.RawElementCount();
}

template<typename QNT>
void RequireSameIndexes(
    const BlockSparseTensorLayout<QNT> &lhs,
    const BlockSparseTensorLayout<QNT> &rhs,
    const char *func
) {
  if (lhs.IsDefault() || rhs.IsDefault()) {
    throw std::invalid_argument(std::string(func) +
                                ": layouts must not be default.");
  }
  if (lhs.indexes != rhs.indexes) {
    throw std::invalid_argument(std::string(func) +
                                ": tensor indexes do not match.");
  }
}

template<typename QNT>
BlockSparseTensorLayout<QNT> UnionLayouts(
    const BlockSparseTensorLayout<QNT> &lhs,
    const BlockSparseTensorLayout<QNT> &rhs,
    const char *func
) {
  if (lhs.IsDefault()) {
    return rhs;
  }
  if (rhs.IsDefault()) {
    return lhs;
  }
  RequireSameIndexes(lhs, rhs, func);
  BlockSparseTensorLayout<QNT> result = lhs;
  for (const auto &entry : rhs.blocks) {
    const auto out_it = result.blocks.find(entry.first);
    if (out_it == result.blocks.end()) {
      result.blocks.emplace(entry.first, entry.second);
      continue;
    }
    if (out_it->second.block_coords != entry.second.block_coords ||
        out_it->second.shape != entry.second.shape ||
        out_it->second.element_count != entry.second.element_count) {
      throw std::invalid_argument(std::string(func) +
                                  ": incompatible block topology.");
    }
  }
  size_t offset = 0;
  for (auto &entry : result.blocks) {
    entry.second.data_offset = offset;
    offset += entry.second.element_count;
  }
  return result;
}

template<typename ElemT, typename QNT>
size_t LhsTransposeTempBytesForTasks(
    const std::vector<RawDataCtrctTask> &tasks,
    const BlockSparseDataTensor<ElemT, QNT> &bsdt,
    const bool needs_transpose
) {
  if (!needs_transpose) {
    return 0;
  }
  std::unordered_set<size_t> seen;
  size_t bytes = 0;
  const auto &blocks = bsdt.GetBlkIdxDataBlkMap();
  for (const auto &task : tasks) {
    if (seen.insert(task.a_blk_idx).second) {
      bytes += blocks.at(task.a_blk_idx).size * sizeof(ElemT);
    }
  }
  return bytes;
}

template<typename ElemT, typename QNT>
size_t RhsTransposeTempBytesForTasks(
    const std::vector<RawDataCtrctTask> &tasks,
    const BlockSparseDataTensor<ElemT, QNT> &bsdt,
    const bool needs_transpose
) {
  if (!needs_transpose) {
    return 0;
  }
  std::unordered_set<size_t> seen;
  size_t bytes = 0;
  const auto &blocks = bsdt.GetBlkIdxDataBlkMap();
  for (const auto &task : tasks) {
    if (seen.insert(task.b_blk_idx).second) {
      bytes += blocks.at(task.b_blk_idx).size * sizeof(ElemT);
    }
  }
  return bytes;
}

template<typename ElemT, typename QNT>
void AddGemmCost(
    TensorOpCost<QNT> &cost,
    const size_t m,
    const size_t k,
    const size_t n,
    const size_t count = 1,
    const bool reads_existing_output = true
) {
  const size_t elem_size = sizeof(ElemT);
  const double flops_per_gemm =
      DenseMultiplyAddFlops<ElemT>() *
      static_cast<double>(m) *
      static_cast<double>(k) *
      static_cast<double>(n);
  cost.flops += static_cast<double>(count) * flops_per_gemm;
  cost.gemm_count += count;
  cost.read_bytes += count * (m * k + k * n) * elem_size;
  if (reads_existing_output) {
    cost.read_bytes += count * (m * n) * elem_size;
  }
  cost.write_bytes += count * (m * n) * elem_size;
}

template<typename ElemT, typename QNT>
void AddContractTaskCost(
    TensorOpCost<QNT> &cost,
    const RawDataCtrctTask &task
) {
  AddGemmCost<ElemT>(cost, task.m, task.k, task.n, 1, task.beta != 0.0);
}

template<typename ElemT, typename QNT>
void AddRank2BlockCost(
    TensorOpCost<QNT> &cost,
    const ShapeT &input_shape,
    const ShapeT &output_shape,
    const size_t axis,
    const dmrg::Rank2AxisApplyGemmMode mode
) {
  const size_t input_axis_dim = input_shape[axis];
  const size_t output_axis_dim = output_shape[axis];
  const size_t inner_size = dmrg::detail::AxisInnerSize(input_shape, axis);
  const size_t outer_size = ShapeProductBeforeAxis(input_shape, axis);

  if (axis == 0) {
    AddGemmCost<ElemT>(
        cost, output_axis_dim, input_axis_dim, inner_size, 1, true);
    return;
  }
  if (axis + 1 == input_shape.size()) {
    AddGemmCost<ElemT>(
        cost, outer_size, input_axis_dim, output_axis_dim, 1, true);
    return;
  }
  AddGemmCost<ElemT>(
      cost, output_axis_dim, input_axis_dim, inner_size, outer_size, true);
  if (mode == dmrg::Rank2AxisApplyGemmMode::kBatch) {
    cost.temp_peak_bytes = std::max(
        cost.temp_peak_bytes,
        output_axis_dim * input_axis_dim * sizeof(ElemT));
  }
}

template<typename ElemT, typename QNT>
void AddLogicalRawCopyCost(
    TensorOpCost<QNT> &cost,
    const size_t elem_count
) {
  cost.read_bytes += elem_count * sizeof(ElemT);
  cost.write_bytes += elem_count * sizeof(ElemT);
}

template<typename ElemT>
double ElementwiseFlops(const size_t elem_count) {
  return static_cast<double>(elem_count) * LinearCombineElementFlops<ElemT>();
}

}  // namespace tensor_cost_detail

/**
 * @brief Estimate a general tensor contraction using Contract() task topology.
 */
template<typename ElemT, typename QNT>
TensorOpCost<QNT> EstimateContractCost(
    const QLTensor<ElemT, QNT> &lhs,
    const QLTensor<ElemT, QNT> &rhs,
    const std::vector<std::vector<size_t>> &axes_set
) {
  tensor_cost_detail::RequireContractInputs(lhs, rhs, axes_set);

  const auto saved_axes_set =
      TenCtrctGenSavedAxesSet(lhs.Rank(), rhs.Rank(), axes_set);
  std::vector<int> lhs_trans_orders;
  std::vector<int> rhs_trans_orders;
  const bool lhs_needs_transpose =
      TenCtrctNeedTransCheck(saved_axes_set[0], axes_set[0],
                             lhs_trans_orders);
  const bool rhs_needs_transpose =
      TenCtrctNeedTransCheck(axes_set[1], saved_axes_set[1],
                             rhs_trans_orders);

  QLTensor<ElemT, QNT> output;
  TenCtrctInitResTen(&lhs, &rhs, saved_axes_set, &output);
  auto tasks = output.GetBlkSparDataTen().DataBlkGenForTenCtrct(
      lhs.GetBlkSparDataTen(),
      rhs.GetBlkSparDataTen(),
      axes_set,
      saved_axes_set
  );

  TensorOpCost<QNT> cost;
  cost.candidate_block_pair_count =
      lhs.GetBlkSparDataTen().GetBlkIdxDataBlkMap().size() *
      rhs.GetBlkSparDataTen().GetBlkIdxDataBlkMap().size();
  cost.executable_block_pair_count = tasks.size();
  cost.output_layout = MakeBlockSparseTensorLayout(output);
  tensor_cost_detail::PopulateOutputStats(cost);

  for (const auto &task : tasks) {
    tensor_cost_detail::AddContractTaskCost<ElemT>(cost, task);
  }

  const size_t lhs_transpose_bytes =
      tensor_cost_detail::LhsTransposeTempBytesForTasks<ElemT>(
          tasks, lhs.GetBlkSparDataTen(), lhs_needs_transpose);
  const size_t rhs_transpose_bytes =
      tensor_cost_detail::RhsTransposeTempBytesForTasks<ElemT>(
          tasks, rhs.GetBlkSparDataTen(), rhs_needs_transpose);
  cost.temp_peak_bytes = lhs_transpose_bytes + rhs_transpose_bytes;
  cost.read_bytes += cost.temp_peak_bytes;
  cost.write_bytes += cost.temp_peak_bytes;
  return cost;
}

/**
 * @brief Estimate a general tensor contraction from pair-form axes.
 */
template<typename ElemT, typename QNT>
TensorOpCost<QNT> EstimateContractCost(
    const QLTensor<ElemT, QNT> &lhs,
    const QLTensor<ElemT, QNT> &rhs,
    const std::vector<std::pair<size_t, size_t>> &contract_axes
) {
  return EstimateContractCost(
      lhs,
      rhs,
      tensor_cost_detail::AxesSetFromPairs(contract_axes));
}

/**
 * @brief Estimate ApplyRank2ToAxisPreserveOrder(), including loop/batch mode.
 */
template<typename ElemT, typename QNT>
TensorOpCost<QNT> EstimateRank2AxisApplyCost(
    const QLTensor<ElemT, QNT> &input,
    const QLTensor<ElemT, QNT> &rank2_op,
    const size_t target_axis,
    const dmrg::Rank2AxisApplyGemmMode mode =
        dmrg::Rank2AxisApplyGemmMode::kLoop
) {
  static_assert(!Fermionicable<QNT>::IsFermionic(),
                "EstimateRank2AxisApplyCost is bosonic-only.");
  constexpr const char *kFunc = "EstimateRank2AxisApplyCost";
  dmrg::detail::RequireUsableAxis(input, target_axis, kFunc);
  dmrg::detail::RequireRank2OpMatchesAxis(
      input.GetIndex(target_axis), rank2_op, kFunc);

  const auto output_indexes =
      dmrg::detail::Rank2OutputIndexes(input, target_axis, rank2_op);
  dmrg::AxisOpStats stats;
  const auto output_topology =
      dmrg::detail::GenerateRank2OutputBlockTopology(
          input, target_axis, rank2_op, output_indexes, &stats);
  QLTensor<ElemT, QNT> output(output_indexes);
  if (!output_topology.blk_coors_s.empty()) {
    output.GetBlkSparDataTen().DataBlksInsert(output_topology.blk_idxs,
                                              output_topology.blk_coors_s,
                                              false,
                                              false);
  }

  TensorOpCost<QNT> cost;
  cost.candidate_block_pair_count =
      input.GetBlkSparDataTen().GetBlkIdxDataBlkMap().size() *
      rank2_op.GetBlkSparDataTen().GetBlkIdxDataBlkMap().size();
  cost.executable_block_pair_count = stats.rank2_topology_block_pair_visits;
  cost.output_layout = MakeBlockSparseTensorLayout(output);
  tensor_cost_detail::PopulateOutputStats(cost);

  const auto &input_blocks = input.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  const auto &op_blocks = rank2_op.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  const auto &output_blocks = output.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  const auto op_blk_idxs_by_input_sector =
      dmrg::detail::Rank2OpBlockIndicesByInputSector(rank2_op);

  for (const auto &input_entry : input_blocks) {
    const auto &input_blk = input_entry.second;
    const auto &matching_op_blk_idxs =
        op_blk_idxs_by_input_sector[input_blk.blk_coors[target_axis]];
    for (const size_t op_blk_idx : matching_op_blk_idxs) {
      const auto &op_blk = op_blocks.at(op_blk_idx);
      CoorsT output_blk_coors_for_pair = input_blk.blk_coors;
      output_blk_coors_for_pair[target_axis] = op_blk.blk_coors[1];
      const size_t output_blk_idx =
          output.GetBlkSparDataTen().BlkCoorsToBlkIdx(
              output_blk_coors_for_pair);
      const auto &output_blk = output_blocks.at(output_blk_idx);
      tensor_cost_detail::AddRank2BlockCost<ElemT>(
          cost,
          input_blk.shape,
          output_blk.shape,
          target_axis,
          mode);
    }
  }
  return cost;
}

/**
 * @brief Estimate two-rank2 axis application output topology and work.
 *
 * Default builds use a block-local GEMM workspace and two rank-2 GEMM
 * applications per executable input/op1/op2 block triple. Defining
 * `QLTEN_DMRG_ENABLE_SCALAR_TWO_RANK2_FALLBACK` switches this estimate to the
 * opt-in scalar reference kernel.
 */
template<typename ElemT, typename QNT>
TensorOpCost<QNT> EstimateTwoRank2AxesApplyCost(
    const QLTensor<ElemT, QNT> &input,
    const QLTensor<ElemT, QNT> &op1,
    const size_t axis1,
    const QLTensor<ElemT, QNT> &op2,
    const size_t axis2
) {
  static_assert(!Fermionicable<QNT>::IsFermionic(),
                "EstimateTwoRank2AxesApplyCost is bosonic-only.");
  constexpr const char *kFunc = "EstimateTwoRank2AxesApplyCost";
  if (axis1 == axis2) {
    throw std::invalid_argument(
        std::string(kFunc) + ": axes must be distinct.");
  }
  dmrg::detail::RequireUsableAxis(input, axis1, kFunc);
  dmrg::detail::RequireUsableAxis(input, axis2, kFunc);
  dmrg::detail::RequireRank2OpMatchesAxis(input.GetIndex(axis1), op1, kFunc);
  dmrg::detail::RequireRank2OpMatchesAxis(input.GetIndex(axis2), op2, kFunc);

  const auto output_indexes =
      dmrg::detail::TwoRank2OutputIndexes(input, axis1, op1, axis2, op2);
  const auto output_topology =
      dmrg::detail::GenerateTwoRank2OutputBlockTopology(
          input, op1, axis1, op2, axis2, output_indexes);
  QLTensor<ElemT, QNT> output(output_indexes);
  if (!output_topology.blk_coors_s.empty()) {
    output.GetBlkSparDataTen().DataBlksInsert(output_topology.blk_idxs,
                                              output_topology.blk_coors_s,
                                              false,
                                              false);
  }

  TensorOpCost<QNT> cost;
  cost.candidate_block_pair_count =
      input.GetBlkSparDataTen().GetBlkIdxDataBlkMap().size() *
      op1.GetBlkSparDataTen().GetBlkIdxDataBlkMap().size() *
      op2.GetBlkSparDataTen().GetBlkIdxDataBlkMap().size();
  cost.output_layout = MakeBlockSparseTensorLayout(output);
  tensor_cost_detail::PopulateOutputStats(cost);

  const auto &input_blocks = input.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  const auto &op1_blocks = op1.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  const auto &op2_blocks = op2.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  const auto &output_bsdt = output.GetBlkSparDataTen();
  const auto &output_blocks = output_bsdt.GetBlkIdxDataBlkMap();
  const auto op1_blk_idxs_by_input_sector =
      dmrg::detail::Rank2OpBlockIndicesByInputSector(op1);
  const auto op2_blk_idxs_by_input_sector =
      dmrg::detail::Rank2OpBlockIndicesByInputSector(op2);

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
        ++cost.executable_block_pair_count;
#ifdef QLTEN_DMRG_ENABLE_SCALAR_TWO_RANK2_FALLBACK
        const size_t updates =
            input_blk.size * op1_blk.shape[1] * op2_blk.shape[1];
        cost.flops +=
            static_cast<double>(updates) *
            tensor_cost_detail::FusedTwoRank2ElementFlops<ElemT>();
        cost.read_bytes +=
            (input_blk.size + op1_blk.size + op2_blk.size + updates) *
            sizeof(ElemT);
        cost.write_bytes += updates * sizeof(ElemT);
#else
        CoorsT output_blk_coors = input_blk.blk_coors;
        output_blk_coors[axis1] = op1_blk.blk_coors[1];
        output_blk_coors[axis2] = op2_blk.blk_coors[1];
        const size_t output_blk_idx =
            output_bsdt.BlkCoorsToBlkIdx(output_blk_coors);
        const auto &output_blk = output_blocks.at(output_blk_idx);
        ShapeT intermediate_shape = input_blk.shape;
        intermediate_shape[axis1] = op1_blk.shape[1];
        const size_t intermediate_size =
            std::accumulate(intermediate_shape.begin(),
                            intermediate_shape.end(),
                            size_t(1),
                            std::multiplies<size_t>());
        cost.write_bytes += intermediate_size * sizeof(ElemT);
        cost.temp_peak_bytes =
            std::max(cost.temp_peak_bytes,
                     intermediate_size * sizeof(ElemT));
        tensor_cost_detail::AddRank2BlockCost<ElemT>(
            cost,
            input_blk.shape,
            intermediate_shape,
            axis1,
            dmrg::Rank2AxisApplyGemmMode::kBatch);
        tensor_cost_detail::AddRank2BlockCost<ElemT>(
            cost,
            intermediate_shape,
            output_blk.shape,
            axis2,
            dmrg::Rank2AxisApplyGemmMode::kBatch);
#endif
      }
    }
  }
  return cost;
}

/**
 * @brief Estimate diagonal axis metadata application.
 */
template<typename ElemT, typename QNT>
TensorOpCost<QNT> EstimateAxisMetadataApplyCost(
    const QLTensor<ElemT, QNT> &input,
    const size_t axis,
    const dmrg::AxisDiagonalOp<ElemT, QNT> &op
) {
  static_assert(!Fermionicable<QNT>::IsFermionic(),
                "EstimateAxisMetadataApplyCost is bosonic-only.");
  constexpr const char *kFunc = "EstimateAxisMetadataApplyCost";
  dmrg::detail::RequireUsableAxis(input, axis, kFunc);
  dmrg::detail::RequireDiagonalOpMatchesAxis(input.GetIndex(axis), op, kFunc);

  TensorOpCost<QNT> cost;
  cost.output_layout = MakeBlockSparseTensorLayout(input);
  tensor_cost_detail::PopulateOutputStats(cost);
  const size_t elem_count = cost.output_raw_elem_count;
  cost.flops =
      static_cast<double>(elem_count) *
      tensor_cost_detail::ScaleElementFlops<ElemT>();
  tensor_cost_detail::AddLogicalRawCopyCost<ElemT>(cost, elem_count);
  return cost;
}

/**
 * @brief Estimate sector-scalar axis metadata application.
 */
template<typename ElemT, typename QNT>
TensorOpCost<QNT> EstimateAxisMetadataApplyCost(
    const QLTensor<ElemT, QNT> &input,
    const dmrg::AxisSectorScalar<ElemT, QNT> &scale
) {
  static_assert(!Fermionicable<QNT>::IsFermionic(),
                "EstimateAxisMetadataApplyCost is bosonic-only.");
  constexpr const char *kFunc = "EstimateAxisMetadataApplyCost";
  dmrg::detail::RequireUsableAxis(input, scale.axis, kFunc);
  if (scale.values_by_sector.size() !=
      input.GetIndex(scale.axis).GetQNSctNum()) {
    throw std::invalid_argument(
        std::string(kFunc) +
        ": values_by_sector size must match the number of sectors.");
  }

  TensorOpCost<QNT> cost;
  cost.output_layout = MakeBlockSparseTensorLayout(input);
  tensor_cost_detail::PopulateOutputStats(cost);
  const size_t elem_count = cost.output_raw_elem_count;
  cost.flops =
      static_cast<double>(elem_count) *
      tensor_cost_detail::ScaleElementFlops<ElemT>();
  tensor_cost_detail::AddLogicalRawCopyCost<ElemT>(cost, elem_count);
  return cost;
}

/**
 * @brief Estimate monomial/scatter axis metadata application.
 */
template<typename ElemT, typename QNT>
TensorOpCost<QNT> EstimateAxisMetadataApplyCost(
    const QLTensor<ElemT, QNT> &input,
    const size_t axis,
    const dmrg::AxisMonomialOp<ElemT, QNT> &op
) {
  static_assert(!Fermionicable<QNT>::IsFermionic(),
                "EstimateAxisMetadataApplyCost is bosonic-only.");
  constexpr const char *kFunc = "EstimateAxisMetadataApplyCost";
  dmrg::detail::RequireUsableAxis(input, axis, kFunc);
  dmrg::detail::RequireMonomialOpMatchesAxis(input.GetIndex(axis), op, kFunc);

  const auto output_indexes =
      dmrg::detail::MonomialOutputIndexes(input, axis, op);
  const auto output_topology =
      dmrg::detail::GenerateMonomialOutputBlockTopology(
          input, axis, op, output_indexes);
  QLTensor<ElemT, QNT> output(output_indexes);
  if (!output_topology.blk_coors_s.empty()) {
    output.GetBlkSparDataTen().DataBlksInsert(output_topology.blk_idxs,
                                              output_topology.blk_coors_s,
                                              false,
                                              false);
  }

  TensorOpCost<QNT> cost;
  cost.output_layout = MakeBlockSparseTensorLayout(output);
  tensor_cost_detail::PopulateOutputStats(cost);

  const auto &input_blocks = input.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  for (const auto &input_entry : input_blocks) {
    const auto &input_blk = input_entry.second;
    const size_t inner_size = dmrg::detail::AxisInnerSize(input_blk.shape, axis);
    const size_t outer_size =
        tensor_cost_detail::ShapeProductBeforeAxis(input_blk.shape, axis);
    for (const auto &op_entry : op.entries) {
      if (op_entry.in_sector != input_blk.blk_coors[axis]) {
        continue;
      }
      ++cost.executable_block_pair_count;
      const size_t touched = outer_size * inner_size;
      cost.flops +=
          tensor_cost_detail::ElementwiseFlops<ElemT>(touched);
      cost.read_bytes += touched * sizeof(ElemT);
      cost.write_bytes += touched * sizeof(ElemT);
    }
  }
  return cost;
}

/**
 * @brief Estimate tensor linear combination from metadata-only layouts.
 */
template<typename ElemT, typename QNT>
TensorOpCost<QNT> EstimateTensorLinearCombineCost(
    const std::vector<BlockSparseTensorLayout<QNT>> &layouts
) {
  if (layouts.empty()) {
    throw std::invalid_argument(
        "EstimateTensorLinearCombineCost: layouts must not be empty.");
  }
  TensorOpCost<QNT> cost;
  cost.output_layout = layouts.front();
  if (cost.output_layout.IsDefault()) {
    throw std::invalid_argument(
        "EstimateTensorLinearCombineCost: layouts must not be default.");
  }
  cost.output_layout.blocks.clear();
  cost.output_layout.scalar_raw_elem_count =
      layouts.front().IsScalar() ? 1 : 0;

  for (const auto &layout : layouts) {
    tensor_cost_detail::RequireSameIndexes(
        layouts.front(), layout, "EstimateTensorLinearCombineCost");
    for (const auto &entry : layout.blocks) {
      cost.output_layout.blocks.emplace(entry.first, entry.second);
      cost.flops +=
          tensor_cost_detail::ElementwiseFlops<ElemT>(
              entry.second.element_count);
      tensor_cost_detail::AddLogicalRawCopyCost<ElemT, QNT>(
          cost, entry.second.element_count);
    }
    if (layout.IsScalar()) {
      cost.flops += tensor_cost_detail::LinearCombineElementFlops<ElemT>();
      tensor_cost_detail::AddLogicalRawCopyCost<ElemT, QNT>(cost, 1);
    }
  }
  size_t offset = 0;
  for (auto &entry : cost.output_layout.blocks) {
    entry.second.data_offset = offset;
    offset += entry.second.element_count;
  }
  tensor_cost_detail::PopulateOutputStats(cost);
  return cost;
}

/**
 * @brief Estimate tensor linear combination from an initializer list of layouts.
 */
template<typename ElemT, typename QNT>
TensorOpCost<QNT> EstimateTensorLinearCombineCost(
    std::initializer_list<BlockSparseTensorLayout<QNT>> layouts
) {
  return EstimateTensorLinearCombineCost<ElemT>(
      std::vector<BlockSparseTensorLayout<QNT>>(layouts));
}

/**
 * @brief Estimate tensor linear combination from concrete tensor topologies.
 */
template<typename ElemT, typename QNT>
TensorOpCost<QNT> EstimateTensorLinearCombineCost(
    const std::vector<const QLTensor<ElemT, QNT> *> &tensors
) {
  if (tensors.empty()) {
    throw std::invalid_argument(
        "EstimateTensorLinearCombineCost: tensors must not be empty.");
  }
  std::vector<BlockSparseTensorLayout<QNT>> layouts;
  layouts.reserve(tensors.size());
  for (const auto *tensor : tensors) {
    if (tensor == nullptr) {
      throw std::invalid_argument(
          "EstimateTensorLinearCombineCost: tensor pointer is null.");
    }
    layouts.push_back(MakeBlockSparseTensorLayout(*tensor));
  }
  return EstimateTensorLinearCombineCost<ElemT>(layouts);
}

/**
 * @brief Estimate accumulating one layout into an existing output layout.
 */
template<typename ElemT, typename QNT>
TensorOpCost<QNT> EstimateTensorAccumulationCost(
    const BlockSparseTensorLayout<QNT> &output_layout,
    const BlockSparseTensorLayout<QNT> &input_layout
) {
  tensor_cost_detail::RequireSameIndexes(
      output_layout, input_layout, "EstimateTensorAccumulationCost");
  TensorOpCost<QNT> cost;
  cost.output_layout = tensor_cost_detail::UnionLayouts(
      output_layout, input_layout, "EstimateTensorAccumulationCost");
  for (const auto &entry : input_layout.blocks) {
    const auto out_it = output_layout.blocks.find(entry.first);
    if (out_it != output_layout.blocks.end()) {
      cost.read_bytes += out_it->second.element_count * sizeof(ElemT);
    }
    cost.flops +=
        tensor_cost_detail::ElementwiseFlops<ElemT>(
            entry.second.element_count);
    tensor_cost_detail::AddLogicalRawCopyCost<ElemT, QNT>(
        cost, entry.second.element_count);
  }
  if (input_layout.IsScalar()) {
    cost.read_bytes += sizeof(ElemT);
    cost.flops += tensor_cost_detail::LinearCombineElementFlops<ElemT>();
    tensor_cost_detail::AddLogicalRawCopyCost<ElemT, QNT>(cost, 1);
  }
  tensor_cost_detail::PopulateOutputStats(cost);
  return cost;
}

/**
 * @brief Estimate direct contraction accumulation into an existing output.
 */
template<typename ElemT, typename QNT>
TensorOpCost<QNT> EstimateContractAccumulateCost(
    const QLTensor<ElemT, QNT> &lhs,
    const QLTensor<ElemT, QNT> &rhs,
    const std::vector<std::vector<size_t>> &axes_set,
    const QLTensor<ElemT, QNT> &existing_output,
    const bool allow_output_topology_expansion = true
) {
  auto cost = EstimateContractCost(lhs, rhs, axes_set);
  if (existing_output.IsDefault()) {
    return cost;
  }

  const auto existing_layout = MakeBlockSparseTensorLayout(existing_output);
  tensor_cost_detail::RequireSameIndexes(
      existing_layout, cost.output_layout, "EstimateContractAccumulateCost");
  bool needs_output_topology_expansion = false;
  if (!allow_output_topology_expansion) {
    for (const auto &entry : cost.output_layout.blocks) {
      if (existing_layout.blocks.find(entry.first) ==
          existing_layout.blocks.end()) {
        throw std::invalid_argument(
            "EstimateContractAccumulateCost: output topology requires expansion.");
      }
    }
  } else {
    for (const auto &entry : cost.output_layout.blocks) {
      if (existing_layout.blocks.find(entry.first) ==
          existing_layout.blocks.end()) {
        needs_output_topology_expansion = true;
        break;
      }
    }
  }
  if (needs_output_topology_expansion) {
    for (const auto &entry : existing_layout.blocks) {
      const size_t copy_bytes = entry.second.element_count * sizeof(ElemT);
      cost.read_bytes += copy_bytes;
      cost.write_bytes += copy_bytes;
    }
  }
  for (const auto &entry : cost.output_layout.blocks) {
    const auto existing_it = existing_layout.blocks.find(entry.first);
    if (existing_it != existing_layout.blocks.end()) {
      cost.read_bytes += existing_it->second.element_count * sizeof(ElemT);
    }
  }
  cost.output_layout = tensor_cost_detail::UnionLayouts(
      existing_layout, cost.output_layout, "EstimateContractAccumulateCost");
  tensor_cost_detail::PopulateOutputStats(cost);
  return cost;
}

/**
 * @brief Estimate direct contraction accumulation from pair-form axes.
 */
template<typename ElemT, typename QNT>
TensorOpCost<QNT> EstimateContractAccumulateCost(
    const QLTensor<ElemT, QNT> &lhs,
    const QLTensor<ElemT, QNT> &rhs,
    const std::vector<std::pair<size_t, size_t>> &contract_axes,
    const QLTensor<ElemT, QNT> &existing_output,
    const bool allow_output_topology_expansion = true
) {
  return EstimateContractAccumulateCost(
      lhs,
      rhs,
      tensor_cost_detail::AxesSetFromPairs(contract_axes),
      existing_output,
      allow_output_topology_expansion);
}

/**
 * @brief Estimate tail/head contiguous contraction accumulation.
 */
template<typename ElemT, typename QNT>
TensorOpCost<QNT> EstimateContractAccumulateCost(
    const QLTensor<ElemT, QNT> &lhs,
    const QLTensor<ElemT, QNT> &rhs,
    const size_t lhs_axes_start,
    const size_t rhs_axes_start,
    const size_t axes_size,
    const QLTensor<ElemT, QNT> &existing_output,
    const bool allow_output_topology_expansion = true
) {
  if (lhs.Rank() == 0 || rhs.Rank() == 0) {
    throw std::invalid_argument(
        "EstimateContractAccumulateCost: input ranks must be positive.");
  }
  std::vector<std::vector<size_t>> axes_set(2);
  axes_set[0].reserve(axes_size);
  axes_set[1].reserve(axes_size);
  for (size_t i = 0; i < axes_size; ++i) {
    axes_set[0].push_back((lhs_axes_start + i) % lhs.Rank());
    axes_set[1].push_back((rhs_axes_start + i) % rhs.Rank());
  }
  return EstimateContractAccumulateCost(
      lhs, rhs, axes_set, existing_output, allow_output_topology_expansion);
}

}  // namespace qlten

#endif /* ifndef QLTEN_TENSOR_MANIPULATION_TENSOR_OP_COST_H */
