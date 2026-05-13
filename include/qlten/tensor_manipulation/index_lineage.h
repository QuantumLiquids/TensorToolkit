// SPDX-License-Identifier: LGPL-3.0-only
//
// Pure helpers for tracking output-index lineage through tensor manipulations.

#ifndef QLTEN_TENSOR_MANIPULATION_INDEX_LINEAGE_H
#define QLTEN_TENSOR_MANIPULATION_INDEX_LINEAGE_H

#include <algorithm>
#include <cassert>
#include <utility>
#include <vector>

namespace qlten {

struct IndexLineage {
  std::vector<size_t> leaf_ids;
};

inline bool operator==(const IndexLineage &lhs, const IndexLineage &rhs) {
  return lhs.leaf_ids == rhs.leaf_ids;
}

inline bool operator!=(const IndexLineage &lhs, const IndexLineage &rhs) {
  return !(lhs == rhs);
}

using IndexLineages = std::vector<IndexLineage>;

namespace index_lineage_detail {

inline void AssertPermutation(const size_t rank,
                              const std::vector<size_t> &axes) {
  assert(axes.size() == rank);
  std::vector<size_t> sorted_axes = axes;
  std::sort(sorted_axes.begin(), sorted_axes.end());
  for (size_t i = 0; i < rank; ++i) {
    assert(sorted_axes[i] == i);
  }
}

inline std::vector<size_t> FuseIndexTransposeAxes(const size_t rank,
                                                  const size_t idx1,
                                                  const size_t idx2) {
  assert(idx1 < idx2 && idx2 < rank);

  std::vector<size_t> transpose_axes(rank);
  transpose_axes[0] = idx1;
  transpose_axes[1] = idx2;
  for (size_t i = 2; i < idx1 + 2; ++i) {
    transpose_axes[i] = i - 2;
  }
  for (size_t i = idx1 + 2; i < idx2 + 1; ++i) {
    transpose_axes[i] = i - 1;
  }
  for (size_t i = idx2 + 1; i < rank; ++i) {
    transpose_axes[i] = i;
  }
  return transpose_axes;
}

inline void AppendLeafIds(IndexLineage &dst, const IndexLineage &src) {
  dst.leaf_ids.insert(dst.leaf_ids.end(), src.leaf_ids.begin(),
                      src.leaf_ids.end());
}

}  // namespace index_lineage_detail

inline IndexLineages TransposeLineages(const IndexLineages &lineages,
                                       const std::vector<size_t> &axes) {
  index_lineage_detail::AssertPermutation(lineages.size(), axes);

  IndexLineages result;
  result.reserve(axes.size());
  for (const size_t axis : axes) {
    result.push_back(lineages[axis]);
  }
  return result;
}

inline IndexLineages FuseIndexLineages(const IndexLineages &lineages,
                                       const size_t idx1,
                                       const size_t idx2) {
  const std::vector<size_t> transpose_axes =
      index_lineage_detail::FuseIndexTransposeAxes(lineages.size(), idx1, idx2);
  IndexLineages output_lineages = TransposeLineages(lineages, transpose_axes);

  IndexLineage fused_lineage = output_lineages[0];
  index_lineage_detail::AppendLeafIds(fused_lineage, output_lineages[1]);
  output_lineages.erase(output_lineages.begin());
  output_lineages[0] = std::move(fused_lineage);
  return output_lineages;
}

/**
 * Returns output index lineages for ordinary ten_ctrct.h contraction order:
 * A uncontracted axes in natural order, then B uncontracted axes in natural
 * order.
 *
 * @warning This does not describe matrix-based contraction ordering in
 * ten_ctrct_based_mat_trans.h; that path uses cyclic saved-axis order and
 * cannot be derived from this axes-set-only signature.
 */
inline IndexLineages ContractOutputLineages(
    const IndexLineages &a_lineages,
    const IndexLineages &b_lineages,
    const std::vector<std::vector<size_t>> &axes_set) {
  assert(axes_set.size() == 2);
  assert(axes_set[0].size() == axes_set[1].size());

  std::vector<bool> a_contracted(a_lineages.size(), false);
  std::vector<bool> b_contracted(b_lineages.size(), false);
  for (const size_t axis : axes_set[0]) {
    assert(axis < a_lineages.size());
    assert(!a_contracted[axis]);
    a_contracted[axis] = true;
  }
  for (const size_t axis : axes_set[1]) {
    assert(axis < b_lineages.size());
    assert(!b_contracted[axis]);
    b_contracted[axis] = true;
  }

  IndexLineages result;
  result.reserve(a_lineages.size() + b_lineages.size() -
                 2 * axes_set[0].size());
  for (size_t i = 0; i < a_lineages.size(); ++i) {
    if (!a_contracted[i]) {
      result.push_back(a_lineages[i]);
    }
  }
  for (size_t i = 0; i < b_lineages.size(); ++i) {
    if (!b_contracted[i]) {
      result.push_back(b_lineages[i]);
    }
  }
  return result;
}

}  // namespace qlten

#endif  // QLTEN_TENSOR_MANIPULATION_INDEX_LINEAGE_H
