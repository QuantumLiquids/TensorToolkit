// SPDX-License-Identifier: LGPL-3.0-only
//
// Pure helpers for tracking leaf order in a fused tensor index.

#ifndef QLTEN_TENSOR_MANIPULATION_INDEX_LINEAGE_H
#define QLTEN_TENSOR_MANIPULATION_INDEX_LINEAGE_H

#include <cstddef>
#include <vector>

namespace qlten {

template<typename T = std::size_t>
struct IndexLineage {
  std::vector<T> leaf_ids;
};

template<typename T>
inline bool operator==(const IndexLineage<T> &lhs,
                       const IndexLineage<T> &rhs) {
  return lhs.leaf_ids == rhs.leaf_ids;
}

template<typename T>
inline bool operator!=(const IndexLineage<T> &lhs,
                       const IndexLineage<T> &rhs) {
  return !(lhs == rhs);
}

template<typename T = std::size_t>
inline IndexLineage<T> FuseIndexLineage(
    const IndexLineage<T> &left_lineage,
    const IndexLineage<T> &right_lineage) {
  IndexLineage<T> fused_lineage = left_lineage;
  fused_lineage.leaf_ids.insert(fused_lineage.leaf_ids.end(),
                                right_lineage.leaf_ids.begin(),
                                right_lineage.leaf_ids.end());
  return fused_lineage;
}

}  // namespace qlten

#endif  // QLTEN_TENSOR_MANIPULATION_INDEX_LINEAGE_H
