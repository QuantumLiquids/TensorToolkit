// SPDX-License-Identifier: LGPL-3.0-only
/*
* Description: MPI sum reductions for block-sparse QLTensor objects.
*/

/**
@file mpi_reduce_sum.h
@brief MPI sum reductions for block-sparse QLTensor objects.
*/
#ifndef QLTEN_MPI_TENSOR_MANIPULATION_REDUCTION_MPI_REDUCE_SUM_H
#define QLTEN_MPI_TENSOR_MANIPULATION_REDUCTION_MPI_REDUCE_SUM_H

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include "mpi.h"

#include "qlten/framework/hp_numeric/mpi_fun.h"
#include "qlten/framework/mem_ops.h"
#include "qlten/qltensor_all.h"

namespace qlten {
namespace detail {

template<typename T>
struct QLFreeDeleter {
  void operator()(T *ptr) const {
    if (ptr != nullptr) {
      QLFree(ptr);
    }
  }
};

template<typename T>
using QLRawBuffer = std::unique_ptr<T, QLFreeDeleter<T>>;

template<typename ElemT>
std::string MPIElemTypeName() {
  if constexpr (std::is_same<ElemT, QLTEN_Double>::value) {
    return "QLTEN_Double";
  } else if constexpr (std::is_same<ElemT, QLTEN_Float>::value) {
    return "QLTEN_Float";
  } else if constexpr (std::is_same<ElemT, QLTEN_Complex>::value) {
    return "QLTEN_Complex";
  } else if constexpr (std::is_same<ElemT, QLTEN_ComplexFloat>::value) {
    return "QLTEN_ComplexFloat";
  } else {
    return typeid(ElemT).name();
  }
}

inline void CheckMPIRoot(const int root, const int mpi_size, const char *api_name) {
  if (root < 0 || root >= mpi_size) {
    throw std::runtime_error(
        std::string(api_name) + " received an invalid root rank.");
  }
}

inline void RequireAllRanks(
    const bool local_ok,
    const char *message,
    const MPI_Comm &comm
) {
  int local_flag = local_ok ? 1 : 0;
  int global_flag = 0;
  HANDLE_MPI_ERROR(::MPI_Allreduce(
      &local_flag, &global_flag, 1, MPI_INT, MPI_MIN, comm));
  if (global_flag == 0) {
    throw std::runtime_error(message);
  }
}

inline std::vector<size_t> AllGatherSizeTVector(
    const std::vector<size_t> &local,
    const MPI_Comm &comm
) {
  int mpi_size = 0;
  MPI_Comm_size(comm, &mpi_size);

  const size_t local_size = local.size();
  std::vector<size_t> sizes(mpi_size);
  HANDLE_MPI_ERROR(::MPI_Allgather(
      &local_size,
      1,
      hp_numeric::GetMPIDataType<size_t>(),
      sizes.data(),
      1,
      hp_numeric::GetMPIDataType<size_t>(),
      comm));

  std::vector<int> counts(mpi_size, 0);
  std::vector<int> displs(mpi_size, 0);
  size_t total_size = 0;
  for (int i = 0; i < mpi_size; ++i) {
    if (sizes[i] > static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error(
          "MPI QLTensor reduction metadata is too large for MPI_Allgatherv.");
    }
    counts[i] = static_cast<int>(sizes[i]);
    if (total_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error(
          "MPI QLTensor reduction metadata displacement is too large.");
    }
    displs[i] = static_cast<int>(total_size);
    total_size += sizes[i];
  }

  std::vector<size_t> gathered(total_size);
  HANDLE_MPI_ERROR(::MPI_Allgatherv(
      local.data(),
      static_cast<int>(local_size),
      hp_numeric::GetMPIDataType<size_t>(),
      gathered.data(),
      counts.data(),
      displs.data(),
      hp_numeric::GetMPIDataType<size_t>(),
      comm));
  return gathered;
}

inline std::vector<std::string> AllGatherString(
    const std::string &local,
    const MPI_Comm &comm
) {
  int mpi_size = 0;
  MPI_Comm_size(comm, &mpi_size);

  const size_t local_size = local.size();
  std::vector<size_t> sizes(mpi_size);
  HANDLE_MPI_ERROR(::MPI_Allgather(
      &local_size,
      1,
      hp_numeric::GetMPIDataType<size_t>(),
      sizes.data(),
      1,
      hp_numeric::GetMPIDataType<size_t>(),
      comm));

  std::vector<int> counts(mpi_size, 0);
  std::vector<int> displs(mpi_size, 0);
  size_t total_size = 0;
  for (int i = 0; i < mpi_size; ++i) {
    if (sizes[i] > static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error(
          "MPI QLTensor reduction signature is too large for MPI_Allgatherv.");
    }
    counts[i] = static_cast<int>(sizes[i]);
    if (total_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error(
          "MPI QLTensor reduction signature displacement is too large.");
    }
    displs[i] = static_cast<int>(total_size);
    total_size += sizes[i];
  }

  std::vector<char> gathered(total_size);
  HANDLE_MPI_ERROR(::MPI_Allgatherv(
      local.data(),
      static_cast<int>(local_size),
      MPI_CHAR,
      gathered.data(),
      counts.data(),
      displs.data(),
      MPI_CHAR,
      comm));

  std::vector<std::string> strings;
  strings.reserve(mpi_size);
  for (int i = 0; i < mpi_size; ++i) {
    strings.emplace_back(
        gathered.data() + displs[i],
        gathered.data() + displs[i] + counts[i]);
  }
  return strings;
}

template<typename TenElemT, typename QNT>
std::string TensorLayoutSignature(const QLTensor<TenElemT, QNT> &tensor) {
  std::ostringstream os;
  os << "elem_type " << MPIElemTypeName<TenElemT>() << "\n";
  os << "qn_type " << typeid(QNT).name() << "\n";
  os << "is_default " << tensor.IsDefault() << "\n";
  if (tensor.IsDefault()) {
    return os.str();
  }
  os << "rank " << tensor.Rank() << "\n";
  for (const auto &index : tensor.GetIndexes()) {
    index.StreamWrite(os);
  }
  return os.str();
}

template<typename TenElemT, typename QNT>
void ValidateCompatibleTensorLayouts(
    const QLTensor<TenElemT, QNT> &local,
    const MPI_Comm &comm,
    const char *api_name
) {
  const auto signatures = AllGatherString(TensorLayoutSignature(local), comm);
  for (size_t i = 1; i < signatures.size(); ++i) {
    if (signatures[i] != signatures[0]) {
      std::ostringstream msg;
      msg << api_name
          << " tensor layout mismatch: rank "
          << i
          << " has a different default state, rank, index layout, element type, "
             "or quantum-number type than rank 0.";
      throw std::runtime_error(msg.str());
    }
  }
}

template<typename QNT>
struct MPIBlockEntry {
  size_t blk_idx = 0;
  CoorsT blk_coors;
  size_t data_offset = 0;
  size_t data_size = 0;
};

struct MPIReduceSumPackTask {
  size_t src_data_offset = 0;
  size_t dest_data_offset = 0;
  size_t data_size = 0;
};

template<typename TenElemT, typename QNT>
std::vector<MPIBlockEntry<QNT>> ExtractBlockEntries(
    const QLTensor<TenElemT, QNT> &tensor
) {
  std::vector<MPIBlockEntry<QNT>> entries;
  if (tensor.IsDefault() || tensor.IsScalar()) {
    return entries;
  }
  const auto &block_map = tensor.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  entries.reserve(block_map.size());
  for (const auto &idx_block : block_map) {
    entries.push_back({
        idx_block.first,
        idx_block.second.blk_coors,
        idx_block.second.data_offset,
        idx_block.second.size});
  }
  return entries;
}

inline bool AllStringsEqual(const std::vector<std::string> &strings) {
  for (size_t i = 1; i < strings.size(); ++i) {
    if (strings[i] != strings[0]) {
      return false;
    }
  }
  return true;
}

template<typename QNT>
bool SameBlockEntries(
    const std::vector<MPIBlockEntry<QNT>> &lhs,
    const std::vector<MPIBlockEntry<QNT>> &rhs
) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i].blk_idx != rhs[i].blk_idx ||
        lhs[i].blk_coors != rhs[i].blk_coors ||
        lhs[i].data_offset != rhs[i].data_offset ||
        lhs[i].data_size != rhs[i].data_size) {
      return false;
    }
  }
  return true;
}

template<typename QNT>
std::vector<size_t> SerializeBlockEntries(
    const std::vector<MPIBlockEntry<QNT>> &entries,
    const size_t rank
) {
  std::vector<size_t> serialized;
  serialized.reserve(1 + entries.size() * (1 + rank));
  serialized.push_back(entries.size());
  for (const auto &entry : entries) {
    serialized.push_back(entry.blk_idx);
    serialized.insert(
        serialized.end(),
        entry.blk_coors.begin(),
        entry.blk_coors.end());
  }
  return serialized;
}

template<typename QNT>
std::string BlockTopologySignature(
    const std::vector<MPIBlockEntry<QNT>> &entries,
    const size_t rank,
    const size_t actual_raw_data_size,
    const bool is_scalar
) {
  std::ostringstream os;
  os << "rank " << rank << "\n";
  os << "is_scalar " << is_scalar << "\n";
  os << "actual_raw_data_size " << actual_raw_data_size << "\n";
  os << "block_count " << entries.size() << "\n";
  for (const auto &entry : entries) {
    os << entry.blk_idx << " "
       << entry.data_offset << " "
       << entry.data_size << "\n";
    for (const auto blk_coor : entry.blk_coors) {
      os << blk_coor << "\n";
    }
  }
  return os.str();
}

template<typename TenElemT, typename QNT>
std::vector<MPIBlockEntry<QNT>> BuildUnionBlockEntries(
    const QLTensor<TenElemT, QNT> &local,
    const std::vector<MPIBlockEntry<QNT>> &local_entries,
    const MPI_Comm &comm
) {
  const size_t rank = local.Rank();
  const auto gathered = AllGatherSizeTVector(
      SerializeBlockEntries(local_entries, rank),
      comm);

  std::map<size_t, CoorsT> union_blocks;
  size_t cursor = 0;
  while (cursor < gathered.size()) {
    const size_t block_count = gathered[cursor++];
    const size_t record_size = 1 + rank;
    if (block_count > (gathered.size() - cursor) / record_size) {
      throw std::runtime_error(
          "MPI QLTensor reduction received malformed block topology metadata.");
    }
    for (size_t i = 0; i < block_count; ++i) {
      const size_t blk_idx = gathered[cursor++];
      CoorsT blk_coors(rank);
      for (size_t axis = 0; axis < rank; ++axis) {
        blk_coors[axis] = gathered[cursor++];
      }
      const auto insert_res = union_blocks.emplace(blk_idx, blk_coors);
      if (!insert_res.second && insert_res.first->second != blk_coors) {
        throw std::runtime_error(
            "MPI QLTensor reduction block topology mismatch: identical block "
            "index maps to different block coordinates.");
      }
    }
  }

  std::vector<MPIBlockEntry<QNT>> union_entries;
  union_entries.reserve(union_blocks.size());
  size_t data_offset = 0;
  for (const auto &idx_coors : union_blocks) {
    const DataBlk<QNT> data_blk(idx_coors.second, local.GetIndexes());
    union_entries.push_back({
        idx_coors.first,
        idx_coors.second,
        data_offset,
        data_blk.size});
    data_offset += data_blk.size;
  }
  return union_entries;
}

template<typename QNT>
size_t CalcRawDataSize(const std::vector<MPIBlockEntry<QNT>> &entries) {
  size_t size = 0;
  for (const auto &entry : entries) {
    size += entry.data_size;
  }
  return size;
}

template<typename QNT>
std::vector<MPIReduceSumPackTask> BuildPackTasks(
    const std::vector<MPIBlockEntry<QNT>> &local_entries,
    const std::vector<MPIBlockEntry<QNT>> &union_entries
) {
  std::map<size_t, size_t> union_offsets;
  for (const auto &entry : union_entries) {
    union_offsets[entry.blk_idx] = entry.data_offset;
  }

  std::vector<MPIReduceSumPackTask> tasks;
  tasks.reserve(local_entries.size());
  for (const auto &entry : local_entries) {
    tasks.push_back({
        entry.data_offset,
        union_offsets.at(entry.blk_idx),
        entry.data_size});
  }
  return tasks;
}

template<typename TenElemT, typename QNT>
void PrepareReductionResultShell(
    const QLTensor<TenElemT, QNT> &local,
    const std::vector<MPIBlockEntry<QNT>> &union_entries,
    const size_t union_raw_size,
    QLTensor<TenElemT, QNT> *result
) {
  *result = QLTensor<TenElemT, QNT>(local.GetIndexes());
  if (local.IsScalar()) {
    if (union_raw_size != 0) {
      result->SetElem({}, TenElemT(0));
    }
    return;
  }
  if (union_entries.empty()) {
    return;
  }

  std::vector<size_t> block_idxs;
  std::vector<CoorsT> block_coors;
  block_idxs.reserve(union_entries.size());
  block_coors.reserve(union_entries.size());
  for (const auto &entry : union_entries) {
    block_idxs.push_back(entry.blk_idx);
    block_coors.push_back(entry.blk_coors);
  }
  result->GetBlkSparDataTen().DataBlksInsert(
      block_idxs,
      block_coors,
      true,
      false);
}

template<typename TenElemT, typename QNT>
QLRawBuffer<TenElemT> PackLocalRawDataToUnionLayout(
    const QLTensor<TenElemT, QNT> &local,
    const std::vector<MPIBlockEntry<QNT>> &local_entries,
    const std::vector<MPIBlockEntry<QNT>> &union_entries,
    const size_t union_raw_size
) {
  QLRawBuffer<TenElemT> packed;
  if (union_raw_size == 0) {
    return packed;
  }
  packed.reset(static_cast<TenElemT *>(
      QLCalloc(union_raw_size, sizeof(TenElemT))));

  const auto &local_bsdt = local.GetBlkSparDataTen();
  const TenElemT *local_raw_data = local_bsdt.GetActualRawDataPtr();
  if (local.IsScalar()) {
    if (local_bsdt.GetActualRawDataSize() == 1) {
      QLMemcpy(packed.get(), local_raw_data, sizeof(TenElemT));
    }
    return packed;
  }

  std::map<size_t, size_t> union_offsets;
  for (const auto &entry : union_entries) {
    union_offsets[entry.blk_idx] = entry.data_offset;
  }
  for (const auto &entry : local_entries) {
    const auto dest_offset = union_offsets.at(entry.blk_idx);
    QLMemcpy(
        packed.get() + dest_offset,
        local_raw_data + entry.data_offset,
        entry.data_size * sizeof(TenElemT));
  }
  return packed;
}

template<typename TenElemT, typename QNT>
void CopyLocalContributionToUnionBuffer(
    const QLTensor<TenElemT, QNT> &local,
    const std::vector<MPIReduceSumPackTask> &pack_tasks,
    TenElemT *dest,
    const size_t union_raw_size
) {
  if (union_raw_size == 0) {
    return;
  }
  QLMemset(dest, 0, union_raw_size * sizeof(TenElemT));

  const auto &local_bsdt = local.GetBlkSparDataTen();
  const TenElemT *local_raw_data = local_bsdt.GetActualRawDataPtr();
  if (local.IsScalar()) {
    if (local_bsdt.GetActualRawDataSize() == 1) {
      QLMemcpy(dest, local_raw_data, sizeof(TenElemT));
    }
    return;
  }

  for (const auto &task : pack_tasks) {
    QLMemcpy(
        dest + task.dest_data_offset,
        local_raw_data + task.src_data_offset,
        task.data_size * sizeof(TenElemT));
  }
}

template<typename TenElemT>
void ZeroRawBuffer(TenElemT *data, const size_t data_size) {
  if (data_size != 0) {
    QLMemset(data, 0, data_size * sizeof(TenElemT));
  }
}

template<typename TenElemT>
void MPIReduceRawSum(
    const TenElemT *send_buf,
    TenElemT *recv_buf,
    const size_t data_size,
    const int root,
    const MPI_Comm &comm,
    const bool root_in_place = false,
    const int rank = -1
) {
  if (data_size == 0) {
    return;
  }
  constexpr size_t chunk_count = hp_numeric::kMPIMaxChunkSize / sizeof(TenElemT);
  size_t offset = 0;
  while (offset < data_size) {
    const size_t this_count = std::min(chunk_count, data_size - offset);
    const void *actual_send_buf = (root_in_place && rank == root)
                                  ? MPI_IN_PLACE
                                  : static_cast<const void *>(send_buf + offset);
    HANDLE_MPI_ERROR(::MPI_Reduce(
        actual_send_buf,
        recv_buf == nullptr ? nullptr : recv_buf + offset,
        static_cast<int>(this_count),
        hp_numeric::GetMPIDataType<TenElemT>(),
        MPI_SUM,
        root,
        comm));
    offset += this_count;
  }
}

template<typename TenElemT>
void MPIAllreduceRawSum(
    const TenElemT *send_buf,
    TenElemT *recv_buf,
    const size_t data_size,
    const MPI_Comm &comm
) {
  if (data_size == 0) {
    return;
  }
  constexpr size_t chunk_count = hp_numeric::kMPIMaxChunkSize / sizeof(TenElemT);
  size_t offset = 0;
  while (offset < data_size) {
    const size_t this_count = std::min(chunk_count, data_size - offset);
    HANDLE_MPI_ERROR(::MPI_Allreduce(
        send_buf + offset,
        recv_buf + offset,
        static_cast<int>(this_count),
        hp_numeric::GetMPIDataType<TenElemT>(),
        MPI_SUM,
        comm));
    offset += this_count;
  }
}

template<typename TenElemT, typename QNT>
size_t ScalarReductionRawSize(const QLTensor<TenElemT, QNT> &local) {
  return local.IsScalar() ? 1 : 0;
}

}  // namespace detail

/**
 * @brief Controls how the root rank contributes to an MPI_ReduceSum operation.
 */
enum class MPIReduceSumRootMode {
  ///< Root contributes the `local` tensor, like a normal MPI_Reduce caller.
  LocalContribution,
  ///< Root contributes zero and uses the result buffer as an MPI_IN_PLACE zero input.
  RootZero,
  ///< Root first copies/packs `local` into the result buffer, then uses MPI_IN_PLACE.
  InPlaceLocalContribution
};

/**
 * @brief Cached plan for repeated QLTensor MPI sum reductions with stable layout.
 *
 * The constructor validates rank/index compatibility, probes whether all ranks
 * have identical block topology, and caches the union layout plus local-to-union
 * copy tasks when needed. Repeated ReduceSum calls then avoid metadata
 * allgather, union construction, offset-map construction, and pack-buffer
 * allocation.
 */
template<typename TenElemT, typename QNT>
class MPIReduceSumPlan {
 public:
  using Tensor = QLTensor<TenElemT, QNT>;

  MPIReduceSumPlan(
      const Tensor &local_template,
      const int root,
      const MPI_Comm &comm
  ) : root_(root), comm_(comm) {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &mpi_size_);
    detail::CheckMPIRoot(root_, mpi_size_, "MPIReduceSumPlan");
    detail::ValidateCompatibleTensorLayouts(
        local_template,
        comm_,
        "MPIReduceSumPlan");

    is_default_ = local_template.IsDefault();
    if (is_default_) {
      return;
    }

    indexes_ = local_template.GetIndexes();
    is_scalar_ = local_template.IsScalar();
    local_actual_raw_data_size_ = local_template.GetActualDataSize();
    local_entries_ = detail::ExtractBlockEntries(local_template);

    const auto topology_signature = detail::BlockTopologySignature(
        local_entries_,
        local_template.Rank(),
        local_actual_raw_data_size_,
        is_scalar_);
    const auto topology_signatures =
        detail::AllGatherString(topology_signature, comm_);
    same_block_topology_ = detail::AllStringsEqual(topology_signatures);

    if (same_block_topology_) {
      union_entries_ = local_entries_;
      union_raw_size_ = is_scalar_
                        ? detail::ScalarReductionRawSize(local_template)
                        : local_actual_raw_data_size_;
    } else {
      union_entries_ = detail::BuildUnionBlockEntries(
          local_template,
          local_entries_,
          comm_);
      union_raw_size_ = is_scalar_
                        ? detail::ScalarReductionRawSize(local_template)
                        : detail::CalcRawDataSize(union_entries_);
    }

    pack_tasks_ = detail::BuildPackTasks(local_entries_, union_entries_);
  }

  bool IsSameBlockTopology() const { return same_block_topology_; }

  size_t GetUnionRawDataSize() const { return union_raw_size_; }

  /**
   * @brief Sum local tensors using the cached plan.
   *
   * Each rank's `local` tensor must have the same indexes and block topology as
   * the tensor used to construct the plan on that rank. Non-root ranks may pass
   * nullptr for `root_result`.
   */
  void ReduceSum(
      const Tensor &local,
      Tensor *root_result,
      const MPIReduceSumRootMode root_mode =
          MPIReduceSumRootMode::LocalContribution
  ) {
    ValidateReduceInputs_(local, root_result);

    if (is_default_) {
      if (rank_ == root_) {
        *root_result = Tensor();
      }
      return;
    }

    const bool root_in_place =
        rank_ == root_ &&
        (root_mode == MPIReduceSumRootMode::RootZero ||
            root_mode == MPIReduceSumRootMode::InPlaceLocalContribution);

    TenElemT *recv_buf = nullptr;
    if (rank_ == root_) {
      detail::PrepareReductionResultShell(
          local,
          union_entries_,
          union_raw_size_,
          root_result);
      if (union_raw_size_ != 0) {
        recv_buf = root_result->GetBlkSparDataTen().GetActualRawDataPtr();
        if (root_mode == MPIReduceSumRootMode::RootZero) {
          detail::ZeroRawBuffer(recv_buf, union_raw_size_);
        } else if (root_mode ==
            MPIReduceSumRootMode::InPlaceLocalContribution) {
          CopyRootLocalContributionToResult_(local, recv_buf);
        }
      }
    }

    const TenElemT *send_buf = nullptr;
    if (!root_in_place) {
      send_buf = LocalSendBuffer_(local);
    }

    detail::MPIReduceRawSum(
        send_buf,
        recv_buf,
        union_raw_size_,
        root_,
        comm_,
        root_in_place,
        rank_);
  }

 private:
  int root_ = 0;
  int rank_ = 0;
  int mpi_size_ = 0;
  MPI_Comm comm_ = MPI_COMM_WORLD;
  bool is_default_ = false;
  bool is_scalar_ = false;
  bool same_block_topology_ = false;
  size_t local_actual_raw_data_size_ = 0;
  size_t union_raw_size_ = 0;
  IndexVec<QNT> indexes_;
  std::vector<detail::MPIBlockEntry<QNT>> local_entries_;
  std::vector<detail::MPIBlockEntry<QNT>> union_entries_;
  std::vector<detail::MPIReduceSumPackTask> pack_tasks_;
  detail::QLRawBuffer<TenElemT> pack_buffer_;

  void ValidateReduceInputs_(
      const Tensor &local,
      const Tensor *root_result
  ) const {
    const bool root_result_ok = rank_ != root_ || root_result != nullptr;
    const bool root_alias_ok =
        rank_ != root_ ||
            root_result == nullptr ||
            static_cast<const void *>(root_result) !=
                static_cast<const void *>(&local);
    const bool default_ok = local.IsDefault() == is_default_;

    bool indexes_ok = true;
    bool topology_ok = true;
    if (default_ok && !is_default_) {
      indexes_ok = local.GetIndexes() == indexes_;
      if (indexes_ok && !is_scalar_) {
        const auto local_entries = detail::ExtractBlockEntries(local);
        topology_ok =
            local.GetActualDataSize() == local_actual_raw_data_size_ &&
                detail::SameBlockEntries(local_entries, local_entries_);
      }
    }

    const std::array<int, 5> local_checks{
        root_result_ok ? 1 : 0,
        root_alias_ok ? 1 : 0,
        default_ok ? 1 : 0,
        indexes_ok ? 1 : 0,
        topology_ok ? 1 : 0};
    std::array<int, 5> global_checks{};
    HANDLE_MPI_ERROR(::MPI_Allreduce(
        local_checks.data(),
        global_checks.data(),
        static_cast<int>(local_checks.size()),
        MPI_INT,
        MPI_MIN,
        comm_));

    if (global_checks[0] == 0) {
      throw std::runtime_error(
          "MPIReduceSumPlan::ReduceSum requires root_result to be non-null "
          "on the root rank.");
    }
    if (global_checks[1] == 0) {
      throw std::runtime_error(
          "MPIReduceSumPlan::ReduceSum does not support aliasing local and "
          "root_result on the root rank.");
    }
    if (global_checks[2] == 0) {
      throw std::runtime_error(
          "MPIReduceSumPlan::ReduceSum local tensor default state does not "
          "match the plan template.");
    }
    if (global_checks[3] == 0) {
      throw std::runtime_error(
          "MPIReduceSumPlan::ReduceSum local tensor indexes do not match the "
          "plan template.");
    }
    if (global_checks[4] == 0) {
      throw std::runtime_error(
          "MPIReduceSumPlan::ReduceSum local tensor block topology does not "
          "match the plan template.");
    }
  }

  void EnsurePackBuffer_() {
    if (union_raw_size_ != 0 && pack_buffer_ == nullptr) {
      pack_buffer_.reset(static_cast<TenElemT *>(
          QLMalloc(union_raw_size_ * sizeof(TenElemT))));
    }
  }

  void CopyRootLocalContributionToResult_(
      const Tensor &local,
      TenElemT *result_raw_data
  ) {
    if (same_block_topology_ && !is_scalar_) {
      QLMemcpy(
          result_raw_data,
          local.GetBlkSparDataTen().GetActualRawDataPtr(),
          union_raw_size_ * sizeof(TenElemT));
      return;
    }
    detail::CopyLocalContributionToUnionBuffer(
        local,
        pack_tasks_,
        result_raw_data,
        union_raw_size_);
  }

  const TenElemT *LocalSendBuffer_(const Tensor &local) {
    if (union_raw_size_ == 0) {
      return nullptr;
    }
    if (same_block_topology_ && !is_scalar_) {
      return local.GetBlkSparDataTen().GetActualRawDataPtr();
    }

    EnsurePackBuffer_();
    detail::CopyLocalContributionToUnionBuffer(
        local,
        pack_tasks_,
        pack_buffer_.get(),
        union_raw_size_);
    return pack_buffer_.get();
  }
};

template<typename TenElemT, typename QNT>
MPIReduceSumPlan<TenElemT, QNT> MakeMPIReduceSumPlan(
    const QLTensor<TenElemT, QNT> &local_template,
    const int root,
    const MPI_Comm &comm
) {
  return MPIReduceSumPlan<TenElemT, QNT>(local_template, root, comm);
}

/**
 * @brief Sum block-sparse QLTensor objects across MPI ranks onto one root rank.
 *
 * All ranks must call this function with compatible tensor index layouts and
 * template element/quantum-number types. Non-root ranks may pass nullptr for
 * `root_result`. The root result must not alias `local`.
 */
template<typename TenElemT, typename QNT>
void MPI_ReduceSum(
    const QLTensor<TenElemT, QNT> &local,
    QLTensor<TenElemT, QNT> *root_result,
    const int root,
    const MPI_Comm &comm
) {
  auto plan = MakeMPIReduceSumPlan(local, root, comm);
  plan.ReduceSum(local, root_result);
}

template<typename TenElemT, typename QNT>
void MPI_ReduceSum(
    const QLTensor<TenElemT, QNT> &local,
    QLTensor<TenElemT, QNT> *root_result,
    const int root,
    const MPI_Comm &comm,
    const MPIReduceSumRootMode root_mode
) {
  auto plan = MakeMPIReduceSumPlan(local, root, comm);
  plan.ReduceSum(local, root_result, root_mode);
}

/**
 * @brief Sum block-sparse QLTensor objects across MPI ranks onto every rank.
 *
 * All ranks must call this function with compatible tensor index layouts and
 * template element/quantum-number types. `result` must be non-null and must not
 * alias `local`.
 */
template<typename TenElemT, typename QNT>
void MPI_AllreduceSum(
    const QLTensor<TenElemT, QNT> &local,
    QLTensor<TenElemT, QNT> *result,
    const MPI_Comm &comm
) {
  detail::RequireAllRanks(
      result != nullptr,
      "MPI_AllreduceSum requires result to be non-null on every rank.",
      comm);
  detail::RequireAllRanks(
      static_cast<const void *>(result) != static_cast<const void *>(&local),
      "MPI_AllreduceSum does not support aliasing local and result.",
      comm);
  detail::ValidateCompatibleTensorLayouts(local, comm, "MPI_AllreduceSum");

  if (local.IsDefault()) {
    *result = QLTensor<TenElemT, QNT>();
    return;
  }

  const auto local_entries = detail::ExtractBlockEntries(local);
  const auto topology_signature = detail::BlockTopologySignature(
      local_entries,
      local.Rank(),
      local.GetActualDataSize(),
      local.IsScalar());
  const bool same_block_topology =
      detail::AllStringsEqual(detail::AllGatherString(topology_signature, comm));
  const auto union_entries = same_block_topology
                             ? local_entries
                             : detail::BuildUnionBlockEntries(
                                 local,
                                 local_entries,
                                 comm);
  const bool is_fast_path = same_block_topology && !local.IsScalar();
  const size_t union_raw_size = local.IsScalar()
                                ? detail::ScalarReductionRawSize(local)
                                : detail::CalcRawDataSize(union_entries);

  detail::PrepareReductionResultShell(
      local,
      union_entries,
      union_raw_size,
      result);

  const TenElemT *send_buf = nullptr;
  detail::QLRawBuffer<TenElemT> packed;
  if (is_fast_path) {
    send_buf = local.GetBlkSparDataTen().GetActualRawDataPtr();
  } else {
    packed = detail::PackLocalRawDataToUnionLayout(
        local,
        local_entries,
        union_entries,
        union_raw_size);
    send_buf = packed.get();
  }

  TenElemT *recv_buf = union_raw_size == 0
                       ? nullptr
                       : result->GetBlkSparDataTen().GetActualRawDataPtr();
  detail::MPIAllreduceRawSum(send_buf, recv_buf, union_raw_size, comm);
}

}  // namespace qlten

#endif /* ifndef QLTEN_MPI_TENSOR_MANIPULATION_REDUCTION_MPI_REDUCE_SUM_H */
