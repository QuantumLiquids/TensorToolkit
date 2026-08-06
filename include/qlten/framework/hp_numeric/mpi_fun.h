// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2021-09-27 12:34
*
* Description: QuantumLiquids/tensor project. MPI wrapper.
*/

/**
@file mpi_fun.h
@brief Wrapper for the MPI related functions.
*/

#ifndef QLTEN_FRAMEWORK_HP_NUMERIC_MPI_FUN_H
#define QLTEN_FRAMEWORK_HP_NUMERIC_MPI_FUN_H

#include <algorithm>                      // min
#include <cstddef>                        // size_t
#include <cstdint>                        // uint64_t
#include <cstdlib>                        // exit
#include <iostream>                       // cerr
#include <limits>                         // numeric_limits
#include <memory>                         // shared_ptr
#include <stdexcept>                      // invalid_argument, runtime_error
#include <string>                         // string
#include <type_traits>                    // is_same_v
#include <utility>                        // move
#include <vector>                         // vector

#include "qlten/framework/value_t.h"      // QLTEN_Double, QLTEN_Complex
#include "mpi.h"                          // MPI_Send, MPI_Recv, MPI_Bcast

namespace qlten {
template<typename ElemT, typename QNT>
class QLTensor;

/// High performance numerical functions.
namespace hp_numeric {
// MPI master's rank - preferred location
const size_t kMPIMasterRank = 0;
#ifdef QLTEN_MPI_MAX_CHUNK_SIZE
inline constexpr size_t kMPIMaxChunkSize = QLTEN_MPI_MAX_CHUNK_SIZE;
#else
inline constexpr size_t kMPIMaxChunkSize =
    static_cast<size_t>(std::numeric_limits<int>::max());
#endif
static_assert(kMPIMaxChunkSize > 0,
              "QLTEN_MPI_MAX_CHUNK_SIZE must be greater than zero.");
static_assert(kMPIMaxChunkSize <=
                  static_cast<size_t>(std::numeric_limits<int>::max()),
              "QLTEN_MPI_MAX_CHUNK_SIZE must fit in an MPI int count.");

// Function to handle MPI errors
inline void HandleMPIError(int error_code, int line) {
  if (error_code != MPI_SUCCESS) {
    char error_string[MPI_MAX_ERROR_STRING];
    int length_of_error_string;

    MPI_Error_string(error_code, error_string, &length_of_error_string);
    std::cerr << "MPI Error at line " << line << ": "
              << std::string(error_string, length_of_error_string)
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#define HANDLE_MPI_ERROR(x) hp_numeric::HandleMPIError(x, __LINE__)

/**
 * @brief Move-only owner of active nonblocking MPI send requests.
 *
 * The request may also retain type-erased staging storage whose addresses are
 * referenced by the active MPI operations. Moving the handle does not move
 * that storage. A default-constructed or moved-from handle is complete.
 *
 * @warning Destroying an incomplete handle waits for all sends to complete.
 *          This deliberately prevents handle-owned buffers from being freed
 *          while MPI still references them. Move-assigning over an incomplete
 *          handle also waits. Complete or destroy handles before
 *          `MPI_Finalize` when they may still be active.
 * @note Destruction during exception unwinding has the same wait-on-destroy
 *       behavior. A fatal-error path that intends to call `MPI_Abort` must keep
 *       its in-flight handles in a scope that survives until the catch handler,
 *       then call `MPI_Abort` before those handles are destroyed. Otherwise,
 *       unwinding may wait for peer communication before the handler can abort.
 *       After `MPI_Finalize`, destruction performs no MPI call and only releases
 *       local state; reaching finalization with active requests violates the
 *       handle's lifetime precondition.
 * @note `IsComplete()` only reports locally observed state; call `Test()` to
 *       make nonblocking MPI progress or `Wait()` to complete the operation.
 *       Calling `Test()` or `Wait()` on an incomplete handle after
 *       `MPI_Finalize` throws `std::runtime_error`.
 * @note MPI errors follow TensorToolkit's existing `HANDLE_MPI_ERROR` policy:
 *       a diagnostic is printed and the process exits.
 */
class MPISendRequest {
 public:
  MPISendRequest() noexcept = default;

  MPISendRequest(const MPISendRequest &) = delete;
  MPISendRequest &operator=(const MPISendRequest &) = delete;

  MPISendRequest(MPISendRequest &&other) noexcept { MoveFrom_(std::move(other)); }

  MPISendRequest &operator=(MPISendRequest &&other) noexcept {
    if (this != &other) {
      CompleteBeforeRelease_();
      MoveFrom_(std::move(other));
    }
    return *this;
  }

  ~MPISendRequest() { CompleteBeforeRelease_(); }

  /** @brief Test whether every component send has completed. */
  bool Test() {
    if (complete_) { return true; }
    EnsureMPIActive_("MPISendRequest::Test");

    bool all_complete = true;
    size_t offset = 0;
    while (offset < requests_.size()) {
      const size_t batch_size = std::min(
          requests_.size() - offset,
          static_cast<size_t>(std::numeric_limits<int>::max()));
      int batch_complete = 0;
      HANDLE_MPI_ERROR(::MPI_Testall(
          static_cast<int>(batch_size),
          requests_.data() + offset,
          &batch_complete,
          MPI_STATUSES_IGNORE));
      all_complete = all_complete && (batch_complete != 0);
      offset += batch_size;
    }

    if (all_complete) { MarkComplete_(); }
    return complete_;
  }

  /** @brief Wait for every component send. Repeated calls are harmless. */
  void Wait() {
    if (complete_) { return; }
    EnsureMPIActive_("MPISendRequest::Wait");
    WaitAll_();
    MarkComplete_();
  }

  /** @brief Whether completion has already been observed by this handle. */
  bool IsComplete() const noexcept { return complete_; }

  /** @brief Total bytes submitted to MPI, including transport metadata. */
  size_t PayloadBytes() const noexcept { return payload_bytes_; }

  /** @brief Number of component MPI requests created for the transfer. */
  size_t RequestCount() const noexcept { return request_count_; }

  /** @brief Approximate handle-owned staging bytes, excluding borrowed data. */
  size_t StagingBytes() const noexcept { return staging_bytes_; }

 private:
  template<typename ElemT, typename QNT>
  friend class ::qlten::QLTensor;

  MPISendRequest(std::vector<MPI_Request> requests,
                 std::shared_ptr<const void> owned_storage,
                 const size_t payload_bytes,
                 const size_t staging_bytes) noexcept
      : requests_(std::move(requests)),
        owned_storage_(std::move(owned_storage)),
        payload_bytes_(payload_bytes),
        staging_bytes_(staging_bytes),
        request_count_(requests_.size()),
        complete_(requests_.empty()) {}

  void MoveFrom_(MPISendRequest &&other) noexcept {
    requests_ = std::move(other.requests_);
    owned_storage_ = std::move(other.owned_storage_);
    payload_bytes_ = other.payload_bytes_;
    staging_bytes_ = other.staging_bytes_;
    request_count_ = other.request_count_;
    complete_ = other.complete_;

    other.payload_bytes_ = 0;
    other.staging_bytes_ = 0;
    other.request_count_ = 0;
    other.complete_ = true;
  }

  static void EnsureMPIActive_(const char *operation) {
    int initialized = 0;
    HANDLE_MPI_ERROR(::MPI_Initialized(&initialized));
    int finalized = 0;
    if (initialized) {
      HANDLE_MPI_ERROR(::MPI_Finalized(&finalized));
    }
    if (!initialized || finalized) {
      throw std::runtime_error(
          std::string(operation) + " requires active MPI before MPI_Finalize");
    }
  }

  void WaitAll_() noexcept {
    size_t offset = 0;
    while (offset < requests_.size()) {
      const size_t batch_size = std::min(
          requests_.size() - offset,
          static_cast<size_t>(std::numeric_limits<int>::max()));
      HANDLE_MPI_ERROR(::MPI_Waitall(
          static_cast<int>(batch_size),
          requests_.data() + offset,
          MPI_STATUSES_IGNORE));
      offset += batch_size;
    }
  }

  void MarkComplete_() noexcept {
    complete_ = true;
    requests_.clear();
    owned_storage_.reset();
  }

  void CompleteBeforeRelease_() noexcept {
    if (complete_) { return; }

    int initialized = 0;
    HANDLE_MPI_ERROR(::MPI_Initialized(&initialized));
    if (!initialized) {
      MarkComplete_();
      return;
    }
    int finalized = 0;
    HANDLE_MPI_ERROR(::MPI_Finalized(&finalized));
    if (finalized) {
      MarkComplete_();
      return;
    }

    WaitAll_();
    MarkComplete_();
  }

  std::vector<MPI_Request> requests_;
  std::shared_ptr<const void> owned_storage_;
  size_t payload_bytes_ = 0;
  size_t staging_bytes_ = 0;
  size_t request_count_ = 0;
  bool complete_ = true;
};

template<typename ElemT>
MPI_Datatype GetMPIDataType();

template<>
inline MPI_Datatype GetMPIDataType<QLTEN_Double>() { return MPI_DOUBLE; }
template<>
inline MPI_Datatype GetMPIDataType<QLTEN_Float>() { return MPI_FLOAT; }
template<>
inline MPI_Datatype GetMPIDataType<QLTEN_Complex>() { return MPI_CXX_DOUBLE_COMPLEX; }
template<>
inline MPI_Datatype GetMPIDataType<QLTEN_ComplexFloat>() { return MPI_CXX_FLOAT_COMPLEX; }
template<>
inline MPI_Datatype GetMPIDataType<char>() { return MPI_CHAR; }

template<>
inline MPI_Datatype GetMPIDataType<size_t>() {
  static_assert(
      std::is_same_v<size_t, unsigned> ||
      std::is_same_v<size_t, unsigned long> ||
      std::is_same_v<size_t, unsigned long long>,
      "Unsupported size_t underlying type for MPI datatype mapping.");

  if constexpr (std::is_same_v<size_t, unsigned>) {
    return MPI_UNSIGNED;
  } else if constexpr (std::is_same_v<size_t, unsigned long>) {
    return MPI_UNSIGNED_LONG;
  } else {
    return MPI_UNSIGNED_LONG_LONG;
  }
}

/** @brief Return the communicator's largest valid point-to-point tag. */
inline int GetMPITagUpperBound(const MPI_Comm &comm) {
  int *tag_upper_bound = nullptr;
  int found = 0;
  HANDLE_MPI_ERROR(::MPI_Comm_get_attr(
      comm, MPI_TAG_UB, &tag_upper_bound, &found));
  if (!found || tag_upper_bound == nullptr) {
    throw std::runtime_error("MPI communicator does not expose MPI_TAG_UB");
  }
  return *tag_upper_bound;
}

/** @brief Validate a contiguous MPI tag span before posting any operations. */
inline void ValidateMPITagRange(const int first_tag,
                                const size_t tag_count,
                                const MPI_Comm &comm) {
  if (first_tag < 0) {
    throw std::invalid_argument("MPI send tag must be nonnegative");
  }
  if (tag_count == 0) { return; }

  const uint64_t tag_upper_bound =
      static_cast<uint64_t>(GetMPITagUpperBound(comm));
  const uint64_t first_tag_value = static_cast<uint64_t>(first_tag);
  if (first_tag_value > tag_upper_bound ||
      tag_count - 1 > tag_upper_bound - first_tag_value) {
    throw std::out_of_range("MPI send requires tags beyond MPI_TAG_UB");
  }
}

template<typename ElemT>
constexpr size_t MPILargeMessageChunkElementCount() {
  constexpr size_t chunk_count = kMPIMaxChunkSize / sizeof(ElemT);
  static_assert(chunk_count > 0,
                "MPI chunk byte limit is smaller than one element.");
  return chunk_count;
}

/** @brief Number of point-to-point requests used by a chunked buffer send. */
template<typename ElemT>
size_t MPILargeMessageRequestCount(const size_t data_size) {
  constexpr size_t chunk_count = MPILargeMessageChunkElementCount<ElemT>();
  const size_t full_chunks = data_size / chunk_count;
  const size_t remainder = data_size % chunk_count;
  return full_chunks + ((remainder > 0 || data_size == 0) ? 1 : 0);
}

namespace detail {

template<typename ElemT>
void MPI_IsendChunksUnchecked(const ElemT *data,
                              const size_t data_size,
                              const int dest,
                              const int tag,
                              const MPI_Comm &comm,
                              std::vector<MPI_Request> &requests) {
  constexpr size_t chunk_count = MPILargeMessageChunkElementCount<ElemT>();
  const size_t full_chunks = data_size / chunk_count;
  const size_t remainder = data_size % chunk_count;
  for (size_t i = 0; i < full_chunks; ++i) {
    requests.push_back(MPI_REQUEST_NULL);
    HANDLE_MPI_ERROR(::MPI_Isend(
        data + i * chunk_count,
        static_cast<int>(chunk_count),
        GetMPIDataType<ElemT>(),
        dest,
        tag + static_cast<int>(i),
        comm,
        &requests.back()));
  }

  if (remainder > 0 || data_size == 0) {
    const ElemT *chunk_start = data;
    if (full_chunks > 0) {
      chunk_start = data + full_chunks * chunk_count;
    }
    requests.push_back(MPI_REQUEST_NULL);
    HANDLE_MPI_ERROR(::MPI_Isend(
        chunk_start,
        static_cast<int>(remainder),
        GetMPIDataType<ElemT>(),
        dest,
        tag + static_cast<int>(full_chunks),
        comm,
        &requests.back()));
  }
}

}  // namespace detail

/**
 * @brief Append nonblocking, large-count sends for an existing raw buffer.
 *
 * Each chunk uses the next consecutive tag, matching `MPI_Recv` below. The
 * caller owns both `data` and the appended requests until completion.
 */
template<typename ElemT>
void MPI_Isend(const ElemT *data,
               const size_t data_size,
               const int dest,
               const int tag,
               const MPI_Comm &comm,
               std::vector<MPI_Request> &requests) {
  const size_t request_count = MPILargeMessageRequestCount<ElemT>(data_size);
  ValidateMPITagRange(tag, request_count, comm);
  if (request_count > std::numeric_limits<size_t>::max() - requests.size()) {
    throw std::overflow_error("MPI request count overflow");
  }
  requests.reserve(requests.size() + request_count);
  detail::MPI_IsendChunksUnchecked(data, data_size, dest, tag, comm, requests);
}

///< Send a solely number with type of size_t
inline void MPI_Send(const size_t n,
                     const int dest,
                     const int tag,
                     const MPI_Comm &comm) {
  HANDLE_MPI_ERROR(::MPI_Send((const void *) (&n), 1, GetMPIDataType<size_t>(), dest, tag, comm));
}

inline ::MPI_Status MPI_Recv(size_t &n,
                             const int source,
                             const int tag,
                             const MPI_Comm &comm) {
  ::MPI_Status status;
  HANDLE_MPI_ERROR(::MPI_Recv((void *) (&n), 1, GetMPIDataType<size_t>(), source, tag, comm, &status));
  return status;
}

template<typename ElemT>
void MPI_Send(const ElemT *data,
              const size_t data_size,
              const int dest,
              const int tag,
              const MPI_Comm &comm
) {
  constexpr size_t chunk_count = kMPIMaxChunkSize / sizeof(ElemT); // Max number of elements per chunk
  size_t num_chunks = data_size / chunk_count;
  size_t remainder = data_size % chunk_count;
  for (size_t i = 0; i < num_chunks; i++) {
    const ElemT *chunk_start = data + i * chunk_count;
    HANDLE_MPI_ERROR(::MPI_Send(chunk_start, chunk_count, GetMPIDataType<ElemT>(), dest, tag + i, comm));
  }

  if (remainder > 0 || data_size == 0) {
    const ElemT *chunk_start = data + num_chunks * chunk_count;
    HANDLE_MPI_ERROR(::MPI_Send(chunk_start, remainder, GetMPIDataType<ElemT>(), dest, tag + num_chunks, comm));
  }
}

template<typename ElemT>
MPI_Status MPI_Recv(ElemT *data,
                    const size_t data_size,
                    const int source,
                    const int tag,
                    const MPI_Comm &comm
) {
  assert(source != MPI_ANY_SOURCE);
  constexpr size_t chunk_count = kMPIMaxChunkSize / sizeof(ElemT); // Max number of elements per chunk
  size_t num_chunks = data_size / chunk_count;
  size_t remainder = data_size % chunk_count;
  ::MPI_Status status;
  for (size_t i = 0; i < num_chunks; i++) {
    ElemT *chunk_start = data + i * chunk_count;
    HANDLE_MPI_ERROR(::MPI_Recv(chunk_start, chunk_count, GetMPIDataType<ElemT>(), source, tag + i, comm, &status));
  }

  //communication when data_size == 0 will write the status
  if (remainder > 0 || data_size == 0) {
    ElemT *chunk_start = data + num_chunks * chunk_count;
    HANDLE_MPI_ERROR(::MPI_Recv(chunk_start,
                                remainder,
                                GetMPIDataType<ElemT>(),
                                source,
                                tag + num_chunks,
                                comm,
                                &status));
  }
  return status;
}

template<typename ElemT>
inline void MPI_Bcast(ElemT *data,
                      const size_t data_size,
                      const int root,
                      const MPI_Comm &comm) {
  constexpr size_t chunk_count = kMPIMaxChunkSize / sizeof(ElemT); // Max number of elements per chunk
  size_t num_chunks = data_size / chunk_count;
  size_t remainder = data_size % chunk_count;

  for (size_t i = 0; i < num_chunks; i++) {
    ElemT *chunk_start = data + i * chunk_count;
    HANDLE_MPI_ERROR(::MPI_Bcast(chunk_start, chunk_count, GetMPIDataType<ElemT>(), root, comm));
  }

  if (remainder > 0) {
    ElemT *chunk_start = data + num_chunks * chunk_count;
    HANDLE_MPI_ERROR(::MPI_Bcast(chunk_start, remainder, GetMPIDataType<ElemT>(), root, comm));
  }
}

///< all messages in different process have same length
inline void GatherAndPrintErrorMessages(
    const std::string &local_msg,
    const MPI_Comm &comm
) {
  std::vector<char> global_msgs;
  int rank, mpi_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &mpi_size);
  if (rank == kMPIMasterRank) {
    global_msgs.resize(local_msg.size() * mpi_size);
  }

  HANDLE_MPI_ERROR(::MPI_Gather(local_msg.data(), local_msg.size(), MPI_CHAR,
                                global_msgs.data(), local_msg.size(), MPI_CHAR,
                                kMPIMasterRank, comm));

  if (rank == kMPIMasterRank) {
//    for (int r = 0; r < mpi_size; ++r) {
//      std::cerr << &global_msgs[r * local_msg.size()];
//    }
    std::cerr << global_msgs.data();
  }
}

}//hp_numeric
}//qlten

#endif
