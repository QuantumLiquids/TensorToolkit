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

#include "qlten/framework/value_t.h"      // QLTEN_Double, QLTEN_Complex
#include "mpi.h"                          // MPI_Send, MPI_Recv, MPI_Bcast

namespace qlten {
/// High performance numerical functions.
// MPI master's rank
[[deprecated("Use qlten::hp_numeric::kMPIMasterRank instead")]]
const size_t kMPIMasterRank = 0;
namespace hp_numeric {
// MPI master's rank - preferred location
const size_t kMPIMasterRank = 0;
const size_t kMPIMaxChunkSize = ((size_t) 1 << 31) - 1;  // byte

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

template<typename ElemT>
MPI_Datatype GetMPIDataType();

template<>
inline MPI_Datatype GetMPIDataType<QLTEN_Double>() { return MPI_DOUBLE; }
template<>
inline MPI_Datatype GetMPIDataType<QLTEN_Complex>() { return MPI_CXX_DOUBLE_COMPLEX; }

///< Send a solely number with type of size_t
inline void MPI_Send(const size_t n,
                     const int dest,
                     const int tag,
                     const MPI_Comm &comm) {
  HANDLE_MPI_ERROR(::MPI_Send((const void *) (&n), 1, MPI_UNSIGNED_LONG_LONG, dest, tag, comm));
}

inline ::MPI_Status MPI_Recv(size_t &n,
                             const int source,
                             const int tag,
                             const MPI_Comm &comm) {
  ::MPI_Status status;
  HANDLE_MPI_ERROR(::MPI_Recv((void *) (&n), 1, MPI_UNSIGNED_LONG_LONG, source, tag, comm, &status));
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