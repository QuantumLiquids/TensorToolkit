// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2021-09-27 12:34
*
* Description: QuantumLiquids/tensor project. MPI related
*/

/**
@file blas_level1.h
@brief Wrapper for the MPI related functions.
*/

#ifndef QLTEN_FRAMEWORK_HP_NUMERIC_MPI_FUN_H
#define QLTEN_FRAMEWORK_HP_NUMERIC_MPI_FUN_H

#include "qlten/framework/value_t.h"      // QLTEN_Double, QLTEN_Complex
#include "mpi.h"                          // MPI_Send, MPI_Recv...

namespace qlten {
/// High performance numerical functions.
namespace hp_numeric {
const size_t kAssumedMPICommunicationMaxDataLength = 4e9;      // char, multiples of 16 (size of complex number)
const size_t kAssumedMPICommunicationMaxDoubleDataSize = kAssumedMPICommunicationMaxDataLength / sizeof(QLTEN_Double);
const size_t kAssumedMPICommunicationMaxComplexDataSize = kAssumedMPICommunicationMaxDataLength / sizeof(QLTEN_Complex);

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

///< Block point-to-point communication wrapper
///< The upperbound of size_t is much large than int that is used in MPI API.
inline void MPI_Send(const QLTEN_Double *data,
                     const size_t data_size,
                     const int dest,
                     const int tag,
                     const MPI_Comm &comm
) {
  if (data_size <= kAssumedMPICommunicationMaxDoubleDataSize) {
    HANDLE_MPI_ERROR(::MPI_Send((const void *) data, data_size, MPI_DOUBLE, dest, tag, comm));
  } else {
    size_t num_fragments = data_size / kAssumedMPICommunicationMaxDoubleDataSize + 1;
    for (size_t i = 0; i < num_fragments - 1; i++) {
      char *fragment_start = (char *) data + i * kAssumedMPICommunicationMaxDataLength;
      HANDLE_MPI_ERROR(::MPI_Send((const void *) fragment_start,
                                  kAssumedMPICommunicationMaxDoubleDataSize,
                                  MPI_DOUBLE,
                                  dest,
                                  tag + i,
                                  comm));
    }
    size_t remain_data_size =
        data_size - kAssumedMPICommunicationMaxDoubleDataSize * (num_fragments - 1);
    char *fragment_start = (char *) data + (num_fragments - 1) * kAssumedMPICommunicationMaxDataLength;
    HANDLE_MPI_ERROR(::MPI_Send((const void *) fragment_start,
                                remain_data_size,
                                MPI_DOUBLE,
                                dest,
                                tag + num_fragments - 1,
                                comm));
  }
}

inline void MPI_Send(const QLTEN_Complex *data,
                     const size_t data_size,
                     const int dest,
                     const int tag,
                     const MPI_Comm &comm
) {
  if (data_size <= kAssumedMPICommunicationMaxComplexDataSize) {
    HANDLE_MPI_ERROR(::MPI_Send((const void *) data, data_size, MPI_CXX_DOUBLE_COMPLEX, dest, tag, comm));
  } else {
    size_t num_fragment = data_size / kAssumedMPICommunicationMaxComplexDataSize + 1;
    for (size_t i = 0; i < num_fragment - 1; i++) {
      char *fragment_start = (char *) data + i * kAssumedMPICommunicationMaxDataLength;
      HANDLE_MPI_ERROR(::MPI_Send((const void *) fragment_start,
                                  kAssumedMPICommunicationMaxComplexDataSize,
                                  MPI_CXX_DOUBLE_COMPLEX,
                                  dest,
                                  tag + i,
                                  comm));
    }
    size_t remain_data_size =
        data_size - kAssumedMPICommunicationMaxComplexDataSize * (num_fragment - 1);
    char *fragment_start = (char *) data + (num_fragment - 1) * kAssumedMPICommunicationMaxDataLength;
    HANDLE_MPI_ERROR(::MPI_Send((const void *) fragment_start,
                                remain_data_size,
                                MPI_CXX_DOUBLE_COMPLEX,
                                dest,
                                tag + num_fragment - 1,
                                comm));
  }
}

inline MPI_Status MPI_Recv(QLTEN_Double *data,
                           const size_t data_size,
                           const int source,
                           const int tag,
                           const MPI_Comm &comm
) {
  assert(source != MPI_ANY_SOURCE);
  ::MPI_Status status;
  if (data_size <= kAssumedMPICommunicationMaxDoubleDataSize) {
    HANDLE_MPI_ERROR(::MPI_Recv((void *) data, data_size, MPI_DOUBLE, source, tag, comm, &status));
  } else {
    size_t num_fragment = data_size / kAssumedMPICommunicationMaxDoubleDataSize + 1;
    for (size_t i = 0; i < num_fragment - 1; i++) {
      char *fragment_start = (char *) data + i * kAssumedMPICommunicationMaxDataLength;
      HANDLE_MPI_ERROR(::MPI_Recv((void *) fragment_start,
                                  kAssumedMPICommunicationMaxDoubleDataSize,
                                  MPI_DOUBLE,
                                  source,
                                  tag + i,
                                  comm,
                                  &status));
    }
    size_t remain_data_size =
        data_size - kAssumedMPICommunicationMaxDoubleDataSize * (num_fragment - 1);
    char *fragment_start = (char *) data + (num_fragment - 1) * kAssumedMPICommunicationMaxDataLength;
    HANDLE_MPI_ERROR(::MPI_Recv((void *) fragment_start,
                                remain_data_size,
                                MPI_DOUBLE,
                                source,
                                tag + num_fragment - 1,
                                comm,
                                &status));
  }
  return status;
}

///< note sizeof(QLTEN_Complex) = 16, while sizeof(MPI_CXX_DOUBLE_COMPLEX) = 8
inline MPI_Status MPI_Recv(QLTEN_Complex *data,
                           const size_t data_size,
                           const size_t source,
                           const int tag,
                           const MPI_Comm &comm) {
  ::MPI_Status status;
  if (data_size <= kAssumedMPICommunicationMaxComplexDataSize) {
    HANDLE_MPI_ERROR(::MPI_Recv((void *) data, data_size, MPI_CXX_DOUBLE_COMPLEX, source, tag, comm, &status));
  } else {
    size_t num_fragment = data_size / kAssumedMPICommunicationMaxComplexDataSize + 1;
    for (size_t i = 0; i < num_fragment - 1; i++) {
      char *fragment_start = (char *) data + i * kAssumedMPICommunicationMaxDataLength;
      HANDLE_MPI_ERROR(::MPI_Recv((void *) fragment_start,
                                  kAssumedMPICommunicationMaxComplexDataSize,
                                  MPI_CXX_DOUBLE_COMPLEX,
                                  source,
                                  tag + i,
                                  comm,
                                  &status));
    }
    size_t remain_data_size =
        data_size - kAssumedMPICommunicationMaxComplexDataSize * (num_fragment - 1);
    char *fragment_start = (char *) data + (num_fragment - 1) * kAssumedMPICommunicationMaxDataLength;
    HANDLE_MPI_ERROR(::MPI_Recv((void *) fragment_start,
                                remain_data_size,
                                MPI_CXX_DOUBLE_COMPLEX,
                                source,
                                tag + num_fragment - 1,
                                comm,
                                &status));
  }
  return status;
}

inline void MPI_Bcast(QLTEN_Double *data,
                      const size_t data_size,
                      const int root,
                      const MPI_Comm &comm) {
  if (data_size <= kAssumedMPICommunicationMaxDoubleDataSize) {
    HANDLE_MPI_ERROR(::MPI_Bcast((void *) data, data_size, MPI_DOUBLE, root, comm));
  } else {
    size_t times_of_sending = data_size / kAssumedMPICommunicationMaxDoubleDataSize + 1;
    for (size_t i = 0; i < times_of_sending - 1; i++) {
      char *fragment_start = (char *) data + i * kAssumedMPICommunicationMaxDataLength;
      HANDLE_MPI_ERROR(::MPI_Bcast((void *) fragment_start,
                                   kAssumedMPICommunicationMaxDoubleDataSize,
                                   MPI_DOUBLE,
                                   root,
                                   comm));
    }
    size_t remain_data_size =
        data_size - kAssumedMPICommunicationMaxDoubleDataSize * (times_of_sending - 1);
    char *fragment_start = (char *) data + (times_of_sending - 1) * kAssumedMPICommunicationMaxDataLength;
    HANDLE_MPI_ERROR(::MPI_Bcast((void *) fragment_start,
                                 remain_data_size,
                                 MPI_DOUBLE,
                                 root,
                                 comm));
  }
}

inline void MPI_Bcast(QLTEN_Complex *data,
                      const size_t data_size,
                      const int root,
                      const MPI_Comm &comm) {
  if (data_size <= kAssumedMPICommunicationMaxComplexDataSize) {
    HANDLE_MPI_ERROR(::MPI_Bcast((void *) data, data_size, MPI_CXX_DOUBLE_COMPLEX, root, comm));
  } else {
    size_t times_of_sending = data_size / kAssumedMPICommunicationMaxComplexDataSize + 1;
    for (size_t i = 0; i < times_of_sending - 1; i++) {
      char *fragment_start = (char *) data + i * kAssumedMPICommunicationMaxDataLength;
      HANDLE_MPI_ERROR(::MPI_Bcast((void *) fragment_start,
                                   kAssumedMPICommunicationMaxComplexDataSize,
                                   MPI_CXX_DOUBLE_COMPLEX,
                                   root,
                                   comm));
    }
    size_t remain_data_size =
        data_size - kAssumedMPICommunicationMaxComplexDataSize * (times_of_sending - 1);
    char *fragment_start = (char *) data + (times_of_sending - 1) * kAssumedMPICommunicationMaxDataLength;
    HANDLE_MPI_ERROR(::MPI_Bcast((void *) fragment_start,
                                 remain_data_size,
                                 MPI_CXX_DOUBLE_COMPLEX,
                                 root,
                                 comm));
  }
}

}//hp_numeric
}//qlten

#endif