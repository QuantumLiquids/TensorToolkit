// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Codex
*
* Description: Unittests for MPI datatype helpers.
*/

#include <cstddef>
#include <type_traits>

#include "gtest/gtest.h"
#include "qlten/framework/hp_numeric/mpi_fun.h"

using namespace qlten;

namespace {

MPI_Datatype ExpectedMPIDataTypeForSizeT() {
  if constexpr (std::is_same_v<size_t, unsigned>) {
    return MPI_UNSIGNED;
  } else if constexpr (std::is_same_v<size_t, unsigned long>) {
    return MPI_UNSIGNED_LONG;
  } else if constexpr (std::is_same_v<size_t, unsigned long long>) {
    return MPI_UNSIGNED_LONG_LONG;
  } else {
    ADD_FAILURE() << "Unsupported size_t underlying type.";
    return MPI_DATATYPE_NULL;
  }
}

}  // namespace

TEST(TestMPIFun, SizeTMapsToMatchingMPIUnsignedType) {
  EXPECT_EQ(hp_numeric::GetMPIDataType<size_t>(), ExpectedMPIDataTypeForSizeT());
}
