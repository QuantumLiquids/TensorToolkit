// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Codex
*
* Description: Unittests for CPU numerics thread configuration.
*/

#include "gtest/gtest.h"
#include "qlten/framework/hp_numeric/omp_set.h"

using namespace qlten;

TEST(TestCpuNumericsThreads, SetsTensorManipulationThreadsForCpuBuilds) {
  hp_numeric::SetCpuNumericsThreads(1);
#ifndef USE_GPU
  EXPECT_EQ(hp_numeric::GetTensorManipulationThreads(), 1);
#endif
}
