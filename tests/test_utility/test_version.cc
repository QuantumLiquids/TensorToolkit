// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Codex
*
* Description: Unittests for TensorToolkit version macros.
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlten/version.h"

static_assert(QLTEN_VERSION_MAJOR_INT >= 0, "MAJOR must be non-negative");
static_assert(QLTEN_VERSION_MINOR_INT >= 0, "MINOR must be non-negative");
static_assert(QLTEN_VERSION_PATCH_INT >= 0, "PATCH must be non-negative");

TEST(TestVersion, PublicHeadersExposeSemanticVersionMacros) {
  EXPECT_STREQ(QLTEN_VERSION_MAJOR, "0");
  EXPECT_STREQ(QLTEN_VERSION_MINOR, "1");
  EXPECT_STREQ(QLTEN_VERSION_PATCH, "0");
  EXPECT_EQ(QLTEN_VERSION_MAJOR_INT, 0);
  EXPECT_EQ(QLTEN_VERSION_MINOR_INT, 1);
  EXPECT_EQ(QLTEN_VERSION_PATCH_INT, 0);
  EXPECT_STREQ(QLTEN_VERSION, "0.1.0");
}
