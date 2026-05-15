#include <gtest/gtest.h>

#include "qlten/tensor_manipulation/index_lineage.h"

using namespace qlten;

namespace {

enum class PhysicalParity { kEven, kOdd };

struct PhysTag {
  size_t site;
  PhysicalParity parity;
};

bool operator==(const PhysTag &lhs, const PhysTag &rhs) {
  return lhs.site == rhs.site && lhs.parity == rhs.parity;
}

}  // namespace

TEST(IndexLineage, FuseIndexLineageConcatenatesSizeTLeaves) {
  const IndexLineage<> left_lineage = {{0, 2}};
  const IndexLineage<> right_lineage = {{3}};
  const IndexLineage<> expected = {{0, 2, 3}};

  EXPECT_EQ(FuseIndexLineage(left_lineage, right_lineage), expected);
}

TEST(IndexLineage, FuseIndexLineageAcceptsStructuredLeafTags) {
  const IndexLineage<PhysTag> left_lineage = {
      {{0, PhysicalParity::kEven}, {1, PhysicalParity::kOdd}}
  };
  const IndexLineage<PhysTag> right_lineage = {{{2, PhysicalParity::kEven}}};
  const IndexLineage<PhysTag> expected = {
      {{0, PhysicalParity::kEven},
       {1, PhysicalParity::kOdd},
       {2, PhysicalParity::kEven}}
  };

  EXPECT_EQ(FuseIndexLineage(left_lineage, right_lineage), expected);
}
