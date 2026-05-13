#include <gtest/gtest.h>

#include "qlten/tensor_manipulation/index_lineage.h"

using namespace qlten;

TEST(IndexLineage, TransposeLineages) {
  const IndexLineages lineages = {{{0}}, {{1, 10}}, {{2}}, {{3}}};
  const std::vector<size_t> axes = {2, 0, 3, 1};
  const IndexLineages expected = {{{2}}, {{0}}, {{3}}, {{1, 10}}};

  EXPECT_EQ(TransposeLineages(lineages, axes), expected);
}

TEST(IndexLineage, FuseIndexLineagesAdjacent) {
  const IndexLineages lineages = {{{0}}, {{1}}, {{2}}, {{3}}};
  const IndexLineages expected = {{{0, 1}}, {{2}}, {{3}}};

  EXPECT_EQ(FuseIndexLineages(lineages, 0, 1), expected);
}

TEST(IndexLineage, FuseIndexLineagesNonAdjacent) {
  const IndexLineages lineages = {{{0}}, {{1}}, {{2}}, {{3}}};
  const IndexLineages expected = {{{1, 3}}, {{0}}, {{2}}};

  EXPECT_EQ(FuseIndexLineages(lineages, 1, 3), expected);
}

TEST(IndexLineage, ContractOutputLineagesNaturalOrdinaryContractOrder) {
  const IndexLineages a_lineages = {{{0}}, {{1}}, {{2}}, {{3}}};
  const IndexLineages b_lineages = {{{10}}, {{11}}, {{12}}, {{13}}};
  const std::vector<std::vector<size_t>> axes = {{2, 0}, {3, 1}};
  const IndexLineages expected = {{{1}}, {{3}}, {{10}}, {{12}}};

  EXPECT_EQ(ContractOutputLineages(a_lineages, b_lineages, axes), expected);
}

TEST(IndexLineage, ContractOutputLineagesCanReturnScalarLineage) {
  const IndexLineages a_lineages = {{{0}}, {{1}}};
  const IndexLineages b_lineages = {{{10}}, {{11}}};
  const std::vector<std::vector<size_t>> axes = {{0, 1}, {1, 0}};

  EXPECT_TRUE(ContractOutputLineages(a_lineages, b_lineages, axes).empty());
}
