# Index Lineage Design

This document records the design contract for the lightweight index lineage API
in TensorToolkit. The API lets callers track the ordered logical leaves inside
one fused tensor index without making `QLTensor` own downstream metadata.

## Motivation

Downstream tensor-network applications sometimes need operation provenance in
addition to numerical tensor data. A typical example is fermionic PEPS code
that fuses physical legs and later needs to recover the order of the physical
leaves in the fused leg.

TensorToolkit already handles numerical fermion signs inside tensor operations.
Index lineage is separate caller-owned metadata: it records which logical leaves
participated in a fused index and in what order.

## Scope

Index lineage is deliberately small:

- It tracks the ordered leaves of one index, not every index of a tensor.
- It is independent of quantum numbers and tensor element types.
- It is opt-in metadata supplied by callers, not state stored inside
  `QLTensor`.
- It does not change numerical tensor behavior.
- It does not implement downstream fermionic operator signs.

## Core Types

`IndexLineage<T>` stores an ordered list of caller-defined leaf labels for one
tensor index. The default label type is `size_t` for compatibility with simple
integer labels.

```cpp
namespace qlten {

template<typename T = size_t>
struct IndexLineage {
  std::vector<T> leaf_ids;
};

}  // namespace qlten
```

The label type can be richer than an integer. For example, downstream PEPS code
can use a physical-leg tag:

```cpp
struct PhysTag {
  SiteIdx site;
  PhysicalParity parity;
};

using PhysLineage = qlten::IndexLineage<PhysTag>;
```

TensorToolkit only copies, moves, stores, and compares labels when callers use
`operator==` on the lineage.

## Pure Helper

The pure helper concatenates the two lineages that participate in an index
fusion:

```cpp
template<typename T = size_t>
IndexLineage<T> FuseIndexLineage(
    const IndexLineage<T> &left_lineage,
    const IndexLineage<T> &right_lineage
);
```

The result preserves fuse order:

```cpp
result.leaf_ids = left_lineage.leaf_ids + right_lineage.leaf_ids;
```

There is intentionally no `IndexLineages` alias and no transpose or contraction
helper. Callers that need to move full-tensor metadata should own that mapping
in downstream code.

## Fuse Metadata

`IndexFusionInfo<QNT>` describes the quantum-number sector metadata produced by
index fusion:

```cpp
template<typename QNT>
struct IndexFusionInfo {
  Index<QNT> left_index;
  Index<QNT> right_index;
  Index<QNT> fused_index;
  std::vector<QNSctsOffsetInfo> sector_offsets;
};
```

## FuseIndex Result

The lineage-aware fuse overload returns both sector-fusion metadata and the
concatenated fused lineage:

```cpp
template<typename QNT, typename LineageT = size_t>
struct FuseIndexResult {
  IndexFusionInfo<QNT> index_fusion;
  IndexLineage<LineageT> fused_lineage;
};
```

The ordinary overload keeps the concise API surface:

```cpp
IndexFusionInfo<QNT> FuseIndex(size_t idx1, size_t idx2);
```

The lineage-aware overload applies the same numerical operation and additionally
returns the fused lineage:

```cpp
template<typename LineageT = size_t>
FuseIndexResult<QNT, LineageT> FuseIndex(
    size_t idx1,
    size_t idx2,
    const IndexLineage<LineageT> &left_lineage,
    const IndexLineage<LineageT> &right_lineage
);
```

Semantics:

- `idx1` and `idx2` are tensor-axis positions and remain `size_t`.
- `left_lineage` is the lineage associated with the index at `idx1`.
- `right_lineage` is the lineage associated with the index at `idx2`.
- The numerical tensor transformation is identical to the two-argument
  `FuseIndex` overload.
- `fused_lineage` is `FuseIndexLineage(left_lineage, right_lineage)`.

## Header Layout

- `include/qlten/tensor_manipulation/index_lineage.h`
  - Includes only standard-library headers.
  - Defines `IndexLineage<T>` and `FuseIndexLineage<T>`.
- `include/qlten/tensor_manipulation/ten_fuse_index.h`
  - Includes `index_lineage.h`, `index_combine.h`, and the tensor headers
    already needed by the fuse implementation.
  - Defines `IndexFusionInfo<QNT>` and `FuseIndexResult<QNT, LineageT>`.
  - Defines both `QLTensor::FuseIndex` overloads.
- `include/qlten/qltensor/qltensor.h`
  - Includes `index_lineage.h`, because the lineage-aware overload takes
    `const IndexLineage<LineageT> &`.
  - Forward-declares `IndexFusionInfo<QNT>` and
    `FuseIndexResult<QNT, LineageT>`.

## Test Coverage

Focused tests live under `tests/test_tensor_manipulation/`:

- `FuseIndexLineage` concatenates default `size_t` leaves.
- `FuseIndexLineage` accepts structured caller-defined leaf tags.
- The lineage-aware `FuseIndex` returns the same `IndexFusionInfo` information
  as the non-lineage overload and the expected fused lineage.
