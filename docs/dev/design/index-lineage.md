# Index Lineage Design

This document records the long-term design contract for the index lineage API
in TensorToolkit. The API lets callers track how logical leaf axes move through
selected tensor-manipulation operations without making `QLTensor` own
downstream metadata.

## Motivation

Downstream tensor-network applications sometimes need operation provenance in
addition to numerical tensor data. A typical example is a fermionic PEPS
measurement where selected trivial physical indices represent site-level modes.
The downstream code needs to know the final order of those leaves after
transpose, fuse, and contraction steps so that it can apply its own operator
sign conventions consistently.

TensorToolkit already handles numerical fermion signs inside tensor operations.
Index lineage is a separate concern: for each current tensor axis, which
original leaf axes does it contain, and in what order?

## Scope

Index lineage is deliberately lightweight:

- It tracks provenance and order of logical leaves.
- It is independent of quantum numbers, fermion parity, and model semantics.
- It is opt-in metadata supplied by callers, not state stored inside
  `QLTensor`.
- It does not change numerical tensor behavior.
- It does not implement downstream fermionic operator signs.

## Header Layout

The header layout avoids adding downstream labels to `Index<QNT>` and avoids
cycles between core tensor headers and tensor-manipulation implementation
headers.

- `include/qlten/tensor_manipulation/index_lineage.h`
  - Includes only standard-library headers.
  - Defines `IndexLineage` and `IndexLineages`.
  - Defines pure lineage helpers that do not depend on `QLTensor`.
- `include/qlten/tensor_manipulation/ten_fuse_index.h`
  - Includes `index_lineage.h`, `index_combine.h`, and the tensor headers
    already needed by the fuse implementation.
  - Defines `IndexFusionInfo<QNT>` and `FuseIndexResult<QNT>`.
  - Defines both `QLTensor::FuseIndex` overloads.
- `include/qlten/qltensor/qltensor.h`
  - Includes `index_lineage.h`, because the lineage-aware overload takes
    `const IndexLineages &`.
  - Forward-declares `template<typename QNT> struct IndexFusionInfo;`.
  - Forward-declares `template<typename QNT> struct FuseIndexResult;`.
  - Declares both `FuseIndex` overloads without including
    `ten_fuse_index.h`.

This keeps `qltensor.h` aware of the lightweight lineage argument type without
making it depend on the full fuse implementation.

## Core Types

`IndexLineage` stores an ordered list of logical leaf ids for one tensor axis:

```cpp
namespace qlten {

struct IndexLineage {
  std::vector<size_t> leaf_ids;
};

using IndexLineages = std::vector<IndexLineage>;

}  // namespace qlten
```

`IndexLineage` is intentionally not templated on `QNT`. It describes only
provenance and order. Downstream code can encode richer labels, such as
`(site, spin)` or `site_id`, into `size_t`.

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

The name is intentionally narrower than `FuseInfo`: it describes index-sector
fusion metadata, not all possible provenance produced by a fuse operation.
There is no `FuseInfo` compatibility alias.

## FuseIndex Result

The lineage-aware fuse overload returns both sector-fusion metadata and output
lineages:

```cpp
template<typename QNT>
struct FuseIndexResult {
  IndexFusionInfo<QNT> index_fusion;
  IndexLineages output_lineages;
};
```

The result does not include a separate `fused_lineage` field because the fused
lineage is `output_lineages[0]` under the current `FuseIndex` output
convention.

## FuseIndex Contract

The concise overload keeps the ordinary API surface:

```cpp
IndexFusionInfo<QNT> FuseIndex(size_t idx1, size_t idx2);
```

The lineage-aware overload applies the same numerical operation and additionally
returns transformed lineage metadata:

```cpp
FuseIndexResult<QNT> FuseIndex(
    size_t idx1,
    size_t idx2,
    const IndexLineages &input_lineages
);
```

Semantics:

- `input_lineages.size() == Rank()` before the fuse.
- `input_lineages[i]` corresponds to `GetIndex(i)` before the fuse.
- The numerical tensor transformation is identical to the two-argument
  `FuseIndex` overload.
- `FuseIndex(idx1, idx2)` first transposes the tensor so `idx1` and `idx2`
  become axes `0` and `1`, in that order, then fuses those leading axes into
  output axis `0`.
- This transpose-to-axes-0-and-1 order is part of the public API contract. For
  fermionic tensors, any exchange sign comes from that pre-fusion transpose;
  fusion then assumes the two axes are adjacent and does not introduce another
  exchange.
- The fused lineage is
  `input_lineages[idx1].leaf_ids + input_lineages[idx2].leaf_ids`.
- The remaining output lineages follow the same output index order as the
  tensor after `FuseIndex`.

Example:

```cpp
input_lineages = [a, b, c, d]
FuseIndex(1, 3, input_lineages)

output tensor axes: [fuse(b, d), a, c]
output_lineages:   [b + d,      a, c]
```

## Pure Lineage Helpers

Pure helpers compute metadata without touching tensor data:

```cpp
IndexLineages TransposeLineages(
    const IndexLineages &input_lineages,
    const std::vector<size_t> &axes
);

IndexLineages FuseIndexLineages(
    const IndexLineages &input_lineages,
    size_t idx1,
    size_t idx2
);

IndexLineages ContractOutputLineages(
    const IndexLineages &a_lineages,
    const IndexLineages &b_lineages,
    const std::vector<std::vector<size_t>> &axes_set
);
```

`TransposeLineages` applies the same permutation convention as
`QLTensor::Transpose`.

`FuseIndexLineages` applies the same output-axis convention as
`QLTensor::FuseIndex`: the fused axis is first, then the remaining original axes
appear in the order produced by the current implementation.

`ContractOutputLineages` mirrors only the ordinary `ten_ctrct.h::Contract`
output order: all uncontracted axes from tensor A in natural axis order,
followed by all uncontracted axes from tensor B in natural axis order. This is
the order produced by `TenCtrctGenSavedAxesSet()` before ordinary `Contract`
calls `TenCtrctInitResTen()`.

`ContractOutputLineages` does not cover matrix-based contraction in
`contract_contiguous_axes.h`. That path builds cyclic saved-axis order from
its contraction starts, and the result order cannot be reproduced from the
axes-set-only helper signature. If lineage support is needed there later, add a
separate helper that accepts the matrix-based contraction parameters or the
explicit saved axes.

## Error Handling

Lineage helpers and lineage-aware operations use assertions in the same style
as nearby tensor-manipulation code:

- lineage count must equal tensor rank where a tensor is involved;
- transpose axes must be a full permutation;
- fuse axes must satisfy `idx1 < idx2`;
- contraction axes must be valid for the two lineage vectors.

## Test Coverage

Focused tests live under `tests/test_tensor_manipulation/`:

- `TransposeLineages` reorders leaf ids using the tensor transpose convention.
- `FuseIndexLineages` matches the `FuseIndex` output order for adjacent and
  non-adjacent axes.
- The lineage-aware `FuseIndex` returns the same `IndexFusionInfo` information
  as the non-lineage overload and the expected output lineage order.
- `ContractOutputLineages` matches ordinary `ten_ctrct.h::Contract` output
  order.
- In-repository references use `IndexFusionInfo`; no `FuseInfo` compatibility
  alias remains.

## Downstream Use

Downstream PEPS code can assign site-level leaves to projected trivial physical
indices, pass those lineages through selected tensor manipulation steps, and
inspect the final leaf order. The downstream estimator remains responsible for
mapping leaf ids to physical sites or modes and for applying fermionic operator
signs.
