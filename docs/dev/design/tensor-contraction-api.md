# Tensor Contraction API Design

**Status:** Implemented in the 2026-05 tensor contraction refactor.
**Audience:** TensorToolkit maintainers and downstream maintainers in
UltraDMRG/PEPS.

This document records the long-term design contract for TensorToolkit's tensor
contraction APIs. It intentionally keeps the API rationale and migration
hazards, while omitting the step-by-step implementation plan used during the
original refactor.

## Summary

TensorToolkit has two contraction surfaces with different intended uses:

- `qlten::Contract(&a, &b, axes_set, &c)` is the general contraction API. It
  accepts arbitrary axis lists and uses the standard tensor-transpose-based
  implementation path.
- `qlten::ContractContiguousAxes(a, b, a_start, b_start, size, c)` is the
  preferred API for the performance-oriented contiguous-axes contraction path.
  It requires the contracted axes to be contiguous and ascending, and it is
  designed for tensor-network update patterns where preserving or choosing a
  favorable index order can reduce explicit tensor transposes.

The old bool-tagged contiguous-axes `Contract<T, QNT, bool, bool>(...)`
overload remains as a source-compatibility wrapper for existing downstream
code, but new code should use `ContractContiguousAxes`.

DMRG-specialized helpers live under `qlten::dmrg`. They are included by the
general umbrella `qlten/qlten.h` and the non-MPI tensor manipulation bundle for
downstream convenience, but remain an expert API surface.

## General vs. contiguous-axes contraction

The general `Contract` API is the right default when the contracted axes are not
known to be contiguous, or when the caller wants a simple arbitrary-axis
interface. It handles the tensor-index rearrangement internally and remains the
semantic baseline for contraction correctness.

`ContractContiguousAxes` exists for a narrower performance reason. In common
DMRG and PEPS paths, contracted axes often form contiguous ranges. When those
ranges are exposed to the matrix-based contraction path, TensorToolkit can avoid
or reduce explicit tensor transpose work and express the contraction as matrix
operations. That lets optimized BLAS/MKL/cuBLAS-style vendor kernels handle the
matrix multiplication and, for some layouts, absorb transpose choices into the
matrix operation rather than forcing TensorToolkit to materialize a separate
tensor transpose first.

This optimization was introduced because UltraDMRG VMPS paths previously used
the general `Contract` path in places where the index pattern was more
structured. Switching those call sites to the contiguous-axes path reduced the
number of explicit transpose operations in the update flow. We do not currently
maintain a standardized benchmark comparing old-VMPS and new-VMPS end-to-end
runtime, so performance claims should stay qualitative unless such a benchmark
is added.

## `ContractContiguousAxes`

Header:

```cpp
#include "qlten/tensor_manipulation/contract_contiguous_axes.h"
```

Preferred API shape:

```cpp
namespace qlten {

enum class CtrctSide { Head, Tail };

template<typename TenElemT,
         typename QNT,
         CtrctSide ASide = CtrctSide::Tail,
         CtrctSide BSide = CtrctSide::Head>
void ContractContiguousAxes(
    const QLTensor<TenElemT, QNT> &a,
    const QLTensor<TenElemT, QNT> &b,
    size_t a_ctrct_axes_start,
    size_t b_ctrct_axes_start,
    size_t ctrct_axes_size,
    QLTensor<TenElemT, QNT> &c);

}  // namespace qlten
```

The side template parameters are part of the performance contract. They describe
which side of each tensor's index range contains the contracted contiguous axes:

- `ASide = CtrctSide::Tail`: A's contracted axes are at the tail side.
- `ASide = CtrctSide::Head`: A's contracted axes are at the head side.
- `BSide = CtrctSide::Head`: B's contracted axes are at the head side.
- `BSide = CtrctSide::Tail`: B's contracted axes are at the tail side.

The default `Tail, Head` case matches the most common tensor-network pattern:
contract the tail of `A` with the head of `B`.

The caller is responsible for using this API only when the contracted axes are
contiguous and ascending. For arbitrary axis lists, use the general
`Contract(&a, &b, axes_set, &c)` API instead.

## Legacy bool wrapper

The previous contiguous-axes API used the same function name as the general
`Contract` API:

```cpp
template<typename TenElemT,
         typename QNT,
         bool a_ctrct_tail = true,
         bool b_ctrct_head = true>
void Contract(
    const QLTensor<TenElemT, QNT> &a,
    const QLTensor<TenElemT, QNT> &b,
    size_t a_ctrct_axes_start,
    size_t b_ctrct_axes_start,
    size_t ctrct_axes_size,
    QLTensor<TenElemT, QNT> &c);
```

The bool template arguments are hard to read at call sites and are asymmetric:

| Legacy bool tag | Preferred enum tag |
| --- | --- |
| `a_ctrct_tail == true` | `ASide = CtrctSide::Tail` |
| `a_ctrct_tail == false` | `ASide = CtrctSide::Head` |
| `b_ctrct_head == true` | `BSide = CtrctSide::Head` |
| `b_ctrct_head == false` | `BSide = CtrctSide::Tail` |

The wrapper is retained because UltraDMRG and PEPS contain many existing
bool-tagged call sites. Keeping the function-level compatibility wrapper lets
downstreams migrate call sites gradually while preserving behavior.

This wrapper is not intended to be permanent. The expected compatibility window
is at most two minor releases, after which removing the bool-tagged overload can
be considered if downstream call sites have migrated.

## Mixed precision

The 2026-05 refactor mirrors the existing Double/Complex and Complex/Double
mixed-precision contiguous-axes overloads for both `ContractContiguousAxes` and
the legacy bool wrapper.

Float/ComplexFloat and ComplexFloat/Float support is a pre-existing API coverage
gap in the contiguous-axes path. Adding those overloads is a feature addition,
not part of the compatibility refactor, and should be handled by a separate API
completion patch with focused tests.

## Header compatibility

The refactor deliberately does not keep forwarding headers for the old paths:

- `qlten/tensor_manipulation/ten_ctrct_based_mat_trans.h`
- `qlten/tensor_manipulation/ten_ctrct_1sct.h`
- `qlten/tensor_manipulation/ten_block_expand.h`

The new paths are:

- `qlten/tensor_manipulation/contract_contiguous_axes.h`
- `qlten/tensor_manipulation/dmrg/contract_1sector.h`
- `qlten/tensor_manipulation/dmrg/block_expand.h`

This is a source-incompatible header-path change under strict semantic
versioning. In TensorToolkit's current pre-1.0 package surface, the change is
acceptable as a coordinated minor-version evolution because the affected
downstreams are controlled and can be updated synchronously. Downstream package
constraints should require the TensorToolkit version that contains the new
headers once those constraints are used for package consumption.

Recommended release handling for this refactor:

- bump TensorToolkit to the next minor version, because old include paths were
  removed;
- update UltraDMRG's minimum TensorToolkit constraint to that new TensorToolkit
  version, because UltraDMRG now includes the new `dmrg/` headers directly;
- leave PEPS unchanged unless its direct package constraints need to require a
  synchronized UltraDMRG release for unrelated reasons.

Do not reintroduce forwarding headers unless TensorToolkit commits to supporting
uncontrolled third-party consumers across the old and new include paths.

## DMRG-only API surface

`Contract1Sector`, `TensorContraction1SectorExecutor`, `ExpandQNBlocks`, and
related helpers are specialized hooks for DMRG/TDVP-style algorithms. They are
used by UltraDMRG for MPI paths and specialized noise/update code, not by normal
TensorToolkit users.

Those APIs live in:

```cpp
#include "qlten/tensor_manipulation/dmrg/contract_1sector.h"
#include "qlten/tensor_manipulation/dmrg/block_expand.h"
```

and in namespace:

```cpp
namespace qlten::dmrg { ... }
```

They are included by `qlten/qlten.h` and `tensor_manipulation_all.h` for
downstream compatibility. They remain namespaced under `qlten::dmrg` so the
general tensor manipulation namespace does not advertise these DMRG
optimization hooks as ordinary APIs.

## Known debt: `ExpandQNBlocks` const-correctness

`ExpandQNBlocks` currently advertises const input tensors but internally uses
`const_cast` and temporary mutation during the algorithm before restoring state.
This is a pre-existing defect and should not be silently cleaned up as part of a
header or namespace refactor.

There are two plausible real fixes, both non-trivial:

- change the public signature to drop `const`, which is an explicit downstream
  API break;
- remove the temporary mutation internally, which needs a performance-sensitive
  algorithmic review.

The previous abandoned direction of making defensive deep copies is not an
acceptable casual fix because it risks a large performance regression in DMRG
paths. Treat this as a separate refactor with dedicated performance review and
downstream validation.

## Testing expectations

Changes to the contiguous-axes API should keep these checks in place:

- legacy bool wrapper tests for all four bool combinations;
- `ContractContiguousAxes` tests for the equivalent four `CtrctSide`
  combinations;
- fermionic tensor coverage for the same mapping;
- mixed-precision tests for each supported operand order.

Changes to the DMRG-only APIs should build and test UltraDMRG in addition to
TensorToolkit, because UltraDMRG is the real downstream consumer of the
`qlten::dmrg` surface.
