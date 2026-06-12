# QLTensor Core

Location: `include/qlten/qltensor/`

## What lives here

- `Index` and `QNSector` definitions
- `QLTensor` core data structure
- Block-sparse storage and block metadata
- Built-in quantum number types under `special_qn/`

## Key headers

- `qlten/qltensor/index.h`
- `qlten/qltensor/qnsct.h`
- `qlten/qltensor/qltensor.h`
- `qlten/qltensor/special_qn/special_qn_all.h`

## Responsibilities

- Define symmetry-aware tensor types
- Track divergence and block structure
- Provide tensor properties and basic operations (`Random`, `Fill`, `Dag`, norms)

## Downstream API guidance

Using `QLTensor::GetBlkIdxDataBlkMap()` + `GetActualRawDataSize()`, or raw-data pointers
through that object is acceptable for diagnostics, tests, and short-lived
downstream experiments, but it is not the standard interface for long-lived
downstream code.
 
If a downstream algorithm needs stable block traversal,
backend-aware raw-data processing, packing metadata, or another storage-related
operation that is not exposed today, add a canonical TensorToolkit interface
instead of depending directly on the internal block map and raw buffer. This
keeps downstream code from accidentally assuming CPU-only memory or a
backend-specific layout.

## Related tests

- `tests/test_qltensor/`
- `tests/test_qltensor/test_special_qn/`
