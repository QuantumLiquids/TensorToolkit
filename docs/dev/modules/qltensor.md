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

## Related tests

- `tests/test_qltensor/`
- `tests/test_qltensor/test_special_qn/`
