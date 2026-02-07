# Coverage Report (Snapshot)

This page summarizes a coverage review captured on **2025-08-19** for
TensorToolkit version **0.1**. Treat it as a historical snapshot rather than
live coverage metrics.

## Test inventory (snapshot)

- Test files: 28
- Test cases: 154
- Test directories:
  - `tests/test_qltensor/`
  - `tests/test_tensor_manipulation/`
  - `tests/test_tensor_mpi/`
  - `tests/test_utility/`

## Areas with strong coverage

- Core tensor types (`QLTensor`, indices, QN sectors)
- Symmetry-aware tensor operations
- Contractions (including fermionic variants)
- Decompositions (SVD/QR/EVD)
- MPI SVD basics

## Gaps called out in the snapshot

- **GPU coverage**: no dedicated GPU tests
- **Utilities**: timers and memory helpers have limited tests
- **Framework bases**: hashable/showable/streamable classes are not directly tested
- **Performance regression**: no automated performance checks

## Suggested next steps

- Add GPU correctness tests (CPU vs GPU equivalence on small cases)
- Add unit tests for utility helpers
- Add a small performance baseline suite for contraction and SVD
