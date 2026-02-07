# Documentation Migration Map

This table is a historical record of how legacy documentation (for example
`docs/tutorials/*`) was mapped into the current `docs/user/en/` and `docs/dev/`
structure.

Notes:

- The **Source** paths are legacy file locations and may no longer exist in the
  repository after the migration.
- Maintain this file only if you plan to re-run/extend the migration. Otherwise
  it can be treated as read-only (or removed once you no longer need the
  traceability).

## Source-to-Target Map

| Source (legacy) | Target | Rewrite intent | Gaps / notes |
| --- | --- | --- | --- |
| `README.md` | `docs/user/en/README.md` | Short, user-facing overview + navigation | Keep root README short and point to user/dev docs. |
| `docs/mainpage.md` | `docs/mainpage.md` (rewrite) | Doxygen landing page pointing to new user/dev docs | Must link to new indexes with `.md` paths. |
| `docs/tutorials/index.md` | `docs/user/en/tutorials/index.md` | New learning path index | Replace numeric list with task-ordered tutorials. |
| `docs/tutorials/01_installation.md` | `docs/user/en/howto/build-and-install.md` | Task-focused install guide | Separate \"build tests\" into its own how-to. |
| `docs/tutorials/02_quick_start.md` | `docs/user/en/tutorials/quick-start-trg.md` | Hands-on first run | Use examples in `examples/` folder. |
| `docs/tutorials/03_indices_and_quantum_numbers.md` | `docs/user/en/explanation/indices-and-quantum-numbers.md` | Concepts + mental model | Keep for explanation, add glossary references. |
| `docs/tutorials/04_tensor_creation.md` | `docs/user/en/tutorials/tensor-basics-walkthrough.md` | Tutorialized creation + inspection | Use `qlten/qlten.h` import and basic index patterns. |
| `docs/tutorials/05_tensor_unary_operations.md` | `docs/user/en/tutorials/contractions-and-decompositions.md` | Combine with contractions/decomp | Split into tutorial + explanation sections. |
| `docs/tutorials/06_ising_vertex_tensor.md` | `docs/user/en/tutorials/quick-start-trg.md` | Part of TRG demo | Use as a step in the quick start. |
| `docs/tutorials/07_tensor_contractions.md` | `docs/user/en/tutorials/contractions-and-decompositions.md` | Practical contraction workflow | Keep snippets focused and minimal. |
| `docs/tutorials/08_matrix_decomposition.md` | `docs/user/en/explanation/matrix-decomposition-concepts.md` | Conceptual explanation | Provide link to Doxygen for API details. |
| `docs/tutorials/09_ising_trg_example.md` | `docs/user/en/tutorials/quick-start-trg.md` | Reference material | Move heavy math into explanation or appendix if needed. |
| `docs/tutorials/10_mpi_basics.md` | `docs/user/en/tutorials/mpi-parallel-basics.md` | Hands-on MPI tutorial | Pair with how-to for distributed SVD. |
| `docs/tutorials/11_mpi_advanced.md` | `docs/user/en/howto/distributed-svd.md` | Task guide | Focus on usage and constraints. |
| `docs/tutorials/12_gpu_setup.md` | `docs/user/en/howto/enable-gpu.md` | GPU setup steps | Mention cuBLAS/cuSOLVER/cuTENSOR. |
| `docs/tutorials/13_z2_graded_tensor_network.md` | `docs/user/en/explanation/z2-graded-fermions.md` | Conceptual explanation | Rewrite for clarity + link to fermion design. |
| `docs/tutorials_cn/*` | English docs only | Rewritten in English | Chinese docs deferred to later phase. |
| `docs/dev/index.md` | `docs/dev/index.md` (rewrite) | Developer documentation hub | Two axes: doc type + modules. |
| `docs/dev/fermion_design.md` | `docs/dev/design/fermion-design.md` | Design doc | Align with user explanation. |
| `docs/dev/random_seed_scope.md` | `docs/dev/guides/randomness-and-seeding.md` | Developer guide | Clarify scope, include API names. |
| `docs/dev/aocl_backend_changelog.md` | `docs/dev/design/backend-architecture.md` | Backend design notes | Merge with code inspection notes. |
| `docs/dev/test_coverage_report.md` | `docs/dev/testing/coverage-report.md` | Testing report | Reframe as coverage overview. |
| `docs/DISTRIBUTION.md` | `docs/dev/guides/docs-distribution.md` | Doc distribution guide | Rewrite for clarity and brevity. |
| `CMakeLists.txt` | `docs/user/en/reference/cmake-options.md` | CMake option reference | Include defaults + what they do. |
| `examples/*` | `docs/user/en/tutorials/*` | Tutorials reference | Keep examples as canonical runnable code. |
| `include/qlten/*` | `docs/user/en/reference/api-reference.md` | API reference index | Point to Doxygen output. |

## Status

All sections listed above have been created under `docs/user/en/` and `docs/dev/`.
Remaining work is quality checks (Doxygen build, link sweep) and any optional
legacy redirects.
