# Downstream CMake Package Migration

Date: 2026-03-25

This guide records the planned migration from copied CMake helper modules and
manual header discovery to installed CMake package configs with imported
targets.

Status update on 2026-03-26:
- `TensorToolkit` package export is now implemented in this repository.
- The remaining migration work described here applies to downstream repos and to
  the later `UltraDMRG` / `PEPS` package-export steps.

It is written for maintainers of the downstream repository stack:

- `UltraDMRG`
- `PEPS`
- `UltraDMRGModelZoo`
- `HeisenbergVMCPEPS`
- later repos built on the same stack

## Current state snapshot

- `TensorToolkit`: package export is now implemented; top-level build now models
  the public dependency graph for the installed CPU and GPU variants; CPU builds
  can still build HPTT as a compiled bundled dependency.
- `UltraDMRG`: no installed package config yet; currently uses
  `find_path(QLTEN_HEADER_PATH ...)` and copies low-level helper modules such as
  `FindCUTENSOR.cmake` and `MathBackend.cmake`; HPTT handling in tests/benchmark
  is currently direct `find_path`/`find_library`, not a copied `Findhptt.cmake`.
- `PEPS`: no installed package config yet; currently uses
  `find_path(QLTENSOR_HEADER_PATH ...)` and `find_path(QLMPS_HEADER_PATH ...)`;
  owns `FindScaLAPACK.cmake`; also carries copied low-level helpers.
- `UltraDMRGModelZoo`: no installed package config yet; currently uses
  `find_path(TENSOR_HEADER_PATH ...)` and `find_path(MPS_HEADER_PATH ...)`; also
  carries copied `MathBackend.cmake` and `Findhptt.cmake`.
- `HeisenbergVMCPEPS`: no installed package config yet; currently uses
  `find_path(TENSOR_HEADER_PATH ...)`, `find_path(MPS_HEADER_PATH ...)`, and
  `find_path(PEPS_HEADER_PATH ...)`; also carries copied `MathBackend.cmake`,
  `Findhptt.cmake`, and `FindScaLAPACK.cmake`.

## Why this migration exists

The current downstream build pattern has several problems:

- shared files under `cmake/` are copied across repos and have already drifted
- downstream repos use `find_path(...)` for `qlten`, `qlmps`, and `qlpeps`
- include paths and link flags are repeated manually in each repo
- dependency fixes in upstream repos do not automatically propagate downstream

The target state is package-based:

- upstream repos install `*Config.cmake` packages
- downstream repos use `find_package(... CONFIG REQUIRED)`
- build requirements propagate through imported targets instead of copied files

## Scope and ownership

This migration follows two rules.

1. Public dependency model:
   downstream repos consume installed packages with `find_package(... CONFIG REQUIRED)`.
2. Internal ownership model:
   shared helper CMake logic lives only in the package that owns it and is not
   copied into consumers.

Recommended ownership:

- `TensorToolkit` owns shared low-level helper logic such as HPTT, cuTENSOR, and
  math-backend discovery used by the tensor toolkit layer.
- `UltraDMRG` depends on installed `TensorToolkit`.
- `PEPS` depends on installed `UltraDMRG` and `TensorToolkit`.
- `PEPS` owns PEPS-specific helper logic such as ScaLAPACK discovery.
- `UltraDMRGModelZoo` depends on installed `UltraDMRG` and `TensorToolkit`.
- `HeisenbergVMCPEPS` depends on installed `PEPS`.

We are not planning a separate shared `QuantumLiquidsCMake` repository at the
start of this migration. If package export leaves a small set of truly shared,
package-independent helpers later, we can revisit that decision after the main
stack has moved to package configs.

## Target dependency graph

The intended dependency chain is:

- `UltraDMRG -> TensorToolkit`
- `PEPS -> UltraDMRG -> TensorToolkit`
- `UltraDMRGModelZoo -> UltraDMRG -> TensorToolkit`
- `HeisenbergVMCPEPS -> PEPS -> UltraDMRG -> TensorToolkit`

Each arrow means:

- the upstream package must be installed
- the downstream package uses `find_package(<Pkg> CONFIG REQUIRED)`
- the downstream targets link imported targets instead of manually collecting
  include paths and libraries

## Package export targets

The upstream repos should export these package names and imported targets:

- `TensorToolkitConfig.cmake` with `TensorToolkit::TensorToolkit`
- `UltraDMRGConfig.cmake` with `UltraDMRG::UltraDMRG`
- `PEPSConfig.cmake` with `PEPS::PEPS`

The corresponding package configs should use `find_dependency(...)` for
transitive requirements when needed, for example:

- `TensorToolkit` must locate and propagate the public transitive dependencies
  required by the installed variant. For current umbrella-header consumers this
  includes MPI plus the selected backend-facing requirements such as
  BLAS/LAPACK, OpenMP, and CPU HPTT or GPU CUDA/cuBLAS/cuSOLVER/cuTENSOR
  dependencies.
- `UltraDMRG` should use `find_dependency(TensorToolkit CONFIG REQUIRED)`
- `PEPS` should use `find_dependency(UltraDMRG CONFIG REQUIRED)` and, when built
  with distributed MinSR enabled, must also locate and propagate the
  ScaLAPACK dependency for that variant through `PEPS::PEPS`

Target install tree sketch:

```text
<variant-prefix>/
  include/
    qlten/
    qlmps/
    qlpeps/
  <libdir>/
    cmake/
      TensorToolkit/
      UltraDMRG/
      PEPS/
    libhptt.a   # CPU variant when bundled with TensorToolkit
```

## Migration order

Downstream migration should follow this order:

1. `TensorToolkit`
2. `UltraDMRG`
3. `PEPS`
4. `UltraDMRGModelZoo`
5. `HeisenbergVMCPEPS`
6. later downstream application repos

This order matters because each repo should switch only after the package it
depends on is installable and stable.

## Repo-by-repo plan

### 1. TensorToolkit

Goals:

- install headers as before
- install package config files
- export `TensorToolkit::TensorToolkit`
- keep ownership of low-level helper CMake logic

Expected changes:

- add package export support with `CMakePackageConfigHelpers`
- install `TensorToolkitConfig.cmake`, version file, and targets file
- install package-owned helper modules if the config needs them
- for CPU variants that expose HPTT-backed public transpose, install the needed
  HPTT artifact into the same prefix and propagate it through
  `TensorToolkit::TensorToolkit`

Notes:

- because `TensorToolkit` is header-only, the imported target will likely be an
  `INTERFACE` library
- transitive compile definitions and link requirements should be attached to the
  interface target instead of being reconstructed by downstream repos

### 2. UltraDMRG

Goals:

- replace `find_path(QLTEN_HEADER_PATH ...)`
- consume `TensorToolkit::TensorToolkit`
- export `UltraDMRG::UltraDMRG`
- remove copied `FindCUTENSOR.cmake` and `MathBackend.cmake`

Expected changes:

- `find_package(TensorToolkit CONFIG REQUIRED)`
- link imported targets rather than manually injecting `QLTEN_HEADER_PATH`
- move any remaining UltraDMRG-specific requirements into `UltraDMRGConfig.cmake`

### 3. PEPS

Goals:

- replace the current PEPS-local `find_path(QLTENSOR_HEADER_PATH ...)` and
  `find_path(QLMPS_HEADER_PATH ...)` calls
- consume `UltraDMRG::UltraDMRG`
- export `PEPS::PEPS`
- keep only PEPS-owned helpers such as `FindScaLAPACK.cmake`

Expected changes:

- `find_package(UltraDMRG CONFIG REQUIRED)`
- remove copied `FindCUTENSOR.cmake` and `MathBackend.cmake`
- stop manually reconstructing upstream include paths
- export `QLPEPS_HAS_SCALAPACK` and `ScaLAPACK::ScaLAPACK` transitively from
  `PEPS::PEPS` when that installed variant is built with distributed MinSR

### 4. UltraDMRGModelZoo

Goals:

- replace `find_path(TENSOR_HEADER_PATH ...)` and `find_path(MPS_HEADER_PATH ...)`
- consume `UltraDMRG::UltraDMRG`
- remove copied `MathBackend.cmake` and `Findhptt.cmake`

Expected changes:

- `find_package(UltraDMRG CONFIG REQUIRED)`
- application targets link `UltraDMRG::UltraDMRG`

### 5. HeisenbergVMCPEPS

Goals:

- replace `find_path(TENSOR_HEADER_PATH ...)`, `find_path(MPS_HEADER_PATH ...)`,
  and `find_path(PEPS_HEADER_PATH ...)`
- consume `PEPS::PEPS`
- remove copied `MathBackend.cmake`, `Findhptt.cmake`, and any PEPS-shared
  helpers that should come from upstream

Expected changes:

- `find_package(PEPS CONFIG REQUIRED)`
- application targets link `PEPS::PEPS`

## Temporary compatibility policy

During the migration, upstream packages may export temporary compatibility
variables to reduce the size of each downstream patch. Examples:

- `QLTEN_HEADER_PATH`
- `QLMPS_HEADER_PATH`
- `QLPEPS_HEADER_PATH`

These variables are a transition aid only. They should not be treated as the
final public interface. The end state is target-based linking.

Rules for the compatibility window:

- new downstream code should prefer imported targets immediately
- compatibility variables may be used to unblock incremental migration
- copied CMake helper modules should be frozen once equivalent package configs
  are available
- copied modules should be removed after active downstream repos have migrated

## Downstream checklist

For each downstream repo:

1. Replace `find_path(...)` calls for upstream headers with `find_package(... CONFIG REQUIRED)`.
2. Replace manual include propagation with `target_link_libraries(... PRIVATE <Pkg>::<Pkg>)`.
3. Remove copied helper modules that are now owned by upstream packages.
4. Keep only repo-specific helper modules that do not belong upstream.
5. Verify configure, build, and test flows against an installed upstream stack.

## Validation automation sketch

The full stack-validation target, once all layers migrate, is one
install-and-consume step per layer:

1. build and install `TensorToolkit` to a temporary prefix
2. configure `UltraDMRG` against that prefix
3. build and install `UltraDMRG`
4. configure `PEPS` against that prefix
5. build and install `PEPS`
6. configure real downstream repos such as `UltraDMRGModelZoo` or
   `HeisenbergVMCPEPS` against the resulting prefix

Current status in the `TensorToolkit` phase:

- required now: install `TensorToolkit` and verify a minimal external consumer
- required once `UltraDMRG` migrates: configure and build `UltraDMRG` against
  the installed `TensorToolkit` prefix
- later layers follow the same rule upward through the stack

## Example consumer pattern

The intended downstream pattern is:

```cmake
find_package(PEPS CONFIG REQUIRED)

add_executable(vmc_optimize src/vmc_optimize.cpp)
target_link_libraries(vmc_optimize PRIVATE PEPS::PEPS)
```

Application repos should not need to know where `qlten`, `qlmps`, or `qlpeps`
headers are installed if the imported targets are modeled correctly.

For executables, `PRIVATE` linkage is the normal pattern. Intermediate library
layers that re-export upstream usage requirements should use `PUBLIC` where
appropriate.

## Rollout record

- 2026-03-25: migration direction agreed
- 2026-03-26: `TensorToolkit` package export implemented with CPU and GPU
  install-consume smoke verification
- after `TensorToolkit` is stable: migrate `UltraDMRG`
- after `UltraDMRG` is stable: migrate `PEPS`
- after `PEPS` is stable: migrate `UltraDMRGModelZoo` and `HeisenbergVMCPEPS`

This document should be updated as each layer finishes migration so downstream
maintainers can tell which repos are already package-based and which still rely
on compatibility shims.
