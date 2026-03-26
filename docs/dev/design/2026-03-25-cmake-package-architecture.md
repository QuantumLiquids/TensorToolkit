# CMake Package Architecture For The QuantumLiquids Stack

Date: 2026-03-25

## Summary

This document defines the target CMake packaging architecture for the current
QuantumLiquids repository stack:

- `TensorToolkit`
- `UltraDMRG`
- `PEPS`
- `UltraDMRGModelZoo`
- `HeisenbergVMCPEPS`

The goal is to replace copied `cmake/` helper modules and manual
`find_path(...)` integration with installed CMake package configs and imported
targets.

The migration rollout and downstream sequencing are tracked separately in
`docs/dev/guides/2026-03-25-downstream-cmake-package-migration.md`.

## Problem statement

The current downstream integration model has several problems:

- shared CMake helper files are copied between repositories and have already
  drifted
- downstream repos use `find_path(...)` for upstream headers such as `qlten`,
  `qlmps`, and `qlpeps`
- include paths, compile definitions, and link requirements are rebuilt by hand
  in each consumer
- fixes in upstream CMake logic do not propagate cleanly to downstream repos

This becomes worse as the stack gets deeper. `HeisenbergVMCPEPS` now repeats the
same pattern that already existed in `PEPS` and `UltraDMRGModelZoo`.

## Goals

- Make `TensorToolkit`, `UltraDMRG`, and `PEPS` installable CMake packages.
- Use `find_package(... CONFIG REQUIRED)` as the supported integration path.
- Export one primary imported target per package.
- Keep package ownership boundaries clear so shared logic is not copied across
  repos.
- Support separate CPU and GPU installed stacks without ambiguous package
  resolution.
- Validate each layer against an installed-prefix consumer immediately, and add
  real downstream-repo validation as the next layer in the stack migrates.

## Non-goals

- Creating a new standalone shared `QuantumLiquidsCMake` repository in the first
  migration phase.
- Supporting CPU and GPU variants from a single install prefix in the first
  phase.
- Keeping copied-module integration and package-first integration as permanent
  dual modes.
- Introducing compiled `TensorToolkit`, `UltraDMRG`, or `PEPS` libraries solely
  for packaging. These packages remain header-only in the first phase.

## Package stack and ownership

The intended dependency graph is:

- `UltraDMRG -> TensorToolkit`
- `PEPS -> UltraDMRG -> TensorToolkit`
- `UltraDMRGModelZoo -> UltraDMRG -> TensorToolkit`
- `HeisenbergVMCPEPS -> PEPS -> UltraDMRG -> TensorToolkit`

Ownership rules:

- `TensorToolkit` owns low-level helper logic that belongs to the tensor layer,
  including the current HPTT, cuTENSOR, and math-backend integration surface.
- `UltraDMRG` depends on installed `TensorToolkit` and should not carry copied
  low-level helpers from it.
- `PEPS` depends on installed `UltraDMRG` and `TensorToolkit`, and owns only
  PEPS-specific public integration such as ScaLAPACK handling.
- Application-style repos such as `UltraDMRGModelZoo` and `HeisenbergVMCPEPS`
  consume the highest-level installed package they actually use.

## Public package interface

The upstream packages should export the following names:

- `TensorToolkitConfig.cmake` with `TensorToolkit::TensorToolkit`
- `UltraDMRGConfig.cmake` with `UltraDMRG::UltraDMRG`
- `PEPSConfig.cmake` with `PEPS::PEPS`

These are the stable `find_package(...)` identifiers:

- `TensorToolkit`
- `UltraDMRG`
- `PEPS`

The package names remain the same across CPU and GPU builds. Variant selection
comes from the chosen install prefix, not from renaming the packages.

## Header-only target model

In the first phase, all three upstream packages export one primary `INTERFACE`
target:

- `TensorToolkit::TensorToolkit`
- `UltraDMRG::UltraDMRG`
- `PEPS::PEPS`

The packages are still header-only. No new compiled `.a` or `.so` for these
repositories is required. The package metadata still needs a standard install
location for:

- `*Config.cmake`
- `*ConfigVersion.cmake`
- `*Targets.cmake`

These files should be installed under `${CMAKE_INSTALL_LIBDIR}/cmake/<Pkg>`.

This does not mean the installed stack has no compiled artifacts at all. The
current CPU transpose path can depend on HPTT, which is a compiled library. The
package architecture therefore distinguishes:

- header-only exported targets for `TensorToolkit`, `UltraDMRG`, and `PEPS`
- compiled third-party artifacts that may still need to be installed alongside
  those packages for specific variants

For the first packaging phase, CPU TensorToolkit variants that expose HPTT-backed
inline transpose through public headers should install the required HPTT artifact
into the same prefix and propagate it transitively through
`TensorToolkit::TensorToolkit`. GPU variants do not use HPTT in that path.

## Installed layout

Use a standard install tree for each package prefix:

- `${CMAKE_INSTALL_INCLUDEDIR}/qlten/...`
- `${CMAKE_INSTALL_INCLUDEDIR}/qlmps/...`
- `${CMAKE_INSTALL_INCLUDEDIR}/qlpeps/...`
- `${CMAKE_INSTALL_LIBDIR}/cmake/TensorToolkit/...`
- `${CMAKE_INSTALL_LIBDIR}/cmake/UltraDMRG/...`
- `${CMAKE_INSTALL_LIBDIR}/cmake/PEPS/...`

If package-owned helper modules must be installed, keep them package-local, for
example:

- `${CMAKE_INSTALL_LIBDIR}/cmake/TensorToolkit/modules/...`
- `${CMAKE_INSTALL_LIBDIR}/cmake/PEPS/modules/...`

Downstream repos should not append those helper paths to `CMAKE_MODULE_PATH`
manually. The owning package config should do that internally if needed.

## What the imported targets carry

### `TensorToolkit::TensorToolkit`

`TensorToolkit::TensorToolkit` should carry:

- the public include root for `qlten`
- public compile definitions required by the installed package variant
- public transitive dependencies that consumers need because the public headers
  require them

Today this is broader than MPI alone. `qlten/qlten.h` reaches public umbrella
headers that pull in:

- `mpi.h` via `qlten/framework/hp_numeric/mpi_fun.h`
- BLAS/LAPACK backend headers via
  `qlten/framework/hp_numeric/backend_selector.h`
- `omp.h` and OpenMP thread-control APIs via
  `qlten/framework/hp_numeric/omp_set.h`
- HPTT headers and symbols on CPU builds, or CUDA/cuTENSOR-facing headers and
  symbols on GPU builds, via `qlten/framework/hp_numeric/ten_trans.h`

That means the installed `TensorToolkit::TensorToolkit` contract must propagate
the selected variant's required public compile definitions and transitive
dependencies. In practice this includes:

- the selected `HP_NUMERIC_BACKEND_*` compile definition, because public headers
  conditionally include backend-facing APIs based on that choice
- `MPI::MPI_CXX`
- the chosen BLAS/LAPACK interface for the installed variant
- `OpenMP::OpenMP_CXX` when required by the public headers/inline code path
- CPU transpose support such as HPTT for CPU variants
- CUDA-related dependencies such as CUDA, cuBLAS, cuSOLVER, and cuTENSOR for
  GPU variants

`TensorToolkit::TensorToolkit` should not carry:

- build-only cache variables
- machine-local probe results that are not part of the package interface
- build-local warning/debug flags such as `-Wall` or `-g`
- duplicated downstream integration instructions that should instead live on the
  imported target

### `UltraDMRG::UltraDMRG`

`UltraDMRG::UltraDMRG` should carry:

- the public include root for `qlmps`
- UltraDMRG-specific public compile definitions if needed
- a transitive dependency on `TensorToolkit::TensorToolkit`

It should not restate TensorToolkit internals if the TensorToolkit imported
target already models them.

### `PEPS::PEPS`

`PEPS::PEPS` should carry:

- the public include root for `qlpeps`
- PEPS-specific public compile definitions or optional public capability flags
  if needed
- a transitive dependency on `UltraDMRG::UltraDMRG`

If a PEPS install is built with ScaLAPACK-enabled MinSR support, that is also a
public exported-target requirement rather than a private helper detail:

- `PEPS::PEPS` must propagate the `QLPEPS_HAS_SCALAPACK` compile definition for
  that variant
- `PEPSConfig.cmake` must locate the ScaLAPACK dependency for that variant
- `PEPS::PEPS` must link `ScaLAPACK::ScaLAPACK` transitively for that variant

This is required because `qlpeps/optimizer/minsr_eigensolve.h` includes
`minsr_scalapack.h` under `QLPEPS_HAS_SCALAPACK`, and the inline Path A code
uses BLACS/ScaLAPACK symbols directly

Application repos should only need:

```cmake
find_package(PEPS CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE PEPS::PEPS)
```

## Dependency modeling rules

Each package config should use `find_dependency(...)` only for true public,
transitive requirements.

Rules:

- if a public header requires a dependency, the package config must require it
- if a dependency is only used to build the package internally, it should not be
  exported as public package API
- backend selection logic should happen when building the package, not again in
  every downstream consumer

This implies a required cleanup in the TensorToolkit build itself. Package
export is not only a `*Config.cmake` task. The top-level TensorToolkit build
must model and discover its public dependencies before export generation, rather
than leaving them only in `tests/` or downstream consumers. In practice that
means the packaging work must add proper top-level dependency handling for the
installed variant's public requirements such as:

- `MPI`
- `BLAS`
- `LAPACK`
- `OpenMP`
- CPU HPTT or GPU CUDA/cuBLAS/cuSOLVER/cuTENSOR dependencies as required

Examples:

- `TensorToolkitConfig.cmake` should locate all public transitive dependencies
  required by the installed variant, not only MPI. For the current umbrella
  header surface this includes MPI plus the selected backend-facing requirements
  such as BLAS/LAPACK, OpenMP, and CPU HPTT or GPU CUDA/cuTENSOR-related
  dependencies.
- `UltraDMRGConfig.cmake` should use
  `find_dependency(TensorToolkit CONFIG REQUIRED)`
- `PEPSConfig.cmake` should use
  `find_dependency(UltraDMRG CONFIG REQUIRED)`

For PEPS variants built with distributed MinSR enabled, `PEPSConfig.cmake`
should also locate the ScaLAPACK dependency and `PEPS::PEPS` should propagate
`ScaLAPACK::ScaLAPACK` transitively.

Package configs should use `include(CMakeFindDependencyMacro)` and stay small,
predictable, and package-owned.

The exported targets should be generated from a build that already knows its own
public dependency graph; the config file should mirror that graph for downstream
consumers, not invent it from scratch after install.

## Backend and feature modeling

The package config describes an already-built package. It should not rerun the
full backend auto-detection logic used when that package was compiled.

Recommended behavior:

- backend selection happens at package build/install time
- downstream packages consume the resulting installed variant
- optional public capability variables may be exported when consumers genuinely
  need them, for example GPU-enabled or ScaLAPACK-enabled variants

If an optional capability changes the public header/API surface or the required
link interface, it is no longer private for that installed variant. That is why
GPU-enabled TensorToolkit variants and ScaLAPACK-enabled PEPS variants must
export those requirements transitively through their imported targets.

Imported targets remain the primary supported interface. Capability variables
are secondary and should be kept minimal.

Backend compile definitions are part of this public target surface when public
headers depend on them. They should move from global `add_definitions(...)`
usage toward target-scoped usage requirements on the exported imported target.

## Prefix strategy and environment activation

Separate build variants use separate install prefixes. In the first phase, that
especially means separate CPU and GPU stacks.

Recommended layout:

- `$HOME/opt/qlstack/cpu`
- `$HOME/opt/qlstack/gpu`

The split is by package variant, not by physical node type. Login nodes, CPU
nodes, and GPU nodes may all see the same `$HOME`, but a given configure should
resolve exactly one stack.

Rules:

- install all packages of one variant into the same prefix
- point `CMAKE_PREFIX_PATH` to only one variant prefix during configure
- do not mix CPU and GPU prefixes in the same downstream configure

The official package model assumes each prefix is self-contained, even if that
duplicates installed headers. If desired, header deduplication can be done later
as a deployment optimization using filesystem links, but it is not part of the
package interface contract.

## Package config behavior

Each `*Config.cmake` file should:

- load exported targets
- resolve true public dependencies with `find_dependency(...)`
- fail clearly if required dependencies or the selected prefix are invalid

Each `*Config.cmake` file should not:

- silently fall back to `find_path(...)` probing for upstream headers
- guess which CPU/GPU stack the user intended
- expose package-internal helper logic as downstream public API

Failure messages should be actionable and should tell the user to set
`CMAKE_PREFIX_PATH` or `<Pkg>_DIR`, and when relevant indicate that the wrong
stack variant or an incomplete prefix was selected.

Relocatability is a design requirement. Package config generation should use
`configure_package_config_file()` from `CMakePackageConfigHelpers` together with
`@PACKAGE_INIT@`, so installed package configs resolve paths relative to the
package prefix rather than embedding unnecessary absolute source/build paths.

## Temporary compatibility policy

During migration, packages may expose temporary compatibility variables to help
downstream repos switch in smaller steps. Examples:

- `QLTEN_HEADER_PATH`
- `QLMPS_HEADER_PATH`
- `QLPEPS_HEADER_PATH`

These are migration shims only. The intended steady-state usage is imported
targets, not raw path variables.

Rules:

- new downstream code should prefer imported targets immediately
- compatibility variables may exist temporarily to reduce patch size
- copied CMake helper modules should be frozen once equivalent package configs
  exist
- copied modules should be removed after active downstream repos have migrated

## Migration strategy

The rollout sequence is:

1. `TensorToolkit`
2. `UltraDMRG`
3. `PEPS`
4. `UltraDMRGModelZoo`
5. `HeisenbergVMCPEPS`

Guardrails:

- each layer should declare a minimum supported upstream package version
- the transition period should be temporary, not permanent dual support
- each phase should end with one working install prefix that a clean downstream
  configure can consume

Detailed downstream sequencing and repo-by-repo migration notes are tracked in
the companion migration guide.

## Versioning

The package migration uses semantic versioning at the package surface:

- `TensorToolkit` starts its packaged version surface at `0.1.0`
- `UltraDMRG` currently uses `0.2.0`
- `PEPS` currently uses `0.1.0`

Each packaged upstream layer should expose its version in two places:

- `project(<Pkg> VERSION x.y.z)` in CMake
- a public `version.h` surface usable by downstream compile-time checks

Minimum supported upstream versions in downstream package configs should refer to
those public package versions rather than to source-tree heuristics.

## Validation and compatibility

Each package layer should validate not only its own build, but also consumption
from an installed prefix.

Required validation types for the `TensorToolkit` package-export phase:

- package builds and installs successfully
- a minimal external consumer configures against the installed prefix

Additional validation types for later migration phases:

- once `UltraDMRG` switches to `find_package(TensorToolkit CONFIG REQUIRED)`, it
  becomes the required real-downstream validation target for TensorToolkit
- once `PEPS` switches to `find_package(UltraDMRG CONFIG REQUIRED)`, it becomes
  the required real-downstream validation target for UltraDMRG
- once application-style repos switch to package-first integration, they become
  the required real-downstream validation targets for PEPS

Real downstream repos are part of the validation strategy after the relevant
layer has migrated:

- `UltraDMRGModelZoo` validates the `UltraDMRG` package surface
- `HeisenbergVMCPEPS` validates the `PEPS` package surface

Clear configure-time failures are preferred over hidden compatibility tricks.
Once the migration is complete, downstream repos should not be able to fall back
to copied modules or `find_path(...)` without explicitly opting into unsupported
legacy behavior.

## Risks and failure modes

The main risks are:

- misclassifying a public dependency as private, causing confusing downstream
  build failures
- exporting too much build-local state on header-only imported targets
- making package configs non-relocatable through careless absolute paths
- letting CPU and GPU variants collide in one prefix
- stopping at toy-consumer validation after downstream repos have already
  migrated

The package design should bias toward explicitness and isolation, even if that
means some duplication in installed headers or stricter configure-time checks.

## End state

At the end of the migration:

- `TensorToolkit`, `UltraDMRG`, and `PEPS` are installable header-only CMake
  packages
- downstream repos consume installed package configs instead of sibling source
  trees
- copied shared helper modules are removed from downstream repos
- `find_path(...)` integration for `qlten`, `qlmps`, and `qlpeps` is retired
- imported targets are the supported integration path for active development
- CPU and GPU stacks are selected by prefix
- package exports are validated first against minimal external consumers, then
  against real downstream consumers as each layer migrates

## Related docs

- [Architecture overview](architecture-overview.md)
- [Backend architecture](backend-architecture.md)
- [Downstream CMake package migration](../guides/2026-03-25-downstream-cmake-package-migration.md)
