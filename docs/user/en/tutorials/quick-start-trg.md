# Quick Start: Z2 Ising TRG

This tutorial gets you from clone to a working Z2 Ising TRG run using the built-in examples.
You will compile two example programs and run the TRG example (multi-beta, multi-iteration by default).

## What you will build

- A Z2-symmetric Ising vertex tensor
- A TRG coarse-graining loop (contract -> reshape -> SVD -> truncate -> reassemble)
- Printed free-energy contributions per iteration

## Prerequisites

- C++17 compiler
- CMake 3.27+
- BLAS/LAPACK
- MPI (required by TensorToolkit)

## Step 1: Build the examples

From the repository root:

```bash
mkdir -p build
cd build
cmake .. \
  -DQLTEN_BUILD_EXAMPLES=ON \
  -DHP_NUMERIC_USE_OPENBLAS=ON
make -j4
```

This builds example binaries from `examples/`.
If you use MKL/AOCL instead of OpenBLAS, see backend setup notes in
`../howto/build-and-install.md`.

## Step 2: Run the examples

From the `build` directory, run the example executables (names may include a prefix depending on your generator):

```bash
./examples/z2_ising_vertex
./examples/z2_ising_trg
```

If your build system places binaries in a different folder, search under `build/` for `z2_ising_*`.

## Step 3: Read the source

Open these two files and follow the flow:

- `examples/z2_ising_tensor.cpp` builds the Z2 Ising vertex tensor at inverse temperature `beta`.
- `examples/z2_ising_trg.cpp` runs TRG across multiple beta values and iterates
  up to `params.max_iterations` (default: 30).

Focus on:

- how Z2 indices are defined
- how the tensor is assembled
- how contraction and SVD are combined for coarse graining

## Conceptual outline

The TRG step follows this skeleton:

1. Build the local vertex tensor `T`.
2. Contract to create a coarse-grained object.
3. Reshape into a matrix and perform SVD.
4. Truncate to a bond dimension `chi`.
5. Reassemble the renormalized tensor.

The example runs repeated coarse-graining steps. The single-step kernel is
`TRGCoarseGrainStep`, and the outer loop is controlled by:

- `beta_values` (defaults to 4 betas)
- `params.max_iterations` (defaults to 30)

If you want a single coarse-graining step, set `beta_values` to one entry and
set `params.max_iterations = 1` in `examples/z2_ising_trg.cpp`.

## Configuration knobs

The TRG example does not expose a runtime config interface. To change the
parameters, edit `examples/z2_ising_trg.cpp`:

- `beta_values` in `main` controls which inverse temperatures are run.
- `params.max_iterations` controls the number of coarse-graining steps.
- The only runtime argument is an optional CSV path for analytic free energy.

## Next steps

- Learn how tensors are created and indexed: [Tensor basics walkthrough](tensor-basics-walkthrough.md)
- Understand contractions and decompositions: [Contractions and decompositions](contractions-and-decompositions.md)
- If you want MPI or GPU, jump ahead to their tutorials
