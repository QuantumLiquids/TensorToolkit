# Backend Architecture

TensorToolkit routes BLAS/LAPACK calls through a centralized backend selector.
This keeps high-performance numerical dependencies consistent across the codebase.

## Backend selector

The header `include/qlten/framework/hp_numeric/backend_selector.h` selects one
backend at compile time via preprocessor definitions:

- `HP_NUMERIC_BACKEND_MKL`
- `HP_NUMERIC_BACKEND_AOCL`
- `HP_NUMERIC_BACKEND_OPENBLAS`
- `HP_NUMERIC_BACKEND_KML` (Huawei Kunpeng Math Library)

The selector includes the appropriate headers and exposes a vendor string via
`hp_numeric_backend::Vendor()`.

## AOCL-specific notes

The AOCL CBLAS interface does not expose certain transpose helpers, so TensorToolkit
invokes BLIS `*omatcopy_` symbols directly. This is why `backend_selector.h` declares
`somatcopy_`, `domatcopy_`, `comatcopy_`, and `zomatcopy_` when `HP_NUMERIC_BACKEND_AOCL`
is set.

## CMake integration

Backend selection is implemented in `cmake/Modules/MathBackend.cmake`, which
defines one of `HP_NUMERIC_BACKEND_*` and configures BLAS include paths. Use
exactly one of:

- `-DHP_NUMERIC_USE_MKL=ON`
- `-DHP_NUMERIC_USE_AOCL=ON`
- `-DHP_NUMERIC_USE_OPENBLAS=ON`

Note:

- `tests/CMakeLists.txt` and `examples/CMakeLists.txt` include `MathBackend.cmake`.
- The top-level `CMakeLists.txt` does **not** include it by default.
- If you integrate TensorToolkit into another project, include
  `cmake/Modules/MathBackend.cmake` yourself or define the `HP_NUMERIC_BACKEND_*`
  compile macro manually. (`HP_NUMERIC_BACKEND_KML` is supported in headers but
  not wired via a CMake option here.)

## Runtime considerations

Ensure that the chosen BLAS/LAPACK shared libraries are visible at runtime.
For AOCL, this typically means:

```
export LD_LIBRARY_PATH=$AOCL_ROOT/lib:$LD_LIBRARY_PATH
```

On macOS, use `DYLD_LIBRARY_PATH` instead of `LD_LIBRARY_PATH`:

```
export DYLD_LIBRARY_PATH=$AOCL_ROOT/lib:$DYLD_LIBRARY_PATH
```

## Related files

- `include/qlten/framework/hp_numeric/`
- `cmake/Modules/MathBackend.cmake`
- `docs/dev/guides/dev-setup.md`
