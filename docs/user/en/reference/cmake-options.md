# CMake Options

This page lists the top-level CMake options defined in `CMakeLists.txt`.

| Option | Default | Description |
| --- | --- | --- |
| `QLTEN_USE_GPU` | `OFF` | Enable GPU code paths for tests and downstream integration (examples are CPU-only). |
| `QLTEN_COMPILE_HPTT_LIB` | `ON` | Build the bundled HPTT library for CPU transpose. Disabled if GPU is ON. |
| `QLTEN_BUILD_UNITTEST` | `OFF` | Build unit tests (requires GoogleTest). |
| `QLTEN_BUILD_EXAMPLES` | `OFF` | Build example programs in `examples/`. |
| `QLTEN_TIMING_MODE` | `ON` | Enable timing utilities and timers. |
| `QLTEN_MPI_TIMING_MODE` | `ON` | Enable MPI timing utilities. |

## Math backend selection (MathBackend.cmake)

These options live in `cmake/Modules/MathBackend.cmake`. They are used by
tests/examples and by downstream projects that include the module.

| Option | Default | Description |
| --- | --- | --- |
| `HP_NUMERIC_USE_MKL` | `OFF` | Select Intel MKL and define `HP_NUMERIC_BACKEND_MKL`. |
| `HP_NUMERIC_USE_AOCL` | `OFF` | Select AMD AOCL and define `HP_NUMERIC_BACKEND_AOCL`. |
| `HP_NUMERIC_USE_OPENBLAS` | `OFF` | Select OpenBLAS and define `HP_NUMERIC_BACKEND_OPENBLAS`. |

## Notes

- If `QLTEN_USE_GPU=ON`, the build forces `QLTEN_COMPILE_HPTT_LIB=OFF`.
- Examples hard-error if `QLTEN_USE_GPU=ON`.
- The default build type is `Release` unless `CMAKE_BUILD_TYPE` is set.
