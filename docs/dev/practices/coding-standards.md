# Coding Standards

TensorToolkit follows the Google C++ Style Guide with a few project-specific
conventions.

## General style

- 2-space indentation
- Clear, descriptive naming
- Avoid heavy macros in public headers
- Prefer `qlten::` namespace and `special_qn` for built-in QNs

## Header-only expectations

Most changes should be header-only. If you introduce new build-system changes,
keep them minimal and documented.

## Doxygen

- Add Doxygen comments to public APIs
- Document non-obvious behavior and constraints
- Keep examples short and compile-oriented

## CMake conventions

- Build out-of-source in `build/`
- If CMake fails due to OpenMP, rerun the same command once

## Tests

- Add tests under the closest matching directory in `tests/`
- Name test files `test_<feature>.cc`
- Prefer small, deterministic test cases with explicit seeds
