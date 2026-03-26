# Use an External HPTT

By default TensorToolkit builds the bundled HPTT in `external/hptt`.
If you want to link against a system-installed HPTT, disable the bundled build
and point CMake to the external headers and library.

## Step 1: Disable bundled HPTT

```bash
cmake .. -DQLTEN_COMPILE_HPTT_LIB=OFF
```

## Step 2: Point CMake to HPTT

`Findhptt.cmake` looks for `hptt.h` and the `hptt` library. You can supply:

- `-Dhptt_INCLUDE_DIR=/path/to/hptt/include`
- `-Dhptt_LIBRARY=/path/to/hptt/lib/libhptt.a` (or `.so`/`.dylib`)

Alternatively, set `CMAKE_PREFIX_PATH` to a prefix that contains `include/` and `lib/`.

```bash
cmake .. \
  -DQLTEN_COMPILE_HPTT_LIB=OFF \
  -Dhptt_INCLUDE_DIR=/opt/hptt/include \
  -Dhptt_LIBRARY=/opt/hptt/lib/libhptt.a
```

## Step 3: Build normally

```bash
make -j4
```

## Step 4: Consume the installed TensorToolkit package

If you install TensorToolkit with `QLTEN_COMPILE_HPTT_LIB=OFF`, the installed
package does not vendor HPTT into the TensorToolkit prefix. Downstream
consumers must also make the external HPTT installation discoverable, for
example by extending `CMAKE_PREFIX_PATH` with both prefixes:

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="/path/to/tensortoolkit-cpu;/opt/hptt"
```

You can also continue to use `hptt_INCLUDE_DIR` and `hptt_LIBRARY` directly in
the downstream configure step.

## Troubleshooting

- If CMake cannot find HPTT, verify that `hptt.h` and `libhptt.*` exist in the paths you provide.
- If you are building shared libraries, ensure your runtime loader can find `libhptt`.
