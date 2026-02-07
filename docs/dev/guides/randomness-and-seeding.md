# Randomness and Seeding

TensorToolkit provides a small, global random number facility used by
`QLTensor::Random` and related helpers.

## Design goals

- Deterministic by default (seed = 0)
- Simple, global control
- Reproducible across runs for a fixed build/toolchain

## API surface

```cpp
#include "qlten/utility/random.h"

qlten::SetRandomSeed(1234);
unsigned long long seed = qlten::GetRandomSeed();

auto &rng = qlten::GlobalRng();
```

The implementation uses `std::mt19937_64` and stores the seed and RNG
as `inline` globals so that all translation units share the same state.

## Reproducibility guarantees

- **CPU**: `std::mt19937_64` is standardized, but helper distributions like
  `std::uniform_real_distribution` are not guaranteed to be bitwise-identical
  across standard library implementations (libstdc++ vs libc++).
- **GPU**: GPU helpers use cuRAND; CPU and GPU random streams are not expected
  to match bitwise, even with the same seed.

## Best practices

- Set an explicit seed in unit tests.
- Treat the seed as part of the experiment configuration.

## Notes on GPU

Use the same seed when comparing CPU/GPU runs to reduce variance, but do not
expect identical random streams or bitwise-equal results.
