/**
* @file random.h
* @brief Modern C++ random number generation utilities for TensorToolkit
*
* This header provides a unified interface for random number generation
* across CPU and GPU operations. It replaces legacy C-style rand() with
* modern std::mt19937_64 for better quality and reproducibility.
*/

#ifndef QLTEN_UTILITY_RANDOM_H
#define QLTEN_UTILITY_RANDOM_H

#include <random>
#include <cstdint>

namespace qlten {

// Internal random state - safely shared across compilation units
namespace tensor_random {
  inline std::mt19937_64 g_rng{0}; // Default seed 0 for determinism
  inline unsigned long long g_seed = 0ULL;
}

// Set global seed. Affects CPU random helpers and GPU curand.
inline void SetRandomSeed(unsigned long long seed) {
  tensor_random::g_seed = seed;
  tensor_random::g_rng.seed(seed);
}

// Get current global seed
inline unsigned long long GetRandomSeed() { 
  return tensor_random::g_seed; 
}

// Access global RNG (for advanced usage)
inline std::mt19937_64& GlobalRng() { 
  return tensor_random::g_rng; 
}

// Uniform [0,1)
inline double Uniform01() {
  static std::uniform_real_distribution<double> dist(0.0, 1.0);
  return dist(tensor_random::g_rng);
}

// Random 32-bit unsigned integer
inline uint32_t RandUint32() {
  return static_cast<uint32_t>(tensor_random::g_rng());
}

} // namespace qlten

#endif // QLTEN_UTILITY_RANDOM_H
