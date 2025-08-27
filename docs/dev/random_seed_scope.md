# 随机种子：作用域与管理

简述：TensorToolkit 提供统一的随机数系统，保证结果可复现、实现可移植。

## 设计要点

- 默认确定性：默认种子 0，未显式更改时跨运行结果一致
- 现代实现：基于 `std::mt19937_64`，统计性质更佳、周期更长
- 全局管理：单一全局种子影响所有随机操作；统一接口 `SetRandomSeed`/`GetRandomSeed`/`GlobalRng`
- 头文件安全：使用 C++17 `inline` 变量，避免 ODR 问题，在各编译单元共享状态

## 实现与 API（精简）
```cpp
// include/qlten/utility/random.h
namespace tensor_random {
  inline std::mt19937_64 g_rng{0};
  inline unsigned long long g_seed = 0ULL;
}

namespace qlten {
  void SetRandomSeed(unsigned long long seed);
  unsigned long long GetRandomSeed();
  std::mt19937_64& GlobalRng();

  inline double Uniform01() {
    static std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(tensor_random::g_rng);
  }
  inline uint32_t RandUint32() {
    static std::uniform_int_distribution<uint32_t> u32;
    return u32(tensor_random::g_rng);
  }
}
```

## 用法
```cpp
#include "qlten/utility/random.h"

qlten::SetRandomSeed(12345);
double x = qlten::Uniform01();
uint32_t i = qlten::RandUint32();

// 高级：自定义分布
auto& rng = qlten::GlobalRng();
std::normal_distribution<double> normal(0.0, 1.0);
double g = normal(rng);
```


## 测试建议

- 在测试 `SetUp` 中显式设置：`qlten::SetRandomSeed(0)`
- 数值比较使用合理容差，避免对 RNG 变化过敏（建议 ≈ `1e-13`）

## 并发与 GPU

- 当前假设单线程；未来可考虑 `thread_local` 生成器或互斥保护
- GPU 通过将相同种子传递给 cuRAND，与 CPU 保持一致

## 最佳实践

- 基准与测试中明确设置种子
- 不混用其他随机源，统一使用本系统
- 将种子视为算法参数并记录
