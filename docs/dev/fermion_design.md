## 费米子设计（Z2 Graded张量网络）

本页面面向开发者，系统说明项目中费米子的判定、宇称传播、符号记账与 API 约定。数学基础请参考教程《Z2-Graded Tensor Network：数学基础》（`docs/tutorials/13_z2_graded_tensor_network.md`）。

### 1. 术语与总体约定

- 统一用术语“fermion/费米子”。量子数携带 Z2 宇称的信息；这一Z2 对称性可以是一个阿贝尔群的子群。比如代表费米子数的U1 对称性的奇偶性的子群是Z2宇称群；也可以是Z2×U1的子群， U1是代表自旋Sz的份量。我们假设在我们的量子数体系中代表费米子Z2 宇称的子群只有一个。
- 索引方向：`IN` ≡ |ket>，`OUT` ≡ <bra|。内积 <bra|ket> 不额外出符号。
- 张量按量子数（含奇偶）块稀疏组织；允许块需满足总量子数守恒（IN 取负、OUT 取正）。

### 2. 何时一个 QNT 被视为费米型？（编译期判定）

- 判定接口：`Fermionicable<QNT>`（`qlten/framework/bases/fermionicable.h`）。
  - `static constexpr bool IsFermionic()`：通过 SFINAE 检测 `QNT` 是否提供成员函数 `IsFermionParityOdd()`；若存在，则视为费米型。
  - `static constexpr ParticleStatistics Statistics()`：返回 `Bosonic` 或 `Fermionic`。
- 典型费米型量子数（`qlten::special_qn`）：
  - `fZ2QN`：显式 Z2 宇称（0/1）。
  - `fU1QN`：以 U1 电荷模 2 的奇偶定义宇称（`IsFermionParityOdd() { return val % 2; }`）。
  - `fZ2U1QN`：成分为 (Z2, U1)，宇称由 Z2 分量给出。
- 非费米型量子数不提供 `IsFermionParityOdd()`，故走玻色路径。

实现要点：若自定义费米型 `QNT`，必须提供：
- `bool IsFermionParityOdd() const;`
- `bool IsFermionParityEven() const;`
- 建议提供 `static QNT Zero()` 与必要的加减/取负运算以参与守恒与块索引。

### 3. 哪些对象“继承费米性”？

- `Index<QNT>`、`QNSector<QNT>`、`QLTensor<ElemT, QNT>` 均继承 `Fermionicable<QNT>`，因此所有“是否费米型”的分支均在编译期通过 `if constexpr (Fermionicable<QNT>::IsFermionic())` 静态选择。
- `QNSector<QNT>` 在费米型时暴露 `IsFermionParityOdd/Even`；在玻色型时这些方法不参与实例化。

### 4. IN/OUT 与量子数聚合（散度）

- 方向枚举：`TenIndexDirType::{IN, OUT, NDIR}`（见 `qlten/qltensor/index.h`）。
- 守恒与散度：对代表性非空块，聚合规则为 OUT 加、IN 减。`InverseIndex()`、`Index::Inverse()`、`QLTensor::Dag()` 会按此规则翻转方向与量子数号。

### 5. 宇称信息的存储与传播

- 每个数据块派生一组布尔宇称位，表示该块各轴是否为奇宇称（`true`）。
- 在转置/重排/矩阵化与收缩路径中，这些宇称位被相应地转置/拼接，用于符号计数。

### 6. 收缩的费米子符号（核心算法）

- 约定：配对 OUT→IN 不出额外交换符号；IN→OUT 在奇宇称时可能出 \(-1\)。
- 实现（`FermionExchangeSignForCtrct(...)`，`qlten/qltensor/blk_spar_data_ten/data_blk_operations.h`）：
  - 将 A、B 的宇称位按当前轴顺序拼接，构造“连接阵列”以定位每对待收缩轴。
  - 对每一对待收缩轴：
    1) 若对应宇称为奇，则统计两轴之间的“奇宇称腿”数，累加交换计数；
    2) 通过 `std::rotate` 将配对轴相邻，并重标其他配对轴位置；
    3) 若该配对在 A 侧索引方向为 `IN`，对交换计数再加 1（对应 IN→OUT 的约定）；
  - 最终符号为 `(-1)^(exchange_num)`。

说明：该算法在分块级 GEMM 前对符号作一次性结算，代价可忽略，兼容任意块顺序与保存轴集合。

### 7. 费米子张量的一元操作

#### Hermitian Conjugate
在 TensorToolkit 中，量子力学/线性代数中的Hermitian Conjugate 由 `Dag()` 实现。它同时提供成员函数与自由函数两个形式，数学含义相同，但前者为就地修改（in-place），后者返回新对象（out-of-place）。

一个良定义的张量 Hermitian Conjugate 应刻画量子力学中 |ket> 与 <bra| 的相互转换，可分解为如下操作：
1. 元素逐点复共轭（complex conjugate）；
2. 将各索引方向在 IN 与 OUT 间翻转；
3. 同时将索引顺序倒序。

关于规则 3：以 rank-3 张量 T(a,b,c) 为例，规则 3 表示 `T^dag(c,b,a)` 与 `T(a,b,c)` 的元素一一对应（相同）。

对于玻色张量，是否显式采用规则 3 仅是实现约定；数学上并无歧义。对费米张量，规则 3 则是数学定义的一部分，因为索引的转置与重排会引入费米子交换符号。

为什么规则 3 对费米张量的Hermitian Conjugate的定义必不可少？可考虑一串 Majorana 费米子算符 `(gamma_1 * gamma_2 * gamma_3 * gamma_4)`，其厄米共轭需将次序反转（4321）。若仍希望维持原先（1234）的书写，需要统计交换次数并引入相应的费米子符号。

在 TensorToolkit 中，`Dag()` 并未采用“倒序后保持倒序”的约定，而是额外包含一步：
4. 在计及费米子交换符号后，将索引顺序恢复为原顺序。

这样设计的原因有二：
1. 兼容早期仅支持玻色的实现——当时并未倒序索引；
2. 不显式倒序可避免内部转置的内存拷贝，仅需结算费米子符号，有利于性能；后续 `Contract` 等操作所需的转置可由外部完成。

**相关：Conj()**
`Conj()` 目前处于一个语义上略显尴尬的状态。早期仅涉及玻色情形的代码中，`Conj()` 的本意是逐元素复共轭，适用场景（如时间反演）也清晰、良定义。引入费米子张量后，`Dag()` 中因算符倒序带来的符号计算被下放到了 `Conj()`；然而，时间反演并不需要这一步。因此从通用语义看，`Conj()` 的含义变得混杂。

经开发者检查，`Conj()` 目前未被外部使用。建议在 Doxygen 中如实标注这一点，并考虑提交 rcf：将 `Conj()` 恢复为“仅逐元素复共轭”，把与费米交换相关的符号处理仅保留在 `Dag()` 中。

**代码实现相关**
- `Conj()`（底层 `BlockSparseDataTensor::Conj()`）
    - 玻色：仅逐元素复共轭。
    - 费米：先逐元素复共轭，然后对那些需要“反向翻转符号”的数据块整体乘以 −1。该“是否翻转”的判断为：统计块上奇宇称轴的数量 `particle_num`，当 `particle_num % 4 ∈ {2,3}` 时乘以 −1，否则乘以 +1（见 `BlkQNInfo::ReverseSign()` 与 `FermionicRawDataConj_`)。
- `Dag()`（`QLTensor::Dag()`）：先对每个索引执行 `Inverse()` 翻转方向（IN↔OUT），再调用上述 `Conj()` 对数据层做复共轭与必要的块级符号翻转。

#### 2-范数
线性代数和量子力学中的向量的2-范数通常有两种定义：
1. 逐元素模的平方求和开根号
2. sqrt(<bra|ket>)

这两种定义是完全一致的。在费米型张量网络中，如果我们将2-范数的概念限制到只考虑波函数的话，这里也不会有要蛾子。所谓的只考虑波函数的含义为，要么只有一个指标来表示|ket>或<bra|，要么有多个指标，但是他们的朝向是相同的，并且我们要求张量的Div是一个偶数（这个限制是TensorToolkit的一个全局假设）。

但当程序扩展至一般的张量场景，玻色子的程序也不会有问题。但是费米子

- 共轭与范数：
  
  - `Get2Norm()` 对费米张量采用“even 减 odd”的分级范数，可能出现被开方负；
  - `GetQuasi2Norm()` 与 `QuasiNormalize()` 不分奇偶，始终非负，建议用于数值稳健化。

#### Eye()


#### ActFermionPOps 的精确定义

`QLTensor::ActFermionPOps()` 调用 `BlockSparseDataTensor::ActFermionPOps()` 实现对“IN 方向的腿”施加宇称算符的等效数值作用：

- 取出所有 `IN` 索引的位置集合 `in_indices`；若为空则直接返回。
- 对每个数据块，计算该块在这些 `in_indices` 上的奇宇称轴个数 `particle_num`。
- 若 `particle_num` 为奇数，则将该块的数据整体乘以 −1；若为偶数则不变（见 `SelectedIndicesParity()` 返回 −1 或 +1）。

直观理解：这相当于对所有 `IN` 腿施加对角宇称算符 P，其作用在每个块上给出 (−1)^{#(IN 上的奇腿)} 的相位。该操作在网络重排或需要显式穿越费米子线的场景中使用。

### 8. 分解（SVD/QR）与中间索引方向

- 方向约定与狄拉克记号一致（参见 `ten_svd.h`、`ten_qr.h`）：
  - 左 `U`：末端追加 `OUT` 中间索引；
  - 中间 `S`：索引为 `IN × OUT`；
  - 右 `V^†`：前端追加 `IN` 中间索引。
- 若 `QNT` 为费米型，这一指标指向和顺序满足分解结果的张量Contract后能回到原来的张量。

### 9. 最佳实践与常见陷阱

- 始终保持 `IN` ≡ |ket>、`OUT` ≡ <bra| 的一致性；<bra|ket> 不应手工加符号。
- 构图与收缩时尽量让配对为 OUT→IN，复杂拓扑交由符号算法自动处理。
- 稳定范数优先 `GetQuasi2Norm()`/`QuasiNormalize()`。
- 自定义费米型 `QNT` 时，实现 `IsFermionParityOdd/Even`，并保证加法/取负与守恒兼容。

### 10. 相关文件与测试

- 判定与基类：`qlten/framework/bases/fermionicable.h`。
- 费米型量子数：`qlten/qltensor/special_qn/fz2qn.h`、`fu1qn.h`、`fz2u1qn.h`。
- 方向与聚合：`qlten/qltensor/index.h`。
- 收缩与符号：`qlten/qltensor/blk_spar_data_ten/data_blk_operations.h`（`FermionExchangeSignForCtrct`）。
- 示例与回归：`tests/test_tensor_manipulation/test_fermion_ten_ctrct.cc`、`tests/test_qltensor/test_fermion_qltensor.cc`、`tests/test_utility/test_fermion_parity_exchange.cc`。

以上设计确保：一旦量子数类型为费米型，符号记账、方向翻转、分解与收缩等均在底层以统一、可组合的方式自动成立。


