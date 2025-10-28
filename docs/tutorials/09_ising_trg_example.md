# Z2 Ising TRG 

本节给出二维正方晶格伊辛模型（零外场、各向同性）的解析自由能密度表达式，用作 TRG 实现的基准，并讨论数值积分与基准耦合方案。

### 解析自由能与临界点（Onsager 解）

在热力学极限下，自由能密度为：

\[
f(T) = -k_B T \left\{ \ln 2 + \tfrac{1}{2} \ln[\sinh(2K)] + \frac{1}{2\pi} \int_0^{\pi} \ln\!\left[ \frac{1 + \sqrt{1 - k^2 \sin^2\theta}}{2} \right] \, d\theta \right\},
\]

其中 \(K = J/(k_B T)\)，并定义

\[
k = \frac{2\,|\sinh(2K)|}{\cosh^2(2K)}.
\]

临界点由

\[
\sinh\bigl(2 K_c\bigr) = 1
\]

给出，即 \(T_c = 2J/\bigl[k_B \ln(1+\sqrt{2})\bigr]\)。以上结果可由 Onsager-Kaufman 方案推导，并与各向同性情形的标准表述等价。

> 参考：L. Onsager, Phys. Rev. 65, 117 (1944); B. M. McCoy and T. T. Wu, The Two-Dimensional Ising Model; R. J. Baxter, Exactly Solved Models in Statistical Mechanics.

We have created a Python script `examples/ising_exact.py` to generate the analytic free energy CSV for 2D Ising (square lattice).