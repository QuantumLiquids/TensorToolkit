# Ising Vertex Tensor Tutorial

This tutorial shows how to construct the Z2-symmetric vertex tensor for the 2D Ising model using TensorToolkit.

## What You'll Learn

- Understand the Z2 symmetry in the Ising model
- Construct vertex tensors with proper quantum numbers
- Implement the character expansion method
- Handle tensor initialization and verification

## From Ising Hamiltonian to Z2-Symmetric Tensor Network

### The Physical Model

The 2D Ising model on a square lattice has the Hamiltonian:

\[ H = -J \sum_{\langle i j \rangle} s_i s_j \]

where \( s_i \in \{+1, -1\} \) are spin variables, \( J > 0 \) is the ferromagnetic coupling, and the sum runs over nearest-neighbor pairs.

The partition function is:

\[ Z = \sum_{\{s\}} \prod_{\langle i j \rangle} e^{\beta J s_i s_j} \]

where \( \beta = 1/T \) is the inverse temperature.

### Character Expansion and Z2 Symmetry

For the Z2 group, we have two irreducible representations labeled by parity \( p \in \{0, 1\} \) with characters:

- \( \chi_0(s) = 1 \) (even/trivial representation)
- \( \chi_1(s) = s \) (odd/sign representation)

The bond Boltzmann factor can be expanded using these characters:

\[ e^{\beta J s_i s_j} = \cosh(\beta J) + (s_i s_j) \sinh(\beta J) = \sum_{p=0,1} \lambda_p \chi_p(s_i) \chi_p(s_j) \]

where \( \lambda_0 = \cosh(\beta J) \) and \( \lambda_1 = \sinh(\beta J) \).

### Half-Edge Factorization

We can factorize each bond weight using half-edge factors:

\[ w(s, p) = \sqrt{2\lambda_p} \frac{\chi_p(s)}{\sqrt{2}} \]

Then each bond weight is recovered by contracting its two half-edges over \( p \):

\[ e^{\beta J s_i s_j} = \sum_{p=0,1} w(s_i, p) w(s_j, p) \]

### Vertex Tensor Construction

At a vertex with four incident bonds carrying parities \( p_l, p_r, p_u, p_d \), summing over the spin at that site yields:

\[ T_{p_l p_r p_u p_d} = \sum_{s=\pm 1} w(s, p_l) w(s, p_r) w(s, p_u) w(s, p_d) \]

This gives us:

\[ T_{p_l p_r p_u p_d} = \left[\prod_{\alpha \in \{l,r,u,d\}} \sqrt{2\lambda_{p_\alpha}}\right] \sum_{s=\pm1} \prod_{\alpha} \frac{\chi_{p_\alpha}(s)}{\sqrt{2}} \]

The sum over spins enforces the **Z2 selection rule**: \( T_{p_l p_r p_u p_d} = 0 \) unless \( (p_l + p_r + p_u + p_d) \bmod 2 = 0 \).

For allowed configurations, the amplitude is:

\[ T_{p_l p_r p_u p_d} = 2 \prod_{\alpha} \sqrt{\lambda_{p_\alpha}} \]

## Tensor Network Setup and Conventions

### Lattice Placement

- **Lattice choice**: We place one rank-4 vertex tensor `T` on each site of the square (direct) lattice
- **Leg order**: `[Left, Right, Up, Down]`, abbreviated `[l, r, u, d]`
- **Index directions**: `Left` and `Down` as `IN`, `Right` and `Up` as `OUT`

### Visual Layout

```text
            Up (OUT)
               ▲
               │ (u)
   Left (IN) ◀─┼─▶ Right (OUT)
          (l)  │  (r)
               │ (d)
               ▼
            Down (IN)
```

### Z2 Conservation Rule

With Z2 symmetry, each leg carries a parity quantum number \( q \in \{0,1\} \). The vertex tensor is non-zero only if the sum of parities on all attached legs is even:

**Even-parity constraint**: \( (q_l + q_r + q_u + q_d) \bmod 2 = 0 \)

## Key Code Implementation

Here are the essential code snippets for constructing the Z2-symmetric vertex tensor:

### 1. Type Aliases and Quantum Number Setup

```cpp
#include "qlten/qlten.h"

using namespace qlten;

// Z2 parity basis: p = 0 (even), 1 (odd)
using Z2QN = special_qn::Z2QN;  
using IndexT = Index<Z2QN>;
using QNSctT = QNSector<Z2QN>;
using DTen = QLTensor<QLTEN_Double, Z2QN>;

// Character-expansion coefficients
const double beta = 1.0, J = 1.0, K = beta * J;
const double lambda0 = std::cosh(K), lambda1 = std::sinh(K);

// Z2 sectors per leg: even (q=0), odd (q=1), degeneracy 1 each
Z2QN q_even(0), q_odd(1);
QNSctT s_even(q_even, 1), s_odd(q_odd, 1);
```

### 2. Index Declaration with Proper Directions

```cpp
// Index directions: L,D = IN; R,U = OUT
IndexT idx_l({s_even, s_odd}, TenIndexDirType::IN);
IndexT idx_r({s_even, s_odd}, TenIndexDirType::OUT);
IndexT idx_u({s_even, s_odd}, TenIndexDirType::OUT);
IndexT idx_d({s_even, s_odd}, TenIndexDirType::IN);

// Vertex tensor T(l, r, u, d) in fixed order [Left, Right, Up, Down]
DTen T({idx_l, idx_r, idx_u, idx_d});
```

### 3. Tensor Element Setting with Z2 Selection Rule

```cpp
// Fill with selection rule: T = 2 Π_α √λ_{p_α} if parity sum is even, else 0
for (size_t l = 0; l < 2; ++l)
    for (size_t r = 0; r < 2; ++r)
        for (size_t u = 0; u < 2; ++u)
            for (size_t d = 0; d < 2; ++d) {
                const bool even_parity = ((l + r + u + d) % 2 == 0);
                if (!even_parity) {
                    T({l, r, u, d}) = 0.0;
                    continue;
                }
                // Allowed sector: product of half-edge factors
                const double amp = 2.0 * sqrt_lambda[l] * sqrt_lambda[r] * sqrt_lambda[u] * sqrt_lambda[d];
                T({l, r, u, d}) = amp;
            }
```

## Key Concepts Demonstrated

- **Quantum Numbers**: Z₂ symmetry for the Ising model (correct choice for spin-1/2 systems)
- **Index Management**: Creating indices with specific quantum number sectors and directions
- **Z2 Selection Rules**: Enforcing symmetry constraints in tensor construction
- **Character Expansion**: Using the character expansion method for efficient tensor construction

## Building and Running

The complete implementation is available in `examples/z2_ising_tensor.cpp`. Build and run it using:

```bash
# Navigate to the root directory of TensorToolkit
mkdir build && cd build

# Configure and build
cmake .. -DQLTEN_BUILD_EXAMPLES=ON
make -j4

# Run the example
./z2_ising_tensor
```

## What's Happening

1. **Tensor Creation**: We create a 4-index tensor representing the Boltzmann weight for nearest-neighbor Ising interactions
2. **Quantum Number Setup**: Define Z2 quantum number sectors for even/odd parity
3. **Index Construction**: Create indices with proper directions (IN/OUT) and quantum number sectors
4. **Tensor Initialization**: Fill the tensor with appropriate weights based on the Z2 selection rule

## Next Steps

This tutorial shows how to construct the basic vertex tensor. To use it in algorithms:

1. **[Tensor Contractions](07_tensor_contractions.html)** - Learn how to contract tensors
2. **[Matrix Decomposition](08_matrix_decomposition.html)** - Master SVD and other decomposition methods
3. **[Complete Ising TRG](09_ising_trg_example.html)** - Use this tensor in the full TRG algorithm

---

*Ready to learn tensor operations? Check out [Tensor Contractions](07_tensor_contractions.html) next!*
