#include "qlten/qlten.h"
#include <cmath>
#include <iostream>

using namespace qlten;

// Type aliases
using Z2QN = special_qn::Z2QN;  // Z2 parity: 0 even, 1 odd
using IndexT = Index<Z2QN>;
using QNSctT = QNSector<Z2QN>;
using DTen = QLTensor<QLTEN_Double, Z2QN>;

int main() {
  const double beta = 1.0;
  const double J = 1.0;
  const double K = beta * J;

  // Character-expansion coefficients λ_p
  const double lambda0 = std::cosh(K);
  const double lambda1 = std::sinh(K);
  const double sqrt_lambda[2] = {std::sqrt(lambda0), std::sqrt(lambda1)};

  // Z2 sectors per leg: even (0), odd (1), deg 1 each
  Z2QN q_even(0), q_odd(1);
  QNSctT s_even(q_even, 1), s_odd(q_odd, 1);

  // Index directions: L,D = IN; R,U = OUT
  IndexT idx_l({s_even, s_odd}, TenIndexDirType::IN);
  IndexT idx_r({s_even, s_odd}, TenIndexDirType::OUT);
  IndexT idx_u({s_even, s_odd}, TenIndexDirType::OUT);
  IndexT idx_d({s_even, s_odd}, TenIndexDirType::IN);

  // Vertex tensor T(l, r, u, d)
  DTen T({idx_l, idx_r, idx_u, idx_d});

  // Fill with selection rule and amplitudes: T = 2 Π_α √λ_{p_α} if parity sum is even, else 0
  for (size_t l = 0; l < 2; ++l)
    for (size_t r = 0; r < 2; ++r)
      for (size_t u = 0; u < 2; ++u)
        for (size_t d = 0; d < 2; ++d) {
          const bool even_parity = ((l + r + u + d) % 2 == 0);
          if (!even_parity) {
            T({l, r, u, d}) = 0.0;
            continue;
          }
          const double amp = 2.0 * sqrt_lambda[l] * sqrt_lambda[r] * sqrt_lambda[u] * sqrt_lambda[d];
          T({l, r, u, d}) = amp;
        }

  std::cout << "Constructed Z2 vertex tensor T with shape: [";
  for (size_t i = 0; i < T.Rank(); ++i) {
    std::cout << T.GetShape()[i] << (i + 1 < T.Rank() ? ", " : "]\n");
  }

  // Print non-zero entries (optional sanity check)
  for (size_t l = 0; l < 2; ++l)
    for (size_t r = 0; r < 2; ++r)
      for (size_t u = 0; u < 2; ++u)
        for (size_t d = 0; d < 2; ++d)
          if (((l + r + u + d) % 2) == 0)
            std::cout << "T[" << l << r << u << d << "] = " << T({l, r, u, d}) << "\n";

  return 0;
}


