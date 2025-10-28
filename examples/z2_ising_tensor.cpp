#include "qlten/qlten.h"
#include <cmath>
#include <iostream>
#include "ising_common.h"

using namespace qlten;

// Type aliases
using Z2QN = special_qn::Z2QN;  // Z2 parity: 0 even, 1 odd
using IndexT = Index<Z2QN>;
using QNSctT = QNSector<Z2QN>;
using DTen = QLTensor<QLTEN_Double, Z2QN>;

int main() {
  const double beta = 1.0;
  const double J = 1.0;

  DTen T = ising_common::MakeZ2IsingVertex(beta, J);

  std::cout << "Constructed Z2 vertex tensor T with shape: [";
  for (size_t i = 0; i < T.Rank(); ++i) {
    std::cout << T.GetShape()[i] << (i + 1 < T.Rank() ? ", " : "]\n");
  }

  // Print non-zero entries (optional sanity check), order: (l, d, r, u)
  for (size_t l = 0; l < 2; ++l)
    for (size_t d = 0; d < 2; ++d)
      for (size_t r = 0; r < 2; ++r)
        for (size_t u = 0; u < 2; ++u)
          if (((l + d + r + u) % 2) == 0)
            std::cout << "T[" << l << d << r << u << "] = " << T({l, d, r, u}) << "\n";

  return 0;
}


