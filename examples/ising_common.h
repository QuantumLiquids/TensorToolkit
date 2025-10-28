#pragma once

#include "qlten/qlten.h"
#include <cmath>

namespace ising_common {

using namespace qlten;

using Z2QN = special_qn::Z2QN;
using IndexT = Index<Z2QN>;
using QNSctT = QNSector<Z2QN>;
using DTen = QLTensor<QLTEN_Double, Z2QN>;

/**
 * @brief Create the Z2-symmetric vertex tensor for the Ising model
 * @param beta Inverse temperature
 * @param J Coupling constant
 * @return DTen The Z2-symmetric vertex tensor 
 *     u
 *     |
 *  l--T--r
 *     |
 *     d
 * indices order: (l, d, r, u)
 */
inline DTen MakeZ2IsingVertex(double beta, double J) {
    const double K = beta * J;
    const double lambda0 = std::cosh(K);
    const double lambda1 = std::sinh(K);
    const double sqrt_lambda[2] = {std::sqrt(lambda0), std::sqrt(lambda1)};

    Z2QN q_even(0), q_odd(1);
    QNSctT s_even(q_even, 1), s_odd(q_odd, 1);

    IndexT idx_l({s_even, s_odd}, TenIndexDirType::IN);
    IndexT idx_d({s_even, s_odd}, TenIndexDirType::IN);
    IndexT idx_r({s_even, s_odd}, TenIndexDirType::OUT);
    IndexT idx_u({s_even, s_odd}, TenIndexDirType::OUT);

    // Order: (l, d, r, u)
    DTen T({idx_l, idx_d, idx_r, idx_u});

    for (size_t l = 0; l < 2; ++l)
        for (size_t d = 0; d < 2; ++d)
            for (size_t r = 0; r < 2; ++r)
                for (size_t u = 0; u < 2; ++u) {
                    const bool even_parity = ((l + d + r + u) % 2 == 0);
                    if (!even_parity) {
                        continue; // skip creating blocks with odd total parity
                    }
                    const double amp = 2.0 * sqrt_lambda[l] * sqrt_lambda[d] * sqrt_lambda[r] * sqrt_lambda[u];
                    T({l, d, r, u}) = amp;
                }

    return T;
}

} // namespace ising_common


