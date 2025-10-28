#include "qlten/qlten.h"
#include "qlten/utility/timer.h"

#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include "ising_common.h"

using namespace qlten;

// Type aliases
using Z2QN = special_qn::Z2QN; // Z2 parity: 0 even, 1 odd
using IndexT = Index<Z2QN>;
using QNSctT = QNSector<Z2QN>;
using DTen = QLTensor<QLTEN_Double, Z2QN>;

/**
 * @brief TRG algorithm parameters
 */
struct TRGParams {
  size_t max_iterations; // Maximum TRG iterations
  size_t max_bond_dim; // Maximum bond dimension
  double truncation_error; // Truncation error threshold
  size_t min_bond_dim; // Minimum bond dimension
  double convergence_threshold; // Convergence threshold for free energy
  bool verbose; // Verbose output
  std::string output_file; // Output file for results
};

/**
 * @brief TRG iteration results
 */
struct TRGIterationResult {
  size_t scale;
  size_t bond_dim;
  double truncation_error;
  double free_energy;
  double free_energy_error;
};

/**
 * @brief Information returned by a single TRG coarse-graining step
 *
 * T is updated in-place; this struct reports auxiliary info for this step.
 */
struct TRGStepInfo {
  double actual_truncation_error; // max of the two SVD truncation errors
  size_t actual_bond_dim; // resulting kept bond dimension (max of two)
  double norm; // normalization factor applied to T
  double svd_time; // total time spent in SVDs in this step (s)
  double contract_time; // total time spent in Contracts in this step (s)
};

/**
 * @brief Construct the Z2-symmetric vertex tensor for Ising model
 *
 * @param beta Inverse temperature
 * @param J Coupling constant
 * @return DTen The rank-4 vertex tensor T(l, r, u, d)
 */
DTen ConstructBareTensor(double beta, double J) {
  return ising_common::MakeZ2IsingVertex(beta, J);
}

/**
 * @brief Perform one TRG coarse-graining step
 *
 * @param T          Input tensor (updated in-place)
 * @param max_bond_dim Maximum bond dimension for truncation
 * @param trunc_err  Target truncation threshold for SVD
 * @return TRGStepInfo Step-wise info: truncation error, kept dim, norm
 */
TRGStepInfo TRGCoarseGrainStep(DTen &T, size_t max_bond_dim, size_t min_bond_dim, double trunc_err) {
  /**   u           u1
   *    |           |
   * l--T--r  = 0l--P0
   *    |            \2
   *    d             n
   *                   \0
   *                    Q0--r2
   *                    |
   *                    d1
   */

  double actual_truncation_error0;
  size_t actual_bond_dim0;
  DTen u, s, vt;
  DTen T0 = T;
  T0.Transpose({0, 3, 1, 2});
  double svd_time = 0.0;
  double contract_time = 0.0;
  {
    Timer __t_svd0;
    SVD(&T0, 2, T0.Div(), trunc_err, min_bond_dim, max_bond_dim, &u, &s, &vt, &actual_truncation_error0, &actual_bond_dim0);
    svd_time += __t_svd0.Elapsed();
  }
  auto s_sqrt = ElementWiseSqrt(s);
  DTen P0, Q0;
  {
    Timer __t_ctr0;
    Contract(&u, &s_sqrt, {{2}, {0}}, &P0);
    contract_time += __t_ctr0.Elapsed();
  }
  {
    Timer __t_ctr1;
    Contract(&s_sqrt, &vt, {{1}, {0}}, &Q0);
    contract_time += __t_ctr1.Elapsed();
  }

  /**
   *    u              u2
   *    |              |
   * l--T--r  =        P1--r1
   *    |             /0
   *    d            n
   *                /2
   *           0l--Q1
   *               |
   *               d1
   */
  double actual_truncation_error1;
  size_t actual_bond_dim1;
  u = DTen();
  s = DTen();
  vt = DTen();
  {
    Timer __t_svd1;
    SVD(&T, 2, T.Div(), trunc_err, min_bond_dim, max_bond_dim, &u, &s, &vt, &actual_truncation_error1, &actual_bond_dim1);
    svd_time += __t_svd1.Elapsed();
  }
  s_sqrt = ElementWiseSqrt(s);
  DTen P1, Q1;
  {
    Timer __t_ctr2;
    Contract(&u, &s_sqrt, {{2}, {0}}, &Q1);
    contract_time += __t_ctr2.Elapsed();
  }
  {
    Timer __t_ctr3;
    Contract(&s_sqrt, &vt, {{1}, {0}}, &P1);
    contract_time += __t_ctr3.Elapsed();
  }

  //Update T with new coarse-grained tensor
  /**
   *
   *    0             2
   *     \           /
   *      Q0--2--0--Q1
   *      |         |               3       2
   *      1         1                \     /
   *      |         |      ---->      T_next
   *      2         1                /     \
   *      |         |               0       1
   *      P1--1--0--P0
   *     /           \
   *    0             2
   */
  DTen tmp0, tmp1, tmp2;
  {
    Timer __t_ctr4;
    Contract(&P1, {1}, &P0, {0}, &tmp0);
    contract_time += __t_ctr4.Elapsed();
  }
  {
    Timer __t_ctr5;
    Contract(&Q1, {0}, &Q0, {2}, &tmp1);
    contract_time += __t_ctr5.Elapsed();
  }
  {
    Timer __t_ctr6;
    Contract(&tmp0, {1, 2}, &tmp1, {3, 0}, &tmp2);
    contract_time += __t_ctr6.Elapsed();
  }

  auto actual_truncation_error = std::max(actual_truncation_error0, actual_truncation_error1);
  auto actual_bond_dim = std::max(actual_bond_dim0, actual_bond_dim1);
  auto norm = tmp2.Normalize();
  T = tmp2;
  return TRGStepInfo{actual_truncation_error, actual_bond_dim, norm, svd_time, contract_time};
}

/**
 * @brief Run the complete TRG algorithm for Ising model
 *
 * @param beta Inverse temperature
 * @param J Coupling constant
 * @param params TRG parameters
 * @return std::vector<TRGIterationResult> Results from each iteration
 */
TRGIterationResult RunTRG(double beta, double J, const TRGParams &params) {
  // Initialize tensor
  DTen T = ConstructBareTensor(beta, J);

  if (params.verbose) {
    std::cout << "Running TRG for β = " << beta << ", J = " << J << "\n";
    std::cout << "Initial tensor shape: [";
    for (size_t i = 0; i < T.Rank(); ++i) {
      std::cout << T.GetShape()[i] << (i + 1 < T.Rank() ? ", " : "]\n");
    }
  }

  // TRG iterations
  size_t cur_scale = 0;
  double lnZ = 0.0;
  double lattice_size = 1.0; // 2^scale
  double prev_free_energy = std::numeric_limits<double>::quiet_NaN();
  TRGStepInfo last_step_info{0.0, 0, 1.0, 0.0, 0.0};
  double last_free_energy = std::numeric_limits<double>::quiet_NaN();
  double last_free_energy_error = std::numeric_limits<double>::infinity();

  for (; cur_scale < params.max_iterations;) {
    Timer iter_timer;
    TRGStepInfo step_info = TRGCoarseGrainStep(T, params.max_bond_dim, params.min_bond_dim, params.truncation_error);
    auto coarse_grained_time = iter_timer.Elapsed();
    cur_scale++;
    lattice_size *= 2.0;
    lnZ += std::log(step_info.norm) / lattice_size;
    const double free_energy = -lnZ / beta; // per-site free energy estimate
    const double free_energy_error =
        std::isnan(prev_free_energy)
          ? std::numeric_limits<double>::infinity()
          : std::abs(free_energy - prev_free_energy);
    prev_free_energy = free_energy;

    // store last values
    last_step_info = step_info;
    last_free_energy = free_energy;
    last_free_energy_error = free_energy_error;

    if (params.verbose) {
      std::cout << "Scale " << cur_scale
          << ": chi = " << step_info.actual_bond_dim
          << ", trunc_err = " << std::scientific << step_info.actual_truncation_error
          << ", norm = " << std::fixed << step_info.norm
          << ", F = " << free_energy
          << ", SVD = " << step_info.svd_time << "s"
          << ", Contract = " << step_info.contract_time << "s"
          << ", TotT = " << coarse_grained_time << "s\n";
    }

    // Convergence check on free energy change (skip at scale 1)
    if (cur_scale > 1 && free_energy_error < params.convergence_threshold) {
      if (params.verbose) {
        std::cout << "Converged: |Δf| = " << free_energy_error
            << " < threshold = " << params.convergence_threshold << "\n";
      }
      break;
    }
  }
  // Record final result
  TRGIterationResult result;
  result.scale = cur_scale;
  result.bond_dim = last_step_info.actual_bond_dim;
  result.truncation_error = last_step_info.actual_truncation_error;
  result.free_energy = last_free_energy;
  result.free_energy_error = last_free_energy_error;

  return result;
}

// Simple CSV loader for analytic free energy: header: beta,f_exact
static bool LoadExactCSV(const std::string &path, std::vector<double> &betas, std::vector<double> &f_exact) {
  std::ifstream fin(path);
  if (!fin) {
    std::cerr << "[warn] Cannot open exact CSV: " << path << "\n";
    return false;
  }
  std::string line;
  // read header
  if (!std::getline(fin, line)) return false;
  while (std::getline(fin, line)) {
    if (line.empty()) continue;
    std::istringstream iss(line);
    std::string s_beta, s_f;
    if (!std::getline(iss, s_beta, ',')) continue;
    if (!std::getline(iss, s_f, ',')) continue;
    try {
      double b = std::stod(s_beta);
      double f = std::stod(s_f);
      betas.push_back(b);
      f_exact.push_back(f);
    } catch (...) {
      // skip malformed
    }
  }
  return !betas.empty() && betas.size() == f_exact.size();
}

static bool FindNearestExact(double beta,
    const std::vector<double> &betas,
    const std::vector<double> &f_exact,
    double *f_match,
    double *matched_beta) {
  if (betas.empty() || betas.size() != f_exact.size()) return false;
  size_t best = 0;
  double best_diff = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < betas.size(); ++i) {
    double d = std::abs(betas[i] - beta);
    if (d < best_diff) {
      best_diff = d;
      best = i;
    }
  }
  if (f_match) *f_match = f_exact[best];
  if (matched_beta) *matched_beta = betas[best];
  return true;
}

int main(int argc, char **argv) {
  std::cout << "=== Z2 Ising TRG Implementation ===\n";
  std::cout << "TensorToolkit Version: " << QLTEN_VERSION_MAJOR << "." << QLTEN_VERSION_MINOR << "\n\n";

  hp_numeric::SetTensorManipulationThreads(4);

  // Optional: load analytic free energy CSV if provided as argv[1]
  std::vector<double> exact_betas;
  std::vector<double> exact_f;
  bool has_exact = false;
  if (argc > 1 && argv[1] && std::string(argv[1]).size() > 0) {
    has_exact = LoadExactCSV(argv[1], exact_betas, exact_f);
    if (has_exact) {
      std::cout << "Loaded analytic CSV: " << argv[1]
          << ", entries = " << exact_betas.size() << "\n";
    }
  }

  // Test parameters
  std::vector<double> beta_values = {0.2, 0.4, 0.7, 1.0};

  // TRG parameters
  TRGParams params;
  params.max_iterations = 30;
  params.truncation_error = 1e-14;
  params.min_bond_dim = 4;
  params.max_bond_dim = 128;
  params.convergence_threshold = 1e-15;
  params.verbose = true;

  // Run TRG for different parameters
  for (double beta : beta_values) {
    std::cout << "\n--- Running TRG for β = " << beta
        << ", χ = " << params.max_bond_dim << " ---\n";

    TRGIterationResult result = RunTRG(beta, 1.0, params);

    // Output final results
    std::cout << "Final free energy per site: "
        << result.free_energy << "\n";
    std::cout << "Final bond dimension: "
        << result.bond_dim << "\n";
    std::cout << "Total iterations: "
        << result.scale << "\n";

    if (has_exact) {
      double f_match = 0.0;
      double matched_beta = beta;
      if (FindNearestExact(beta, exact_betas, exact_f, &f_match, &matched_beta)) {
        double abs_err = std::abs(result.free_energy - f_match);
        double rel_err = abs_err / (std::max(1.0, std::abs(f_match)));
        std::cout << std::setprecision(16)
            << "Exact (β≈" << matched_beta << ") f: " << f_match << "\n"
            << "Abs error: " << abs_err << ", Rel error: " << rel_err << "\n";
      }
    }
  }
  return 0;
}
