#include "qlten/qlten.h"

#include <cmath>
#include <cstdlib>
#include <vector>

namespace {

using MpiInitFn = int (*)(int *, char ***);
using MpiFinalizeFn = int (*)();

volatile MpiInitFn kMpiInitRef = &MPI_Init;
volatile MpiFinalizeFn kMpiFinalizeRef = &MPI_Finalize;

}  // namespace

int main(int argc, char **argv) {
  // Keep explicit MPI symbol references so the smoke validates transitive MPI
  // link propagation without depending on local MPI runtime launch behavior.
  (void)argc;
  (void)argv;
  (void)kMpiInitRef;
  (void)kMpiFinalizeRef;

  qlten::hp_numeric::SetTensorManipulationThreads(1);

  std::vector<size_t> input_shape{2, 2};
  std::vector<int> transpose_order{1, 0};
  std::vector<int> output_shape{2, 2};
  double input_data[4] = {1.0, 2.0, 3.0, 4.0};
  double output_data[4] = {0.0, 0.0, 0.0, 0.0};
  qlten::hp_numeric::TensorTranspose(
      transpose_order, 2, input_data, input_shape, output_data, output_shape, 1.0);
  const double expected_transpose[4] = {1.0, 3.0, 2.0, 4.0};
  for (int i = 0; i < 4; ++i) {
    if (std::abs(output_data[i] - expected_transpose[i]) > 1e-12) {
      return 1;
    }
  }

  double *u = nullptr;
  double *s = nullptr;
  double *vt = nullptr;
  qlten::hp_numeric::MatSVD(input_data, 2, 2, u, s, vt);
  if (u == nullptr || s == nullptr || vt == nullptr) {
    return 1;
  }
  if (std::abs(s[0] - 5.464985704219043) > 1e-10
      || std::abs(s[1] - 0.365966190626257) > 1e-10) {
    std::free(u);
    std::free(s);
    std::free(vt);
    return 1;
  }
  std::free(u);
  std::free(s);
  std::free(vt);

  return 0;
}
