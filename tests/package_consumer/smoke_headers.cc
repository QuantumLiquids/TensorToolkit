#include "qlten/core.h"
#include "qlten/tensor_manipulation_all.h"
#include "qlten/mpi_tensor_manipulation_all.h"

#include <type_traits>

namespace {

using QNT = qlten::special_qn::U1QN;
using Tensor = qlten::QLTensor<qlten::QLTEN_Double, QNT>;

static_assert(std::is_same<decltype(std::declval<qlten::QNSector<QNT>>().GetQn()),
                           QNT>::value,
              "qlten/core.h should expose QNSector.");
static_assert(std::is_base_of<qlten::Executor,
                              qlten::TensorSVDExecutor<qlten::QLTEN_Double, QNT>>::value,
              "qlten/tensor_manipulation_all.h should expose tensor manipulation.");
static_assert(std::is_base_of<qlten::Executor,
                              qlten::dmrg::TensorContraction1SectorExecutor<qlten::QLTEN_Double, QNT>>::value,
              "qlten/tensor_manipulation_all.h should expose DMRG helpers.");

using MPISVDSlaveFn = void (*)(const MPI_Comm &);
static_assert(std::is_same<decltype(&qlten::MPISVDSlave<qlten::QLTEN_Double>),
                           MPISVDSlaveFn>::value,
              "qlten/mpi_tensor_manipulation_all.h should expose MPI helpers.");

}  // namespace

int main() {
  Tensor tensor;
  return tensor.IsDefault() ? 0 : 1;
}
