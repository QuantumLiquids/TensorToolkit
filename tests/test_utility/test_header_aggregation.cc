// SPDX-License-Identifier: LGPL-3.0-only
/*
* Description: Unittests for public umbrella header aggregation.
*/

#include "qlten/qlten.h"

#include <gtest/gtest.h>
#include <stdexcept>
#include <type_traits>

namespace {

using QNT = qlten::special_qn::U1QN;
using QNSector = qlten::QNSector<QNT>;
using Index = qlten::Index<QNT>;
using Tensor = qlten::QLTensor<qlten::QLTEN_Double, QNT>;
using SingularValueTensor = qlten::QLTensor<qlten::QLTEN_Double, QNT>;

TEST(TestHeaderAggregation, QltenUmbrellaExportsDmrgAndMpiEntryPoints) {
  static_assert(std::is_same<decltype(std::declval<QNSector>().GetQn()), QNT>::value,
                "qlten/qlten.h should expose QNSector.");
  static_assert(std::is_same<decltype(std::declval<Index>().GetQNSct(size_t{})),
                             const QNSector &>::value,
                "qlten/qlten.h should expose Index.");

  using TensorContractionExecutor =
      qlten::TensorContractionExecutor<qlten::QLTEN_Double, QNT>;
  using TensorSVDExecutor = qlten::TensorSVDExecutor<qlten::QLTEN_Double, QNT>;
  using TensorQRExecutor = qlten::TensorQRExecutor<qlten::QLTEN_Double, QNT>;
  using TensorLQExecutor = qlten::TensorLQExecutor<qlten::QLTEN_Double, QNT>;
  using SymMatEVDExecutor = qlten::SymMatEVDExecutor<qlten::QLTEN_Double, QNT>;
  using DMRGExecutor =
      qlten::dmrg::TensorContraction1SectorExecutor<qlten::QLTEN_Double, QNT>;

  static_assert(std::is_base_of<qlten::Executor, TensorContractionExecutor>::value,
                "qlten/qlten.h should expose tensor contraction helpers.");
  static_assert(std::is_base_of<qlten::Executor, TensorSVDExecutor>::value,
                "qlten/qlten.h should expose SVD helpers.");
  static_assert(std::is_base_of<std::runtime_error, qlten::EmptySVDResultError>::value,
                "qlten/qlten.h should expose SVD error types.");
  static_assert(std::is_base_of<qlten::Executor, TensorQRExecutor>::value,
                "qlten/qlten.h should expose QR helpers.");
  static_assert(std::is_base_of<qlten::Executor, TensorLQExecutor>::value,
                "qlten/qlten.h should expose LQ helpers.");
  static_assert(std::is_base_of<qlten::Executor, SymMatEVDExecutor>::value,
                "qlten/qlten.h should expose EVD helpers.");
  static_assert(std::is_base_of<qlten::Executor, DMRGExecutor>::value,
                "qlten/qlten.h should expose DMRG tensor manipulation helpers.");

  using ContractFn = void (*)(
      const Tensor *,
      const Tensor *,
      const std::vector<std::vector<size_t>> &,
      Tensor *);
  using SVDFn = void (*)(
      const Tensor *,
      size_t,
      const QNT &,
      qlten::QLTEN_Double,
      size_t,
      size_t,
      Tensor *,
      SingularValueTensor *,
      Tensor *,
      qlten::QLTEN_Double *,
      size_t *);
  using QRFn = void (*)(const Tensor *, size_t, const QNT &, Tensor *, Tensor *);
  using LQFn = void (*)(const Tensor *, size_t, const QNT &, Tensor *, Tensor *);
  using ExpandFn = void (*)(
      const Tensor *,
      const Tensor *,
      const std::vector<size_t> &,
      Tensor *);
  using AxisDiagonalFn = void (*)(
      const Tensor &,
      size_t,
      const qlten::dmrg::AxisDiagonalOp<qlten::QLTEN_Double, QNT> &,
      Tensor &,
      qlten::QLTEN_Double,
      qlten::QLTEN_Double);
  using AxisMonomialFn = void (*)(
      const Tensor &,
      size_t,
      const qlten::dmrg::AxisMonomialOp<qlten::QLTEN_Double, QNT> &,
      Tensor &,
      qlten::QLTEN_Double,
      qlten::QLTEN_Double);
  using ScaleAxisSectorsFn = void (*)(
      Tensor &,
      const qlten::dmrg::AxisSectorScalar<qlten::QLTEN_Double, QNT> &);
  using ProjectAxisFn = void (*)(
      const Tensor &,
      size_t,
      const qlten::dmrg::AxisProjector<QNT> &,
      Tensor &);
  using DMRGContractFn = void (*)(
      const Tensor *,
      size_t,
      size_t,
      const Tensor *,
      const std::vector<std::vector<size_t>> &,
      Tensor *);
  using MPISVDMasterFn = void (*)(
      const Tensor *,
      size_t,
      const QNT &,
      qlten::QLTEN_Double,
      size_t,
      size_t,
      Tensor *,
      SingularValueTensor *,
      Tensor *,
      qlten::QLTEN_Double *,
      size_t *,
      const MPI_Comm &);
  using MPISVDSlaveFn = void (*)(const MPI_Comm &);

  static_assert(std::is_same<decltype(static_cast<ContractFn>(
                                 &qlten::Contract<qlten::QLTEN_Double, QNT>)),
                             ContractFn>::value,
                "qlten/qlten.h should expose Contract.");
  static_assert(std::is_same<decltype(static_cast<SVDFn>(
                                 &qlten::SVD<qlten::QLTEN_Double, QNT>)),
                             SVDFn>::value,
                "qlten/qlten.h should expose SVD.");
  static_assert(std::is_same<decltype(static_cast<QRFn>(
                                 &qlten::QR<qlten::QLTEN_Double, QNT>)),
                             QRFn>::value,
                "qlten/qlten.h should expose QR.");
  static_assert(std::is_same<decltype(static_cast<LQFn>(
                                 &qlten::LQ<qlten::QLTEN_Double, QNT>)),
                             LQFn>::value,
                "qlten/qlten.h should expose LQ.");
  static_assert(std::is_same<decltype(static_cast<ExpandFn>(
                                 &qlten::dmrg::ExpandQNBlocks<qlten::QLTEN_Double, QNT>)),
                             ExpandFn>::value,
                "qlten/qlten.h should expose DMRG block expansion.");
  static_assert(std::is_same<decltype(static_cast<AxisDiagonalFn>(
                                 &qlten::dmrg::ApplyAxisDiagonal<qlten::QLTEN_Double, QNT>)),
                             AxisDiagonalFn>::value,
                "qlten/qlten.h should expose DMRG axis diagonal apply.");
  static_assert(std::is_same<decltype(static_cast<AxisMonomialFn>(
                                 &qlten::dmrg::ApplyAxisMonomial<qlten::QLTEN_Double, QNT>)),
                             AxisMonomialFn>::value,
                "qlten/qlten.h should expose DMRG axis monomial apply.");
  static_assert(std::is_same<decltype(static_cast<ScaleAxisSectorsFn>(
                                 &qlten::dmrg::ScaleAxisSectorsInPlace<qlten::QLTEN_Double, QNT>)),
                             ScaleAxisSectorsFn>::value,
                "qlten/qlten.h should expose DMRG axis sector scaling.");
  static_assert(std::is_same<decltype(static_cast<ProjectAxisFn>(
                                 &qlten::dmrg::ProjectAxis<qlten::QLTEN_Double, QNT>)),
                             ProjectAxisFn>::value,
                "qlten/qlten.h should expose DMRG compact axis projection.");
  static_assert(std::is_same<decltype(static_cast<DMRGContractFn>(
                                 &qlten::dmrg::Contract1Sector<qlten::QLTEN_Double, QNT>)),
                             DMRGContractFn>::value,
                "qlten/qlten.h should expose DMRG restricted contraction.");
  static_assert(std::is_same<decltype(static_cast<MPISVDMasterFn>(
                                 &qlten::MPISVDMaster<qlten::QLTEN_Double, QNT>)),
                             MPISVDMasterFn>::value,
                "qlten/qlten.h should expose MPI SVD master entry point.");
  static_assert(std::is_same<decltype(static_cast<MPISVDSlaveFn>(
                                 &qlten::MPISVDSlave<qlten::QLTEN_Double>)),
                             MPISVDSlaveFn>::value,
                "qlten/qlten.h should expose MPI SVD slave entry point.");

  SUCCEED();
}

}  // namespace
