// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2021-8-20
*
* Description: QuantumLiquids/tensor project. MPI SVD for a symmetric QLTensor.
*/

/**
@file mpi_svd.h
@brief MPI SVD for a symmetric QLTensor.
*/

#ifndef QLTEN_MPI_TENSOR_MANIPULATION_TEN_DECOMP_MPI_SVD_H
#define QLTEN_MPI_TENSOR_MANIPULATION_TEN_DECOMP_MPI_SVD_H

#include "qlten/framework/bases/executor.h"                           // Executor
#include "qlten/qltensor_all.h"
#include "qlten/tensor_manipulation/ten_decomp/ten_decomp_basic.h"    // GenIdxTenDecompDataBlkMats, IdxDataBlkMatMap
#include "qlten/tensor_manipulation/ten_decomp/ten_svd.h"             // TensorSVDExecutor

namespace qlten {
/**
MPI Tensor SVD executor used in master processor.

@tparam TenElemT Element type of tensors.
@tparam QNT Quantum number type of tensors.
*/
template<typename TenElemT, typename QNT>
class MPITensorSVDExecutor : public TensorSVDExecutor<TenElemT, QNT> {
 public:
  MPITensorSVDExecutor(
      const QLTensor<TenElemT, QNT> *,
      const size_t,
      const QNT &,
      const QLTEN_Double, const size_t, const size_t,
      QLTensor<TenElemT, QNT> *,
      QLTensor<QLTEN_Double, QNT> *,
      QLTensor<TenElemT, QNT> *,
      QLTEN_Double *, size_t *,
      const MPI_Comm &
  );

  ~MPITensorSVDExecutor(void) = default;

  void Execute(void) override;

 private:
  const MPI_Comm &mpi_comm_;
};

template<typename TenElemT, typename QNT>
MPITensorSVDExecutor<TenElemT, QNT>::MPITensorSVDExecutor(
    const QLTensor<TenElemT, QNT> *pt,
    const size_t ldims,
    const QNT &lqndiv,
    const QLTEN_Double trunc_err, const size_t Dmin, const size_t Dmax,
    QLTensor<TenElemT, QNT> *pu,
    QLTensor<QLTEN_Double, QNT> *ps,
    QLTensor<TenElemT, QNT> *pvt,
    QLTEN_Double *pactual_trunc_err, size_t *pD,
    const MPI_Comm &comm
): TensorSVDExecutor<TenElemT, QNT>(pt, ldims, lqndiv, trunc_err, Dmin, Dmax, pu, ps, pvt,
                                    pactual_trunc_err, pD), mpi_comm_(comm) {}

/**
MPI Execute tensor SVD calculation.
*/
template<typename TenElemT, typename QNT>
void MPITensorSVDExecutor<TenElemT, QNT>::Execute(void) {
  this->SetStatus(ExecutorStatus::EXEING);

  auto idx_raw_data_svd_res = this->pt_->GetBlkSparDataTen().DataBlkDecompSVDMaster(
      this->idx_ten_decomp_data_blk_mat_map_,
      mpi_comm_
  );
  auto kept_sv_info = this->CalcTruncedSVInfo_(idx_raw_data_svd_res);
  this->ConstructSVDResTens_(kept_sv_info, idx_raw_data_svd_res);
  DeleteDataBlkMatSvdResMap(idx_raw_data_svd_res);

  this->SetStatus(ExecutorStatus::FINISH);
}

/**
 * @brief Distributed (MPI) tensor SVD — master-side entry.
 *
 * Performs the same factorization as SVD but distributes blockwise SVD to
 * worker ranks. This entry point is called on the master rank; other ranks
 * should call MPISVDSlave<TenElemT>(comm) to participate.
 *
 * @tparam TenElemT Element type of the tensors.
 * @tparam QNT Quantum number type of the tensors.
 *
 * @param pt Pointer to tensor \f$ T \f$ (rank >= 2).
 * @param ldims Number of left indices.
 * @param lqndiv Quantum-number divergence assigned to U.
 * @param trunc_err Target truncation error.
 * @param Dmin Minimal kept dimension.
 * @param Dmax Maximal kept dimension.
 * @param pu Output U tensor (default on entry).
 * @param ps Output S matrix (default on entry).
 * @param pvt Output \f$ V^{\dagger} \f$ tensor (default on entry).
 * @param pactual_trunc_err Output achieved truncation error.
 * @param pD Output kept dimension.
 * @param comm MPI communicator.
 *
 * @note Collectively participates with ranks in `comm`. Ensure slaves run
 *       MPISVDSlave<TenElemT>(comm) concurrently.
 * @ref qlten::SVD
 */
template<typename TenElemT, typename QNT>
void MPISVDMaster(
    const QLTensor<TenElemT, QNT> *pt,
    const size_t ldims,
    const QNT &lqndiv,
    const QLTEN_Double trunc_err, const size_t Dmin, const size_t Dmax,
    QLTensor<TenElemT, QNT> *pu,
    QLTensor<QLTEN_Double, QNT> *ps,
    QLTensor<TenElemT, QNT> *pvt,
    QLTEN_Double *pactual_trunc_err, size_t *pD,
    const MPI_Comm &comm
) {
  MPITensorSVDExecutor<TenElemT, QNT> mpi_ten_svd_executor(
      pt,
      ldims,
      lqndiv,
      trunc_err, Dmin, Dmax,
      pu, ps, pvt, pactual_trunc_err, pD,
      comm
  );
  mpi_ten_svd_executor.Execute();
}

/**
 * @brief Distributed (MPI) SVD — worker-side entry.
 *
 * @tparam TenElemT Element type; must match master's type.
 * @param comm MPI communicator used by the master.
 *
 * @note Call concurrently on non-master ranks while master calls MPISVDMaster.
 */
template<typename TenElemT>
inline void MPISVDSlave(
    const MPI_Comm &comm
) {
  DataBlkDecompSVDSlave<TenElemT>(comm);
}

}

#endif