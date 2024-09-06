// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-15 12:17
*
* Description: QuantumLiquids/tensor project. Block sparse data tensor.
*/

/**
@file blk_spar_data_ten.h
@brief Block sparse data tensor.
*/
#ifndef QLTEN_QLTENSOR_BLK_SPAR_DATA_TEN_BLK_SPAR_DATA_TEN_H
#define QLTEN_QLTENSOR_BLK_SPAR_DATA_TEN_BLK_SPAR_DATA_TEN_H

#include "qlten/framework/value_t.h"                                      // CoorsT, ShapeT
#include "qlten/framework/bases/streamable.h"                             // Streamable
#include "qlten/qltensor/index.h"                                         // IndexVec, CalcQNSctNumOfIdxs
#include "qlten/qltensor/blk_spar_data_ten/data_blk.h"                    // DataBlk
#include "qlten/qltensor/blk_spar_data_ten/raw_data_operation_tasks.h"    // RawDataTransposeTask
#include "qlten/qltensor/blk_spar_data_ten/data_blk_mat.h"                // IdxDataBlkMatMap
#include "qlten/utility/utils_inl.h"                                      // CalcEffOneDimArrayOffset, CalcMultiDimDataOffsets, Reorder, ArrayEq, VecMultiSelectElemts

#include "qlten/framework/hp_numeric/mpi_fun.h"
#include <boost/serialization/serialization.hpp>
// #include <boost/serialization/map.hpp>    
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/array_wrapper.hpp>
#include <boost/mpi.hpp>
// #include "qlten/qltensor/qltensor.h"

#include <set>              // set
#include <map>              // map
#include <unordered_map>    // unordered_map
#include <iostream>         // endl, istream, ostream

#include <stdlib.h>     // malloc, free
#include <string.h>     // memcpy, memset

#ifdef Release
#define NDEBUG
#endif

#include <cassert>     // assert
#include <random>

namespace qlten {

template<typename ElemT, typename QNT>
class QLTensor;

/**
Block sparse data tensor.

@tparam ElemT Type of the tensor element.
@tparam QNT   Type of the quantum number.
*/
template<typename ElemT, typename QNT>
class BlockSparseDataTensor : public Streamable {
 public:
  /// Type of block index to data block ordered mapping.
  using BlkIdxDataBlkMap = std::map<size_t, DataBlk<QNT>>;

  // Constructors and destructor.
  BlockSparseDataTensor(const IndexVec<QNT> *);

  BlockSparseDataTensor(const BlockSparseDataTensor &);

  BlockSparseDataTensor &operator=(const BlockSparseDataTensor &);

  ~BlockSparseDataTensor(void);

  // Element level operations.
  ElemT ElemGet(const std::pair<CoorsT, CoorsT> &) const;

  void ElemSet(const std::pair<CoorsT, CoorsT> &, const ElemT);

  // Data block level operations
  typename BlkIdxDataBlkMap::iterator
  DataBlkInsert(const CoorsT &blk_coors, const bool alloc_mem = true);

  // N.B. this is not a complete operation
  typename BlkIdxDataBlkMap::iterator
  DataBlkQuasiInsert(const CoorsT &blk_coors);

  void DataBlksOffsetRefresh();

  void DataBlksInsert(
      const std::vector<size_t> &,
      const std::vector<CoorsT> &,
      const bool,
      const bool = false
  );

  void DataBlksInsert(
      const std::vector<CoorsT> &,
      const bool,
      const bool
  );

  DataBlk<QNT> GetMaxSizeDataBlk(void) const;

  size_t GetMaxDataBlkSize(void) const;

  void DataBlkCopyAndScale(
      const RawDataCopyAndScaleTask<ElemT> &,
      const ElemT *
  );

  std::vector<RawDataCtrctTask> DataBlkGenForTenCtrct(
      const BlockSparseDataTensor &,
      const BlockSparseDataTensor &,
      const std::vector<std::vector<size_t>> &,
      const std::vector<std::vector<size_t>> &
  );

  ///< Boson case
  template<typename T = QNT>
  typename std::enable_if<!Fermionicable<T>::IsFermionic(), std::vector<RawDataCtrctTask>>::type
  DataBlkGenFor1SectTenCtrct(
      const std::map<size_t, qlten::DataBlk<QNT>> &,
      const std::map<size_t, qlten::DataBlk<QNT>> &,
      const std::vector<std::vector<size_t>> &,
      const std::vector<std::vector<size_t>> &,
      std::vector<size_t> &
  );

  ///< Fermion case
  template<typename T = QNT>
  typename std::enable_if<Fermionicable<T>::IsFermionic(), std::vector<RawDataCtrctTask>>::type
  DataBlkGenFor1SectTenCtrct(
      const std::map<size_t, qlten::DataBlk<QNT>> &,
      const std::map<size_t, qlten::DataBlk<QNT>> &,
      const std::vector<std::vector<size_t>> &,
      const std::vector<std::vector<size_t>> &,
      std::vector<size_t> &,
      const std::vector<TenIndexDirType> &
  );

  std::map<size_t, int> CountResidueFermionSignForMatBasedCtrct(
      const std::vector<size_t> &saved_axes_set,
      const size_t trans_critical_axe
  ) const;

  std::map<size_t, DataBlkMatSvdRes<ElemT>> DataBlkDecompSVD(
      const IdxDataBlkMatMap<QNT> &
  ) const;

  std::map<size_t, DataBlkMatSvdRes<ElemT>> DataBlkDecompSVDMaster(
      const IdxDataBlkMatMap<QNT> &,
      const boost::mpi::communicator &
  ) const;

  void DataBlkCopySVDUdata(
      const CoorsT &, const size_t, const size_t,
      const size_t,
      const ElemT *, const size_t, const size_t,
      const std::vector<size_t> &
  );

  void DataBlkCopySVDVtData(
      const CoorsT &, const size_t, const size_t,
      const size_t,
      const ElemT *, const size_t, const size_t,
      const std::vector<size_t> &
  );

  std::vector<SymMatEVDTask> DataBlkGenForSymMatEVD(
      BlockSparseDataTensor &u,
      BlockSparseDataTensor<QLTEN_Double, QNT> &d
  ) const;

  std::map<size_t, DataBlkMatQrRes<ElemT>> DataBlkDecompQR(
      const IdxDataBlkMatMap<QNT> &
  ) const;

  void DataBlkCopyQRQdata(
      const CoorsT &, const size_t, const size_t,
      const size_t,
      const ElemT *, const size_t, const size_t
  );

  void DataBlkCopyQRRdata(
      const CoorsT &, const size_t, const size_t,
      const size_t,
      const ElemT *, const size_t, const size_t
  );

  // Global level operations
  void Clear(void);

  void Allocate(const bool init = false);

  void Random(void);

  void Transpose(const std::vector<size_t> &);

  void FuseFirstTwoIndex(
      const std::vector<std::tuple<size_t, size_t, size_t, size_t>> &
  );

  std::vector<RawDataFermionNormTask> GenFermionNormTask(void) const;

  QLTEN_Double Norm(void);

  QLTEN_Double Quasi2Norm(void);

  QLTEN_Double Normalize(void);

  QLTEN_Double QuasiNormalize(void);

  void Conj(void);

  void AddTwoBSDTAndAssignIn(
      const BlockSparseDataTensor &,
      const BlockSparseDataTensor &
  );

  void AddAndAssignIn(const BlockSparseDataTensor &);

  void MultiplyByScalar(const ElemT);

  void CtrctTwoBSDTAndAssignIn(
      const BlockSparseDataTensor &,
      const BlockSparseDataTensor &,
      std::vector<RawDataCtrctTask> &,
      std::vector<int> &,
      std::vector<int> &
  );

  template<bool a_ctrct_tail, bool b_ctrct_head>
  void CtrctAccordingTask(
      const ElemT *a_raw_data,
      const ElemT *b_raw_data,
      const std::vector<RawDataCtrctTask> &raw_data_ctrct_tasks,
      const std::map<size_t, int> &,
      const std::map<size_t, int> &
  );

  void ConstructExpandedDataOnFirstIndex(
      const BlockSparseDataTensor &,
      const BlockSparseDataTensor &,
      const std::vector<bool> &,
      const std::map<size_t, size_t> &
  );

  void ConstructMCExpandedDataOnFirstIndex(
      const BlockSparseDataTensor &,
      const BlockSparseDataTensor &,
      const std::map<size_t, int> &
  );

  void OutOfPlaceMatrixTransposeForSelectedDataBlk(
      const std::set<size_t> &selected_data_blk_idxs,
      const size_t critical_axe,
      ElemT *transposed_data
  ) const;

  void SymMatEVDRawDataDecomposition(
      BlockSparseDataTensor &u,
      BlockSparseDataTensor<QLTEN_Double, QNT> &d,
      std::vector<SymMatEVDTask> evd_tasks
  ) const;

  bool HasDataBlkQNInfo() {
    return blk_idx_data_blk_map_.begin()->second.HasQNBlkInfo();
  }

  void CopyFromReal(const BlockSparseDataTensor<QLTEN_Double, QNT> &);

  void ElementWiseInv(void);

  void ElementWiseInv(double tolerance);

  void ElementWiseSqrt(void);

  template<typename TenElemT = ElemT, typename std::enable_if<std::is_same<TenElemT,
                                                                           QLTEN_Double>::value>::type * = nullptr>
  void ElementWiseSign(void);

  template<typename TenElemT = ElemT, typename std::enable_if<std::is_same<TenElemT,
                                                                           QLTEN_Complex>::value>::type * = nullptr>
  void ElementWiseSign(void);

  void ElementWiseBoundTo(double bound);

  template<typename RandGenerator>
  void ElementWiseRandSign(std::uniform_real_distribution<double> &dist,
                           RandGenerator &g);

  double GetMaxAbs(void) const;

  // Operators overload
  bool operator==(const BlockSparseDataTensor &) const;

  bool operator!=(const BlockSparseDataTensor &rhs) const {
    return !(*this == rhs);
  }

  // Override base class
  void StreamRead(std::istream &) override;

  void StreamWrite(std::ostream &) const override;

  // Misc
  bool IsScalar(void) const;

  /**
  Calculate block index from block coordinates.

  @param blk_coors Block coordinates.

  @return Corresponding block index.
  */
  size_t BlkCoorsToBlkIdx(const CoorsT &blk_coors) const {
    return CalcEffOneDimArrayOffset(blk_coors, blk_multi_dim_offsets_);
  }

  /// Get block index <-> data block map.
  const BlkIdxDataBlkMap &GetBlkIdxDataBlkMap(void) const {
    return blk_idx_data_blk_map_;
  }

  /// Get the pointer to actual raw data constant.
  const ElemT *GetActualRawDataPtr(void) const { return pactual_raw_data_; }

  /// Get the actual raw data size.
  size_t GetActualRawDataSize(void) const { return actual_raw_data_size_; }

  ///Get the pointer to actual raw data
  ElemT *GetActualRawDataPtr(void) { return pactual_raw_data_; }

  // Static members.
  static void ResetDataOffset(BlkIdxDataBlkMap &);

  static std::vector<size_t> GenBlkIdxQNBlkInfoPartHashMap(
      const BlkIdxDataBlkMap &,
      const std::vector<size_t> &
  );

  static std::vector<size_t> GenBlkIdxQNBlkCoorPartHashMap(
      const BlkIdxDataBlkMap &,
      const std::vector<size_t> &
  );

  void CollectiveLinearCombine(
      const std::vector<const BlockSparseDataTensor *>
  );

  void MPISend(
      const boost::mpi::communicator &,
      const int dest,
      const int tag
  );

  boost::mpi::status MPIRecv(
      const boost::mpi::communicator &,
      const int source,
      const int tag
  );

  boost::mpi::status MPIRecvIdxDataBlkMap(
      const boost::mpi::communicator &,
      const int source,
      const int tag
  );

  boost::mpi::status MPIRecvActualData(
      boost::mpi::communicator &world,
      const int source,
      const int tag
  );

  /// Rank of the tensor.
  size_t ten_rank = 0;
  /// Block shape.
  ShapeT blk_shape;
  /// A pointer which point to the indexes of corresponding QLTensor.
  const IndexVec<QNT> *pqlten_indexes = nullptr;

 private:
  /// Ordered map from block index to data block for existed blocks.
  BlkIdxDataBlkMap blk_idx_data_blk_map_;

  /// Block multi-dimension data offsets;
  std::vector<size_t> blk_multi_dim_offsets_;

  /**
  Size of the raw data in this block sparse data tensor. This size must equal to
  the sum of the size of each existed DataBlk. The only exception is the rank 0
  (scalar) case which number of DataBlk is always zero but raw_data_size_ can
  equals to 1.

  @note This variable will only be changed in DataBlk* member functions.
  */
  size_t raw_data_size_ = 0;

  /**
  Actual size of the raw data. This size must equal to the of pactual_raw_data_.

  @note This variable will only be changed in Constructors and RawData* member functions.
  */
  size_t actual_raw_data_size_ = 0;

  /**
  Pointer which point to the actual one-dimensional data array (the raw data).

  @note This variable will only be changed in Constructors and RawData* member functions.
  */
  ElemT *pactual_raw_data_ = nullptr;

  // Private data block operations.
  void DataBlkClear_(void);

  // Raw data operations.
  void RawDataFree_(void);

  void RawDataDiscard_(void);

  void RawDataAlloc_(const size_t, const bool init = false);

  void RawDataInsert_(const size_t, const size_t, const bool init = false);

  void RawDataCopy_(const std::vector<RawDataCopyTask> &, const ElemT *);

  void RawDataCopy_(const std::vector<ElemT *> &,
                    const std::vector<ElemT *> &,
                    const std::vector<size_t> &);

  void RawDataCopyNoAdd_(const std::vector<RawDataCopyTask> &, const ElemT *);

  void RawDataCopyAndScale_(
      const RawDataCopyAndScaleTask<ElemT> &,
      const ElemT *
  );

  void RawDataSetZeros_(const size_t, const size_t);

  void RawDataSetZeros_(const std::vector<size_t> &, const std::vector<size_t> &);

  void RawDataSetZeros_(const std::vector<RawDataSetZerosTask> &);

  void RawDataDuplicateFromReal_(const QLTEN_Double *, const size_t);

  void RawDataRand_(void);

  void RawDataTranspose_(const std::vector<RawDataTransposeTask> &);

  QLTEN_Double RawDataNorm_(void);

  QLTEN_Double RawDataFermionNorm_(const std::vector<RawDataFermionNormTask> &);

  QLTEN_Double RawDataNormalize_(double norm);  //input the 2-norm

  void RawDataConj_(void);

  void FermionicRawDataConj_(const std::vector<RawDataInplaceReverseTask> &);

  void RawDataMultiplyByScalar_(const ElemT);

  void RawDataTwoMatMultiplyAndAssignIn_(
      const ElemT *,
      const ElemT *,
      const size_t,
      const size_t, const size_t, const size_t,
      const double,
      const ElemT
  );

  void RawDataTwoMatMultiplyAndAssignInBatch_(
      const ElemT **,
      const ElemT **,
      ElemT **,
      const int *, const int *, const int *,
      const ElemT *,
      size_t group_count
  );

  // Related to tensor decomposition
  ElemT *RawDataGenDenseDataBlkMat_(
      const TenDecompDataBlkMat<QNT> &
  ) const;

  void RawDataRead_(std::istream &);

  void RawDataWrite_(std::ostream &) const;

  friend class boost::serialization::access;

  template<class Archive>
  void save(Archive &ar, const unsigned int version) const;

  template<class Archive>
  void load(Archive &ar, const unsigned int version);

  BOOST_SERIALIZATION_SPLIT_MEMBER()

  template<typename ElemT2, typename QNT2>
  friend inline boost::mpi::status recv_qlten(boost::mpi::communicator world,
                                              int source, int tag,
                                              QLTensor<ElemT2, QNT2> &qlten);

  template<typename ElemT2, typename QNT2>
  friend inline void RecvBroadCastQLTensor(boost::mpi::communicator world,
                                           QLTensor<ElemT2, QNT2> &qlten,
                                           const int root);

  friend class BlockSparseDataTensor<QLTEN_Complex, QNT>; //access private data in double BSDT from complex BSDT
};

template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataTwoMatMultiplyAndAssignInBatch_(const ElemT **, const ElemT **, ElemT **,
                                                                               const int *, const int *, const int *,
                                                                               const ElemT *, size_t group_count) {

}

/**
Create a block sparse data tensor using a pointer which point to the indexes
of corresponding QLTensor.

@param pqlten_indexes A pointer which point to the indexes of corresponding
       QLTensor.
*/
template<typename ElemT, typename QNT>
BlockSparseDataTensor<ElemT, QNT>::BlockSparseDataTensor(
    const IndexVec<QNT> *pqlten_indexes
) : pqlten_indexes(pqlten_indexes) {
  ten_rank = pqlten_indexes->size();
  blk_shape = CalcQNSctNumOfIdxs(*pqlten_indexes);
  blk_multi_dim_offsets_ = CalcMultiDimDataOffsets(blk_shape);
}

/**
Copy a block sparse data tensor.

@param bsdt Another block sparse data tensor.
*/
template<typename ElemT, typename QNT>
BlockSparseDataTensor<ElemT, QNT>::BlockSparseDataTensor(
    const BlockSparseDataTensor &bsdt
) :
    ten_rank(bsdt.ten_rank),
    blk_shape(bsdt.blk_shape),
    pqlten_indexes(bsdt.pqlten_indexes),
    blk_idx_data_blk_map_(bsdt.blk_idx_data_blk_map_),
    blk_multi_dim_offsets_(bsdt.blk_multi_dim_offsets_),
    raw_data_size_(bsdt.raw_data_size_),
    actual_raw_data_size_(bsdt.actual_raw_data_size_) {
  if (bsdt.pactual_raw_data_ != nullptr) {
    const size_t data_byte_size = actual_raw_data_size_ * sizeof(ElemT);
    pactual_raw_data_ = (ElemT *) malloc(data_byte_size);
    memcpy(pactual_raw_data_, bsdt.pactual_raw_data_, data_byte_size);
  }
}

/**
Assign a block sparse data tensor.

@param rhs Another block sparse data tensor.
*/
template<typename ElemT, typename QNT>
BlockSparseDataTensor<ElemT, QNT> &
BlockSparseDataTensor<ElemT, QNT>::operator=(const BlockSparseDataTensor &rhs) {
  free(pactual_raw_data_);
  ten_rank = rhs.ten_rank;
  blk_shape = rhs.blk_shape;
  blk_multi_dim_offsets_ = rhs.blk_multi_dim_offsets_;
  blk_idx_data_blk_map_ = rhs.blk_idx_data_blk_map_;
  pqlten_indexes = rhs.pqlten_indexes;
  actual_raw_data_size_ = rhs.actual_raw_data_size_;
  if (rhs.pactual_raw_data_ != nullptr) {
    auto data_byte_size = actual_raw_data_size_ * sizeof(ElemT);
    pactual_raw_data_ = (ElemT *) malloc(data_byte_size);
    memcpy(pactual_raw_data_, rhs.pactual_raw_data_, data_byte_size);
  } else {
    pactual_raw_data_ = nullptr;
  }
  return *this;
}

/// Destroy a block sparse data tensor.
template<typename ElemT, typename QNT>
BlockSparseDataTensor<ElemT, QNT>::~BlockSparseDataTensor(void) {
  RawDataFree_();
}

/**
Get an element using block coordinates and in-block data coordinates (in the
degeneracy space).

@param blk_coors_data_coors Block coordinates and in-block data coordinates pair.

@return The tensor element.
*/
template<typename ElemT, typename QNT>
ElemT BlockSparseDataTensor<ElemT, QNT>::ElemGet(
    const std::pair<CoorsT, CoorsT> &blk_coors_data_coors
) const {
  assert(blk_coors_data_coors.first.size() == ten_rank);
  assert(
      blk_coors_data_coors.first.size() == blk_coors_data_coors.second.size()
  );

  // For scalar case
  if (IsScalar()) {
    if (actual_raw_data_size_ == 1) {
      return *pactual_raw_data_;
    } else {
      return 0.0;
    }
  }

  auto blk_idx_data_blk_it = blk_idx_data_blk_map_.find(
      BlkCoorsToBlkIdx(blk_coors_data_coors.first)
  );
  if (blk_idx_data_blk_it == blk_idx_data_blk_map_.end()) {
    return 0.0;
  } else {
    size_t inblk_data_idx =
        blk_idx_data_blk_it->second.DataCoorsToInBlkDataIdx(
            blk_coors_data_coors.second
        );
    return *(
        pactual_raw_data_ + blk_idx_data_blk_it->second.data_offset + inblk_data_idx
    );
  }
}

/**
Set an element using block coordinates and in-block data coordinates (in the
degeneracy space).

@param blk_coors_data_coors Block coordinates and in-block data coordinates
       pair.

@param elem The value of the tensor element.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::ElemSet(
    const std::pair<CoorsT, CoorsT> &blk_coors_data_coors,
    const ElemT elem
) {
  assert(blk_coors_data_coors.first.size() == ten_rank);
  assert(
      blk_coors_data_coors.first.size() == blk_coors_data_coors.second.size()
  );

  // For scalar case
  if (IsScalar()) {
    if (actual_raw_data_size_ == 0) {
      raw_data_size_ = 1;
      Allocate();
      *pactual_raw_data_ = elem;
    } else {
      *pactual_raw_data_ = elem;
    }
    return;
  }

  auto blk_idx_data_blk_it = blk_idx_data_blk_map_.find(
      BlkCoorsToBlkIdx(blk_coors_data_coors.first)
  );
  if (blk_idx_data_blk_it == blk_idx_data_blk_map_.end()) {
    blk_idx_data_blk_it = DataBlkInsert(blk_coors_data_coors.first);
  }
  size_t inblk_data_idx = blk_idx_data_blk_it->second.DataCoorsToInBlkDataIdx(
      blk_coors_data_coors.second
  );
  *(pactual_raw_data_ + blk_idx_data_blk_it->second.data_offset + inblk_data_idx) = elem;
}

/**
Rank 0 (scalar) check.
*/
template<typename ElemT, typename QNT>
bool BlockSparseDataTensor<ElemT, QNT>::IsScalar(void) const {
#ifdef NDEBUG
  return (ten_rank == 0);
#else     // Do more extra check
  bool is_rank0 = (ten_rank == 0);
  if (is_rank0) {
    assert(blk_shape.empty());
    assert(blk_multi_dim_offsets_.empty());
    assert(blk_idx_data_blk_map_.empty());
    assert((raw_data_size_ == 0) || (raw_data_size_ == 1));
    assert((actual_raw_data_size_ == 0) || (actual_raw_data_size_ == 1));
  }
  return is_rank0;
#endif
}

/**
Read a BlockSparseDataTensor from a stream.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::StreamRead(std::istream &is) {
  size_t data_blk_num;
  is >> data_blk_num;
  std::vector<CoorsT> blk_coors_s;
  blk_coors_s.reserve(data_blk_num);
  for (size_t i = 0; i < data_blk_num; ++i) {
    CoorsT blk_coors(ten_rank);
    for (size_t j = 0; j < ten_rank; ++j) {
      is >> blk_coors[j];
    }
    blk_coors_s.emplace_back(blk_coors);
  }
  DataBlksInsert(blk_coors_s, false, false);

  if (IsScalar()) { raw_data_size_ = 1; }
  Allocate();
  RawDataRead_(is);
}

/**
Write a BlockSparseDataTensor to a stream.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::StreamWrite(std::ostream &os) const {
  os << blk_idx_data_blk_map_.size() << "\n";
  for (auto &blk_idx_data_blk : blk_idx_data_blk_map_) {
    for (auto &blk_coor : blk_idx_data_blk.second.blk_coors) {
      os << blk_coor << "\n";
    }
  }

  if (IsScalar() && actual_raw_data_size_ == 0) {     // Empty scalar case
    ElemT *pscalar0 = new ElemT();
    os.write((char *) pscalar0, sizeof(ElemT));
    os << "\n";
    delete pscalar0;
  } else {
    RawDataWrite_(os);
  }
}

/**
 * @note only save the shell, no data!
 */
template<typename ElemT, typename QNT>
template<class Archive>
void BlockSparseDataTensor<ElemT, QNT>::save(Archive &ar, const unsigned int version) const {
  ar & blk_idx_data_blk_map_.size();
  for (auto &blk_idx_data_blk : blk_idx_data_blk_map_) {
    ar & blk_idx_data_blk.first;
    for (auto &blk_coor : blk_idx_data_blk.second.blk_coors) {
      ar & blk_coor;
    }
  }
}

/** 
 * load the shell and allocate the memory, no data is read.
 */
template<typename ElemT, typename QNT>
template<class Archive>
void BlockSparseDataTensor<ElemT, QNT>::load(Archive &ar, const unsigned int version) {
  size_t data_blk_num;
  ar & data_blk_num;
  std::vector<CoorsT> blk_coors_s(data_blk_num, CoorsT(ten_rank));
  std::vector<size_t> idxs(data_blk_num);
  for (size_t i = 0; i < data_blk_num; ++i) {
    ar & idxs[i];
    for (size_t j = 0; j < ten_rank; ++j) {
      ar & blk_coors_s[i][j];
    }
  }

  DataBlksInsert(idxs, blk_coors_s, false, false);
  if (IsScalar()) { raw_data_size_ = 1; }

  Allocate();
}

template<typename ElemT, typename QNT>
inline void BlockSparseDataTensor<ElemT, QNT>::MPISend(
    const boost::mpi::communicator &world,
    const int dest,
    const int tag) {
  world.send(dest, tag, *this);
  int tag_data = 2 * tag + 1;
  if (IsScalar() && actual_raw_data_size_ == 0) {
    ElemT zero = ElemT(0.0);
    ElemT *data_pointer = &zero;
    int data_size = 1;
    hp_numeric::MPI_Send(data_pointer, data_size, dest, tag_data, MPI_Comm(world));
  } else {
    hp_numeric::MPI_Send(pactual_raw_data_, actual_raw_data_size_, dest, tag_data, MPI_Comm(world));
  }
}

template<typename ElemT, typename QNT>
inline boost::mpi::status BlockSparseDataTensor<ElemT, QNT>::MPIRecv(
    const boost::mpi::communicator &world,
    const int source,
    const int tag) {
  boost::mpi::status recv_ten_status = world.recv(source, tag, *this);
  const int data_source = recv_ten_status.source();
  const int data_tag = 2 * recv_ten_status.tag() + 1;
  hp_numeric::MPI_Recv(pactual_raw_data_, actual_raw_data_size_, data_source, data_tag, MPI_Comm(world));
  return recv_ten_status;
}

template<typename ElemT, typename QNT>
inline boost::mpi::status BlockSparseDataTensor<ElemT, QNT>::MPIRecvIdxDataBlkMap(
    const boost::mpi::communicator &world,
    const int source,
    const int tag
) {
  return world.recv(source, tag, *this);
}

template<typename ElemT, typename QNT>
inline boost::mpi::status BlockSparseDataTensor<ElemT, QNT>::MPIRecvActualData(
    boost::mpi::communicator &world,
    const int source,
    const int tag_data
) {
  assert(source != boost::mpi::any_source);
  boost::mpi::status recv_ten_status = world.recv(source, tag_data, pactual_raw_data_, actual_raw_data_size_);
  return recv_ten_status;
}

/**
Re-calculate and reset the data offset of each data block in a BlkIdxDataBlkMap.

@param blk_idx_data_blk_map A block index <-> data block mapping.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::ResetDataOffset(
    BlkIdxDataBlkMap &blk_idx_data_blk_map
) {
  size_t data_offset = 0;
  for (auto &blk_idx_data_blk : blk_idx_data_blk_map) {
    blk_idx_data_blk.second.data_offset = data_offset;
    data_offset += blk_idx_data_blk.second.size;
  }
}

/**
Generate block index map <-> quantum block info part hash value.

@param blk_idx_data_blk_map A block index <-> data block map.
@param axes Selected axes indexes for part hash.

@return Block index (in even index) <-> part hash value map (in odd index).
*/
template<typename ElemT, typename QNT>
std::vector<size_t>
BlockSparseDataTensor<ElemT, QNT>::GenBlkIdxQNBlkInfoPartHashMap(
    const BlkIdxDataBlkMap &blk_idx_data_blk_map,
    const std::vector<size_t> &axes
) {
  std::vector<size_t> blk_idx_qnblk_info_part_hash_map;
  blk_idx_qnblk_info_part_hash_map.reserve(2 * blk_idx_data_blk_map.size());
  for (auto &blk_idx_data_blk : blk_idx_data_blk_map) {
    blk_idx_qnblk_info_part_hash_map[
        blk_idx_data_blk.first
    ] = blk_idx_data_blk.second.GetBlkQNInfo().PartHash(axes);
  }
  return blk_idx_qnblk_info_part_hash_map;
}

/**
Generate block index <-> block part hash value map.

@param blk_idx_data_blk_map A block index <-> data block map.
@param axes Selected axes indexes for part hash.

@return Block index (in even index) <-> part hash value map (in odd index).
*/
template<typename ElemT, typename QNT>
std::vector<size_t>
BlockSparseDataTensor<ElemT, QNT>::GenBlkIdxQNBlkCoorPartHashMap(
    const BlkIdxDataBlkMap &blk_idx_data_blk_map,
    const std::vector<size_t> &axes
) {
  std::vector<size_t> blk_idx_qnblk_info_part_hash_map;
  blk_idx_qnblk_info_part_hash_map.reserve(2 * blk_idx_data_blk_map.size());
  for (auto &blk_idx_data_blk : blk_idx_data_blk_map) {
    blk_idx_qnblk_info_part_hash_map.push_back(blk_idx_data_blk.first);
    const ShapeT &blk_coors = blk_idx_data_blk.second.blk_coors;
    blk_idx_qnblk_info_part_hash_map.push_back(VecPartHasher(blk_coors, axes));
  }
  return blk_idx_qnblk_info_part_hash_map;
}

/**
Equivalence check. Only check data, not quantum number information.

@param rhs The other BlockSparseDataTensor at the right hand side.

@return Equivalence check result.
*/
template<typename ElemT, typename QNT>
bool BlockSparseDataTensor<ElemT, QNT>::operator==(
    const BlockSparseDataTensor &rhs
) const {
  if (IsScalar() && rhs.IsScalar()) {
    return ElemGet({}) == rhs.ElemGet({});
  }

  auto data_blk_size = blk_idx_data_blk_map_.size();
  if (data_blk_size != rhs.blk_idx_data_blk_map_.size()) { return false; }

//  if constexpr (Fermionicable<QNT>::IsFermionic()) {
//
//
//  } else {  //bosonic case
  auto lhs_idx_blk_it = blk_idx_data_blk_map_.begin();
  auto rhs_idx_blk_it = rhs.blk_idx_data_blk_map_.begin();
  for (size_t i = 0; i < data_blk_size; ++i) {
    // test if the indices of the datablk are the same, as the indices reflect the position and shape of the datablk
    if (lhs_idx_blk_it->first != rhs_idx_blk_it->first) { return false; }
    lhs_idx_blk_it++;
    rhs_idx_blk_it++;
  }
  return ArrayEq(
      pactual_raw_data_, actual_raw_data_size_,
      rhs.pactual_raw_data_, rhs.actual_raw_data_size_
  );
//  }
}
} /* qlten */
#endif /* ifndef QLTEN_QLTENSOR_BLK_SPAR_DATA_TEN_BLK_SPAR_DATA_TEN_H */


// Include other details
#include "qlten/qltensor/blk_spar_data_ten/global_operations.h"
#include "qlten/qltensor/blk_spar_data_ten/data_blk_operations.h"
#include "qlten/qltensor/blk_spar_data_ten/raw_data_operations.h"
