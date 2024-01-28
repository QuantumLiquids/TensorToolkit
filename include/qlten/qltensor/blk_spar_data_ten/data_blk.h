// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-18 16:11
*
* Description: QuantumLiquids/tensor project. Data block in a block sparse data tensor.
*/

/**
@file data_blk.h
@brief Data block in a block sparse data tensor.
*/
#ifndef QLTEN_QLTENSOR_BLK_SPAR_DATA_TEN_DATA_BLK_H
#define QLTEN_QLTENSOR_BLK_SPAR_DATA_TEN_DATA_BLK_H

#include "qlten/framework/value_t.h"                        // CoorsT, ShapeT
#include "qlten/qltensor/blk_spar_data_ten/qnblk_info.h"    // QNBlkInfo
#include "qlten/qltensor/index.h"                           // IndexVec
#include "qlten/utility/utils_inl.h"                        // CalcEffOneDimArrayOffset, CalcMultiDimDataOffsets, Reorder

namespace qlten {

/**
Data block in a block sparse data tensor.

@tparam QNT Type of the quantum number.
*/
template<typename QNT>
class DataBlk {
 public:
  DataBlk(void) = default;

  /**
  Create a data block using its block coordinates and QLTensor indexes
  information. This constructor will construct the corresponding quantum number
  block information.

  @param blk_coors Block coordinates.
  @param qlten_indexes Related QLTensor's indexes.
  */
  DataBlk(
      const CoorsT &blk_coors,
      const IndexVec<QNT> &qlten_indexes
  ) : blk_coors(blk_coors) {
    CreateQNBlkInfo_(qlten_indexes);
    has_qnblk_info_ = true;
    shape = qnblk_info_.CalcDgncSpaceShape();
    CalcSize_();
  }

  DataBlk(
      const CoorsT &blk_coors,
      const ShapeT &shape
  ) :
      blk_coors(blk_coors),
      shape(shape),
      has_qnblk_info_(false) {
    CalcSize_();
  }

  DataBlk(
      const CoorsT &&blk_coors,
      const ShapeT &&shape
  ) :
      blk_coors(std::move(blk_coors)),
      shape(std::move(shape)),
      has_qnblk_info_(false) {
    CalcSize_();
  }

  /// Get quantum number block info.
  const QNBlkInfo<QNT> &GetQNBlkInfo(void) const {
    assert(has_qnblk_info_);
    return qnblk_info_;
  }

  /// Create and set quantum number block info if it doesn't exist.
  void SetQNBlkInfo(const IndexVec<QNT> &qlten_indexes) {
    if (!has_qnblk_info_) {
      CreateQNBlkInfo_(qlten_indexes);
      has_qnblk_info_ = true;
    }
  }

  bool HasQNBlkInfo() {
    return has_qnblk_info_;
  };

  /**
  Transpose the data block (only information).

  @param transed_idxes_order Transposed order of indexes.
  */
  void Transpose(const std::vector<size_t> &transed_idxes_order) {
    InplaceReorder(blk_coors, transed_idxes_order);
    InplaceReorder(shape, transed_idxes_order);

    if (has_qnblk_info_) { qnblk_info_.Transpose(transed_idxes_order); }
  }

  /**
  Convert data coordinates to corresponding index.

  @param data_coors Data coordinates.
  */
  size_t DataCoorsToInBlkDataIdx(const CoorsT &data_coors) const {
    auto data_multi_dim_offsets = CalcMultiDimDataOffsets(shape);
    return CalcEffOneDimArrayOffset(data_coors, data_multi_dim_offsets);
  }

  /// Block coordinates of this data block in the block sparse data tensor.
  CoorsT blk_coors;
  /// Shape of the data block.
  ShapeT shape;
  /// Total number of data elements.
  size_t size = 0;

  /// The start offset of this data block in the 1D data array of block sparse data tensor.
  size_t data_offset;

 private:
  QNBlkInfo<QNT> qnblk_info_;

  /**
  Create quantum number block information for this data block.

  @param qlten_indexes Indexes of the related QLTensor.
  */
  void CreateQNBlkInfo_(const IndexVec<QNT> &qlten_indexes) {
    QNSectorVec<QNT> qnscts;
    qnscts.reserve(blk_coors.size());
    for (size_t i = 0; i < blk_coors.size(); ++i) {
      auto &qnsct = qlten_indexes[i].GetQNSct(blk_coors[i]);
      qnscts.push_back(qnsct);
    }
    qnblk_info_ = QNBlkInfo<QNT>(qnscts);
  }

  bool has_qnblk_info_ = false;

  size_t CalcSize_(void) {
    if (shape.size() == 0) { return 0; }
    size = 1;
    for (auto dim: shape) { size *= dim; }
    return size;
  }
};
} /* qlten */
#endif /* ifndef QLTEN_QLTENSOR_BLK_SPAR_DATA_TEN_DATA_BLK_H */
