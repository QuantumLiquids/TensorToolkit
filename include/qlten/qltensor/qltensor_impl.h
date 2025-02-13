// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-26 19:00
*
* Description: QuantumLiquids/tensor project. Implementation details for symmetry-blocked
* sparse tensor class.
*/

/**
@file qltensor_impl.h
@brief  Implementation details for symmetry-blocked sparse tensor class.
*/
#ifndef QLTEN_QLTENSOR_QLTENSOR_IMPL_H
#define QLTEN_QLTENSOR_QLTENSOR_IMPL_H

#include <iostream>     // cout, endl, istream, ostream
#include <iterator>     // next
#include <algorithm>    // is_sorted

#include "qlten/framework/hp_numeric/mpi_fun.h"

#include "qlten/qltensor/qltensor.h"                                // QLTensor
#include "qlten/qltensor/index.h"                                   // IndexVec, GetQNSctNumOfIdxs, CalcDiv
#include "qlten/qltensor/blk_spar_data_ten/blk_spar_data_ten.h"     // BlockSparseDataTensor
#include "qlten/utility/utils_inl.h"                                // GenAllCoors, Rand, Reorder, CalcScalarNorm, CalcConj

#ifdef Release
#define NDEBUG
#endif

#include <cassert>     // assert

namespace qlten {

/**
Create an empty QLTensor using indexes.

@param indexes Vector of Index of the tensor.
*/
template<typename ElemT, typename QNT>
QLTensor<ElemT, QNT>::QLTensor(
    const IndexVec<QNT> &indexes
) : indexes_(indexes) {
  rank_ = indexes_.size();
  shape_ = CalcShape_();
  size_ = CalcSize_();
  if (!IsDefault()) {
    pblk_spar_data_ten_ = new BlockSparseDataTensor<ElemT, QNT>(&indexes_);
  }
}

template<typename ElemT, typename QNT>
QLTensor<ElemT, QNT>::QLTensor(
    const IndexVec<QNT> &&indexes
) : indexes_(std::move(indexes)) {
  rank_ = indexes_.size();
  shape_ = CalcShape_();
  size_ = CalcSize_();
  if (!IsDefault()) {
    pblk_spar_data_ten_ = new BlockSparseDataTensor<ElemT, QNT>(&indexes_);
  }
}

/**
Create an empty QLTensor by moving indexes.

@param indexes Vector of Index of the tensor.
*/
template<typename ElemT, typename QNT>
QLTensor<ElemT, QNT>::QLTensor(IndexVec<QNT> &&indexes) : indexes_(indexes) {
  rank_ = indexes_.size();
  shape_ = CalcShape_();
  size_ = CalcSize_();
  if (!IsDefault()) {
    pblk_spar_data_ten_ = new BlockSparseDataTensor<ElemT, QNT>(&indexes_);
  }
}

/**
Copy a QLTensor.

@param qlten Another QLTensor.
*/
template<typename ElemT, typename QNT>
QLTensor<ElemT, QNT>::QLTensor(const QLTensor &qlten) :
    rank_(qlten.rank_),
    shape_(qlten.shape_),
    size_(qlten.size_),
    indexes_(qlten.indexes_) {
  if (qlten.IsDefault()) {
    // Do nothing
  } else {
    pblk_spar_data_ten_ = new BlockSparseDataTensor<ElemT, QNT>(
        *qlten.pblk_spar_data_ten_
    );
    pblk_spar_data_ten_->pqlten_indexes = &indexes_;
  }
}

/**
Assign a QLTensor.

@param rhs Another QLTensor.
*/
template<typename ElemT, typename QNT>
QLTensor<ElemT, QNT> &QLTensor<ElemT, QNT>::operator=(const QLTensor &rhs) {
  rank_ = rhs.rank_;
  shape_ = rhs.shape_;
  size_ = rhs.size_;
  indexes_ = rhs.indexes_;
  delete pblk_spar_data_ten_;
  if (rhs.IsDefault()) {
    pblk_spar_data_ten_ = nullptr;
  } else {
    pblk_spar_data_ten_ = new BlockSparseDataTensor<ElemT, QNT>(
        *rhs.pblk_spar_data_ten_
    );
    pblk_spar_data_ten_->pqlten_indexes = &indexes_;
  }
  return *this;
}

/**
Move a QLTensor.

@param qlten Another QLTensor to-be moved.
*/
template<typename ElemT, typename QNT>
QLTensor<ElemT, QNT>::QLTensor(QLTensor &&qlten) noexcept :
    rank_(qlten.rank_),
    shape_(qlten.shape_),
    size_(qlten.size_) {
  if (qlten.IsDefault()) {
    // Do nothing
  } else {
    indexes_ = std::move(qlten.indexes_);
    pblk_spar_data_ten_ = qlten.pblk_spar_data_ten_;
    qlten.pblk_spar_data_ten_ = nullptr;
    pblk_spar_data_ten_->pqlten_indexes = &indexes_;
  }
}

/**
Move and assign a QLTensor.

@param rhs Another QLTensor to-be moved.
*/
template<typename ElemT, typename QNT>
QLTensor<ElemT, QNT> &QLTensor<ElemT, QNT>::operator=(QLTensor &&rhs) noexcept {
  rank_ = rhs.rank_;
  shape_ = rhs.shape_;
  size_ = rhs.size_;
  delete pblk_spar_data_ten_;
  if (rhs.IsDefault()) {
    pblk_spar_data_ten_ = nullptr;
  } else {
    indexes_ = std::move(rhs.indexes_);
    pblk_spar_data_ten_ = rhs.pblk_spar_data_ten_;
    rhs.pblk_spar_data_ten_ = nullptr;
    pblk_spar_data_ten_->pqlten_indexes = &indexes_;
  }
  return *this;
}

template<typename ElemT, typename QNT>
struct QLTensor<ElemT, QNT>::QLTensorElementAccessDeref {
  const QLTensor &const_this_ten;
  std::vector<size_t> coors;

  QLTensorElementAccessDeref(
      const QLTensor &qlten,
      const std::vector<size_t> &coors
  ) : const_this_ten(qlten), coors(coors) {
    assert(const_this_ten.Rank() == coors.size());
  }

  operator ElemT() const {
    return const_this_ten.GetElem(coors);
  }

  void operator=(const ElemT elem) {
    const_cast<QLTensor &>(const_this_ten).SetElem(coors, elem);
  }

  bool operator==(const ElemT elem) const {
    return const_this_ten.GetElem(coors) == elem;
  }

  bool operator!=(const ElemT elem) const {
    return !(*this == elem);
  }
};

/**
Access to the tensor element using its coordinates.

@param coors The coordinates of the element.
*/
template<typename ElemT, typename QNT>
typename QLTensor<ElemT, QNT>::QLTensorElementAccessDeref
QLTensor<ElemT, QNT>::operator()(const std::vector<size_t> &coors) {
  return QLTensorElementAccessDeref(*this, coors);
}

template<typename ElemT, typename QNT>
typename QLTensor<ElemT, QNT>::QLTensorElementAccessDeref
QLTensor<ElemT, QNT>::operator()(const std::vector<size_t> &coors) const {
  return QLTensorElementAccessDeref(*this, coors);
}

/**
Access to the rank 0 (scalar) tensor element.
*/
template<typename ElemT, typename QNT>
typename QLTensor<ElemT, QNT>::QLTensorElementAccessDeref
QLTensor<ElemT, QNT>::operator()(void) {
  assert(IsScalar());
  return QLTensorElementAccessDeref(*this, {});
}

template<typename ElemT, typename QNT>
typename QLTensor<ElemT, QNT>::QLTensorElementAccessDeref
QLTensor<ElemT, QNT>::operator()(void) const {
  assert(IsScalar());
  return QLTensorElementAccessDeref(*this, {});
}

/**
Access to the tensor element of a non-scalar tensor.

@tparam OtherCoorsT The types of the second, third, etc coordinates.
@param coor0 The first coordinate.
@param other_coors The second, third, ... coordiantes. They should be
       non-negative integers.
*/
template<typename ElemT, typename QNT>
template<typename... OtherCoorsT>
typename QLTensor<ElemT, QNT>::QLTensorElementAccessDeref
QLTensor<ElemT, QNT>::operator()(
    const size_t coor0,
    const OtherCoorsT... other_coors
) {
  return QLTensorElementAccessDeref(
      *this,
      {coor0, static_cast<size_t>(other_coors)...}
  );
}

template<typename ElemT, typename QNT>
template<typename... OtherCoorsT>
typename QLTensor<ElemT, QNT>::QLTensorElementAccessDeref
QLTensor<ElemT, QNT>::operator()(
    const size_t coor0,
    const OtherCoorsT... other_coors
) const {
  return QLTensorElementAccessDeref(
      *this,
      {coor0, static_cast<size_t>(other_coors)...}
  );
}

/**
Calculate the quantum number divergence of the QLTensor.

@return The quantum number divergence.
*/
template<typename ElemT, typename QNT>
QNT QLTensor<ElemT, QNT>::Div(void) const {
  assert(!IsDefault());
  if (IsScalar()) {
    std::cout << "Tensor is a scalar. Return empty quantum number."
              << std::endl;
    return QNT();
  } else {
    auto qnblk_num = GetQNBlkNum();
    if (qnblk_num == 0) {
      std::cout << "Tensor does not have a block. Return empty quantum number."
                << std::endl;
      return QNT();
    } else {
      auto blk_idx_data_blk_map = GetBlkSparDataTen().GetBlkIdxDataBlkMap();
      auto indexes = GetIndexes();
      auto first_blk_idx_data_blk = blk_idx_data_blk_map.begin();
      auto div = CalcDiv(indexes, first_blk_idx_data_blk->second.blk_coors);
#ifndef NDEBUG
      for (
          auto it = std::next(first_blk_idx_data_blk);
          it != blk_idx_data_blk_map.end();
          ++it
          ) {
        auto blki_div = CalcDiv(indexes, it->second.blk_coors);
        if (blki_div != div) {
          std::cout << "Tensor does not have a special divergence. Return empty quantum number."
                    << std::endl;
          assert(blki_div == div);
          return QNT();
        }
      }
#endif
      return div;
    }
  }
}

/**
Get the tensor element using its coordinates.

@coors The coordinates of the tensor element. An empty vector for scalar.
*/
template<typename ElemT, typename QNT>
ElemT QLTensor<ElemT, QNT>::GetElem(const std::vector<size_t> &coors) const {
  assert(!IsDefault());
  assert(coors.size() == rank_);
  auto blk_coors_data_coors = CoorsToBlkCoorsDataCoors_(coors);
  return pblk_spar_data_ten_->ElemGet(blk_coors_data_coors);
}

/**
Set the tensor element using its coordinates.

@param coors The coordinates of the tensor element. An empty vector for scalar.
@param elem The value of the tensor element.
*/
template<typename ElemT, typename QNT>
void QLTensor<ElemT, QNT>::SetElem(
    const std::vector<size_t> &coors,
    const ElemT elem
) {
  assert(!IsDefault());
  assert(coors.size() == rank_);
  auto blk_coors_data_coors = CoorsToBlkCoorsDataCoors_(coors);
  pblk_spar_data_ten_->ElemSet(blk_coors_data_coors, elem);
}

/**
Equivalence check.

@param rhs The QLTensor at the right hand side.

@return Equivalence check result.
*/
template<typename ElemT, typename QNT>
bool QLTensor<ElemT, QNT>::operator==(const QLTensor &rhs) const {
  // Default check
  if (IsDefault() || rhs.IsDefault()) {
    if (IsDefault() && rhs.IsDefault()) {
      return true;
    } else {
      return false;
    }
  }
  // Indexes check
  if (indexes_ != rhs.indexes_) { return false; }
  // Block sparse data tensor check
  return (*pblk_spar_data_ten_ == *rhs.pblk_spar_data_ten_);
}

/**
Random set tensor elements in [0, 1] with given quantum number divergence.
Original data of this tensor will be destroyed.

@param div Target quantum number divergence.
*/
template<typename ElemT, typename QNT>
void QLTensor<ElemT, QNT>::Random(const QNT &div) {
  assert(!IsDefault());
  if (IsScalar()) { assert(div == QNT()); }
  pblk_spar_data_ten_->Clear();
  if (IsScalar()) {
    pblk_spar_data_ten_->Random();
    return;
  }
  if (pblk_spar_data_ten_->blk_shape.size() == 2) {
    ShapeT shape = pblk_spar_data_ten_->blk_shape;
    std::vector<CoorsT> blk_coors_s;
    blk_coors_s.reserve(shape[0]);
    for (size_t i = 0; i < shape[0]; i++) {
      for (size_t j = 0; j < shape[1]; j++) {
        if (CalcDiv(indexes_, {i, j}) == div) {
          blk_coors_s.push_back({i, j});
        }
      }
    }
    pblk_spar_data_ten_->DataBlksInsert(blk_coors_s, false, false);     // NO allocate memory on this stage.
  } else {
    for (auto &blk_coors : GenAllCoors(pblk_spar_data_ten_->blk_shape)) {
      if (CalcDiv(indexes_, blk_coors) == div) {
        pblk_spar_data_ten_->DataBlkInsert(blk_coors, false);     // NO allocate memory on this stage.
      }
    }
  }
  pblk_spar_data_ten_->Random();
}

/**
Transpose the tensor using a new indexes order.

@param transed_idxes_order Transposed order of indexes.
*/
template<typename ElemT, typename QNT>
void QLTensor<ElemT, QNT>::Transpose(
    const std::vector<size_t> &transed_idxes_order
) {
  assert(!IsDefault());

  if (IsScalar()) { return; }

  assert(transed_idxes_order.size() == rank_);
  // Give a shorted order, do nothing
  if (std::is_sorted(transed_idxes_order.begin(), transed_idxes_order.end())) {
    return;
  }
  InplaceReorder(shape_, transed_idxes_order);
  InplaceReorder(indexes_, transed_idxes_order);
  pblk_spar_data_ten_->Transpose(transed_idxes_order);
}

/**
 * Calculate the 2-norm of the tensor, square root of summation of element squares
 * @return the 2-norm
 */
template<typename ElemT, typename QNT>
QLTEN_Double QLTensor<ElemT, QNT>::Get2Norm(void) const {
  assert(!IsDefault());
  QLTEN_Double norm = pblk_spar_data_ten_->Norm();
  return norm;
}

template<typename ElemT, typename QNT>
QLTEN_Double QLTensor<ElemT, QNT>::GetQuasi2Norm(void) const {
  assert(!IsDefault());
  QLTEN_Double norm = pblk_spar_data_ten_->Quasi2Norm();
  return norm;
}

/**
Normalize the tensor and return its norm.

@return The norm before the normalization.
*/
template<typename ElemT, typename QNT>
QLTEN_Double QLTensor<ElemT, QNT>::Normalize(void) {
  assert(!IsDefault());
  QLTEN_Double norm = pblk_spar_data_ten_->Normalize();
  return norm;
}

template<typename ElemT, typename QNT>
QLTEN_Double QLTensor<ElemT, QNT>::QuasiNormalize() {
  assert(!IsDefault());
  QLTEN_Double norm = pblk_spar_data_ten_->QuasiNormalize();
  return norm;
}

/**
Switch the direction of the indexes, complex conjugate of the elements.
*/
template<typename ElemT, typename QNT>
void QLTensor<ElemT, QNT>::Dag(void) {
  assert(!IsDefault());
  for (auto &index : indexes_) { index.Inverse(); }
  pblk_spar_data_ten_->Conj();

}

template<typename ElemT, typename QNT>
void QLTensor<ElemT, QNT>::ActFermionPOps() {
  assert(!IsDefault());
  pblk_spar_data_ten_->ActFermionPOps();
}

/**
Calculate \f$ -1 * T \f$.

@return \f$ -T \f$.
*/
template<typename ElemT, typename QNT>
QLTensor<ElemT, QNT> QLTensor<ElemT, QNT>::operator-(void) const {
  assert(!IsDefault());
  QLTensor<ElemT, QNT> res(*this);
  res *= -1.0;
  return res;
}

/**
Add this QLTensor \f$ A \f$ and another QLTensor \f$ B \f$.

@param rhs Another QLTensor \f$ B \f$.

@return \f$ A + B \f$.
*/
template<typename ElemT, typename QNT>
QLTensor<ElemT, QNT> QLTensor<ElemT, QNT>::operator+(
    const QLTensor &rhs
) const {
  assert(!(IsDefault() || rhs.IsDefault()));
  assert(indexes_ == rhs.indexes_);
//  assert(Div() == rhs.Div());
  QLTensor<ElemT, QNT> res(indexes_);
  res.pblk_spar_data_ten_->AddTwoBSDTAndAssignIn(
      *pblk_spar_data_ten_,
      *rhs.pblk_spar_data_ten_
  );
  return res;
}

/**
Add and assign another QLTensor \f$ B \f$ to this tensor.

@param rhs Another QLTensor \f$ B \f$.

@return \f$ A = A + B \f$.
*/
template<typename ElemT, typename QNT>
QLTensor<ElemT, QNT> &QLTensor<ElemT, QNT>::operator+=(const QLTensor &rhs) {
  assert(!(IsDefault() || rhs.IsDefault()));
  assert(indexes_ == rhs.indexes_);
  pblk_spar_data_ten_->AddAndAssignIn(*rhs.pblk_spar_data_ten_);
  return *this;
}

/**
Multiply a QLTensor by a scalar (real/complex number).

@param s A scalar.
*/
template<typename ElemT, typename QNT>
QLTensor<ElemT, QNT> QLTensor<ElemT, QNT>::operator*(const ElemT s) const {
  assert(!IsDefault());
  QLTensor<ElemT, QNT> res(*this);
  res.pblk_spar_data_ten_->MultiplyByScalar(s);
  return res;
}

/**
Multiply a QLTensor by a scalar (real/complex number) and assign back.

@param s A scalar.
*/
template<typename ElemT, typename QNT>
QLTensor<ElemT, QNT> &QLTensor<ElemT, QNT>::operator*=(const ElemT s) {
  assert(!IsDefault());
  pblk_spar_data_ten_->MultiplyByScalar(s);
  return *this;
}

template<typename ElemT, typename QNT>
QLTensor<ElemT, QNT> ElementWiseInv(const QLTensor<ElemT, QNT> &tensor) {
  QLTensor<ElemT, QNT> res(tensor);
  res.ElementWiseInv();
  return res;
}

template<typename ElemT, typename QNT>
QLTensor<ElemT, QNT> ElementWiseInv(const QLTensor<ElemT, QNT> &tensor, const double tolerance) {
  QLTensor<ElemT, QNT> res(tensor);
  res.ElementWiseInv(tolerance);
  return res;
}

template<typename ElemT, typename QNT>
QLTensor<ElemT, QNT> DiagMatInv(const QLTensor<ElemT, QNT> &tensor) {
  QLTensor<ElemT, QNT> res(tensor);
  res.DiagMatInv();
  return res;
}

template<typename ElemT, typename QNT>
QLTensor<ElemT, QNT> DiagMatInv(const QLTensor<ElemT, QNT> &tensor, const double tolerance) {
  QLTensor<ElemT, QNT> res(tensor);
  res.DiagMatInv(tolerance);
  return res;
}

template<typename ElemT, typename QNT>
QLTensor<ElemT, QNT> ElementWiseSqrt(const QLTensor<ElemT, QNT> &tensor) {
  QLTensor<ElemT, QNT> res(tensor);
  res.ElementWiseSqrt();
  return res;
}

template<typename ElemT, typename QNT>
void QLTensor<ElemT, QNT>::StreamRead(std::istream &is) {
  assert(IsDefault());    // Only default tensor can read data
  is >> rank_;
  indexes_ = IndexVec<QNT>(rank_);
  for (auto &index : indexes_) { is >> index; }
  shape_ = CalcShape_();
  size_ = CalcSize_();
  pblk_spar_data_ten_ = new BlockSparseDataTensor<ElemT, QNT>(&indexes_);
  is >> (*pblk_spar_data_ten_);
}

template<typename ElemT, typename QNT>
void QLTensor<ElemT, QNT>::StreamReadShellForMPI(std::istream &is) {
  assert(IsDefault());    // Only default tensor can read data
  is >> rank_;
  indexes_ = IndexVec<QNT>(rank_);
  for (auto &index : indexes_) { is >> index; }
  shape_ = CalcShape_();
  size_ = CalcSize_();
  if (size_ > 0) { //not default
    pblk_spar_data_ten_ = new BlockSparseDataTensor<ElemT, QNT>(&indexes_);
    pblk_spar_data_ten_->StreamReadBlkIdxDataBlkMapForMPI(is);
  }
}

template<typename ElemT, typename QNT>
void QLTensor<ElemT, QNT>::StreamWrite(std::ostream &os) const {
  assert(!IsDefault());
  os << rank_ << "\n";
  for (auto &index : indexes_) { os << index; }
  os << (*pblk_spar_data_ten_);
}

template<typename ElemT, typename QNT>
void QLTensor<ElemT, QNT>::StreamWriteShellForMPI(std::ostream &os) const {
  assert(!IsDefault());
  os << rank_ << "\n";
  for (auto &index : indexes_) { os << index; }
  pblk_spar_data_ten_->StreamWriteBlkIdxDataBlkMapForMPI(os);
}

template<typename ElemT, typename QNT>
std::string QLTensor<ElemT, QNT>::SerializeShell() const {
  std::ostringstream oss(std::ios::binary);
  StreamWriteShellForMPI(oss); // Use your existing StreamWrite function
  return oss.str();
}

template<typename ElemT, typename QNT>
void QLTensor<ElemT, QNT>::DeserializeShell(const std::string &buffer) {
  std::istringstream iss(buffer, std::ios::binary);
  StreamReadShellForMPI(iss);
}

template<typename VecElemT>
inline void VectorPrinter(const std::vector<VecElemT> &vec) {
  auto size = vec.size();
  for (size_t i = 0; i < size; ++i) {
    std::cout << vec[i];
    if (i != size - 1) {
      std::cout << ", ";
    }
  }
}

template<typename ElemT, typename QNT>
void QLTensor<ElemT, QNT>::Show(const size_t indent_level) const {
  std::cout << IndentPrinter(indent_level) << "QLTensor:" << std::endl;
  std::cout << IndentPrinter(indent_level + 1) << "Shape: (";
  VectorPrinter(shape_);
  std::cout << ")" << std::endl;
  std::cout << IndentPrinter(indent_level + 1) << "Indices:" << std::endl;
  for (auto &index : indexes_) {
    index.Show(indent_level + 2);
  }
  if (!IsDefault()) {
    std::cout << IndentPrinter(indent_level + 1) << "Divergence:" << std::endl;
    Div().Show(indent_level + 2);
    std::cout << IndentPrinter(indent_level + 1) << "Nonzero elements:" << std::endl;
    for (auto &coors : GenAllCoors(shape_)) {
      auto elem = GetElem(coors);
      if (elem != ElemT(0.0)) {
        std::cout << IndentPrinter(indent_level + 2) << "[";
        VectorPrinter(coors);
        std::cout << "]: " << elem << std::endl;
      }
    }
  }
}

template<typename ElemT, typename QNT>
QLTensor<ElemT, QNT> &QLTensor<ElemT, QNT>::RemoveTrivialIndexes(void) {
  if (this->IsDefault()) {
    return *this;
  }
  std::vector<size_t> trivial_idx_nums;
  std::vector<Index<QNT>> trivial_idxs;
  for (size_t i = 0; i < rank_; i++) {
    if (shape_[i] == 1) {
      trivial_idx_nums.push_back(i);
      trivial_idxs.push_back(InverseIndex(GetIndex(i)));
    }
  }

  QLTensor scale_ten({trivial_idxs});
  std::vector<size_t> zero_coor(trivial_idx_nums.size(), 0);
  scale_ten(zero_coor) = ElemT(1);
  std::vector<size_t> natural_num_coor(trivial_idx_nums.size());
  for (size_t i = 0; i < natural_num_coor.size(); i++) {
    natural_num_coor[i] = i;
  }
  QLTensor tmp;
  Contract(this, trivial_idx_nums, &scale_ten, natural_num_coor, &tmp);
  *this = std::move(tmp);
  return *this;
}

template<typename ElemT, typename QNT>
QLTensor<ElemT, QNT> &QLTensor<ElemT, QNT>::RemoveTrivialIndexes(const std::vector<size_t> &trivial_idx_axes) {
  if (this->IsDefault()) {
    return *this;
  }
  std::vector<Index<QNT>> trivial_idxs;
  for (auto trivial_idx_axe : trivial_idx_axes) {
    trivial_idxs.push_back(InverseIndex(GetIndex(trivial_idx_axe)));
  }

  QLTensor scale_ten({trivial_idxs});
  std::vector<size_t> zero_coor(trivial_idx_axes.size(), 0);
  scale_ten(zero_coor) = ElemT(1);
  std::vector<size_t> natural_num_coor(trivial_idx_axes.size());
  for (size_t i = 0; i < natural_num_coor.size(); i++) {
    natural_num_coor[i] = i;
  }
  QLTensor tmp;
  Contract(this, trivial_idx_axes, &scale_ten, natural_num_coor, &tmp);
  *this = std::move(tmp);
  return *this;
}

template<typename QNT>
std::string ElemenTypeOfTensor(const QLTensor<QLTEN_Double, QNT> &t) {
  return "QLTEN_Double";
}

template<typename QNT>
std::string ElemenTypeOfTensor(const QLTensor<QLTEN_Complex, QNT> &t) {
  return "QLTEN_Complex";
}

template<typename ElemT, typename QNT>
void QLTensor<ElemT, QNT>::ConciseShow(const size_t indent_level) const {
  using std::cout;
  using std::endl;
  cout << IndentPrinter(indent_level) << "QLTensor Concise Info: " << "\n";
  cout << IndentPrinter(indent_level + 1) << "tensor shape:" << "\t[";
  VectorPrinter(shape_);
  cout << "]\n";
  cout << IndentPrinter(indent_level + 1) << "tensor elementary type:\t"
       << ElemenTypeOfTensor(*this) << "\n";
  cout << IndentPrinter(indent_level + 1)
       << "tensor qumtum number block number:\t"
       << GetQNBlkNum() << "\n";
  cout << IndentPrinter(indent_level + 1)
       << "tensor size(product of shape):\t" << size_ << "\n";
  if (IsDefault()) {
    cout << IndentPrinter(indent_level + 1) << "default tensor" << endl;
    return;
  }
  unsigned data_size = pblk_spar_data_ten_->GetActualRawDataSize();
  cout << IndentPrinter(indent_level + 1)
       << "actual data size:\t" << data_size << "\n";
  cout << IndentPrinter(indent_level + 1)
       << "tensor sparsity:\t" << double(data_size) / double(size_) << endl;
}

template<typename ElemT, typename QNT>
double QLTensor<ElemT, QNT>::GetRawDataMemUsage() const {
  const size_t raw_data_size = pblk_spar_data_ten_->GetActualRawDataSize();
  const size_t bytes = raw_data_size * sizeof(ElemT);
  return double(bytes) / 1024 / 1024 / 1024;
}

// helper
inline bool is_nan(const double &value) {
  return std::isnan(value);
}

inline bool is_nan(const std::complex<double> &value) {
  return std::isnan(value.real()) || std::isnan(value.imag());
}

template<typename ElemT, typename QNT>
bool QLTensor<ElemT, QNT>::HasNan() const {
  const ElemT *raw_data = pblk_spar_data_ten_->GetActualRawDataPtr();
  for (size_t i = 0; i < pblk_spar_data_ten_->GetActualRawDataSize(); i++) {
    if (is_nan(raw_data[i])) {
      return true;
    }
  }
  return false;
}

template<typename ElemT, typename QNT>
size_t QLTensor<ElemT, QNT>::GetActualDataSize(void) const {
  return pblk_spar_data_ten_->GetActualRawDataSize();
}

/**
Calculate shape from tensor rank.
*/
template<typename ElemT, typename QNT>
inline ShapeT QLTensor<ElemT, QNT>::CalcShape_(void) const {
  ShapeT shape(rank_);
  for (size_t i = 0; i < rank_; ++i) {
    shape[i] = indexes_[i].dim();
  }
  return shape;
}

/**
Calculate size from tensor shape.
*/
template<typename ElemT, typename QNT>
inline size_t QLTensor<ElemT, QNT>::CalcSize_(void) const {
  size_t size = 1;
  for (auto dim : shape_) { size *= dim; }
  return size;
}

/**
Convert tensor element coordinates to data block coordinates and in-block data
coordinates.

@param coors Tensor element coordinates.
*/
template<typename ElemT, typename QNT>
inline
std::pair<CoorsT, CoorsT> QLTensor<ElemT, QNT>::CoorsToBlkCoorsDataCoors_(
    const CoorsT &coors
) const {
  assert(coors.size() == rank_);
  CoorsT blk_coors{};
  blk_coors.reserve(rank_);
  CoorsT data_coors{};
  data_coors.reserve(rank_);
  for (size_t i = 0; i < rank_; ++i) {
    auto blk_coor_data_coor = indexes_[i].CoorToBlkCoorDataCoor(coors[i]);
    blk_coors.push_back(blk_coor_data_coor.first);
    data_coors.push_back(blk_coor_data_coor.second);
  }
  return make_pair(blk_coors, data_coors);
}

inline int TensorDataTag(const int tag) {
  return 11 * tag + 1;
}
/**
 * Note use this function rather world.send() directly
 */
template<typename ElemT, typename QNT>
void QLTensor<ElemT, QNT>::MPI_Send(int dest, int tag, const MPI_Comm &comm) const {
  const bool is_default = this->IsDefault();
  ::MPI_Send(&is_default, 1, MPI_CXX_BOOL, dest, tag, comm);
  if (is_default) { return; }
#ifdef QLTEN_MPI_TIMING_MODE
  int mpi_rank;
  MPI_Comm_rank(comm, &mpi_rank);
  Timer send_shell_timer("mpi_send_ten_shell: from rank "
                             + std::to_string(mpi_rank)
                             + " to "
                             + std::to_string(dest)
  );
#endif
  auto buffer = SerializeShell(); // blk_idx_data_blk_map_ info
  size_t buffer_size = buffer.size();
  hp_numeric::MPI_Send(buffer_size, dest, tag, comm);
  ::MPI_Send(buffer.data(), buffer_size, MPI_CHAR, dest, tag, comm);
#ifdef QLTEN_MPI_TIMING_MODE
  send_shell_timer.PrintElapsed();
  Timer send_raw_data_timer("mpi_send_ten_raw_data: from rank "
                                + std::to_string(mpi_rank)
                                + " to "
                                + std::to_string(dest)
                                + ", data size = "
                                + std::to_string(GetBlkSparDataTen().GetActualRawDataSize())
  );
#endif
  GetBlkSparDataTen().RawDataMPISend(comm, dest, TensorDataTag(tag));
#ifdef QLTEN_MPI_TIMING_MODE
  send_raw_data_timer.PrintElapsed();
#endif
}

// function to receive the full tensor include the raw data
template<typename ElemT, typename QNT>
MPI_Status QLTensor<ElemT, QNT>::MPI_Recv(int source, int tag, const MPI_Comm &comm) {
  assert(IsDefault());
  bool is_default;
  MPI_Status status;
  HANDLE_MPI_ERROR(::MPI_Recv(&is_default, 1, MPI_CXX_BOOL, source, tag, comm, &status));
  if (is_default) { return status; }
  source = status.MPI_SOURCE;
  tag = status.MPI_TAG;
#ifdef QLTEN_MPI_TIMING_MODE
  int mpi_rank;
  MPI_Comm_rank(comm, &mpi_rank);
  Timer recv_shell_timer("mpi_recv_ten_shell: from rank "
                             + std::to_string(source)
                             + " to "
                             + std::to_string(mpi_rank)
  );
#endif
  std::string buffer;
  size_t buffer_size;
  MPI_Status mpi_status = hp_numeric::MPI_Recv(buffer_size, source, tag, comm);
  buffer.resize(buffer_size);
  HANDLE_MPI_ERROR(::MPI_Recv(buffer.data(), buffer_size, MPI_CHAR, source, tag, comm, &mpi_status));
  DeserializeShell(buffer);
#ifdef QLTEN_MPI_TIMING_MODE
  recv_shell_timer.PrintElapsed();
  Timer recv_raw_data_timer("mpi_recv_ten_raw_data from rank "
                                + std::to_string(source)
                                + " to "
                                + std::to_string(mpi_rank)
                                + ", data size = "
                                + std::to_string(GetBlkSparDataTen().GetActualRawDataSize())
  );
#endif
  GetBlkSparDataTen().RawDataMPIRecv(comm, source, TensorDataTag(tag));
#ifdef QLTEN_MPI_TIMING_MODE
  recv_raw_data_timer.PrintElapsed();
#endif
  return status;
}

template<typename ElemT, typename QNT>
void QLTensor<ElemT, QNT>::MPI_Bcast(const int root, const MPI_Comm &comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank != root) {
    assert(IsDefault());
  }
  bool is_default;
  if (rank == root) {
    is_default = IsDefault();
  }
  HANDLE_MPI_ERROR(::MPI_Bcast(&is_default, 1, MPI_CXX_BOOL, root, comm));
  if (is_default) {
    return;
  }
  //broad shell
  std::string buffer;
  int buffer_size;
  if (rank == root) {
    buffer = SerializeShell();
    buffer_size = buffer.size();
  }
  HANDLE_MPI_ERROR(::MPI_Bcast(&buffer_size, 1, MPI_INT, root, comm));
  if (rank != root) {
    buffer.resize(buffer_size);
  }
  HANDLE_MPI_ERROR(::MPI_Bcast(buffer.data(), buffer_size, MPI_CHAR, root, comm));
  if (rank != root) {
    DeserializeShell(buffer);
  }

#ifdef QLTEN_MPI_TIMING_MODE
  Timer mpi_bcast_raw_data_timer("mpi_bcast_raw_data: root = "
                                     + std::to_string(root)
                                     + " data size = "
                                     + std::to_string(GetBlkSparDataTen().GetActualRawDataSize())
  );
#endif
  GetBlkSparDataTen().RawDataMPIBcast(comm, root);
#ifdef QLTEN_MPI_TIMING_MODE
  mpi_bcast_raw_data_timer.PrintElapsed();
#endif
}

template<typename ElemT, typename QNT>
void MPI_Bcast(QLTensor<ElemT, QNT> &qlten, const int root, const MPI_Comm &comm) {
  qlten.MPI_Bcast(root, comm);
}

template<typename ElemT, typename QNT>
void QLTensor<ElemT, QNT>::ElementWiseInv(void) {
  pblk_spar_data_ten_->ElementWiseInv();
}

template<typename ElemT, typename QNT>
void QLTensor<ElemT, QNT>::ElementWiseInv(double tolerance) {
  pblk_spar_data_ten_->ElementWiseInv(tolerance);
}

template<typename ElemT, typename QNT>
void QLTensor<ElemT, QNT>::ElementWiseSqrt(void) {
  pblk_spar_data_ten_->ElementWiseSqrt();
}

template<typename ElemT, typename QNT>
void QLTensor<ElemT, QNT>::ElementWiseSign() {
  pblk_spar_data_ten_->ElementWiseSign();
}

template<typename ElemT, typename QNT>
void QLTensor<ElemT, QNT>::ElementWiseBoundTo(double bound) {
  pblk_spar_data_ten_->ElementWiseBoundTo(bound);
}

#ifndef USE_GPU
template<typename ElemT, typename QNT>
void QLTensor<ElemT, QNT>::DiagMatInv(void) {
  for (size_t i = 0; i < GetShape()[0]; i++) {
    (*this)({i, i}) = 1.0 / (*this)({i, i});
  }
}

template<typename ElemT, typename QNT>
void QLTensor<ElemT, QNT>::DiagMatInv(double tolerance) {
  for (size_t i = 0; i < GetShape()[0]; i++) {
    if (std::abs((*this)({i, i})) > tolerance)
      (*this)({i, i}) = 1.0 / (*this)({i, i});
    else
      (*this)({i, i}) = 0.0;
  }
}

template<typename ElemT, typename QNT>
template<typename RandGenerator>
void QLTensor<ElemT, QNT>::ElementWiseRandSign(std::uniform_real_distribution<double> &dist,
                                               RandGenerator &g) {
  pblk_spar_data_ten_->ElementWiseRandSign(dist, g);
}
#endif //not USE_GPU

// generate Identity tensor for bosonic tensors
// or fermionic parity operator for fermionic tensors
template<typename ElemT, typename QNT>
QLTensor<ElemT, QNT> Eye(const Index<QNT> &index0) {
  QLTensor<ElemT, QNT> eye({index0, InverseIndex(index0)});
  for (size_t i = 0; i < index0.dim(); i++) {
    eye({i, i}) = 1.0;
  }
  return eye;
}

} /* qlten */
#endif /* ifndef QLTEN_QLTENSOR_QLTENSOR_IMPL_H */
