// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-13 16:18
*
* Description: QuantumLiquids/tensor project. Symmetry-blocked sparse tensor class and
* some short inline helper functions.
*/

/**
@file qltensor.h
@brief Symmetry-blocked sparse tensor class and some short inline helper functions.
*/
#ifndef QLTEN_QLTENSOR_QLTENSOR_H
#define QLTEN_QLTENSOR_QLTENSOR_H

#include "qlten/framework/value_t.h"                                // CoorsT, ShapeT, QLTEN_Double, QLTEN_Complex
#include "qlten/framework/bases/streamable.h"                       // Streamable
#include "qlten/framework/bases/showable.h"                         // Showable
#include "qlten/qltensor/index.h"                                   // IndexVec
#include "qlten/qltensor/blk_spar_data_ten/blk_spar_data_ten.h"     // BlockSparseDataTensor

#include <vector>       // vector
#include <iostream>     // istream, ostream
namespace qlten {

/**
Symmetry-blocked sparse tensor.

@tparam ElemT Type of the tensor element.
@tparam QNT   Type of the quantum number.
*/
template<typename ElemT, typename QNT>
class QLTensor : public Showable, public Fermionicable<QNT> {
 public:
  // Constructors and destructor.
  /// Default constructor.
  QLTensor(void) = default;

  QLTensor(const IndexVec<QNT> &);

  QLTensor(const IndexVec<QNT> &&);

  QLTensor(IndexVec<QNT> &&);

  QLTensor(const QLTensor &);

  QLTensor &operator=(const QLTensor &);

  QLTensor(QLTensor &&) noexcept;

  QLTensor &operator=(QLTensor &&) noexcept;

  /// Destroy a QLTensor.
  ~QLTensor(void) { delete pblk_spar_data_ten_; }

  // Get or check basic properties of the QLTensor.
  /// Get rank of the QLTensor.
  size_t Rank(void) const { return rank_; }

  /// Get the size of the QLTensor.
  size_t size(void) const { return size_; }

  /// Get indexes of the QLTensor.
  const IndexVec<QNT> &GetIndexes(void) const { return indexes_; }

  /// Get index of the QLTensor.
  const Index<QNT> &GetIndex(const size_t i) const { return indexes_[i]; }

  /// Get the shape of the QLTensor.
  const ShapeT &GetShape(void) const { return shape_; }

  double Sparsity(void) const { return double(pblk_spar_data_ten_->GetActualRawDataSize()) / (double) size_; }
  /// Get the number of quantum number block contained by this QLTensor.
  size_t GetQNBlkNum(void) const {
    if (IsScalar() || IsDefault()) { return 0; }
    return pblk_spar_data_ten_->GetBlkIdxDataBlkMap().size();
  }

  /// Get the block sparse data tensor constant.
  BlockSparseDataTensor<ElemT, QNT> &GetBlkSparDataTen(void) {
    return *pblk_spar_data_ten_;
  }

  /// Get the block sparse data tensor constant.
  const BlockSparseDataTensor<ElemT, QNT> &GetBlkSparDataTen(void) const {
    return *pblk_spar_data_ten_;
  }

  /// Get the pointer which point to block sparse data tensor constant.
  const BlockSparseDataTensor<ElemT, QNT> *GetBlkSparDataTenPtr(void) const {
    return pblk_spar_data_ten_;
  }

  /// Check whether the tensor is a scalar.
  bool IsScalar(void) const { return (rank_ == 0) && (size_ == 1); }

  /// Check whether the tensor is created by the default constructor.
  bool IsDefault(void) const { return size_ == 0; }

  // Calculate properties of the QLTensor.
  QNT Div(void) const;

  // Element getter and setter.
  ElemT GetElem(const std::vector<size_t> &) const;

  void SetElem(const std::vector<size_t> &, const ElemT);

  struct QLTensorElementAccessDeref;

  QLTensorElementAccessDeref operator()(const std::vector<size_t> &);

  QLTensorElementAccessDeref operator()(const std::vector<size_t> &) const;

  QLTensorElementAccessDeref operator()(void);

  QLTensorElementAccessDeref operator()(void) const;

  template<typename... OtherCoorsT>
  QLTensorElementAccessDeref operator()(const size_t, const OtherCoorsT...);

  template<typename... OtherCoorsT>
  QLTensorElementAccessDeref operator()(
      const size_t,
      const OtherCoorsT...
  ) const;

  // Inplace operations.
  void Random(const QNT &);

  void Transpose(const std::vector<size_t> &);

  void FuseIndex(const size_t, const size_t);

  QLTEN_Double Get2Norm(void) const;

  QLTEN_Double GetQuasi2Norm(void) const;

  QLTEN_Double Normalize(void);

  ///< rescale the tensor so that the summation of the square of elements = 1
  QLTEN_Double QuasiNormalize(void);

  void Dag(void);

  ///< act the fermion P operators
  void ActFermionPOps(void);

  void ElementWiseInv(void);

  void ElementWiseInv(double tolerance);

  void DiagMatInv(void);

  void DiagMatInv(double tolerance);

  void ElementWiseSqrt(void);

  void ElementWiseSign(void);

  void ElementWiseBoundTo(double bound);

  template<typename RandGenerator>
  void ElementWiseRandSign(std::uniform_real_distribution<double> &dist,
                           RandGenerator &g);

  double GetMaxAbs(void) const {
    return pblk_spar_data_ten_->GetMaxAbs();
  }
  // Operators overload.
  bool operator==(const QLTensor &) const;

  bool operator!=(const QLTensor &rhs) const { return !(*this == rhs); }

  QLTensor operator-(void) const;

  QLTensor operator+(const QLTensor &) const;

  QLTensor &operator+=(const QLTensor &);

  QLTensor operator*(const ElemT) const;

  QLTensor &operator*=(const ElemT);

  QLTensor &RemoveTrivialIndexes(void);

  QLTensor &RemoveTrivialIndexes(const std::vector<size_t> &trivial_idx_axes);

  // Override base class
  void StreamRead(std::istream &);
  void StreamWrite(std::ostream &) const;

  void MPI_Send(int, int, const MPI_Comm &) const;
  MPI_Status MPI_Recv(int, int, const MPI_Comm &);
  void MPI_Bcast(const int, const MPI_Comm &) ;

  void StreamReadShellForMPI(std::istream &);
  void StreamWriteShellForMPI(std::ostream &) const;
  std::string SerializeShell() const;
  void DeserializeShell(const std::string &);

  void Show(const size_t indent_level = 0) const override;

  void ConciseShow(const size_t indent_level = 0) const;

  // in unit of GB
  double GetRawDataMemUsage() const;

  bool HasNan(void) const;

  size_t GetActualDataSize(void) const;

  const ElemT *GetRawDataPtr(void) const;

  // The following non-const function should be called carefully.
  ElemT *GetRawDataPtr(void);

 private:
  /// The rank of the QLTensor.
  size_t rank_ = 0;
  /// The shape of the QLTensor.
  ShapeT shape_;
  /// The total number of elements of the QLTensor.
  size_t size_ = 0;

  /// Indexes of the QLTensor.
  IndexVec<QNT> indexes_;
  /// The pointer which point to block sparse data tensor.
  BlockSparseDataTensor<ElemT, QNT> *pblk_spar_data_ten_ = nullptr;

  ShapeT CalcShape_(void) const;

  size_t CalcSize_(void) const;

  std::pair<CoorsT, CoorsT> CoorsToBlkCoorsDataCoors_(const CoorsT &) const;
};

// Out-of-class declaration and definition.
template<typename ElemT, typename QNT>
inline QLTensor<ElemT, QNT> operator*(
    const QLTEN_Double scalar,
    const QLTensor<ElemT, QNT> &t
) {
  return t * scalar;
}

template<typename ElemT, typename QNT>
inline QLTensor<ElemT, QNT> operator*(
    const QLTEN_Complex scalar,
    const QLTensor<ElemT, QNT> &t
) {
  return t * scalar;
}

template<typename ElemT, typename QNT>
inline std::istream &operator>>(std::istream &is, QLTensor<ElemT, QNT> &t) {
  t.StreamRead(is);
  return is;
}

template<typename ElemT, typename QNT>
inline std::ostream &operator<<(std::ostream &os, const QLTensor<ElemT, QNT> &t) {
  t.StreamWrite(os);
  return os;
}

} /* qlten */
#endif /* ifndef QLTEN_QLTENSOR_QLTENSOR_H */
