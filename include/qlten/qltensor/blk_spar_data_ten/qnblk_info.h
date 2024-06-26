// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-16 13:05
*
* Description: QuantumLiquids/tensor project. Information of a quantum number block.
*/

/**
@file qnblk_info.h
@brief Information of a quantum number block.
*/
#ifndef QLTEN_QLTENSOR_QNBLK_INFO_H
#define QLTEN_QLTENSOR_QNBLK_INFO_H

#include "qlten/framework/value_t.h"    // CoorsT, ShapeT
#include "qlten/framework/vec_hash.h"   // VecPtrHasher
#include "qlten/qltensor/qnsct.h"       // QNSectorVec
#include "qlten/utility/utils_inl.h"    // Reorder

namespace qlten {

// Dummy class for FermionParityInfo when IsFermionic is false
template<typename QNT, bool IsFermionic>
class FermionParityInfoBase {};

// Specialization for FermionParityInfo when IsFermionic is true
template<typename QNT>
class FermionParityInfoBase<QNT, true> {
 public:
  FermionParityInfoBase(void) = default;
  FermionParityInfoBase(const QNSectorVec<QNT> &qnscts) : qn_parities(qnscts.size()) {
    for (size_t i = 0; i < qnscts.size(); i++) {
      qn_parities[i] = qnscts[i].IsFermionParityOdd();
    }
  }

  std::vector<bool> qn_parities;//true for odd, false for even.
};

// Typedef for convenience
template<typename QNT>
using FermionParityInfo = FermionParityInfoBase<QNT, Fermionicable<QNT>::IsFermionic()>;

/**
Information of a quantum number block.

@tparam QNT Type of the quantum number.
*/
template<typename QNT>
class BlkQNInfo : public FermionParityInfo<QNT> {
 public:
  BlkQNInfo(void) = default;

  // Constructor for fermionic case
  template<typename T = QNT, typename std::enable_if<Fermionicable<T>::IsFermionic(), int>::type = 0>
  BlkQNInfo(const QNSectorVec<QNT> &qnscts) : FermionParityInfo<QNT>(qnscts), qnscts(qnscts) {}

  // Constructor for bosonic case
  template<typename T = QNT, typename std::enable_if<!Fermionicable<T>::IsFermionic(), int>::type = 0>
  BlkQNInfo(const QNSectorVec<QNT> &qnscts) : qnscts(qnscts) {}

  /**
  Calculate the shape of the degeneracy space.
  */
  ShapeT CalcDgncSpaceShape(void) {
    ShapeT dgnc_space_shape;
    dgnc_space_shape.reserve(qnscts.size());
    for (auto &qnsct: qnscts) {
      dgnc_space_shape.push_back(qnsct.GetDegeneracy());
    }
    return dgnc_space_shape;
  }

  size_t PartHash(const std::vector<size_t> &) const;
  size_t QnHash(void) const;

  /**
  Transpose quantum number sectors for fermionic case
  */
  template<typename T = QNT>
  typename std::enable_if<Fermionicable<T>::IsFermionic(), int>::type
  Transpose(const std::vector<size_t> &transed_idxes_order) {
    return FermionicInplaceReorder(qnscts, transed_idxes_order, this->qn_parities);
  }
  /**
  Transpose quantum number sectors for bosonic case
  */
  template<typename T = QNT>
  typename std::enable_if<!Fermionicable<T>::IsFermionic(), int>::type
  Transpose(const std::vector<size_t> &transed_idxes_order) {
    InplaceReorder(qnscts, transed_idxes_order);
    return 1;
  }

  /**
   * sign when reversing all index, using in Dag()
   * @tparam T
   * @param transed_idxes_order
   * @return
   */
  template<typename T = QNT>
  typename std::enable_if<Fermionicable<T>::IsFermionic(), int>::type
  ReverseSign(void) const {
    size_t particle_num = std::count(this->qn_parities.cbegin(), this->qn_parities.cend(), true);
    if (particle_num % 4 == 0 || particle_num % 4 == 1) {
      return 1;
    } else {
      return -1;
    }
  }

  //bosonic case
  template<typename T = QNT>
  typename std::enable_if<!Fermionicable<T>::IsFermionic(), int>::type
  ReverseSign(void) const {
    std::cerr << "going to useless function." << std::endl;
    return 1;
  }
  QNSectorVec<QNT> qnscts;
};

/**
Calculate part hash value using selected axes indexes.


*/
template<typename QNT>
size_t BlkQNInfo<QNT>::PartHash(const std::vector<size_t> &axes) const {
  return VecPartHasher(qnscts, axes);
}

/**
Calculate a hash value only based on quantum numbers but the shape of the degeneracy space.
*/
template<typename QNT>
size_t BlkQNInfo<QNT>::QnHash(void) const {
  std::vector<QNT> qns;
  qns.reserve(qnscts.size());
  for (auto &qnsct: qnscts) { qns.push_back(qnsct.GetQn()); }
  return VecHasher(qns);
}
} /* qlten */
#endif /* ifndef QLTEN_QLTENSOR_QNBLK_INFO_H */
