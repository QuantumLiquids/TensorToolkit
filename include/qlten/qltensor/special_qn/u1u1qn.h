// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2021-9-28
*
* Description: QuantumLiquids/tensor project. `U(1) \cross U(1)` QN class.
*/

/**
@file u1u1qn.h
@brief `U(1) \cross U(1)` QN class = qlten::QN<U1QNVal, U1QNVal>.
 Avoiding virtual function realization. (except Showable)
 Compatible in reading and writing  QN<U1QNVal, U1QNVal> data.
*/

#ifndef QLTEN_QLTENSOR_SPECIAL_QN_U1U1QN_H
#define QLTEN_QLTENSOR_SPECIAL_QN_U1U1QN_H

#include "qlten/framework/vec_hash.h"   //_HASH_XXPRIME_1...
#include "qlten/qltensor/qn/qn.h"   //QNCardVec
#include "qlten/qltensor/qn/qnval.h"    //QNVal
#include "qlten/qltensor/qn/qnval_u1.h" //U1QNVal
#include <boost/serialization/serialization.hpp>

namespace qlten {
namespace special_qn {

class U1U1QN : public Showable {
 public:
  U1U1QN(void);
  U1U1QN(const int val1, const int val2);
  U1U1QN(const std::string &name1, const int val1,
         const std::string &name2, const int val2
  );
  U1U1QN(const U1U1QN &);

  //Compatible
  U1U1QN(const QNCardVec &qncards);

  U1U1QN &operator=(const U1U1QN &);

  ~U1U1QN(void);

  U1U1QN operator-(void) const;
  U1U1QN &operator+=(const U1U1QN &);

  U1U1QN operator+(const U1U1QN &rhs) const;
  U1U1QN operator-(const U1U1QN &rhs) const;

  size_t dim(void) const { return 1; }

  //Compatible
  U1QNVal GetQNVal(const size_t idx) const {
    assert(idx == 0 || idx == 1);
    return U1QNVal(vals_[idx]);
  }

  bool operator==(const U1U1QN &rhs) const {
    return hash_ == rhs.hash_;
  }

  bool operator!=(const U1U1QN &rhs) const {
    return !(*this == rhs);
  }

  //Hashable
  size_t Hash() const { return hash_; }

  void StreamRead(std::istream &);
  void StreamWrite(std::ostream &) const;

  void Show(const size_t indent_level = 0) const override;

  static U1U1QN Zero(void) {
    return U1U1QN(0, 0);
  }

 private:
  size_t CalcHash_(void) const;

  int vals_[2];
  size_t hash_;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version) {
    ar & vals_;
    ar & hash_;
  }
};

inline U1U1QN::U1U1QN(void) : vals_{0, 0}, hash_(CalcHash_()) {}

inline U1U1QN::U1U1QN(const int val1, const int val2) : vals_{val1, val2}, hash_(CalcHash_()) {}

inline U1U1QN::U1U1QN(const std::string &name1, const int val1, const std::string &name2, const int val2) :
    vals_{val1, val2}, hash_(CalcHash_()) {}

inline U1U1QN::U1U1QN(const U1U1QN &rhs) : vals_{rhs.vals_[0], rhs.vals_[1]}, hash_(rhs.hash_) {}

inline U1U1QN::U1U1QN(const QNCardVec &qncards) {
  assert(qncards.size() == 2);
  for (size_t i = 0; i < 2; i++) {
    const int val = qncards[i].GetValPtr()->GetVal();
    vals_[i] = val;
  }
  hash_ = CalcHash_();
}

inline U1U1QN::~U1U1QN() {}

inline U1U1QN &U1U1QN::operator=(const U1U1QN &rhs) {
  for (size_t i = 0; i < 2; i++) {
    vals_[i] = rhs.vals_[i];
  }
  hash_ = rhs.hash_;
  return *this;
}

inline U1U1QN U1U1QN::operator-() const {
  return U1U1QN(-vals_[0], -vals_[1]);
}

inline U1U1QN &U1U1QN::operator+=(const U1U1QN &rhs) {
  vals_[0] += rhs.vals_[0];
  vals_[1] += rhs.vals_[1];
  hash_ = CalcHash_();
  return *this;
}

inline U1U1QN U1U1QN::operator+(const U1U1QN &rhs) const {
  return U1U1QN(vals_[0] + rhs.vals_[0], vals_[1] + rhs.vals_[1]);
}

inline U1U1QN U1U1QN::operator-(const U1U1QN &rhs) const {
  return U1U1QN(vals_[0] - rhs.vals_[0], vals_[1] - rhs.vals_[1]);
}

inline void U1U1QN::StreamRead(std::istream &is) {
  is >> vals_[0];
  is >> vals_[1];
  is >> hash_;
}

inline void U1U1QN::StreamWrite(std::ostream &os) const {
  os << vals_[0] << "\n" << vals_[1] << "\n" << hash_ << "\n";
}

inline void U1U1QN::Show(const size_t indent_level) const {
  std::cout << IndentPrinter(indent_level)
            << "U1U1QN:  ("
            << vals_[0]
            << ", "
            << vals_[1]
            << ")"
            << "\n";
}

inline size_t U1U1QN::CalcHash_() const {
  ///< a faith realization compatible with QN<U1QNVal, U1QNVal>
//  const size_t len = 2;
//  size_t hash_val = _HASH_XXPRIME_5;
//  for(size_t i = 0; i < len; i++){
//    const size_t item_hash_val = vals_[i];
//    hash_val += item_hash_val * _HASH_XXPRIME_2;
//    hash_val = _HASH_XXROTATE(hash_val);
//    hash_val *= _HASH_XXPRIME_1;
//  }
//  hash_val += len ^ _HASH_XXPRIME_5;
//  return hash_val;
  /** a simple realization
   * in 64 bit system size_t has 8 byte = 64 bits.
   * assume -2^30 < u1vals < 2^30, a map is direct
   */
  const size_t segment_const = 1024 * 1024 * 1024; //2^30
  size_t hash_val1 = vals_[0] + segment_const;
  size_t hash_val2 = vals_[1] + segment_const;
  hash_val2 *= (2 * segment_const);
  size_t hash_val = hash_val1 + hash_val2;
  return ((hash_val << 10) | (hash_val >> 54)); // To avoid collide of QNSector
}

inline std::istream &operator>>(std::istream &is, U1U1QN &qn) {
  qn.StreamRead(is);
  return is;
}

inline std::ostream &operator<<(std::ostream &os, const U1U1QN &qn) {
  qn.StreamWrite(os);
  return os;
}

inline size_t Hash(const U1U1QN &qn) { return qn.Hash(); }

}//special_qn
}//qlten
#endif //QLTEN_QLTENSOR_SPECIAL_QN_U1U1QN_H
