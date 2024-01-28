// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 20201-9-28
*
* Description: QuantumLiquids/tensor project. U(1) QN class.
*/

/**
@file u1qn.h
@brief U(1) QN class = qlten::QN<qlten::U1QNVal>.
 Avoiding virtual function realization. (except Show)
 Compatible in reading and writing  QN<U1QNVal> data.
*/

#ifndef QLTEN_QLTENSOR_SPECIAL_QN_U1QN_H
#define QLTEN_QLTENSOR_SPECIAL_QN_U1QN_H

#include "qlten/framework/vec_hash.h"   //_HASH_XXPRIME_1...
#include "qlten/qltensor/qn/qnval.h"    //QNVal
#include "qlten/qltensor/qn/qnval_u1.h" //U1QNVal
#include <boost/serialization/serialization.hpp>

namespace qlten {
namespace special_qn {

//Showable
class U1QN : public Showable {
 public:
  U1QN(void);
  U1QN(const int val);
  U1QN(const std::string &name, const int val);
  U1QN(const U1QN &);
  //Compatible

  U1QN(const QNCardVec &qncards);

  U1QN &operator=(const U1QN &);

  ~U1QN(void);

  U1QN operator-(void) const;
  U1QN &operator+=(const U1QN &);

  U1QN operator+(const U1QN &rhs) const;
  U1QN operator-(const U1QN &rhs) const;

  size_t dim(void) const { return 1; }

  //Compatible
  U1QNVal GetQNVal(const size_t idx) const {
    assert(idx == 0);
    return U1QNVal(val_);
  }

  bool operator==(const U1QN &rhs) const {
    return val_ == rhs.val_;
  }

  bool operator!=(const U1QN &rhs) const {
    return !(*this == rhs);
  }

  //Hashable
  size_t Hash() const { return hash_; }

  void StreamRead(std::istream &);
  void StreamWrite(std::ostream &) const;

  void Show(const size_t indent_level = 0) const override;

  //Generate QN 0 element
  static U1QN Zero(void) {
    return U1QN((const int) 0, (const size_t) 0);
  }

 private:

  U1QN(const int val, const size_t hash) : val_(0), hash_(hash) {}

  size_t CalcHash_(void) const;

  int val_;
  size_t hash_;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version) {
    ar & val_;
    ar & hash_;
  }
};

inline U1QN::U1QN(void) : val_(0), hash_(CalcHash_()) {}

inline U1QN::U1QN(const int val) : val_(val), hash_(CalcHash_()) {}

inline U1QN::U1QN(const std::string &name, const int val) : val_(val), hash_(CalcHash_()) {}

inline U1QN::U1QN(const U1QN &rhs) : val_(rhs.val_), hash_(rhs.hash_) {}

inline U1QN::U1QN(const QNCardVec &qncards) {
  const int val = qncards[0].GetValPtr()->GetVal();
  val_ = val;
  hash_ = CalcHash_();
}

inline U1QN::~U1QN() {}

inline U1QN &U1QN::operator=(const U1QN &rhs) {
  val_ = rhs.val_;
  hash_ = rhs.hash_;
  return *this;
}

inline U1QN U1QN::operator-() const {
  return U1QN(-val_);
}

inline U1QN &U1QN::operator+=(const U1QN &rhs) {
  val_ += rhs.val_;
  hash_ = CalcHash_();
  return *this;
}

inline U1QN U1QN::operator+(const U1QN &rhs) const {
  return U1QN(val_ + rhs.val_);
}

inline U1QN U1QN::operator-(const U1QN &rhs) const {
  return U1QN(val_ - rhs.val_);
}

inline void U1QN::StreamRead(std::istream &is) {
  is >> val_;
  is >> hash_;
}

inline void U1QN::StreamWrite(std::ostream &os) const {
  os << val_ << "\n" << hash_ << "\n";
}

inline void U1QN::Show(const size_t indent_level) const {
  std::cout << IndentPrinter(indent_level) << "U1QN:  " << val_ << "\n";
}

inline size_t U1QN::CalcHash_() const {
  ///< a faith realization compatible with QN<U1QNVal>
//  const size_t len = 1;
//  size_t hash_val = _HASH_XXPRIME_5;
//  const size_t item_hash_val = val_;
//  hash_val += item_hash_val * _HASH_XXPRIME_2;
//  hash_val = _HASH_XXROTATE(hash_val);
//  hash_val *= _HASH_XXPRIME_1;
//  hash_val += len ^ _HASH_XXPRIME_5;
//  return hash_val;
  ///< a simple realization, but not compatible with QN<U1QNVal>
  size_t hash = val_;
  return _HASH_XXROTATE(hash);

}

inline std::istream &operator>>(std::istream &is, U1QN &qn) {
  qn.StreamRead(is);
  return is;
}

inline std::ostream &operator<<(std::ostream &os, const U1QN &qn) {
  qn.StreamWrite(os);
  return os;
}

inline size_t Hash(const U1QN &qn) { return qn.Hash(); }

}//special_qn
}//qlten
#endif //QLTEN_QLTENSOR_SPECIAL_QN_U1QN_H
