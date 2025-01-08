// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2024-6-23
*
* Description: QuantumLiquids/tensor project. Fermionic U(1) QN class.
*/

/**
@file fu1qn.h
@brief fermionic U(1) QN class
*/

#ifndef QLTEN_QLTENSOR_SPECIAL_QN_FfU1QN_H
#define QLTEN_QLTENSOR_SPECIAL_QN_FfU1QN_H

#include "qlten/framework/vec_hash.h"   //_HASH_XXPRIME_1...
#include "qlten/qltensor/qn/qnval.h"    //QNVal
#include "qlten/qltensor/qn/qnval_u1.h" //fU1QNVal

namespace qlten {
namespace special_qn {

//Showable
class fU1QN : public Showable {
 public:
  fU1QN(void);
  fU1QN(const int val);
  fU1QN(const std::string &name, const int val);
  fU1QN(const fU1QN &);

  fU1QN &operator=(const fU1QN &);

  ~fU1QN(void);

  fU1QN operator-(void) const;
  fU1QN &operator+=(const fU1QN &);

  fU1QN operator+(const fU1QN &rhs) const;
  fU1QN operator-(const fU1QN &rhs) const;

  size_t dim(void) const { return 1; }

  bool operator==(const fU1QN &rhs) const {
    return val_ == rhs.val_;
  }

  bool operator!=(const fU1QN &rhs) const {
    return !(*this == rhs);
  }

  //Hashable
  size_t Hash() const { return hash_; }

  bool IsFermionParityOdd() const { return val_ % 2; }
  bool IsFermionParityEven() const { return !(IsFermionParityOdd()); }

  void StreamRead(std::istream &);
  void StreamWrite(std::ostream &) const;

  void Show(const size_t indent_level = 0) const override;

  //Generate QN 0 element
  static fU1QN Zero(void) {
    return fU1QN(0);
  }

 private:

  fU1QN(const int val, const size_t hash) : val_(0), hash_(hash) {}

  size_t CalcHash_(void) const;

  int val_;
  size_t hash_;
};

inline fU1QN::fU1QN(void) : val_(0), hash_(CalcHash_()) {}

inline fU1QN::fU1QN(const int val) : val_(val), hash_(CalcHash_()) {}

inline fU1QN::fU1QN(const std::string &name, const int val) : val_(val), hash_(CalcHash_()) {}

inline fU1QN::fU1QN(const fU1QN &rhs) : val_(rhs.val_), hash_(rhs.hash_) {}

inline fU1QN::~fU1QN() {}

inline fU1QN &fU1QN::operator=(const fU1QN &rhs) {
  val_ = rhs.val_;
  hash_ = rhs.hash_;
  return *this;
}

inline fU1QN fU1QN::operator-() const {
  return fU1QN(-val_);
}

inline fU1QN &fU1QN::operator+=(const fU1QN &rhs) {
  val_ += rhs.val_;
  hash_ = CalcHash_();
  return *this;
}

inline fU1QN fU1QN::operator+(const fU1QN &rhs) const {
  return fU1QN(val_ + rhs.val_);
}

inline fU1QN fU1QN::operator-(const fU1QN &rhs) const {
  return fU1QN(val_ - rhs.val_);
}

inline void fU1QN::StreamRead(std::istream &is) {
  is >> val_;
  is >> hash_;
}

inline void fU1QN::StreamWrite(std::ostream &os) const {
  os << val_ << "\n" << hash_ << "\n";
}

inline void fU1QN::Show(const size_t indent_level) const {
  std::cout << IndentPrinter(indent_level) << "fU1QN:  " << val_ << "\n";
}

inline size_t fU1QN::CalcHash_() const {
  size_t hash = val_;
  return _HASH_XXROTATE(hash);

}

inline std::istream &operator>>(std::istream &is, fU1QN &qn) {
  qn.StreamRead(is);
  return is;
}

inline std::ostream &operator<<(std::ostream &os, const fU1QN &qn) {
  qn.StreamWrite(os);
  return os;
}

inline size_t Hash(const fU1QN &qn) { return qn.Hash(); }

}//special_qn
}//qlten
#endif //QLTEN_QLTENSOR_SPECIAL_QN_FfU1QN_H
