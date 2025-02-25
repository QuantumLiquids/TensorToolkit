// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Haoxin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2025-02-23
*
* Description: QuantumLiquids/tensor project. Fermionic Z2 x U1 quantum number.
*/

#ifndef QLTEN_QLTENSOR_SPECIAL_QN_fZ2U1QN_H
#define QLTEN_QLTENSOR_SPECIAL_QN_fZ2U1QN_H

#include "qlten/framework/vec_hash.h"         //_HASH_XXPRIME_1...
#include "qlten/framework/bases/showable.h"   // Showable

namespace qlten {

namespace special_qn {
/**
Fermionic Z2 x U1 quantum number value.
*/
class fZ2U1QN : public Showable {
 public:
  // Constructors
  fZ2U1QN(void) = default;
  fZ2U1QN(const int z2val, const int u1_val);
  fZ2U1QN(const fZ2U1QN &rhs);

  // Assignment operator
  fZ2U1QN &operator=(const fZ2U1QN &rhs) {
    if (this != &rhs) {
      znval_ = rhs.znval_;
      u1_val_ = rhs.u1_val_;
      hash_ = rhs.hash_;
    }
    return *this;
  }
  ~fZ2U1QN() {}
  fZ2U1QN operator-(void) const {
    return fZ2U1QN(znval_, -u1_val_);
  }
  fZ2U1QN operator+(const fZ2U1QN &rhs) const {
    return fZ2U1QN((znval_ + rhs.znval_) % 2, u1_val_ + rhs.u1_val_);
  }
  fZ2U1QN &operator+=(const fZ2U1QN &rhs) {
    znval_ = (znval_ + rhs.znval_) % 2;
    u1_val_ += rhs.u1_val_;
    hash_ = CalcHash_();
    return *this;
  }

  fZ2U1QN operator-(const fZ2U1QN &rhs) const {
    return *this + (-rhs);
  }
  fZ2U1QN &operator-=(const fZ2U1QN &rhs) {
    znval_ = (znval_ == rhs.znval_) ? 0 : 1;
    u1_val_ -= rhs.u1_val_;
    hash_ = CalcHash_();
    return *this;
  }

  // Operators
  bool operator==(const fZ2U1QN &rhs) const {
    return hash_ == rhs.hash_;
  }

  bool operator!=(const fZ2U1QN &rhs) const { return !(*this == rhs); }

  size_t Hash() const { return hash_; }
  size_t dim(void) const { return 1; }
  bool IsFermionParityOdd() const { return znval_; }
  bool IsFermionParityEven() const { return !(IsFermionParityOdd()); }

  void StreamRead(std::istream &);
  void StreamWrite(std::ostream &) const;
  void Show(const size_t indent_level = 0) const override;

  static fZ2U1QN Zero(void) {
    return fZ2U1QN(0, 0);
  }

 private:
  size_t CalcHash_() const;
  int znval_;    // fermion Z2 parity value
  int u1_val_;   // bosonic U1 quantum number value (like spin)
  size_t hash_;
};

//inline fZ2U1QN::fZ2U1QN(void) : fZ2U1QN(0, 0) {}

inline fZ2U1QN::fZ2U1QN(const int z2val, const int u1_val)
    : znval_(z2val), u1_val_(u1_val), hash_(CalcHash_()) {}

inline fZ2U1QN::fZ2U1QN(const fZ2U1QN &rhs) : znval_(rhs.znval_), u1_val_(rhs.u1_val_), hash_(rhs.hash_) {}

inline void fZ2U1QN::StreamRead(std::istream &is) {
  is >> znval_;
  is >> u1_val_;
  is >> hash_;
}

inline void fZ2U1QN::StreamWrite(std::ostream &os) const {
  os << znval_ << " " << u1_val_ << " " << hash_;
}

inline void fZ2U1QN::Show(const size_t indent_level) const {
  std::cout << IndentPrinter(indent_level)
            << "fZ2U1QN: ("
            << znval_
            << ", "
            << u1_val_ << ") \n";
}

inline std::istream &operator>>(std::istream &is, fZ2U1QN &qn) {
  qn.StreamRead(is);
  return is;
}

inline std::ostream &operator<<(std::ostream &os, const fZ2U1QN &qn) {
  qn.StreamWrite(os);
  return os;
}


inline size_t fZ2U1QN::CalcHash_() const {
  size_t h = znval_ ^ (u1_val_ << 1);
  return _HASH_XXPRIME_1 ^ _HASH_XXROTATE(h);
}

}
} /* namespace qlten */

#endif /* ifndef QLTEN_QLTENSOR_SPECIAL_QN_fZ2U1QN_H */