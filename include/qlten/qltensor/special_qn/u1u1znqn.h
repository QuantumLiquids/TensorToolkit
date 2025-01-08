// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2021-9-29
*
* Description: QuantumLiquids/tensor project. `U(1) \cross U(1) \cross Zn` QN class.
*/

/**
@file u1u1znqn.h
@brief `U(1) \cross U(1) \cross Zn` QN class = qlten::QN<U1QNVal, U1QNVal, ZnQNVal>.
 Avoiding virtual function realization. (except Showable).
*/

#ifndef QLTEN_QLTENSOR_SPECIAL_QN_U1U1ZNQN_H
#define QLTEN_QLTENSOR_SPECIAL_QN_U1U1ZNQN_H

#include "qlten/framework/vec_hash.h"   //_HASH_XXPRIME_1...
#include "qlten/qltensor/qn/qn.h"       //QNCardVec
#include "qlten/qltensor/qn/qnval.h"    //QNVal
#include "qlten/qltensor/qn/qnval_u1.h" //U1QNVal

namespace qlten {
namespace special_qn {

template<size_t n>
class U1U1ZnQN : public Showable {
 public:
  U1U1ZnQN(void);
  U1U1ZnQN(const int u1val1, const int u1val2, const int znval);
  U1U1ZnQN(const std::string &name1, const int u1val1,
           const std::string &name2, const int u1val2,
           const std::string &name3, const int znval
  );
  U1U1ZnQN(const U1U1ZnQN &);

  //Compatible
  U1U1ZnQN(const QNCardVec &qncards);

  U1U1ZnQN &operator=(const U1U1ZnQN &);

  ~U1U1ZnQN(void);

  U1U1ZnQN operator-(void) const;
  U1U1ZnQN &operator+=(const U1U1ZnQN &);

  U1U1ZnQN operator+(const U1U1ZnQN &rhs) const;
  U1U1ZnQN operator-(const U1U1ZnQN &rhs) const;

  size_t dim(void) const { return 1; }

  //Compatible
//  U1QNVal GetQNVal(const size_t idx) const {
//    if(idx < 2){
//      return U1QNVal(u1vals_[idx]);
//    }else{//idx=2
//      return ZnQNVal<n>(znval_);
//    }
//  }

  bool operator==(const U1U1ZnQN &rhs) const {
    return hash_ == rhs.hash_;
  }

  bool operator!=(const U1U1ZnQN &rhs) const {
    return !(*this == rhs);
  }

  //Hashable
  size_t Hash() const { return hash_; }

  void StreamRead(std::istream &);
  void StreamWrite(std::ostream &) const;

  void Show(const size_t indent_level = 0) const override;

  static U1U1ZnQN<n> Zero(void) {
    return U1U1ZnQN<n>(0, 0, 0);
  }

 private:
  size_t CalcHash_(void) const;

  int u1vals_[2];
  int znval_; //0,1,2,....,n-1
  size_t hash_;
};

template<size_t n>
U1U1ZnQN<n>::U1U1ZnQN(void) : u1vals_{0, 0}, znval_(0), hash_(CalcHash_()) {}

template<size_t n>
U1U1ZnQN<n>::U1U1ZnQN(const int u1val1, const int u1val2, const int znval) :
    u1vals_{u1val1, u1val2}, znval_(znval), hash_(CalcHash_()) {}

template<size_t n>
U1U1ZnQN<n>::U1U1ZnQN(const std::string &name1,
                      const int u1val1,
                      const std::string &name2,
                      const int u1val2,
                      const std::string &name3,
                      const int znval
) : u1vals_{u1val1, u1val2}, znval_(znval), hash_(CalcHash_()) {}

template<size_t n>
U1U1ZnQN<n>::U1U1ZnQN(const U1U1ZnQN<n> &rhs) :
    u1vals_{rhs.u1vals_[0], rhs.u1vals_[1]}, znval_(rhs.znval_), hash_(rhs.hash_) {}

template<size_t n>
inline U1U1ZnQN<n>::U1U1ZnQN(const QNCardVec &qncards) {
  assert(qncards.size() == 3);
  for (size_t i = 0; i < 2; i++) {
    const int val = qncards[i].GetValPtr()->GetVal();
    u1vals_[i] = val;
  }
  znval_ = qncards[3].GetValPtr()->GetVal();
  hash_ = CalcHash_();
}

template<size_t n>
U1U1ZnQN<n>::~U1U1ZnQN() {}

template<size_t n>
U1U1ZnQN<n> &U1U1ZnQN<n>::operator=(const U1U1ZnQN<n> &rhs) {
  for (size_t i = 0; i < 2; i++) {
    u1vals_[i] = rhs.u1vals_[i];
  }
  znval_ = rhs.znval_;
  hash_ = rhs.hash_;
  return *this;
}

template<size_t n>
U1U1ZnQN<n> U1U1ZnQN<n>::operator-() const {
  int minus_zn = (znval_ == 0) ? 0 : (n - znval_);
  return U1U1ZnQN(-u1vals_[0], -u1vals_[1], minus_zn);
}

template<size_t n>
U1U1ZnQN<n> &U1U1ZnQN<n>::operator+=(const U1U1ZnQN<n> &rhs) {
  u1vals_[0] += rhs.u1vals_[0];
  u1vals_[1] += rhs.u1vals_[1];
  znval_ += rhs.znval_;
  if (znval_ >= n) {
    znval_ = znval_ - n;
  }
  hash_ = CalcHash_();
  return *this;
}

template<size_t n>
U1U1ZnQN<n> U1U1ZnQN<n>::operator+(const U1U1ZnQN<n> &rhs) const {
  U1U1ZnQN<n> res(*this);
  res += rhs;
  return res;
}

template<size_t n>
inline U1U1ZnQN<n> U1U1ZnQN<n>::operator-(const U1U1ZnQN<n> &rhs) const {
  U1U1ZnQN<n> res(*this);
  res += (-rhs);
  return res;
}

template<size_t n>
void U1U1ZnQN<n>::StreamRead(std::istream &is) {
  is >> u1vals_[0];
  is >> u1vals_[1];
  is >> znval_;
  is >> hash_;
}

template<size_t n>
void U1U1ZnQN<n>::StreamWrite(std::ostream &os) const {
  os << u1vals_[0] << "\n"
     << u1vals_[1] << "\n"
     << znval_ << "\n"
     << hash_ << "\n";
}

template<size_t n>
void U1U1ZnQN<n>::Show(const size_t indent_level) const {
  std::cout << IndentPrinter(indent_level)
            << "U1U1ZnQN:  ("
            << u1vals_[0]
            << ", "
            << u1vals_[1]
            << ", "
            << znval_
            << ")"
            << "\n";
}

template<size_t n>
size_t U1U1ZnQN<n>::CalcHash_() const {
  ///< a faith realization compatible with general realization
  /*
  const size_t len = 3;
  size_t hash_val = _HASH_XXPRIME_5;
  for(size_t i = 0; i < 2; i++){
    const size_t item_hash_val = u1vals_[i];
    hash_val += item_hash_val * _HASH_XXPRIME_2;
    hash_val = _HASH_XXROTATE(hash_val);
    hash_val *= _HASH_XXPRIME_1;
  }
  const size_t item_hash_val = znval_;
  hash_val += item_hash_val * _HASH_XXPRIME_2;
  hash_val = _HASH_XXROTATE(hash_val);
  hash_val *= _HASH_XXPRIME_1;

  hash_val += len ^ _HASH_XXPRIME_5;
  return hash_val;
   */
  /** a simple realization
   * in 64 bit system size_t has 8 byte = 64 bits.
   * assume -2^20 < u1val < 2^20, 0<= znval < 2^20, a map is direct
   */
  const size_t segment_const = 1048576; //2^20
  size_t hash_val1 = u1vals_[0] + segment_const;
  size_t hash_val2 = u1vals_[1] + segment_const;
  hash_val2 *= (2 * segment_const);
  size_t hash_val3 = 4 * znval_ * segment_const * segment_const;
  size_t hash_val = hash_val1 + hash_val2 + hash_val3;
  return ((hash_val << 10) | (hash_val >> 54)); // To avoid collide of QNSector
}

template<size_t n>
inline std::istream &operator>>(std::istream &is, U1U1ZnQN<n> &qn) {
  qn.StreamRead(is);
  return is;
}

template<size_t n>
inline std::ostream &operator<<(std::ostream &os, const U1U1ZnQN<n> &qn) {
  qn.StreamWrite(os);
  return os;
}

template<size_t n>
inline size_t Hash(const U1U1ZnQN<n> &qn) { return qn.Hash(); }

}//special_qn
}//qlten

#endif //QLTEN_QLTENSOR_SPECIAL_QN_U1U1ZNQN_H
