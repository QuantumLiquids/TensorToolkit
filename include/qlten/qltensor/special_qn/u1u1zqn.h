// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2023-6-18
*
* Description: QuantumLiquids/tensor project. `U(1) \cross U(1) \cross Zn` QN class, with dynamic value of n
*/


#ifndef QLTEN_QLTENSOR_SPECIAL_QN_U1U1ZQN_H
#define QLTEN_QLTENSOR_SPECIAL_QN_U1U1ZQN_H

#include "qlten/framework/vec_hash.h"   //_HASH_XXPRIME_1...
#include "qlten/qltensor/qn/qn.h"       //QNCardVec
#include "qlten/qltensor/qn/qnval.h"    //QNVal
#include "qlten/qltensor/qn/qnval_u1.h" //U1QNVal

namespace qlten {
namespace special_qn {

class U1U1ZQN : public Showable {
 public:
  U1U1ZQN(void);
  U1U1ZQN(const size_t n, const int u1val1, const int u1val2, const int znval);
  U1U1ZQN(const size_t n, const std::string &name1, const int u1val1,
          const std::string &name2, const int u1val2,
          const std::string &name3, const int znval
  );
  U1U1ZQN(const U1U1ZQN &);

  //Compatible
  U1U1ZQN(const size_t n, const QNCardVec &qncards);

  U1U1ZQN &operator=(const U1U1ZQN &);

  ~U1U1ZQN(void);

  U1U1ZQN operator-(void) const;
  U1U1ZQN &operator+=(const U1U1ZQN &);

  U1U1ZQN operator+(const U1U1ZQN &rhs) const;
  U1U1ZQN operator-(const U1U1ZQN &rhs) const;

  size_t dim(void) const { return 1; }

  //Compatible
//  U1QNVal GetQNVal(const size_t idx) const {
//    if(idx < 2){
//      return U1QNVal(u1vals_[idx]);
//    }else{//idx=2
//      return ZnQNVal<n>(znval_);
//    }
//  }

  bool operator==(const U1U1ZQN &rhs) const {
    assert(n == rhs.n);
    return hash_ == rhs.hash_;
  }

  bool operator!=(const U1U1ZQN &rhs) const {
    return !(*this == rhs);
  }

  //Hashable
  size_t Hash() const { return hash_; }

  void StreamRead(std::istream &);
  void StreamWrite(std::ostream &) const;

  void Show(const size_t indent_level = 0) const override;

  static U1U1ZQN Zero(size_t n) {
    return U1U1ZQN(n, 0, 0, 0);
  }

 private:
  size_t CalcHash_(void) const;

  size_t n; //order the Zn group
  int u1vals_[2];
  int znval_; //0,1,2,....,n-1
  size_t hash_;
};

inline U1U1ZQN::U1U1ZQN(void) : n(1), u1vals_{0, 0}, znval_(0), hash_(CalcHash_()) {}

inline U1U1ZQN::U1U1ZQN(const size_t n, const int u1val1, const int u1val2, const int znval) :
    n(n), u1vals_{u1val1, u1val2}, znval_(znval), hash_(CalcHash_()) {}

inline U1U1ZQN::U1U1ZQN(const size_t n,
                        const std::string &name1,
                        const int u1val1,
                        const std::string &name2,
                        const int u1val2,
                        const std::string &name3,
                        const int znval
) : n(n), u1vals_{u1val1, u1val2}, znval_(znval), hash_(CalcHash_()) {}

inline U1U1ZQN::U1U1ZQN(const U1U1ZQN &rhs) :
    n(rhs.n), u1vals_{rhs.u1vals_[0], rhs.u1vals_[1]}, znval_(rhs.znval_), hash_(rhs.hash_) {}

inline U1U1ZQN::U1U1ZQN(const size_t n, const QNCardVec &qncards) :
    n(n) {
  assert(qncards.size() == 3);
  for (size_t i = 0; i < 2; i++) {
    const int val = qncards[i].GetValPtr()->GetVal();
    u1vals_[i] = val;
  }
  znval_ = qncards[3].GetValPtr()->GetVal();
  hash_ = CalcHash_();
}

inline U1U1ZQN::~U1U1ZQN() {}

inline U1U1ZQN &U1U1ZQN::operator=(const U1U1ZQN &rhs) {
  n = rhs.n;
  for (size_t i = 0; i < 2; i++) {
    u1vals_[i] = rhs.u1vals_[i];
  }
  znval_ = rhs.znval_;
  hash_ = rhs.hash_;
  return *this;
}

inline U1U1ZQN U1U1ZQN::operator-() const {
  int minus_zn = (znval_ == 0) ? 0 : (n - znval_);
  return U1U1ZQN(n, -u1vals_[0], -u1vals_[1], minus_zn);
}

inline U1U1ZQN &U1U1ZQN::operator+=(const U1U1ZQN &rhs) {
  assert(n == rhs.n);
  u1vals_[0] += rhs.u1vals_[0];
  u1vals_[1] += rhs.u1vals_[1];
  znval_ += rhs.znval_;
  if (znval_ >= (int) n) {
    znval_ = znval_ - n;
  }
  hash_ = CalcHash_();
  return *this;
}

inline U1U1ZQN U1U1ZQN::operator+(const U1U1ZQN &rhs) const {
  U1U1ZQN res(*this);
  res += rhs;
  return res;
}

inline U1U1ZQN U1U1ZQN::operator-(const U1U1ZQN &rhs) const {
  U1U1ZQN res(*this);
  res += (-rhs);
  return res;
}

inline void U1U1ZQN::StreamRead(std::istream &is) {
  is >> n;
  is >> u1vals_[0];
  is >> u1vals_[1];
  is >> znval_;
  is >> hash_;
}

inline void U1U1ZQN::StreamWrite(std::ostream &os) const {
  os << n << "\n"
     << u1vals_[0] << "\n"
     << u1vals_[1] << "\n"
     << znval_ << "\n"
     << hash_ << "\n";
}

inline void U1U1ZQN::Show(const size_t indent_level) const {
  std::cout << IndentPrinter(indent_level)
            << "U1U1Z" << n << "QN:  ("
            << u1vals_[0]
            << ", "
            << u1vals_[1]
            << ", "
            << znval_
            << ")"
            << "\n";
}

inline size_t U1U1ZQN::CalcHash_() const {
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

inline std::istream &operator>>(std::istream &is, U1U1ZQN &qn) {
  qn.StreamRead(is);
  return is;
}

inline std::ostream &operator<<(std::ostream &os, const U1U1ZQN &qn) {
  qn.StreamWrite(os);
  return os;
}

inline size_t Hash(const U1U1ZQN &qn) { return qn.Hash(); }

}//special_qn
}//qlten



#endif //QLTEN_QLTENSOR_SPECIAL_QN_U1U1ZQN_H
