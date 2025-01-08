// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2023-6-18
*
* Description: QuantumLiquids/tensor project. `U(1) \cross Zn` QN class, with dynamic value of n
*/


#ifndef QLTEN_QLTENSOR_SPECIAL_QN_U1ZQN_H
#define QLTEN_QLTENSOR_SPECIAL_QN_U1ZQN_H

#include "qlten/framework/vec_hash.h"   //_HASH_XXPRIME_1...
#include "qlten/qltensor/qn/qn.h"       //QNCardVec
#include "qlten/qltensor/qn/qnval.h"    //QNVal
#include "qlten/qltensor/qn/qnval_u1.h" //U1QNVal

namespace qlten {
namespace special_qn {

class U1ZQN : public Showable {
 public:
  U1ZQN(void);

  U1ZQN(const size_t n, const int u1val, const int znval);

  U1ZQN(const size_t n, const std::string &name1, const int u1val,
        const std::string &name2, const int znval
  );

  U1ZQN(const U1ZQN &);

  //Compatible
  U1ZQN(const size_t n, const QNCardVec &qncards);

  U1ZQN &operator=(const U1ZQN &);

  ~U1ZQN(void);

  U1ZQN operator-(void) const;

  U1ZQN &operator+=(const U1ZQN &);

  U1ZQN operator+(const U1ZQN &rhs) const;

  U1ZQN operator-(const U1ZQN &rhs) const;

  size_t dim(void) const { return 1; }

  bool operator==(const U1ZQN &rhs) const {
    assert(n == rhs.n);
    return hash_ == rhs.hash_;
  }

  bool operator!=(const U1ZQN &rhs) const {
    return !(*this == rhs);
  }

  //Hashable
  size_t Hash() const { return hash_; }

  void StreamRead(std::istream &);

  void StreamWrite(std::ostream &) const;

  void Show(const size_t indent_level = 0) const override;

  static U1ZQN Zero(size_t n) {
    return U1ZQN(n, 0, 0);
  }

 private:
  size_t CalcHash_(void) const;

  size_t n; //order the Zn group
  int u1val_;
  int znval_; //0,1,2,....,n-1
  size_t hash_;
};

inline U1ZQN::U1ZQN(void) : n(1), u1val_(0), znval_(0), hash_(CalcHash_()) {}

inline U1ZQN::U1ZQN(const size_t n, const int u1val, const int znval) :
    n(n), u1val_(u1val), znval_(znval), hash_(CalcHash_()) {}

inline U1ZQN::U1ZQN(const size_t n,
                    const std::string &name1,
                    const int u1val,
                    const std::string &name2,
                    const int znval
) : n(n), u1val_(u1val), znval_(znval), hash_(CalcHash_()) {}

inline U1ZQN::U1ZQN(const U1ZQN &rhs) :
    n(rhs.n), u1val_(rhs.u1val_), znval_(rhs.znval_), hash_(rhs.hash_) {}

inline U1ZQN::U1ZQN(const size_t n, const QNCardVec &qncards) :
    n(n) {
  assert(qncards.size() == 2);
  u1val_ = qncards[0].GetValPtr()->GetVal();
  znval_ = qncards[2].GetValPtr()->GetVal();
  hash_ = CalcHash_();
}

inline U1ZQN::~U1ZQN() {}

inline U1ZQN &U1ZQN::operator=(const U1ZQN &rhs) {
  n = rhs.n;
  u1val_ = rhs.u1val_;

  znval_ = rhs.znval_;
  hash_ = rhs.hash_;
  return *this;
}

inline U1ZQN U1ZQN::operator-() const {
  int minus_zn = (znval_ == 0) ? 0 : (n - znval_);
  return U1ZQN(n, -u1val_, minus_zn);
}

inline U1ZQN &U1ZQN::operator+=(const U1ZQN &rhs) {
  assert(n == rhs.n);
  u1val_ += rhs.u1val_;
  znval_ += rhs.znval_;
  if (znval_ >= (int) n) {
    znval_ = znval_ - n;
  }
  hash_ = CalcHash_();
  return *this;
}

inline U1ZQN U1ZQN::operator+(const U1ZQN &rhs) const {
  U1ZQN res(*this);
  res += rhs;
  return res;
}

inline U1ZQN U1ZQN::operator-(const U1ZQN &rhs) const {
  U1ZQN res(*this);
  res += (-rhs);
  return res;
}

inline void U1ZQN::StreamRead(std::istream &is) {
  is >> n;
  is >> u1val_;
  is >> znval_;
  is >> hash_;
}

inline void U1ZQN::StreamWrite(std::ostream &os) const {
  os << n << "\n"
     << u1val_ << "\n"
     << znval_ << "\n"
     << hash_ << "\n";
}

inline void U1ZQN::Show(const size_t indent_level) const {
  std::cout << IndentPrinter(indent_level)
            << "U1Z" << n << "QN:  ("
            << u1val_
            << ", "
            << znval_
            << ")"
            << "\n";
}

inline size_t U1ZQN::CalcHash_() const {
  const size_t segment_const = 1024 * 1024 * 1024; //2^30
  size_t hash_val1 = u1val_ + segment_const;
  size_t hash_val2 = znval_ * (2 * segment_const);
  size_t hash_val = hash_val1 + hash_val2;
  return ((hash_val << 10) | (hash_val >> 54)); // To avoid collide of QNSector
}

inline std::istream &operator>>(std::istream &is, U1ZQN &qn) {
  qn.StreamRead(is);
  return is;
}

inline std::ostream &operator<<(std::ostream &os, const U1ZQN &qn) {
  qn.StreamWrite(os);
  return os;
}

inline size_t Hash(const U1ZQN &qn) { return qn.Hash(); }

}//special_qn
}//qlten



#endif //QLTEN_QLTENSOR_SPECIAL_QN_U1ZQN_H
