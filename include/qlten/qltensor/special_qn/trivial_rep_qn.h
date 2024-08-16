// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2024-8-16
*
* Description: QuantumLiquids/tensor project. QN class for trivial representations.
*/

#ifndef QLTEN_QLTENSOR_SPECIAL_QN_TRIVIAL_REP_QN_H
#define QLTEN_QLTENSOR_SPECIAL_QN_TRIVIAL_REP_QN_H

#include "qlten/framework/vec_hash.h"   //_HASH_XXPRIME_1...
#include "qlten/qltensor/qn/qnval.h"    //QNVal
#include <boost/serialization/serialization.hpp>

namespace qlten {
namespace special_qn {

//Showable
class TrivialRepQN : public Showable {
 public:
  TrivialRepQN(void);
  TrivialRepQN(const TrivialRepQN &);

  TrivialRepQN &operator=(const TrivialRepQN &);

  ~TrivialRepQN(void) override;

  TrivialRepQN operator-(void) const;
  TrivialRepQN &operator+=(const TrivialRepQN &);

  TrivialRepQN operator+(const TrivialRepQN &rhs) const;
  TrivialRepQN operator-(const TrivialRepQN &rhs) const;

  static size_t dim(void) { return 1; }

  bool operator==(const TrivialRepQN &rhs) const {
    return true;
  }

  bool operator!=(const TrivialRepQN &rhs) const {
    return !(*this == rhs);
  }

  //Hashable
  static size_t Hash() { return 0; }

  void StreamRead(std::istream &);
  void StreamWrite(std::ostream &) const;

  void Show(const size_t indent_level = 0) const override;

  //Generate QN 0 element
  static TrivialRepQN Zero(void) {
    return {};
  }

 private:

  static size_t CalcHash_(void) ;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version) {
  }
};

inline TrivialRepQN::TrivialRepQN(void) {}

inline TrivialRepQN::TrivialRepQN(const TrivialRepQN &rhs) {}

inline TrivialRepQN::~TrivialRepQN() {}

inline TrivialRepQN &TrivialRepQN::operator=(const TrivialRepQN &rhs) {
  return *this;
}

inline TrivialRepQN TrivialRepQN::operator-() const {
  return {};
}

inline TrivialRepQN &TrivialRepQN::operator+=(const TrivialRepQN &rhs) {
  return *this;
}

inline TrivialRepQN TrivialRepQN::operator+(const TrivialRepQN &rhs) const {
  return {};
}

inline TrivialRepQN TrivialRepQN::operator-(const TrivialRepQN &rhs) const {
  return {};
}

inline void TrivialRepQN::StreamRead(std::istream &is) {
}

inline void TrivialRepQN::StreamWrite(std::ostream &os) const {
}

inline void TrivialRepQN::Show(const size_t indent_level) const {
  std::cout << IndentPrinter(indent_level) << "TrivialRepQN." << "\n";
}

inline size_t TrivialRepQN::CalcHash_() {
  return 0;
}

inline std::istream &operator>>(std::istream &is, TrivialRepQN &qn) {
  qn.StreamRead(is);
  return is;
}

inline std::ostream &operator<<(std::ostream &os, const TrivialRepQN &qn) {
  qn.StreamWrite(os);
  return os;
}

inline size_t Hash(const TrivialRepQN &qn) { return qn.Hash(); }

}//special_qn
}//qlten
#endif //QLTEN_QLTENSOR_SPECIAL_QN_TRIVIAL_REP_QN_H
