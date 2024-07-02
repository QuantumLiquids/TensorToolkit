/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2024-6-23
*
* Description: QuantumLiquids/tensor project. fermionic Z_2 QN class.
*/

/**
@file fz2qn.h
@brief Fermionic Z2 QN class which has additional attribution function IsFermionParityOdd()
       serving as the marker of fermionic quantum numbers.
*/

#ifndef QLTEN_QLTENSOR_SPECIAL_QN_fZ2QN_H
#define QLTEN_QLTENSOR_SPECIAL_QN_fZ2QN_H

namespace qlten {
namespace special_qn {

class fZ2QN : public Showable {
 public:
  //Constructor
  fZ2QN(void);
  fZ2QN(const int znval);
  fZ2QN(const fZ2QN &);
  //Copy
  fZ2QN &operator=(const fZ2QN &);
  //Destructor
  ~fZ2QN() {}
  //Overload
  fZ2QN operator-(void) const {
    int minus_zn = (znval_ == 0) ? 0 : (n - znval_);
    return fZ2QN(minus_zn);
  }
  fZ2QN operator+(const fZ2QN &rhs) const;
  fZ2QN operator-(const fZ2QN &rhs) const {
    return *this + (-rhs);
  }
  fZ2QN &operator+=(const fZ2QN &rhs);
  bool operator==(const fZ2QN &rhs) const {
    return hash_ == rhs.hash_;
  }
  bool operator!=(const fZ2QN &rhs) const {
    return !(*this == rhs);
  }
  size_t Hash() const { return hash_; }
  size_t dim(void) const { return 1; }

  //Marker for fermionic quantum number
  bool IsFermionParityOdd() const { return znval_; }
  bool IsFermionParityEven() const { return !(IsFermionParityOdd()); }

  void StreamRead(std::istream &);
  void StreamWrite(std::ostream &) const;
  void Show(const size_t indent_level = 0) const override;

  static fZ2QN Zero(void) {
    return fZ2QN(0);
  }
 private:
  size_t CalcHash_(void) const;
  int znval_;
  size_t hash_;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version) {
    ar & znval_;
    ar & hash_;
  }
  static const size_t n = 2;
};

inline fZ2QN::fZ2QN(void) : fZ2QN(0) {}

inline fZ2QN::fZ2QN(const int znval_) : znval_(znval_), hash_(CalcHash_()) {}

inline fZ2QN::fZ2QN(const fZ2QN &rhs) : znval_(rhs.znval_), hash_(rhs.hash_) {}

inline fZ2QN &fZ2QN::operator=(const fZ2QN &rhs) {
  znval_ = rhs.znval_;
  hash_ = rhs.hash_;
  return (*this);
}

inline fZ2QN fZ2QN::operator+(const fZ2QN &rhs) const {
  return fZ2QN((this->znval_ + rhs.znval_) % n);
}

inline fZ2QN &fZ2QN::operator+=(const fZ2QN &rhs) {
  znval_ += rhs.znval_;
  if (znval_ >= n) {
    znval_ = znval_ - n;
  }
  hash_ = CalcHash_();
  return *this;
}

inline void fZ2QN::StreamRead(std::istream &is) {
  is >> znval_;
  is >> hash_;
}

inline void fZ2QN::StreamWrite(std::ostream &os) const {
  os << znval_ << "\n"
     << hash_ << "\n";
}

inline void fZ2QN::Show(const size_t indent_level) const {
  std::cout << IndentPrinter(indent_level)
            << "fZ2QN:  ("
            << znval_
            << ")"
            << "\n";
}

inline size_t fZ2QN::CalcHash_() const {
  return znval_;
}

inline std::istream &operator>>(std::istream &is, fZ2QN &qn) {
  qn.StreamRead(is);
  return is;
}

inline std::ostream &operator<<(std::ostream &os, const fZ2QN &qn) {
  qn.StreamWrite(os);
  return os;
}

inline size_t Hash(const fZ2QN &qn) { return qn.Hash(); }

}
}

#endif //QLTEN_QLTENSOR_SPECIAL_QN_fZ2QN_H
