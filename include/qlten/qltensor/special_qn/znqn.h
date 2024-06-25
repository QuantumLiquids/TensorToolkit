/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2024-6-23
*
* Description: QuantumLiquids/tensor project. Z_n QN class.
*/

#ifndef QLTEN_QLTENSOR_SPECIAL_QN_ZNQN_H
#define QLTEN_QLTENSOR_SPECIAL_QN_ZNQN_H

namespace qlten {
namespace special_qn {

template<size_t n>
class ZnQN : public Showable {
 public:
  //Constructor
  ZnQN(void);
  ZnQN(const int znval);
  ZnQN(const ZnQN &);
  //Copy
  ZnQN &operator=(const ZnQN &);
  //Destructor
  ~ZnQN() {}
  //Overload
  ZnQN operator-(void) const {
    int minus_zn = (znval_ == 0) ? 0 : (n - znval_);
    return ZnQN(minus_zn);
  }
  ZnQN operator+(const ZnQN &rhs) const;
  ZnQN operator-(const ZnQN &rhs) const {
    return *this + (-rhs);
  }
  ZnQN &operator+=(const ZnQN &rhs);
  bool operator==(const ZnQN &rhs) const {
    return hash_ == rhs.hash_;
  }
  bool operator!=(const ZnQN &rhs) const {
    return !(*this == rhs);
  }
  size_t Hash() const { return hash_; }
  size_t dim(void) const { return 1; }

  void StreamRead(std::istream &);
  void StreamWrite(std::ostream &) const;
  void Show(const size_t indent_level = 0) const override;

  static ZnQN<n> Zero(void) {
    return ZnQN(0);
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
};

template<size_t n>
ZnQN<n>::ZnQN(void) :ZnQN(0) {}

template<size_t n>
ZnQN<n>::ZnQN(const int znval_) : znval_(znval_), hash_(CalcHash_()) {}

template<size_t n>
ZnQN<n>::ZnQN(const ZnQN<n> &rhs): znval_(rhs.znval_), hash_(rhs.hash_) {}

template<size_t n>
ZnQN<n> &ZnQN<n>::operator=(const ZnQN<n> &rhs) {
  znval_ = rhs.znval_;
  hash_ = rhs.hash_;
  return (*this);
}

template<size_t n>
ZnQN<n> ZnQN<n>::operator+(const ZnQN<n> &rhs) const {
  return ZnQN<n>((this->znval_ + rhs.znval_) % n);
}

template<size_t n>
ZnQN<n> &ZnQN<n>::operator+=(const ZnQN<n> &rhs) {
  znval_ += rhs.znval_;
  if (znval_ >= n) {
    znval_ = znval_ - n;
  }
  hash_ = CalcHash_();
  return *this;
}
template<size_t n>
void ZnQN<n>::StreamRead(std::istream &is) {
  is >> znval_;
  is >> hash_;
}

template<size_t n>
void ZnQN<n>::StreamWrite(std::ostream &os) const {
  os << znval_ << "\n"
     << hash_ << "\n";
}

template<size_t n>
void ZnQN<n>::Show(const size_t indent_level) const {
  std::cout << IndentPrinter(indent_level)
            << "ZnQN:  ("
            << znval_
            << ")"
            << "\n";
}

template<size_t n>
size_t ZnQN<n>::CalcHash_() const {
  return znval_;
}

template<size_t n>
inline std::istream &operator>>(std::istream &is, ZnQN<n> &qn) {
  qn.StreamRead(is);
  return is;
}

template<size_t n>
inline std::ostream &operator<<(std::ostream &os, const ZnQN<n> &qn) {
  qn.StreamWrite(os);
  return os;
}

template<size_t n>
inline size_t Hash(const ZnQN<n> &qn) { return qn.Hash(); }

using Z2QN = ZnQN<2>;

}//special_qn
}//qlten
#endif //QLTEN_QLTENSOR_SPECIAL_QN_ZNQN_H
