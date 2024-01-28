// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-10-28 09:13
*
* Description: QuantumLiquids/tensor project. Abstract base class for streamable object.
*/

/**
@file streamable.h
@brief Abstract base class for streamable object.
*/
#ifndef QLTEN_FRAMEWORK_BASES_STREAMABLE_H
#define QLTEN_FRAMEWORK_BASES_STREAMABLE_H


#include <iostream>     // istream, ostream


namespace qlten {


/**
Abstract base class for streamable object.
*/
class Streamable {
public:
  Streamable(void) = default;
  virtual ~Streamable(void) = default;

  /// Read from a stream.
  virtual void StreamRead(std::istream &) = 0;

  /// Write to a stream.
  virtual void StreamWrite(std::ostream &) const = 0;
};


/**
Overload input stream operator for streamable object.

@param is Input stream.
@param obj Streamable object.
*/
inline std::istream &operator>>(std::istream &is, Streamable &obj) {
  obj.StreamRead(is);
  return is;
}


/**
Overload output stream operator for streamable object.

@param os Output stream.
@param obj Streamable object.
*/
inline std::ostream &operator<<(std::ostream &os, const Streamable &obj) {
  obj.StreamWrite(os);
  return os;
}
} /* qlten */
#endif /* ifndef QLTEN_FRAMEWORK_BASES_STREAMABLE_H */
