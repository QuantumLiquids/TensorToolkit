// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-10-28 12:16
*
* Description: QuantumLiquids/tensor project. Abstract base class for quantum number value.
*/

/**
@file qnval.h
@brief Abstract base class for quantum number value.
*/
#ifndef QLTEN_QLTENSOR_QN_QNVAL_H
#define QLTEN_QLTENSOR_QN_QNVAL_H


#include "qlten/framework/bases/hashable.h"       // Hashable
#include "qlten/framework/bases/streamable.h"     // Streamable
#include "qlten/framework/bases/showable.h"       // Showable
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>

#include <vector>     // vector
#include <memory>     // shared_ptr


namespace qlten {


/**
Abstract base class for quantum number value.
*/
class QNVal : public Hashable, public Streamable, public Showable {
public:
  QNVal(void) = default;
  virtual ~QNVal(void) = default;

  /**
  Clone a quantum number value instance.

  @return A pointer which point to the new instance.
  */
  virtual QNVal *Clone(void) const = 0;

  /**
  Get the dimension of the representation labeled by this quantum number vaule.

  @return The dimension of the representation.
  */
  virtual size_t dim(void) const = 0;

  /**
  Get the value of the quantum number value.
  @return The value.
  */
  virtual int GetVal(void) const = 0;

  /**
  Calculate the value of the minus of the quantum number value.

  @return The minus of the quantum number value.
  */
  virtual QNVal *Minus(void) const = 0;

  /**
  Add the other quantum number value to this quantum number value.
  */
  virtual void AddAssign(const QNVal *) = 0;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version){;}
};

using QNValPtrVec = std::vector<QNVal *>;
using QNValSharedPtr = std::shared_ptr<QNVal>;
} /* qlten */
BOOST_SERIALIZATION_ASSUME_ABSTRACT(qlten::QNVal)
#endif /* ifndef QLTEN_QLTENSOR_QN_QNVAL_H */
