// SPDX-License-Identifier: LGPL-3.0-only
/*
* Description: TensorToolkit public exception types.
*/
#ifndef QLTEN_FRAMEWORK_EXCEPTIONS_H
#define QLTEN_FRAMEWORK_EXCEPTIONS_H

#include <stdexcept>    // logic_error, runtime_error
#include <string>       // string

namespace qlten {

/**
Base class for TensorToolkit runtime failures.
*/
class QLTENRuntimeError : public std::runtime_error {
 public:
  explicit QLTENRuntimeError(const std::string &what_arg)
      : std::runtime_error(what_arg) {}

  explicit QLTENRuntimeError(const char *what_arg)
      : std::runtime_error(what_arg) {}
};

/**
Base class for TensorToolkit API misuse or invalid internal state errors.
*/
class QLTENLogicError : public std::logic_error {
 public:
  explicit QLTENLogicError(const std::string &what_arg)
      : std::logic_error(what_arg) {}

  explicit QLTENLogicError(const char *what_arg)
      : std::logic_error(what_arg) {}
};

/**
Thrown when blockwise tensor SVD produces no singular-value sectors.
*/
class EmptySVDResultError : public QLTENRuntimeError {
 public:
  explicit EmptySVDResultError(const std::string &what_arg)
      : QLTENRuntimeError(what_arg) {}

  explicit EmptySVDResultError(const char *what_arg)
      : QLTENRuntimeError(what_arg) {}
};

/**
Thrown when an Index stores a direction outside TenIndexDirType.
*/
class InvalidIndexDirectionError : public QLTENLogicError {
 public:
  explicit InvalidIndexDirectionError(const std::string &what_arg)
      : QLTENLogicError(what_arg) {}

  explicit InvalidIndexDirectionError(const char *what_arg)
      : QLTENLogicError(what_arg) {}
};

/**
Thrown when a DataBlk accessor is invalid for the quantum-number statistics.
*/
class InvalidDataBlkAccessError : public QLTENLogicError {
 public:
  explicit InvalidDataBlkAccessError(const std::string &what_arg)
      : QLTENLogicError(what_arg) {}

  explicit InvalidDataBlkAccessError(const char *what_arg)
      : QLTENLogicError(what_arg) {}
};

}  // namespace qlten

#endif /* ifndef QLTEN_FRAMEWORK_EXCEPTIONS_H */
