/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2024-6-23
*
* Description: QuantumLiquids/tensor project. Abstract base class for boosting classes to fermion class
* The class who inherits the Fermionicable class can be boosted to fermion class
*/


#ifndef QLTEN_FRAMEWORK_BASES_FERMIONICABLE_H
#define QLTEN_FRAMEWORK_BASES_FERMIONICABLE_H

#include <type_traits>
namespace qlten {

enum ParticleStatistics {
  Bosonic,
  Fermionic
};

/**
 * usage:
 * class A, which represent Tensor, or Index can inherites Fermionicable class,
 * and justify its particle statistics by
 * A::IsFermionic().
 * e.g.
 * static_assert(A::IsFermionic(), "A is a fermionic object");
 *
 * @tparam QNT
 */
template<typename QNT>
class Fermionicable {
 public:

  static constexpr bool IsFermionic() {
    return decltype(HasIsFermionParityOdd<QNT>(0))::value;
  }

  static constexpr ParticleStatistics Statistics() {
    if (IsFermionic()) {
      return Fermionic;
    } else {
      return Bosonic;
    }
  }
 private:
  // Helper struct to detect if QNT has a member function IsFermionParityOdd
  template<typename T>
  static auto HasIsFermionParityOdd(int) -> decltype(std::declval<T>().IsFermionParityOdd(), std::true_type());

  template<typename T>
  static std::false_type HasIsFermionParityOdd(...);

};

}
#endif //QLTEN_FRAMEWORK_BASES_FERMIONICABLE_H
