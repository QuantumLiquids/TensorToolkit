// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2024-6-23
*
* Description: QuantumLiquids/tensor project. Fermionic U(1)_charge \prod U(1)_spin QN class.
*/

/**
@file fu1u1qn.h
@brief Fermionic U(1)_charge \prod U(1)_spin QN class.
*/

#ifndef QLTEN_QLTENSOR_SPECIAL_QN_FfU1U1QN_H
#define QLTEN_QLTENSOR_SPECIAL_QN_FfU1U1QN_H

#include "qlten/qltensor/special_qn/u1u1qn.h"

namespace qlten {
namespace special_qn {

/// The first u1 val mush be the particle number.
class fU1U1QN : public U1U1QN {
 public:
  bool IsFermionParityOdd() const { return vals_[0] % 2; }
  bool IsFermionParityEven() const { return !(IsFermionParityOdd()); }
};
}//special_qn
}//qlten
#endif //QLTEN_QLTENSOR_SPECIAL_QN_FfU1U1QN_H
