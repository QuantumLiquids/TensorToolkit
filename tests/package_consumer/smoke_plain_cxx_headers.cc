#include "qlten/qlten.h"

#include <type_traits>

namespace {

using QNT = qlten::special_qn::U1QN;
using Tensor = qlten::QLTensor<qlten::QLTEN_Complex, QNT>;

static_assert(std::is_same<typename Tensor::value_type, qlten::QLTEN_Complex>::value,
              "A plain C++ consumer should compile qlten/qlten.h from a GPU package.");

}  // namespace

int main() {
  return 0;
}
