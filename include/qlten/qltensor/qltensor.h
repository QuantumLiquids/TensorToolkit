// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-13 16:18
*
* Description: QuantumLiquids/tensor project. Symmetry-blocked sparse tensor class and
* some short inline helper functions.
*/

/**
@file qltensor.h
@brief Symmetry-blocked sparse tensor class and some short inline helper functions.
*/
#ifndef QLTEN_QLTENSOR_QLTENSOR_H
#define QLTEN_QLTENSOR_QLTENSOR_H

#include "qlten/framework/value_t.h"                                // CoorsT, ShapeT, QLTEN_Double, QLTEN_Complex
#include "qlten/framework/bases/streamable.h"                       // Streamable
#include "qlten/framework/bases/showable.h"                         // Showable
#include "qlten/qltensor/index.h"                                   // IndexVec
#include "qlten/qltensor/blk_spar_data_ten/blk_spar_data_ten.h"     // BlockSparseDataTensor

#include <vector>       // vector
#include <iostream>     // istream, ostream
namespace qlten {

/**
 * Symmetry-blocked sparse tensor.
 *
 * @tparam ElemT Type of the tensor element, QLTEN_Double or QLTEN_Complex
 * @tparam QNT   Type of the quantum number used for symmetry blocking
 *
 * @warning **CUDA Lifetime Warning**: Objects of this class manage CUDA memory resources.
 *          Declaring global/static instances may lead to undefined behavior due to destruction
 *          order issues. The CUDA driver typically shuts down before global destructors run,
 *          making cudaFree() calls in destructors unsafe. Prefer stack/heap allocation within
 *          main()'s scope.
 *
 * @note **Safe/Unsafe Usage Pattern**:
 * @code
 * #define USE_GPU 1
 * QLTensor<float, QNT> global_tensor(indexes);   // Unsafe: may be destroyed after CUDA shutdown
 * int main() {
 *     QLTensor<float, QNT> local_tensor(indexes);  // Safe: destroyed before CUDA shutdown
 *     // ... use tensors ...
 *     return 0; // Destructor of local_tensor runs here while CUDA is active
 * }  // Destructor of global_tensor runs here while CUDA is not active
 * @endcode
 */
template<typename ElemT, typename QNT>
class QLTensor : public Showable, public Fermionicable<QNT> {
 public:
  // Constructors and destructor.
  /**
   * @brief Default constructor.
   *
   * Constructs a default (empty) tensor. A default tensor has rank 0 and size 0
   * and does not own any underlying block-sparse storage until constructed with
   * indices.
   *
   * @note Many member functions require a non-default tensor; assertions will
   *       trigger if they are invoked on a default instance.
   */
  QLTensor(void) = default;

  /**
   * @brief Construct an empty tensor with the provided indices.
   *
   * Initializes internal block-sparse storage consistent with the symmetry
   * sectors described by `indexes`.
   *
   * @param indexes Indices (with quantum-number sectors and directions) that
   *                define the tensor hilbert space.
   */
  QLTensor(const IndexVec<QNT> &);

  /**
   * @brief Construct an empty tensor by moving the provided indices.
   * @param indexes Rvalue-reference to indices to take ownership of.
   */
  QLTensor(const IndexVec<QNT> &&);

  /**
   * @brief Construct an empty tensor by moving the provided indices.
   * @param indexes Indices to move into this tensor.
   */
  QLTensor(IndexVec<QNT> &&);

  /**
   * @brief Copy constructor.
   * @param other Source tensor to copy from.
   */
  QLTensor(const QLTensor &);

  /**
   * @brief Copy assignment.
   * @param rhs Source tensor to copy from.
   * @return Reference to this tensor after assignment.
   */
  QLTensor &operator=(const QLTensor &);

  /**
   * @brief Move constructor.
   * @param other Source tensor to move from. `other` will be left in a valid
   *              but unspecified state.
   */
  QLTensor(QLTensor &&) noexcept;

  /**
   * @brief Move assignment.
   * @param rhs Source tensor to move from.
   * @return Reference to this tensor after assignment.
   */
  QLTensor &operator=(QLTensor &&) noexcept;

  /**
   * @brief Destructor. Releases block-sparse storage if allocated.
   * @note See the class-level CUDA lifetime warning regarding global/static
   *       objects when GPU is enabled.
   */
  ~QLTensor(void) { delete pblk_spar_data_ten_; }

  // Get or check basic properties of the QLTensor.
  /** @brief Get the rank (number of indices). */
  size_t Rank(void) const { return rank_; }

  /** @brief Get the total number of elements (product of dimensions). */
  size_t size(void) const { return size_; }

  /** @brief Get all indices of the tensor. */
  const IndexVec<QNT> &GetIndexes(void) const { return indexes_; }

  /**
   * @brief Get a single index by position.
   * @param i Zero-based index position.
   */
  const Index<QNT> &GetIndex(const size_t i) const { return indexes_[i]; }

  /** @brief Get the tensor shape (dimensions of each index). */
  const ShapeT &GetShape(void) const { return shape_; }

  /**
   * @brief Sparsity ratio of the underlying block-sparse data.
   * @return Actual nonzero element count divided by total size.
   */
  double Sparsity(void) const { return double(pblk_spar_data_ten_->GetActualRawDataSize()) / (double) size_; }
  /** @brief Number of quantum-number blocks contained by this tensor. */
  size_t GetQNBlkNum(void) const {
    if (IsScalar() || IsDefault()) { return 0; }
    return pblk_spar_data_ten_->GetBlkIdxDataBlkMap().size();
  }

  /** @brief Access the underlying block-sparse data tensor. */
  BlockSparseDataTensor<ElemT, QNT> &GetBlkSparDataTen(void) {
    return *pblk_spar_data_ten_;
  }

  /** @brief Const access to the underlying block-sparse data tensor. */
  const BlockSparseDataTensor<ElemT, QNT> &GetBlkSparDataTen(void) const {
    return *pblk_spar_data_ten_;
  }

  /** @brief Raw pointer to the underlying block-sparse data tensor. */
  const BlockSparseDataTensor<ElemT, QNT> *GetBlkSparDataTenPtr(void) const {
    return pblk_spar_data_ten_;
  }

  /** @brief Check whether this tensor is a scalar (rank 0, size 1). */
  bool IsScalar(void) const { return (rank_ == 0) && (size_ == 1); }

  /** @brief Check whether this tensor is default-constructed (no storage). */
  bool IsDefault(void) const { return size_ == 0; }

  // Calculate properties of the QLTensor.
  /**
   * @brief Compute the quantum-number divergence of this tensor.
   *        Defined as minus the sum of IN-sector QNs plus the sum of OUT-sector QNs
   *        for a representative non-empty block.
   * @return The divergence value aggregated from index directions and sectors.
   * @note For non-scalar tensors without blocks or with inconsistent block
   *       divergence, an empty quantum number may be returned in debug mode.
   */
  QNT Div(void) const;

  // Element getter and setter.
  /**
   * @brief Get an element by its coordinate in the full (dense) space.
   * @param coors Coordinates vector (size equals rank). Use empty vector for scalar.
   * @return The element value.
   */
  ElemT GetElem(const std::vector<size_t> &) const;

  /**
   * @brief Set an element by its coordinate in the full (dense) space.
   * @param coors Coordinates vector (size equals rank). Use empty vector for scalar.
   * @param value Value to assign.
   */
  void SetElem(const std::vector<size_t> &, const ElemT);

  struct QLTensorElementAccessDeref;

  /**
   * @brief Element access proxy for chained assignment or read.
   * @param coors Coordinates vector.
   */
  QLTensorElementAccessDeref operator()(const std::vector<size_t> &);

  QLTensorElementAccessDeref operator()(const std::vector<size_t> &) const;

  /**
   * @brief Element access proxy for scalar tensor.
   */
  QLTensorElementAccessDeref operator()(void);

  QLTensorElementAccessDeref operator()(void) const;

  template<typename... OtherCoorsT>
  /**
   * @brief Variadic element access for convenience on non-scalar tensors.
   * @param coor0 First coordinate.
   * @param other_coors Remaining coordinates.
   */
  QLTensorElementAccessDeref operator()(const size_t, const OtherCoorsT...);

  template<typename... OtherCoorsT>
  QLTensorElementAccessDeref operator()(
      const size_t,
      const OtherCoorsT...
  ) const;

  // Inplace operations.
  /**
   * @brief Fill tensor with random values in [0,1] consistent with a target divergence.
   * @param div Target quantum-number divergence to respect when populating blocks.
   */
  void Random(const QNT &);

  /**
   * @brief Fill tensor with a constant value on all allowed blocks consistent with a target divergence.
   * @param div Target quantum-number divergence.
   * @param value Value to assign.
   */
  void Fill(const QNT &, const ElemT &);

  /**
   * @brief Transpose the tensor indices in-place.
   * @param transed_idxes_order New order of indices (a permutation of 0..rank-1).
   */
  void Transpose(const std::vector<size_t> &);

  /**
   * @brief Fuse two adjacent indices in-place.
   * @param left_axis The left index position to fuse.
   * @param right_axis The right index position to fuse.
   */
  void FuseIndex(const size_t, const size_t);

  /** @brief 2-norm of the tensor.
   *
   * Definitions :
   * - Bosonic tensors: \( \|A\|_2 = \sqrt{\sum_i |a_i|^2} \).
   * - Fermionic tensors (Z2-graded, let E/O denote even/odd fermionic parity blocks):
   *   \[
   *   \|A\|_{2,\mathrm{graded}} = \sqrt{\sum_{i\in E} |a_i|^2\; - \; \sum_{i\in O} |a_i|^2}.
   *   \]
   *   This graded norm can be ill-defined if the odd contribution exceeds the even one,
   *   in which case the quantity under the square root becomes negative.
   *
   * This definition is consistent with `sqrt(Contract(A, A.Dag()))`.
   * @note For fermionic tensors, prefer `GetQuasi2Norm()` or `QuasiNormalize()` when you
   *       need a conventional non-negative norm.
   */
  QLTEN_Double Get2Norm(void) const;

  /** @brief Quasi 2-norm of the tensor.
   * For both bosonic and fermionic tensors,
   * \( \|A\|_{2,\mathrm{quasi}} = \sqrt{\sum_i |a_i|^2} \).
   * This is always non-negative and coincides with the standard Euclidean norm.
   */
  QLTEN_Double GetQuasi2Norm(void) const;

  /**
   * @brief Normalize tensor by its 2-norm, returning the original norm.
   * @return Norm before normalization.
   * @note The definition of 2-norm of fermionic tensors refer to Get2Norm().
   *       If the graded norm is ill-defined (negative under the square root), 
   *       the result may be NaN. In such cases prefer QuasiNormalize().
   */
  QLTEN_Double Normalize(void);

  /** @brief Rescale the tensor so that \(\sum_i |a_i|^2 = 1\) (uses quasi 2-norm) */
  QLTEN_Double QuasiNormalize(void);

  /**
   * @brief Hermitian conjugate: invert index directions and conjugate elements (if complex).
   */
  void Dag(void);

  ///< act the fermion P operators
  void ActFermionPOps(void);

  /** @brief In-place element-wise inverse with tolerance 0. */
  void ElementWiseInv(void);

  /** @brief In-place element-wise inverse with tolerance for near-zeros. */
  void ElementWiseInv(double tolerance);

  /** @brief In-place element-wise product with another tensor (same indices). */
  void ElementWiseMultiply(const QLTensor &);

  /** @brief In-place inverse of a diagonal matrix stored as a tensor.
   * The input tensor should be a matrix form, that means the number of indices is 2, and the two indices are the same except direction.
   */
  void DiagMatInv(void);

  /** @brief In-place inverse of a diagonal matrix with tolerance.
   * The input tensor should be a matrix form, that means the number of indices is 2, and the two indices are the same except direction.
   */
  void DiagMatInv(double tolerance);

  /** @brief In-place element-wise square-root. */
  void ElementWiseSqrt(void);

  /** @brief In-place element-wise square. */
  void ElementWiseSquare(void);

  /** @brief In-place element-wise sign.
   * For real elements: sign(x) in {-1, 0, 1}.
   * For complex elements: preserves the complex phase, maps magnitude to 1 (0 if x==0).
   */
  void ElementWiseSign(void);

  /// @brief bound the tensor element to the bound
  /// bound of real number is sign * bound,
  /// bound of complex number is the same angle of complex number but the magnitude is bound
  void ElementWiseBoundTo(double bound);

  template<typename RandGenerator>
  /**
   * @brief In-place randomize elements using a RNG but keep the sign of the elements.
   * @param dist Uniform real distribution in [0,1) or similar.
   * @param g Random generator instance.
   */
  void ElementWiseRandomizeMagnitudePreservePhase(std::uniform_real_distribution<double> &dist,
                           RandGenerator &g);

  /** @brief Maximum absolute value among stored (non-zero) elements. */
  double GetMaxAbs(void) const {
    return pblk_spar_data_ten_->GetMaxAbs();
  }

  /** @brief First non-zero element encountered in storage order. */
  ElemT GetFirstNonZeroElement(void) const { return pblk_spar_data_ten_->GetFirstNonZeroElement(); }

  // Operators overload.
  /**
   * @brief Equality comparison: checks indices and block-sparse contents.
   */
  bool operator==(const QLTensor &) const;

  bool operator!=(const QLTensor &rhs) const { return !(*this == rhs); }

  /** @brief Unary minus: returns a copy scaled by -1. */
  QLTensor operator-(void) const;

  /** @brief Tensor addition. Indices must match exactly. */
  QLTensor operator+(const QLTensor &) const;

  /** @brief In-place tensor addition. Indices must match exactly. */
  QLTensor &operator+=(const QLTensor &);

  /** @brief Scale by a scalar value, returning a new tensor. */
  QLTensor operator*(const ElemT) const;

  /** @brief In-place scale by a scalar value. */
  QLTensor &operator*=(const ElemT);

  /**
   * @brief Remove all trivial (dimension-1) indices by contracting with a
   *        unit scale tensor, preserving data.
   * @return Reference to this tensor after modification.
   */
  QLTensor &RemoveTrivialIndexes(void);

  /**
   * @brief Remove selected trivial (dimension-1) indices by axes.
   * @param trivial_idx_axes Axes to remove. Each must have dimension 1.
   * @return Reference to this tensor after modification.
   */
  QLTensor &RemoveTrivialIndexes(const std::vector<size_t> &trivial_idx_axes);

  // Override base class
  /** @brief Read tensor (indices and data) from stream. Requires default tensor. */
  void StreamRead(std::istream &);
  /** @brief Write tensor (indices and data) to stream. */
  void StreamWrite(std::ostream &) const;

  /** @brief Send the full tensor through MPI. */
  void MPI_Send(int, int, const MPI_Comm &) const;
  /** @brief Receive the full tensor through MPI into a default tensor. */
  MPI_Status MPI_Recv(int, int, const MPI_Comm &);
  /** @brief Broadcast the full tensor through MPI. */
  void MPI_Bcast(const int, const MPI_Comm &);

  /** @brief Serialize/deserialize only the structural shell for MPI workflows. */
  void StreamReadShellForMPI(std::istream &);
  void StreamWriteShellForMPI(std::ostream &) const;
  std::string SerializeShell() const;
  void DeserializeShell(const std::string &);

  /** @brief Verbose human-readable print. */
  void Show(const size_t indent_level = 0) const override;

  /** @brief Concise one-shot summary print. */
  void ConciseShow(const size_t indent_level = 0) const;

  // in unit of GB
  /** @brief Estimated memory usage of raw data buffer (GB). */
  double GetRawDataMemUsage() const;

  /** @brief Whether any element is NaN. */
  bool HasNan(void) const;

  /** @brief Actual number of stored elements in block-sparse buffer. */
  size_t GetActualDataSize(void) const;

  /**
   * @brief Set an element by quantum-number sector and in-block coordinates.
   * @param qn_sector Block quantum numbers in index order.
   * @param blk_coors In-block coordinates.
   * @param value Value to assign.
   */
  void SetElemByQNSector(
      const std::vector<QNT> &qn_sector,     // quantum number block {QN0, QN1, ...}
      const std::vector<size_t> &blk_coors,  // block inner coordinates
      const ElemT value                      // value to set
  ) {
    assert(!IsDefault());
    pblk_spar_data_ten_->SetElemByQNSector(qn_sector, blk_coors, value);
  }

 private:
  /// The rank of the QLTensor.
  size_t rank_ = 0;
  /// The shape of the QLTensor.
  ShapeT shape_;
  /// The total number of elements of the QLTensor.
  size_t size_ = 0;

  /// Indexes of the QLTensor.
  IndexVec<QNT> indexes_;
  /// The pointer which point to block sparse data tensor.
  BlockSparseDataTensor<ElemT, QNT> *pblk_spar_data_ten_ = nullptr;

  ShapeT CalcShape_(void) const;

  size_t CalcSize_(void) const;

  std::pair<CoorsT, CoorsT> CoorsToBlkCoorsDataCoors_(const CoorsT &) const;
};

// Out-of-class declaration and definition.
template<typename ElemT, typename QNT>
inline QLTensor<ElemT, QNT> operator*(
    const QLTEN_Double scalar,
    const QLTensor<ElemT, QNT> &t
) {
  return t * scalar;
}

template<typename ElemT, typename QNT>
inline QLTensor<ElemT, QNT> operator*(
    const QLTEN_Complex scalar,
    const QLTensor<ElemT, QNT> &t
) {
  return t * scalar;
}

template<typename ElemT, typename QNT>
inline std::istream &operator>>(std::istream &is, QLTensor<ElemT, QNT> &t) {
  t.StreamRead(is);
  return is;
}

template<typename ElemT, typename QNT>
inline std::ostream &operator<<(std::ostream &os, const QLTensor<ElemT, QNT> &t) {
  t.StreamWrite(os);
  return os;
}

} /* qlten */
#endif /* ifndef QLTEN_QLTENSOR_QLTENSOR_H */
