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

// Forward declaration for FuseInfo (defined in ten_fuse_index.h)
template<typename QNT>
struct FuseInfo;

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
  using RealType = typename RealTypeTrait<ElemT>::type;
 public:
  using value_type = ElemT;
  using qn_type = QNT;

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
   * @brief Fuse two indices in-place and return fusion information.
   *
   * Fuses the indices at positions `idx1` and `idx2` into a single combined
   * index placed at position 0. Both indices must have the same direction.
   *
   * @param idx1 Position of the first index to fuse (must be < idx2).
   * @param idx2 Position of the second index to fuse.
   * @return FuseInfo<QNT> containing information needed for potential SplitIndex.
   *
   * @note For fermionic tensors, the fermion exchange sign is correctly handled
   *       during the internal transpose step.
   *
   * @see FuseInfo, ten_fuse_index.h for detailed documentation.
   */
  FuseInfo<QNT> FuseIndex(const size_t, const size_t);

  /** @brief 2-norm of the tensor.
   *
   * Definitions :
   * - Bosonic tensors: \f$ \|A\|_2 = \sqrt{\sum_i |a_i|^2} \f$.
   * - Fermionic tensors (Z2-graded, let E/O denote even/odd fermionic parity blocks):
   *   \f[
   *   \|A\|_{2,\mathrm{graded}} = \sqrt{\sum_{i\in E} |a_i|^2\; - \; \sum_{i\in O} |a_i|^2}.
   *   \f]
   *   This graded norm can be ill-defined if the odd contribution exceeds the even one,
   *   in which case the quantity under the square root becomes negative.
   *
   * This definition is consistent with `sqrt(Contract(A, A.Dag()))`.
   * @note For fermionic tensors, prefer `GetQuasi2Norm()` or `QuasiNormalize()` when you
   *       need a conventional non-negative norm.
   * @throws std::runtime_error if the graded \f$\|A\|_{2,\mathrm{graded}}^2\f$ is negative.
   */
  auto Get2Norm(void) const;

  /** @brief Quasi 2-norm of the tensor.
   * For both bosonic and fermionic tensors,
   * \f$ \|A\|_{2,\mathrm{quasi}} = \sqrt{\sum_i |a_i|^2} \f$.
   * This is always non-negative and coincides with the standard Euclidean norm.
   */
  auto GetQuasi2Norm(void) const;

  /**
   * @brief Normalize tensor by its 2-norm, returning the original norm.
   * @return Norm before normalization.
   * @note The definition of 2-norm of fermionic tensors refer to Get2Norm().
   *       If the graded norm is ill-defined (negative under the square root),
   *       Normalize() throws. In such cases prefer QuasiNormalize().
   */
  auto Normalize(void);

  /** @brief Rescale the tensor so that \f$\sum_i |a_i|^2 = 1\f$ (uses quasi 2-norm) */
  auto QuasiNormalize(void);

  /**
   * @brief Hermitian conjugate: invert index directions and conjugate elements (if complex).
   */
  void Dag(void);

  ///< act the fermion P operators
  void ActFermionPOps(void);

  /** @brief In-place element-wise inverse with tolerance 0. */
  void ElementWiseInv(void);

  /** @brief In-place element-wise inverse with tolerance for near-zeros. */
  void ElementWiseInv(RealType tolerance);

  /** @brief In-place element-wise product with another tensor (same indices). */
  void ElementWiseMultiply(const QLTensor &);

  /**
   * @brief In-place binary element-wise assignment using this tensor's stored blocks.
   *
   * Iterates over the current tensor's stored block layout. For each lhs element,
   * the rhs element with the same block coordinates and in-block coordinates is
   * used when present; otherwise `rhs_default` is supplied to `op`.
   *
   * @param rhs Tensor with the same indices and coordinate meaning.
   * @param rhs_default Value used when rhs does not store a matching block.
   * @param op Callable with signature `ElemT op(ElemT lhs, ElemT rhs)`.
   */
  template<typename BinaryOp>
  void ElementWiseBinaryAssignByLhsLayout(const QLTensor &, const ElemT &, BinaryOp);

  /**
   * @brief In-place shifted element-wise division using this tensor's stored blocks.
   *
   * For each stored lhs element `x`, assigns `x / (shift - rhs)`. Missing rhs
   * blocks are interpreted as structural zeros, so the denominator is `shift`.
   * If `abs(shift - rhs) < tolerance`, the lhs element is set to zero.
   */
  void ElementWiseShiftedDivideBy(const QLTensor &, const ElemT, RealType tolerance = RealType(0));

  /** @brief In-place inverse of a diagonal matrix stored as a tensor.
   * The input tensor should be a matrix form, that means the number of indices is 2, and the two indices are the same except direction.
   */
  void DiagMatInv(void);

  /** @brief In-place inverse of a diagonal matrix with tolerance.
   * The input tensor should be a matrix form, that means the number of indices is 2, and the two indices are the same except direction.
   */
  void DiagMatInv(RealType tolerance);

  /** @brief In-place element-wise square-root. */
  void ElementWiseSqrt(void);

  /** @brief In-place element-wise square. */
  void ElementWiseSquare(void);

  /** @brief In-place element-wise squared norm (|z|^2 for each element).
   * For real types this is identical to ElementWiseSquare (x^2).
   * For complex types this stores the squared magnitude a^2+b^2 (as a real-valued complex).
   */
  void ElementWiseSquaredNorm(void);

  /** @brief In-place element-wise sign extraction.
   * For real elements: returns -1, 0, or 1 based on sign.
   * For complex elements: returns (±1 or 0, ±1 or 0) based on signs of real and imaginary parts separately.
   * 
   * Note: This does NOT preserve complex phase. For phase preservation, use other methods.
   */
  void ElementWiseSign(void);

  /** @brief Clip tensor elements to specified limit (magnitude bound).
   * 
   * For real numbers: clips to ±limit while preserving sign
   * For complex numbers: clips magnitude to limit while preserving phase
   * 
   * @param limit The magnitude limit to clip to
   */
  void ElementWiseClipTo(RealType limit);

  template<typename RandGenerator>
  /**
   * @brief In-place randomize elements using a RNG but keep the sign of the elements.
   * @param dist Uniform real distribution in [0,1) or similar.
   * @param g Random generator instance.
   */
  void ElementWiseRandomizeMagnitudePreservePhase(std::uniform_real_distribution<RealType> &dist,
                           RandGenerator &g);

  /** @brief Maximum absolute value among stored (non-zero) elements. */
  auto GetMaxAbs(void) const {
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

  /**
   * @brief Append this tensor to an MPI wire-format byte buffer.
   *
   * @note Unsupported when built with `USE_GPU`; this method throws
   *       `std::runtime_error` if called in that configuration.
   * @warning This routine appends both the tensor shell and the raw tensor
   *          payload to `buffer`. The caller temporarily holds the original
   *          tensor storage and the packed copy at the same time. For large
   *          dense tensors or large operator payloads, this can noticeably
   *          increase peak host memory usage.
   * @warning When packing many tensors into one shared buffer, avoiding
   *          repeated `std::vector<char>` reallocations requires a higher
   *          layer to reserve the total payload size in advance. A per-call
   *          reserve inside this function cannot fully eliminate cross-call
   *          growth copies.
   *
   * Layout per tensor:
   *   [uint8_t  is_default]           - 1 byte
   *   [uint64_t shell_size]           - 8 bytes (0 if default)
   *   [char[]   shell_data]           - shell_size bytes
   *   [uint64_t raw_data_elem_count]  - 8 bytes (0 if default)
   *   [ElemT[]  raw_data]             - raw_data_elem_count * sizeof(ElemT) bytes
   *
   * @param buffer Output buffer; data is appended.
   */
  void PackForMPI(std::vector<char> &buffer) const;

  /**
   * @brief Unpack this tensor from an MPI wire-format byte buffer.
   *
   * @pre The target tensor is default-constructed.
   * @note Unsupported when built with `USE_GPU`; this method throws
   *       `std::runtime_error` if called in that configuration.
   * @warning This routine reads from a contiguous packed buffer and copies the
   *          raw tensor payload into this tensor's storage. During unpack, the
   *          receiver may temporarily hold both the packed bytes and the
   *          reconstructed tensor data. For large dense tensors or operators,
   *          account for this peak host memory overhead.
   * @param cursor Pointer into buffer; advanced past consumed bytes.
   * @param end One-past-end pointer for bounds checking.
   * @throws std::runtime_error on malformed or truncated input.
   */
  void UnpackForMPI(const char *&cursor, const char *end);

  /**
   * @brief Append this tensor's shell metadata to an MPI wire-format byte buffer.
   *
   * Same header layout as PackForMPI but WITHOUT the trailing raw data bytes:
   *   [uint8_t  is_default]           - 1 byte
   *   [uint64_t shell_size]           - 8 bytes (0 if default)
   *   [char[]   shell_data]           - shell_size bytes
   *   [uint64_t raw_data_elem_count]  - 8 bytes (0 if default)
   *
   * raw_data_elem_count is included so the receiver can verify its allocation
   * matches before a subsequent RawDataMPIBcast. Empty scalar tensors keep the
   * legacy `0` count in the packed shell, and `UnpackShellForMPI` retains a
   * one-element scratch allocation on the receiver so follow-on raw-data MPI
   * calls still match the scalar transport convention.
   *
   * @param buffer Output buffer; data is appended.
   */
  void PackShellForMPI(std::vector<char> &buffer) const;

  /**
   * @brief Unpack this tensor's shell from an MPI wire-format byte buffer.
   *
   * Reconstructs the tensor shell and allocates raw data memory, but does NOT
   * fill raw data. The caller must follow with RawDataMPIBcast (or equivalent)
   * to populate the raw data buffer. Empty scalar tensors retain a one-element
   * scratch allocation so that subsequent raw-data MPI calls match the sender's
   * scalar transport convention.
   *
   * @pre The target tensor is default-constructed.
   * @param cursor Pointer into buffer; advanced past consumed bytes.
   * @param end One-past-end pointer for bounds checking.
   * @throws std::runtime_error on malformed or truncated input.
   */
  void UnpackShellForMPI(const char *&cursor, const char *end);

  /** @brief Verbose human-readable print. */
  void Show(const size_t indent_level = 0) const override;

  /** @brief Concise one-shot summary print. */
  void ConciseShow(const size_t indent_level = 0) const;

  // in unit of GB
  /** @brief Estimated memory usage of raw data buffer (GB). */
  double GetRawDataMemUsage() const;

  /** @brief Whether any element is NaN. */
  bool HasNan(void) const;

  /** @brief Whether every tensor element is finite. */
  bool AllFinite(void) const;

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
    const QLTEN_Float scalar,
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
inline QLTensor<ElemT, QNT> operator*(
    const QLTEN_ComplexFloat scalar,
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
