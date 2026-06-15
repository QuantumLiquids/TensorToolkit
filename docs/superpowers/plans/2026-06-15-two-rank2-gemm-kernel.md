# Two Rank2 GEMM Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the default `ApplyTwoRank2ToAxesPreserveOrder` scalar fused CPU path with a GEMM/cuBLAS-backed block-local kernel.

**Architecture:** Keep the existing block-sparse topology code and replace only the numeric block kernel. Each input/op1/op2 block triple computes through a temporary block-local workspace with two `AddRank2AxisBlock<Rank2AxisApplyGemmMode::kBatch>` calls, so CPU uses BLAS/batch GEMM and GPU uses existing cuBLAS paths. The old scalar fused kernel is retained only behind an explicit opt-in macro for debugging/reference work.

**Tech Stack:** Header-only C++17, TensorToolkit `QLTensor`, `QLMalloc`/`QLMemset`/`QLFree`, `hp_numeric::MatMultiply`/batch GEMM, GoogleTest.

---

### Task 1: Lock In Default GEMM Behavior

**Files:**
- Modify: `tests/test_tensor_manipulation/test_dmrg_axis_ops.cc`

- [ ] Change `ApplyTwoRank2ToAxesPreserveOrderMatchesSequential` so non-GPU builds expect no scalar fused updates and at least one GEMM call.
- [ ] Add an assertion for a new stats counter showing block-local GEMM execution.
- [ ] Run:

```bash
./build/tests/test_dmrg_axis_ops --gtest_filter='DmrgAxisOpsTest.ApplyTwoRank2ToAxesPreserveOrderMatchesSequential'
```

Expected before implementation: fail because the current CPU path reports scalar fused updates and zero GEMM calls.

### Task 2: Add Block-Local GEMM Kernel

**Files:**
- Modify: `include/qlten/tensor_manipulation/dmrg/axis_ops.h`

- [ ] Add `two_rank2_block_gemm_hits` and `two_rank2_block_workspace_bytes` to `AxisOpStats` and `AddAxisOpStats`.
- [ ] Add a helper that allocates an intermediate block with `QLMalloc`, zeros it with `QLMemset`, applies `op1` to `axis1`, applies `op2` to `axis2`, and frees the workspace.
- [ ] Replace default `ApplyTwoRank2ToAxesPreserveOrder` numeric dispatch with the new helper.
- [ ] Wrap the existing `AddTwoRank2AxesBlock` scalar helper and its call site in `QLTEN_DMRG_ENABLE_SCALAR_TWO_RANK2_FALLBACK`.

### Task 3: Verify

**Files:**
- Test: `tests/test_tensor_manipulation/test_dmrg_axis_ops.cc`

- [ ] Run the focused test and confirm it passes.
- [ ] Run all DMRG axis op tests:

```bash
./build/tests/test_dmrg_axis_ops
```

Expected after implementation: all tests pass on local CPU build. GPU runtime validation will be done later on the cluster.
