Change Log (AOCL Backend Integration)
Date: Oct 20, 2025

  1. Common backend selector

     Added include/qlten/framework/hp_numeric/backend_selector.h so BLAS/LAPACK headers and macros are chosen in one place. This header
  also declares the BLIS Fortran somatcopy_/domatcopy_/comatcopy_/zomatcopy_ symbols because AOCL’s CBLAS interface does not expose
  cblas_*omatcopy; we therefore invoke the BLIS entry points directly.
  2. AOCL-aware HP numeric wrappers

     blas_extensions.h, blas_level1.h, blas_level3.h, lapack.h, and omp_set.h now include backend_selector.h and branch on
  HP_NUMERIC_BACKEND_{MKL,AOCL,OPENBLAS} instead of hard-coded headers.
    • For AOCL, the batched row-major transpose helpers call BLIS *omatcopy_ with swapped dimensions and leading dimensions, matching
      AOCL’s column-major expectations.
    • OpenBLAS and MKL paths remain unchanged.
  3. CMake plumbing

     Introduced cmake/Modules/MathBackend.cmake and updated top-level, example, and test CMakeLists.txt to detect AOCL, enforce a single
   BLAS vendor, and propagate the correct include/lib/search paths and preprocessor defines.
  4. Test updates

     Tensor-contraction tests (test_ten_ctrct.cc, test_fermion_ten_ctrct.cc) use CBLAS_ORDER so they compile against AOCL’s CBLAS
  headers, and the auxiliary HP numeric test header reflects the new backend macros.
  5. Runtime note

     Documentation/comments clarify that AOCL builds require LD_LIBRARY_PATH=$AOCL_ROOT/lib (or an equivalent rpath) so the BLIS/FLAME
  shared libraries are found at test or run time.
