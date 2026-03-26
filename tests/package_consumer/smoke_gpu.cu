#include "qlten/qlten.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int ReportCudaFailure(const char *step, cudaError_t status) {
  std::fprintf(stderr, "%s failed: %s (%s)\n",
               step, cudaGetErrorName(status), cudaGetErrorString(status));
  return 1;
}

int ReportGpuUnavailable(cudaError_t status) {
  std::fprintf(stderr, "Skipping GPU smoke: %s (%s)\n",
               cudaGetErrorName(status), cudaGetErrorString(status));
  return 77;
}

using MpiInitFn = int (*)(int *, char ***);
using MpiFinalizeFn = int (*)();

volatile MpiInitFn kMpiInitRef = &MPI_Init;
volatile MpiFinalizeFn kMpiFinalizeRef = &MPI_Finalize;

}  // namespace

int main(int argc, char **argv) {
  // Keep explicit MPI symbol references so the smoke validates transitive MPI
  // link propagation without depending on local MPI runtime launch behavior.
  (void)argc;
  (void)argv;
  (void)kMpiInitRef;
  (void)kMpiFinalizeRef;

#ifndef USE_GPU
#error "TensorToolkit GPU package consumer requires USE_GPU"
#endif

  int device_count = 0;
  cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess) {
    return ReportGpuUnavailable(status);
  }
  if (device_count < 1) {
    std::fprintf(stderr, "Skipping GPU smoke: no CUDA devices available.\n");
    return 77;
  }

  std::vector<size_t> input_shape{2, 2};
  std::vector<int> transpose_order{1, 0};
  std::vector<int> output_shape{2, 2};

  double host_input[4] = {1.0, 2.0, 3.0, 4.0};
  double *device_input = nullptr;
  double *device_output = nullptr;
  status = cudaMalloc(reinterpret_cast<void **>(&device_input), sizeof(host_input));
  if (status != cudaSuccess) {
    return ReportCudaFailure("cudaMalloc(device_input)", status);
  }
  status = cudaMalloc(reinterpret_cast<void **>(&device_output), sizeof(host_input));
  if (status != cudaSuccess) {
    return ReportCudaFailure("cudaMalloc(device_output)", status);
  }
  status = cudaMemcpy(device_input, host_input, sizeof(host_input), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    return ReportCudaFailure("cudaMemcpy(H2D input)", status);
  }

  qlten::hp_numeric::TensorTranspose(
      transpose_order, 2, device_input, input_shape, device_output, output_shape, 1.0);
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    return ReportCudaFailure("cudaDeviceSynchronize after TensorTranspose", status);
  }
  double host_output[4] = {0.0, 0.0, 0.0, 0.0};
  status = cudaMemcpy(host_output, device_output, sizeof(host_output), cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    return ReportCudaFailure("cudaMemcpy(D2H transpose)", status);
  }
  const double expected_transpose[4] = {1.0, 3.0, 2.0, 4.0};
  for (int i = 0; i < 4; ++i) {
    if (std::abs(host_output[i] - expected_transpose[i]) > 1e-10) {
      return 1;
    }
  }

  double *u = nullptr;
  double *s = nullptr;
  double *vt = nullptr;
  qlten::hp_numeric::MatSVD(device_input, 2, 2, u, s, vt);
  if (u == nullptr || s == nullptr || vt == nullptr) {
    std::fprintf(stderr, "MatSVD returned a null output pointer.\n");
    return 1;
  }
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    return ReportCudaFailure("cudaDeviceSynchronize after MatSVD", status);
  }
  double host_s[2] = {0.0, 0.0};
  status = cudaMemcpy(host_s, s, sizeof(host_s), cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    return ReportCudaFailure("cudaMemcpy(D2H singular values)", status);
  }
  if (std::abs(host_s[0] - 5.464985704219043) > 1e-8
      || std::abs(host_s[1] - 0.365966190626257) > 1e-8) {
    return 1;
  }

  status = cudaFree(vt);
  if (status != cudaSuccess) {
    return ReportCudaFailure("cudaFree(vt)", status);
  }
  status = cudaFree(s);
  if (status != cudaSuccess) {
    return ReportCudaFailure("cudaFree(s)", status);
  }
  status = cudaFree(u);
  if (status != cudaSuccess) {
    return ReportCudaFailure("cudaFree(u)", status);
  }
  status = cudaFree(device_output);
  if (status != cudaSuccess) {
    return ReportCudaFailure("cudaFree(device_output)", status);
  }
  status = cudaFree(device_input);
  if (status != cudaSuccess) {
    return ReportCudaFailure("cudaFree(device_input)", status);
  }

  return 0;
}
