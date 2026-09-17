// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/utils.h"
#include "mlx/backend/cuda/allocator.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/dtype_utils.h"

#include <fmt/format.h>
#include <cuda/cmath>
#include <vector>

namespace mlx::core {

// An out-of-memory from anywhere -- a pool allocation, a graph instantiation,
// a kernel launch reserving its resources -- is the device telling us the
// allocator is holding back too little for the consumers that cannot use its
// pool. Feed it back before the throw so whatever retries runs under a reserve
// that accounts for it. The guard is for the error paths the allocator itself
// sits on: its constructor and its own CUDA calls run through this function,
// and re-entering the allocator from there would recurse or deadlock.
static void report_out_of_memory_once() {
  static thread_local bool reporting = false;
  if (reporting || !cu::allocator_live().load(std::memory_order_acquire)) {
    return;
  }
  reporting = true;
  try {
    cu::allocator().report_out_of_memory();
  } catch (...) {
  }
  reporting = false;
}

void check_cuda_error(const char* name, cudaError_t err) {
  if (err != cudaSuccess) {
    if (err == cudaErrorMemoryAllocation) {
      report_out_of_memory_once();
    }
    throw std::runtime_error(
        fmt::format("{} failed: {}", name, cudaGetErrorString(err)));
  }
}

void check_cuda_error(const char* name, CUresult err) {
  if (err != CUDA_SUCCESS) {
    if (err == CUDA_ERROR_OUT_OF_MEMORY) {
      report_out_of_memory_once();
    }
    const char* err_str = "Unknown error";
    cuGetErrorString(err, &err_str);
    throw std::runtime_error(fmt::format("{} failed: {}", name, err_str));
  }
}

const char* dtype_to_cuda_type(const Dtype& dtype) {
  switch (dtype) {
    case bool_:
      return "bool";
    case int8:
      return "int8_t";
    case int16:
      return "int16_t";
    case int32:
      return "int32_t";
    case int64:
      return "int64_t";
    case uint8:
      return "uint8_t";
    case uint16:
      return "uint16_t";
    case uint32:
      return "uint32_t";
    case uint64:
      return "uint64_t";
    case float16:
      return "__half";
    case bfloat16:
      return "__nv_bfloat16";
    case float32:
      return "float";
    case float64:
      return "double";
    case complex64:
      return "mlx::core::cu::complex64_t";
    default:
      return "unknown";
  }
}

CudaGraph::CudaGraph(cu::Device& device) {
  device.make_current();
  CHECK_CUDA_ERROR(cudaGraphCreate(&handle_, 0));
}

void CudaGraph::end_capture(cudaStream_t stream) {
  CHECK_CUDA_ERROR(cudaStreamEndCapture(stream, &handle_));
}

void CudaGraphExec::instantiate(cudaGraph_t graph) {
  assert(handle_ == nullptr);
  CHECK_CUDA_ERROR(cudaGraphInstantiate(&handle_, graph, nullptr, nullptr, 0));
}

CudaStream::CudaStream(cu::Device& device) {
  device.make_current();
  CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&handle_, cudaStreamNonBlocking));
}

void* allocate_workspace(cu::CommandEncoder& encoder, size_t workspace_size) {
  if (workspace_size == 0) {
    return nullptr;
  }

  // Workspace allocation should not be captured.
#ifndef NDEBUG
  cudaStreamCaptureStatus status;
  CHECK_CUDA_ERROR(cudaStreamIsCapturing(encoder.stream(), &status));
  assert(status == cudaStreamCaptureStatusNone);
#endif

  // Ensure workspace is 256-byte aligned.
  int nbytes = cuda::ceil_div(workspace_size, 256) * 256;
  array workspace(cu::malloc_async(nbytes, encoder), {nbytes}, int8);
  encoder.add_temporary(workspace);
  return gpu_ptr<void>(workspace);
}

} // namespace mlx::core
