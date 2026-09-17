// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/allocator.h"
#include "mlx/backend/common/buffer_cache.h"
#include "mlx/backend/cuda/cuda_utils.h"

#include <cuda_runtime.h>
#include <atomic>
#include <mutex>
#include <set>
#include <utility>

namespace mlx::core::cu {

class CommandEncoder;

using allocator::Buffer;

// Stores cuda-managed unified memory.
struct CudaBuffer {
  void* data;
  size_t size;
  int device; // -1 for managed
};

class SmallSizePool {
 private:
  union Block {
    Block* next;
    CudaBuffer buf;
  };

  Block* buffer_{nullptr};
  void* data_{nullptr};
  Block* next_free_{nullptr};

 public:
  SmallSizePool();
  ~SmallSizePool();

  SmallSizePool(const SmallSizePool&) = delete;
  SmallSizePool& operator=(const SmallSizePool&) = delete;

  CudaBuffer* malloc();
  void free(CudaBuffer* buf);
  bool in_pool(CudaBuffer* buf);
};

class CudaAllocator : public allocator::Allocator {
 public:
  Buffer malloc(size_t size) override;
  Buffer malloc_async(size_t size, int device, cudaStream_t stream);
  void free(Buffer buffer) override;
  size_t size(Buffer buffer) const override;

  // Replace the memory of |buf| with unified memory (managed memory or pinned
  // host memory), and copy the data over. Pass |stream| to copy asynchronously.
  void move_to_unified_memory(CudaBuffer& buf, cudaStream_t stream = nullptr);

  size_t get_active_memory() const;
  size_t get_peak_memory() const;
  void reset_peak_memory();
  // The limit actually in force: the requested budget, clamped to what the
  // device has left after the reserve. Derived rather than stored, because the
  // reserve it is measured against moves.
  size_t memory_limit() const;
  size_t get_memory_limit();
  size_t set_memory_limit(size_t limit);
  size_t get_cache_memory() const;
  size_t set_cache_limit(size_t limit);
  void clear_cache();
  // Record that the device refused an allocation: the one moment the free
  // memory a consumer outside this pool actually needed can be read, at the
  // instant it decided the outcome. Never throws, never takes mutex_: it runs
  // on the error path, where the caller may hold it.
  void report_out_of_memory();

 private:
  void free_cuda_buffer(CudaBuffer* buf);
  void free_async(CudaBuffer& buf, cudaStream_t stream = nullptr);
  // Called without mutex_ held, with |device| current.
  void wait_for_physical_memory(size_t size, int device);
  // Device memory nobody holds -- the only room every consumer can reach, and
  // so the only figure worth gating an allocation on. The pool's own unused
  // reservation is deliberately not reported alongside it: it is off the
  // device's free list, and stream-ordered reuse means it cannot be counted on
  // to serve an arbitrary request even when it dwarfs one. Sampling also raises
  // `free_limit_` when it shows more memory held outside the pool than the
  // reserve currently accounts for. Called without mutex_ held.
  size_t observe_device_memory(int device);
  // Raise `free_limit_` to |reserve| if it is larger, bounded by half the
  // card. Lock-free: the reserve only ever grows, so a lost race is a sample
  // another will retake.
  void raise_reserve(size_t reserve);
  // Print the device-side memory split to stderr under MLX_CUDA_MEMORY_TRACE.
  // Non-throwing and rate-limited -- the first 32 events then every 256th --
  // because it runs on paths a failing run reaches thousands of times. |seen|
  // is the call site's own counter, so a site that fires rarely is not crowded
  // out of the trace by one that fires constantly.
  void trace_device_memory(
      const char* where,
      std::atomic<size_t>& seen,
      size_t size);
  // Return every memory pool's unused reservation to the device. Freeing a
  // buffer hands it back to the CUDA async pool, which keeps the pages
  // reserved for its own future allocations; anything that allocates outside
  // the pool -- a graph instantiation, a cuDNN workspace, a kernel's launch
  // resources -- cannot touch that reservation and fails with an out-of-memory
  // the pool's own statistics contradict. Trimming is what turns a released
  // buffer back into device memory those consumers can actually use.
  void trim_pools();

  CudaAllocator();
  friend CudaAllocator& allocator();

  std::mutex mutex_;
  // Device memory that must stay genuinely free -- not merely unused inside
  // the async pool's reservation -- because the consumers that need it cannot
  // allocate from the pool: the CUDA context, NVRTC modules, cuDNN workspaces,
  // instantiated graph executables. It is the cap on how much the pool may keep
  // reserved, which makes it the only figure in this allocator that decides how
  // much of the card those consumers can reach. Seeded from what the device had
  // already handed out before MLX allocated anything, raised by samples that
  // see more held outside the pool, and raised again by every refusal. It only
  // ever grows -- giving headroom back would hand away memory a measurement
  // just proved another consumer needs.
  std::atomic<size_t> free_limit_{0};
  // What a caller asked the memory limit to be. A budget, not a fact about the
  // device: `memory_limit()` honours it only down to what the reserve leaves.
  // Atomic because `memory_limit()` is read from the refusal path, which must
  // not take mutex_ -- the caller there may already hold it.
  std::atomic<size_t> requested_limit_{0};
  size_t total_memory_{0};
  size_t max_pool_size_{0};
  BufferCache<CudaBuffer> buffer_cache_;
  size_t active_memory_{0};
  size_t peak_memory_{0};
  std::vector<CudaStream> free_streams_;
  std::vector<cudaMemPool_t> mem_pools_;
  SmallSizePool scalar_pool_;
};

CudaAllocator& allocator();

// Whether `allocator()`'s singleton has finished constructing. Consulted by
// the CUDA error path, which the constructor's own CUDA calls run through.
std::atomic<bool>& allocator_live();

Buffer malloc_async(size_t size, CommandEncoder& encoder);

} // namespace mlx::core::cu
