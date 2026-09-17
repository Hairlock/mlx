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
  size_t get_memory_limit();
  size_t set_memory_limit(size_t limit);
  size_t get_cache_memory() const;
  size_t set_cache_limit(size_t limit);
  void clear_cache();
  // Record that the device refused an allocation. Grows the reserve this
  // allocator holds back for consumers that cannot draw on its pool, so a
  // retry finds the room this attempt did not. Never throws, never takes
  // mutex_: it runs on the error path, where the caller may hold it.
  void report_out_of_memory();

 private:
  void free_cuda_buffer(CudaBuffer* buf);
  void free_async(CudaBuffer& buf, cudaStream_t stream = nullptr);
  // Called without mutex_ held, with |device| current.
  void wait_for_physical_memory(size_t size, int device);
  // What the device can still hand out, split by who is able to take it.
  // |pooled| is this allocator's own unused reservation, which only
  // `cudaMallocAsync` on |device| can draw on; |free| is device memory nobody
  // holds, which every consumer can. Sampling also raises `foreign_reserve_`
  // when the sample shows more memory held outside the pool than the reserve
  // currently accounts for. Called without mutex_ held.
  struct Availability {
    size_t free;
    size_t pooled;
  };
  Availability observe_device_memory(int device);
  // Raise `foreign_reserve_` to |reserve| if it is larger. Lock-free: the
  // reserve only ever grows, so a lost race is a sample another will retake.
  void raise_reserve(size_t reserve);
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
  size_t memory_limit_;
  size_t free_limit_;
  size_t total_memory_;
  size_t max_pool_size_;
  BufferCache<CudaBuffer> buffer_cache_;
  size_t active_memory_{0};
  size_t peak_memory_{0};
  // Device memory that must stay free after any allocation this allocator
  // makes, because the consumers that need it cannot allocate from the pool:
  // the CUDA context, NVRTC modules, cuDNN workspaces, instantiated graph
  // executables. Seeded from what the device had already handed out before
  // MLX allocated anything, raised by samples that see more held outside the
  // pool, and raised again whenever the device refuses an allocation. It only
  // ever grows -- giving headroom back would hand away memory a measurement
  // just proved another consumer needs.
  std::atomic<size_t> foreign_reserve_{0};
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
