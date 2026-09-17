// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/allocator.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/utils.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/memory.h"
#include "mlx/scheduler.h"
#include "mlx/utils.h"

#include <cuda_runtime.h>
#include <fmt/format.h>

#include <algorithm>
#include <cassert>
#include <fstream>
#include <string>

namespace mlx::core {

namespace cu {

constexpr int page_size = 16384;

// Any allocations smaller than this will try to use the small pool
constexpr int small_block_size = 8;

// The small pool size in bytes. This should be a multiple of the host page
// size and small_block_size.
constexpr int small_pool_size = 4 * page_size;

bool supports_managed_memory() {
  static bool managed_memory = []() {
    int device_count = gpu::device_count();
    for (int i = 0; i < device_count; ++i) {
      auto& d = cu::device(i);
      // Empirically on Windows (and WSL) if there is no concurrentManagedAccess
      // the managed memory also does not work.
      // The same has been observed on NVIDIA Tegra, typically on Jetson Orin
      // Nano boards.
      // NVIDIA documentation describes Windows, WSL, and Tegra as having a
      // specific unified memory paradigm called "Limited unified memory
      // support", which corresponds to concurrentManagedAccess = 0.
      // See:
      //   https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#table-unified-memory-levels
      //   https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#um-legacy-devices
      if (!d.concurrent_managed_access()) {
        return false;
      }
    }
    return true;
  }();
  return managed_memory;
}

inline void* unified_malloc(size_t size) {
  void* data = nullptr;
  if (supports_managed_memory()) {
    CHECK_CUDA_ERROR(cudaMallocManaged(&data, size));
  } else {
    CHECK_CUDA_ERROR(cudaMallocHost(&data, size));
  }
  return data;
}

inline void unified_free(void* data) {
  if (supports_managed_memory()) {
    CHECK_CUDA_ERROR(cudaFree(data));
  } else {
    CHECK_CUDA_ERROR(cudaFreeHost(data));
  }
}

#if CUDART_VERSION >= 13000
inline cudaMemLocation cuda_mem_loc(int i) {
  cudaMemLocation loc;
  loc.type = cudaMemLocationTypeDevice;
  loc.id = i;
  return loc;
}
#else
inline int cuda_mem_loc(int i) {
  return i;
}
#endif // CUDART_VERSION >= 13000

SmallSizePool::SmallSizePool() {
  auto num_blocks = small_pool_size / small_block_size;
  buffer_ = new Block[num_blocks];
  next_free_ = buffer_;

  data_ = unified_malloc(small_pool_size);
  if (supports_managed_memory()) {
    int device_count = gpu::device_count();
    for (int i = 0; i < device_count; ++i) {
      if (device(i).concurrent_managed_access()) {
        auto loc = cuda_mem_loc(i);
        CHECK_CUDA_ERROR(cudaMemAdvise(
            data_, small_pool_size, cudaMemAdviseSetAccessedBy, loc));
      }
    }
  }

  auto curr = next_free_;
  for (size_t i = 1; i < num_blocks; ++i) {
    curr->next = buffer_ + i;
    curr = curr->next;
  }
  curr->next = nullptr;
}

SmallSizePool::~SmallSizePool() {
  unified_free(data_);
  delete[] buffer_;
}

CudaBuffer* SmallSizePool::malloc() {
  if (next_free_ == nullptr) {
    return nullptr;
  }
  Block* b = next_free_;
  uint64_t i = next_free_ - buffer_;
  next_free_ = next_free_->next;
  b->buf.data = static_cast<char*>(data_) + i * small_block_size;
  b->buf.size = small_block_size;
  b->buf.device = -1;
  return &b->buf;
}

void SmallSizePool::free(CudaBuffer* buf) {
  auto b = reinterpret_cast<Block*>(buf);
  b->next = next_free_;
  next_free_ = b;
}

bool SmallSizePool::in_pool(CudaBuffer* buf) {
  constexpr int num_blocks = (small_pool_size / small_block_size);
  auto b = reinterpret_cast<Block*>(buf);
  int64_t block_num = b - buffer_;
  return block_num >= 0 && block_num < num_blocks;
}

CudaAllocator::CudaAllocator()
    : buffer_cache_(
          page_size,
          [](CudaBuffer* buf) { return buf->size; },
          [this](CudaBuffer* buf) { free_cuda_buffer(buf); }) {
  size_t free = 0;
  CHECK_CUDA_ERROR(cudaMemGetInfo(&free, &total_memory_));
  // Memory the device has already handed to someone else -- the CUDA context
  // above all -- is never ours to allocate, so a limit derived from the total
  // alone is a limit that can never be reached. Clamp to what is actually
  // free, and let the difference widen the reserve `free_limit_` protects.
  // Nothing is ours yet, so everything the device has already handed out is
  // the first measurement of the consumers that never allocate from this
  // pool. Seeding the reserve with it is what keeps `set_memory_limit` and
  // every later allocation from spending memory that was never ours.
  foreign_reserve_.store(total_memory_ - free, std::memory_order_relaxed);
  memory_limit_ = std::min(static_cast<size_t>(total_memory_ * 0.95), free);
  free_limit_ = total_memory_ - memory_limit_;
  max_pool_size_ = memory_limit_;

  int device_count = gpu::device_count();
  free_streams_.resize(device_count);
  mem_pools_.resize(device_count);
  for (int i = 0; i < device_count; ++i) {
    auto& d = device(i);
    if (d.memory_pools()) {
      free_streams_[i] = CudaStream(d);
      CHECK_CUDA_ERROR(cudaDeviceGetDefaultMemPool(&mem_pools_[i], i));
    }
  }
}

Buffer
CudaAllocator::malloc_async(size_t size, int device, cudaStream_t stream) {
  if (size == 0) {
    return Buffer{new CudaBuffer{nullptr, 0, -1}};
  }

  if (size <= small_block_size) {
    size = 8;
  } else if (size < page_size) {
    size = next_power_of_2(size);
  } else {
    size = page_size * ((size + page_size - 1) / page_size);
  }

  if (size <= small_block_size || stream == nullptr) {
    device = -1;
  }

  // Find available buffer from cache.
  std::unique_lock lock(mutex_);
  CudaBuffer* buf = buffer_cache_.reuse_from_cache(size);

  // Backpressure for device allocations: while work is in flight, its
  // completion returns memory to the cache, so waiting for it is strictly
  // better than allocating past the limit and racing the device for the
  // physical headroom the limit exists to protect. Only device allocations
  // wait; they are issued by the thread walking the eval tape, which never
  // runs inside a task the wait could depend on. Nothing here commits the
  // open graph — the caller is mid-way through encoding it.
  // Cached buffers count against the limit because they are memory MLX holds
  // and the device cannot hand to anyone else: a freed buffer goes back to the
  // CUDA async pool, not to the device. Waiting on `active_memory_` alone lets
  // the real footprint reach the limit twice over, which is how a run whose
  // limit was 37.75 GB peaked at 43.09 GB with 15.28 GB sitting in cache.
  if (!buf && device >= 0) {
    while (active_memory_ + get_cache_memory() + size > memory_limit_ &&
           scheduler::n_active_tasks() > 0) {
      lock.unlock();
      scheduler::wait_for_completion();
      lock.lock();
      buf = buffer_cache_.reuse_from_cache(size);
      if (buf) {
        break;
      }
    }
  }

  if (!buf) {
    // If we have a lot of memory pressure try to reclaim memory from the cache.
    int64_t mem_to_free =
        get_active_memory() + get_cache_memory() + size - memory_limit_;
    if (mem_to_free > 0) {
      buffer_cache_.release_cached_buffers(mem_to_free);
    }

    // Try the scalar pool first
    if (size <= small_block_size) {
      buf = scalar_pool_.malloc();
    }
    lock.unlock();
    if (!buf) {
      void* data = nullptr;
      if (device == -1) {
        data = unified_malloc(size);
      } else {
        cu::device(device).make_current();
        if (mem_pools_[device]) { // supports memory pools
          wait_for_physical_memory(size, device);
          CHECK_CUDA_ERROR(cudaMallocAsync(&data, size, stream));
        } else {
          CHECK_CUDA_ERROR(cudaMalloc(&data, size));
        }
      }
      if (!data) {
        std::ostringstream msg;
        msg << "[malloc] Unable to allocate " << size << " bytes.";
        throw std::runtime_error(msg.str());
      }
      buf = new CudaBuffer{data, size, device};
    }
    lock.lock();

    // If any cuda memory pool has too much reserved memory, clear some
    // memory from the cache. This prevents graph / kernel execution failing
    // from OOM.
    //
    // Releasing cached buffers alone cannot do that: a released buffer is
    // handed back to the async pool, which keeps its pages reserved, so the
    // reserved figure this very condition tests is unchanged by the release.
    // Trimming is the step that turns the freed reservation back into device
    // memory a graph instantiation or a kernel's launch resources can use.
    for (auto p : mem_pools_) {
      if (p) {
        size_t reserved = 0;
        CHECK_CUDA_ERROR(cudaMemPoolGetAttribute(
            p, cudaMemPoolAttrReservedMemCurrent, &reserved));
        if (reserved > (total_memory_ - free_limit_)) {
          buffer_cache_.release_cached_buffers(free_limit_);
          trim_pools();
          break;
        }
      }
    }
  }
  active_memory_ += buf->size;
  peak_memory_ = std::max(active_memory_, peak_memory_);

  // Maintain the cache below the requested limit.
  if (get_cache_memory() > max_pool_size_) {
    buffer_cache_.release_cached_buffers(get_cache_memory() - max_pool_size_);
  }
  lock.unlock();
  // Copy to unified memory here if the buffer is not on the right device.
  if (buf->device >= 0 && buf->device != device) {
    move_to_unified_memory(*buf, stream);
  }
  return Buffer{buf};
}

void CudaAllocator::trim_pools() {
  for (auto p : mem_pools_) {
    if (p) {
      CHECK_CUDA_ERROR(cudaMemPoolTrimTo(p, 0));
    }
  }
}

void CudaAllocator::raise_reserve(size_t reserve) {
  size_t current = foreign_reserve_.load(std::memory_order_relaxed);
  while (reserve > current &&
         !foreign_reserve_.compare_exchange_weak(
             current, reserve, std::memory_order_relaxed)) {
  }
}

// The device refusing an allocation is the one measurement sampling can never
// make: a sample sees the allocations outside this pool that succeeded, never
// the one that failed. So treat the refusal as the measurement it is and grow
// the reserve, so whatever retries finds the room this attempt did not. The
// step is a fraction of the reserve rather than a constant because the right
// figure is a property of the card and the graphs a run instantiates, and a
// few refusals converge on it from a measured starting point instead of
// guessing it up front.
void CudaAllocator::report_out_of_memory() {
  size_t current = foreign_reserve_.load(std::memory_order_relaxed);
  raise_reserve(current + current / 4 + page_size);
}

// Report what the device can still hand out, keeping the two kinds of room
// apart: the pool's unused reservation is already off the device's free list,
// so spending it costs no free memory and only `cudaMallocAsync` can do it,
// while anything else has to come out of free memory that every consumer
// competes for. Sampling doubles as a measurement of the reserve -- whatever
// the device holds beyond this allocator's reservation belongs to consumers
// the pool cannot serve, so a sample showing more of it than the reserve holds
// back proves the reserve too small. Resident managed memory and the scalar
// pool land on that side of the subtraction too; both are small, and counting
// them here only makes the reserve larger than it strictly has to be, which is
// the direction that is safe to be wrong in.
CudaAllocator::Availability CudaAllocator::observe_device_memory(int device) {
  auto pool = mem_pools_[device];
  size_t free = 0;
  size_t total = 0;
  CHECK_CUDA_ERROR(cudaMemGetInfo(&free, &total));
  size_t reserved = 0;
  size_t used = 0;
  CHECK_CUDA_ERROR(cudaMemPoolGetAttribute(
      pool, cudaMemPoolAttrReservedMemCurrent, &reserved));
  CHECK_CUDA_ERROR(
      cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used));

  size_t held = total - free;
  raise_reserve(held > reserved ? held - reserved : 0);
  return Availability{free, reserved - used};
}

// The memory limit bounds what this allocator counts, not what the device
// holds: the CUDA context, instantiated graph executables and pool
// fragmentation all sit on top of it. An allocation is safe only once the
// device can serve it without eating into the reserve -- either out of the
// pool's own reservation, which costs no free memory, or out of free memory
// with the reserve still left behind. Counting the reservation as available
// for any allocation, as an earlier version of this did, is what lets a run
// fill the card to the last byte of its pool and then die on the next graph
// launch with an out-of-memory the pool's own statistics contradict. Hand
// cached buffers back to the device and wait for in-flight work instead, for
// as long as there is work to wait for.
void CudaAllocator::wait_for_physical_memory(size_t size, int device) {
  auto servable = [this, device, size]() {
    auto avail = observe_device_memory(device);
    return avail.pooled >= size ||
        avail.free >= size + foreign_reserve_.load(std::memory_order_relaxed);
  };
  while (!servable() && scheduler::n_active_tasks() > 0) {
    {
      std::lock_guard lock(mutex_);
      buffer_cache_.release_cached_buffers(get_cache_memory());
      trim_pools();
    }
    scheduler::wait_for_completion();
  }
  // Nothing is left in flight to wait for, so this is the last chance to make
  // the reservation usable before the allocation is attempted anyway.
  if (!servable()) {
    std::lock_guard lock(mutex_);
    buffer_cache_.release_cached_buffers(get_cache_memory());
    trim_pools();
  }
}

Buffer CudaAllocator::malloc(size_t size) {
  return malloc_async(size, -1, nullptr);
}

void CudaAllocator::free(Buffer buffer) {
  auto* buf = static_cast<CudaBuffer*>(buffer.ptr());
  if (!buf) {
    return;
  }
  if (buf->size == 0) {
    delete buf;
    return;
  }

  std::unique_lock lock(mutex_);
  active_memory_ -= buf->size;
  if (get_cache_memory() < max_pool_size_) {
    buffer_cache_.recycle_to_cache(buf);
  } else {
    free_cuda_buffer(buf);
  }
}

size_t CudaAllocator::size(Buffer buffer) const {
  auto* buf = static_cast<CudaBuffer*>(buffer.ptr());
  if (!buf) {
    return 0;
  }
  return buf->size;
}

void CudaAllocator::move_to_unified_memory(
    CudaBuffer& buf,
    cudaStream_t stream) {
  if (buf.device == -1) {
    return;
  }
  void* data = unified_malloc(buf.size);
  cudaMemcpyKind kind =
      supports_managed_memory() ? cudaMemcpyDefault : cudaMemcpyDeviceToHost;
  if (stream && mem_pools_[buf.device]) {
    CHECK_CUDA_ERROR(cudaMemcpyAsync(data, buf.data, buf.size, kind, stream));
    free_async(buf, stream);
  } else {
    CHECK_CUDA_ERROR(cudaMemcpy(data, buf.data, buf.size, kind));
    free_async(buf);
  }
  buf.data = data;
  buf.device = -1;
}

// This must be called with mutex_ acquired
void CudaAllocator::free_cuda_buffer(CudaBuffer* buf) {
  if (scalar_pool_.in_pool(buf)) {
    scalar_pool_.free(buf);
  } else {
    free_async(*buf);
    delete buf;
  }
}

void CudaAllocator::free_async(CudaBuffer& buf, cudaStream_t stream) {
  if (buf.device == -1) {
    unified_free(buf.data);
  } else {
    // Free asynchronously when memory pools is supported.
    if (mem_pools_[buf.device]) {
      if (!stream) {
        stream = free_streams_[buf.device];
      }
      CHECK_CUDA_ERROR(cudaFreeAsync(buf.data, stream));
    } else {
      CHECK_CUDA_ERROR(cudaFree(buf.data));
    }
  }
}

size_t CudaAllocator::get_active_memory() const {
  return active_memory_;
}

size_t CudaAllocator::get_peak_memory() const {
  return peak_memory_;
}

void CudaAllocator::reset_peak_memory() {
  std::lock_guard lock(mutex_);
  peak_memory_ = 0;
}

size_t CudaAllocator::get_memory_limit() {
  return memory_limit_;
}

// A caller asking for a limit is expressing a budget, not a fact about the
// device. Honour it only down to what the device has actually left us: a
// framework that computes its ceiling from the device total would otherwise
// undo the constructor's clamp and spend the reserve, and get back the
// out-of-memory the reserve exists to prevent. Returns the limit that was in
// force.
size_t CudaAllocator::set_memory_limit(size_t limit) {
  std::lock_guard lock(mutex_);
  size_t reserve = foreign_reserve_.load(std::memory_order_relaxed);
  size_t bound = total_memory_ > reserve ? total_memory_ - reserve : 0;
  limit = std::min(limit, bound);
  std::swap(limit, memory_limit_);
  free_limit_ = total_memory_ - memory_limit_;
  return limit;
}

size_t CudaAllocator::get_cache_memory() const {
  return buffer_cache_.cache_size();
}

size_t CudaAllocator::set_cache_limit(size_t limit) {
  std::lock_guard lk(mutex_);
  std::swap(limit, max_pool_size_);
  return limit;
}

void CudaAllocator::clear_cache() {
  std::lock_guard lk(mutex_);
  buffer_cache_.clear();
}

// The singleton below is built by a function-local static, and its constructor
// makes CUDA calls that run through `check_cuda_error`. Reaching back into
// `allocator()` from that error path would re-enter an initialization already
// in progress, so the error path asks this first and stays out until the
// singleton exists.
std::atomic<bool>& allocator_live() {
  static std::atomic<bool> live{false};
  return live;
}

CudaAllocator& allocator() {
  static auto* allocator_ = []() {
    // Ensure scheduler is created before allocator.
    scheduler::scheduler();
    // By creating the |allocator_| on heap, the destructor of CudaAllocator
    // will not be called on exit and buffers in the cache will be leaked. This
    // can save some time at program exit.
    auto* a = new CudaAllocator();
    allocator_live().store(true, std::memory_order_release);
    return a;
  }();
  return *allocator_;
}

Buffer malloc_async(size_t size, CommandEncoder& encoder) {
  return allocator().malloc_async(
      size, encoder.device().cuda_device(), encoder.stream());
}

} // namespace cu

namespace allocator {

Allocator& allocator() {
  return cu::allocator();
}

void* Buffer::raw_ptr() {
  if (!ptr_) {
    return nullptr;
  }
  auto& cbuf = *static_cast<cu::CudaBuffer*>(ptr_);
  cu::allocator().move_to_unified_memory(cbuf);
  return cbuf.data;
}

bool can_reuse_alien_buffer(void* ptr) {
  return true;
}

} // namespace allocator

size_t get_active_memory() {
  return cu::allocator().get_active_memory();
}
size_t get_peak_memory() {
  return cu::allocator().get_peak_memory();
}
void reset_peak_memory() {
  return cu::allocator().reset_peak_memory();
}
size_t set_memory_limit(size_t limit) {
  return cu::allocator().set_memory_limit(limit);
}
size_t get_memory_limit() {
  return cu::allocator().get_memory_limit();
}
size_t get_cache_memory() {
  return cu::allocator().get_cache_memory();
}
size_t set_cache_limit(size_t limit) {
  return cu::allocator().set_cache_limit(limit);
}
void clear_cache() {
  cu::allocator().clear_cache();
}

// Not supported in CUDA.
size_t set_wired_limit(size_t) {
  return 0;
}

} // namespace mlx::core
