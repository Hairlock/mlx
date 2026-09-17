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
#include <cstdlib>
#include <fstream>
#include <limits>
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
  // Two floors, and the larger wins. What the device has already handed out is
  // a direct measurement of the consumers that never allocate from this pool,
  // and it is never ours to spend. A twentieth of the card is the historical
  // reserve this allocator kept, and on a card where the context is small that
  // is still the better starting guess for the graph executables and launch
  // resources that appear later. The reserve only goes up from here.
  free_limit_.store(
      std::max(total_memory_ - free, total_memory_ / 20),
      std::memory_order_relaxed);
  requested_limit_.store(
      std::numeric_limits<size_t>::max(), std::memory_order_relaxed);
  max_pool_size_ = memory_limit();

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

  // The baseline every later line is read against: what the device had already
  // handed out before this allocator took a single byte.
  static std::atomic<size_t> seen{0};
  trace_device_memory("construct", seen, 0);
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
    while (active_memory_ + get_cache_memory() + size > memory_limit() &&
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
        get_active_memory() + get_cache_memory() + size - memory_limit();
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
          // A timeline of the device split across a run, so the trace shows
          // how foreign memory grows between the step that works and the step
          // that dies -- not only the moment it has already gone wrong.
          static std::atomic<size_t> seen{0};
          trace_device_memory("malloc", seen, size);
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

    // If the device has dropped below the reserve, hand the cache back and
    // trim. This is what keeps graph instantiation and kernel launch resources
    // -- none of which allocate from this pool -- from failing with an
    // out-of-memory the pool's own statistics contradict.
    //
    // Releasing cached buffers alone cannot do that: a released buffer is
    // handed back to the async pool, which keeps its pages reserved, so the
    // device's free figure is unchanged by the release. Trimming is the step
    // that turns the freed reservation back into device memory those consumers
    // can reach.
    //
    // The condition is asked of the device rather than of the pool's reserved
    // figure, because the two are not related by `total_memory_` alone. What
    // the device actually has free is `total - pool_reserved - foreign`, and an
    // earlier version of this gate tested `pool_reserved > total - reserve`,
    // which therefore only ever delivered `reserve - foreign` of free memory --
    // short by the foreign term on every run, measured on an L40S as a 3.17 GB
    // reserve that left the device 2.38 GB. Reading `free` directly is the same
    // measurement the reserve's own growth rule already uses, and it carries
    // the foreign term and pool fragmentation without this allocator having to
    // account for either.
    size_t reserve = free_limit_.load(std::memory_order_relaxed);
    size_t free = 0;
    size_t total = 0;
    if (cudaMemGetInfo(&free, &total) == cudaSuccess && free < reserve) {
      buffer_cache_.release_cached_buffers(reserve);
      trim_pools();
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

static bool memory_trace_enabled();

// Trimming is the only step that turns pool reservation back into device
// memory, so whether it moves a byte is the single most load-bearing fact in
// this allocator -- and two runs in a row ended with `pool_reserved` frozen at
// byte-identical 44560285696 despite hundreds of trims. Report the reservation
// either side of the call so the trace answers that directly instead of
// leaving it to be inferred from the figures a later allocation happens to
// print.
void CudaAllocator::trim_pools() {
  bool trace = memory_trace_enabled();
  for (auto p : mem_pools_) {
    if (p) {
      size_t before = 0;
      if (trace) {
        cudaMemPoolGetAttribute(p, cudaMemPoolAttrReservedMemCurrent, &before);
      }
      CHECK_CUDA_ERROR(cudaMemPoolTrimTo(p, 0));
      if (trace) {
        size_t after = 0;
        size_t used = 0;
        size_t free = 0;
        size_t total = 0;
        cudaMemPoolGetAttribute(p, cudaMemPoolAttrReservedMemCurrent, &after);
        cudaMemPoolGetAttribute(p, cudaMemPoolAttrUsedMemCurrent, &used);
        cudaMemGetInfo(&free, &total);
        static std::atomic<size_t> seen{0};
        size_t n = seen.fetch_add(1, std::memory_order_relaxed);
        if (n < 16 || n % 64 == 0) {
          fmt::print(
              stderr,
              "[mlx][cuda-mem] trim n={} reserved={}->{} released={} "
              "pool_used={} free={}\n",
              n,
              before,
              after,
              before > after ? before - after : 0,
              used,
              free);
        }
      }
    }
  }
}

// Every figure the allocator reports about itself -- active, cache, peak,
// limit -- describes only what it allocated. What decides whether the next
// graph instantiation succeeds is the device's own free figure and how much
// of the card is held by consumers this allocator never sees. Set
// MLX_CUDA_MEMORY_TRACE=1 to have both printed at the moments that matter:
// when the device refuses an allocation, and when a wait gives up and
// allocates anyway. Rate-limited, because a failing run produces thousands.
static bool memory_trace_enabled() {
  static const bool enabled = []() {
    const char* v = std::getenv("MLX_CUDA_MEMORY_TRACE");
    return v && *v && *v != '0';
  }();
  return enabled;
}

void CudaAllocator::trace_device_memory(
    const char* where,
    std::atomic<size_t>& seen,
    size_t size) {
  if (!memory_trace_enabled()) {
    return;
  }
  size_t n = seen.fetch_add(1, std::memory_order_relaxed);
  if (n >= 32 && n % 256 != 0) {
    return;
  }
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess || device < 0 ||
      static_cast<size_t>(device) >= mem_pools_.size()) {
    return;
  }
  size_t free = 0;
  size_t total = 0;
  size_t reserved = 0;
  size_t used = 0;
  cudaMemGetInfo(&free, &total);
  if (auto pool = mem_pools_[device]) {
    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved);
    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used);
  }
  size_t held = total - free;
  // Everything the device has handed out that did not come from this pool.
  // This is the figure no other diagnostic in the stack reports, and the one
  // the reserve is trying to predict.
  size_t foreign = held > reserved ? held - reserved : 0;
  fmt::print(
      stderr,
      "[mlx][cuda-mem] {} n={} size={} free={} total={} pool_reserved={} "
      "pool_used={} foreign={} reserve={} active={} cache={} limit={}\n",
      where,
      n,
      size,
      free,
      total,
      reserved,
      used,
      foreign,
      free_limit_.load(std::memory_order_relaxed),
      active_memory_,
      get_cache_memory(),
      memory_limit());
}

void CudaAllocator::raise_reserve(size_t reserve) {
  // Half the card bounds what any single measurement may claim. Past that the
  // pool cap it feeds leaves too little to train under, and a sample taken
  // while another process holds most of the device would otherwise set a cap
  // this allocator can never satisfy.
  reserve = std::min(reserve, total_memory_ / 2);
  size_t current = free_limit_.load(std::memory_order_relaxed);
  while (reserve > current &&
         !free_limit_.compare_exchange_weak(
             current, reserve, std::memory_order_relaxed)) {
  }
}

// A refusal is the one moment sampling can never reach on its own: a sample
// sees the out-of-pool allocations that succeeded, never the one that failed.
// It is also the only evidence that exists about how much free memory a graph
// instantiation or a kernel's launch resources actually wanted, because that
// demand never passes through this allocator at all. So treat it as the
// measurement it is and widen the reserve, which lowers the cap on what the
// pool may keep reserved, so the next trim hands the difference back to the
// device and the retry finds the room this attempt did not.
//
// An earlier version of this grew the reserve the same way and ran away
// instead of converging, because the only thing the reserve fed back then was
// a precondition in `wait_for_physical_memory` that the pool's idle
// reservation short-circuited -- the raises never freed a byte, so they never
// stopped. Growth converges only when each step actually moves memory.
//
// The step is taken from what the device had free at the refusal rather than
// from the reserve's own previous value, because that is the quantity the
// failed allocation just proved insufficient -- it wanted more than |free| and
// did not get it. Compounding the reserve instead would make the size of the
// raise depend on how many refusals happened to arrive before the next trim,
// and a failing step produces them in bursts of thousands; anchoring it to a
// measurement makes a burst converge on one answer instead of exponentiating
// through it.
void CudaAllocator::report_out_of_memory() {
  static std::atomic<size_t> seen{0};
  trace_device_memory("refused", seen, 0);
  size_t free = 0;
  size_t total = 0;
  if (cudaMemGetInfo(&free, &total) != cudaSuccess) {
    return;
  }
  raise_reserve(free + free / 4 + page_size);
}

// Report what the device can still hand out to anyone, and take a measurement
// of the reserve while the numbers are in hand: whatever the device holds
// beyond this allocator's own reservation belongs to consumers the pool cannot
// serve, so a sample showing more of it than the reserve holds back proves the
// reserve too small. Resident managed memory and the scalar pool land on that
// side of the subtraction too; both are small, and counting them here only
// makes the reserve larger than it strictly has to be, which is the direction
// that is safe to be wrong in.
size_t CudaAllocator::observe_device_memory(int device) {
  auto pool = mem_pools_[device];
  size_t free = 0;
  size_t total = 0;
  CHECK_CUDA_ERROR(cudaMemGetInfo(&free, &total));
  size_t reserved = 0;
  CHECK_CUDA_ERROR(cudaMemPoolGetAttribute(
      pool, cudaMemPoolAttrReservedMemCurrent, &reserved));

  size_t held = total - free;
  raise_reserve(held > reserved ? held - reserved : 0);
  return free;
}

// The memory limit bounds what this allocator counts, not what the device
// holds: the CUDA context, instantiated graph executables and pool
// fragmentation all sit on top of it. So the only figure worth testing here is
// the device's own free memory, which already contains every one of those
// consumers without this allocator having to model any of them.
//
// Idle reservation inside the pool is deliberately not counted as available,
// though an earlier version of this did count it. Two measurements killed that
// idea. It is not sound: a run died on an 80 MB `cudaMallocAsync` while the
// pool held 13.56 GB reserved-but-unused, because reuse is stream-ordered and
// idle blocks belong to whichever stream freed them. And counting it is what
// disables this function entirely -- a pool with gigabytes idle answers "yes"
// to every request, so the loop below never runs, the cache is never handed
// back, in-flight work is never drained, and the device is free to sit below
// its reserve indefinitely. That short-circuit is also why trimming looked
// powerless: the only trims left running were the ones against a hot pool
// where nearly every granule still holds something live. The trim that
// reclaims is the one after `wait_for_completion` has drained the work that
// was pinning those granules, and reaching it requires asking the device --
// not the pool -- whether there is room.
void CudaAllocator::wait_for_physical_memory(size_t size, int device) {
  auto servable = [this, device, size]() {
    size_t free = observe_device_memory(device);
    return free >= size + free_limit_.load(std::memory_order_relaxed);
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
    static std::atomic<size_t> seen{0};
    trace_device_memory("fallthrough", seen, size);
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

// Derived rather than stored, because the reserve it is measured against
// moves. A caller asking for a limit is expressing a budget, not a fact about
// the device: honour it only down to what the device has actually left us,
// otherwise a framework that computes its ceiling from the device total spends
// the reserve and gets back the out-of-memory the reserve exists to prevent.
size_t CudaAllocator::memory_limit() const {
  size_t reserve = free_limit_.load(std::memory_order_relaxed);
  size_t bound = total_memory_ > reserve ? total_memory_ - reserve : 0;
  return std::min(requested_limit_.load(std::memory_order_relaxed), bound);
}

size_t CudaAllocator::get_memory_limit() {
  return memory_limit();
}

// Returns the limit that was in force.
size_t CudaAllocator::set_memory_limit(size_t limit) {
  std::lock_guard lock(mutex_);
  size_t previous = memory_limit();
  requested_limit_.store(limit, std::memory_order_relaxed);
  return previous;
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
