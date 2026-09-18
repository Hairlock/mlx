#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/quantized/qmm/qmm.h"
#include "mlx/dtype_utils.h"

#include <cooperative_groups.h>
#include <cuda/cmath>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

// Segment sorted expert indices into fixed-height tiles.
//
// `rhs_indices` is sorted, so the pairs routed to one expert are a contiguous
// run. Chopping each run into tiles of `tile_m` rows turns the whole gather
// into a batch of `tile_m`-row matmuls, one expert per batch entry, which is
// the shape `qmm_sm80` is efficient at.
//
// Written for one block: the histogram needs a single shared counter array,
// and the per-expert tile offsets are a scan over `group_count` — at most a
// few thousand entries, once per projection.
template <int N_READS>
__global__ void qmm_rhs_tile_map(
    const uint32_t* rhs_indices,
    int64_t size,
    int group_count,
    int tile_m,
    int max_tiles,
    uint32_t* tile_expert,
    uint32_t* tile_src_base,
    uint32_t* tile_src_count) {
  auto block = cg::this_thread_block();

  // [0, group_count): the end offset of each expert's run.
  // [group_count, 2 * group_count): the first tile each expert owns.
  // [2 * group_count]: how many tiles are occupied in total.
  extern __shared__ uint32_t shared[];
  uint32_t* cum_histo = shared;
  uint32_t* tile_off = shared + group_count;
  uint32_t* total_tiles = shared + 2 * group_count;

  int tid = block.thread_rank();
  int num_threads = block.num_threads();

  for (int e = tid; e < group_count; e += num_threads) {
    cum_histo[e] = 0;
  }
  block.sync();

  // The indices are sorted, so a position where the value changes is the
  // cumulative count of every expert it stepped over. Experts with no pairs
  // land on the same offset as their predecessor and so get an empty run.
  int64_t elems_per_block = int64_t(num_threads) * N_READS;
  for (int64_t r = 0; r < cuda::ceil_div(size, elems_per_block); ++r) {
    for (int i = 0; i < N_READS; ++i) {
      int64_t pos = r * elems_per_block + int64_t(tid) * N_READS + i;
      if (pos >= size) {
        break;
      }
      uint32_t elem = rhs_indices[pos];
      uint32_t next = pos < size - 1 ? rhs_indices[pos + 1]
                                     : static_cast<uint32_t>(group_count);
      while (elem < next) {
        cum_histo[elem] = static_cast<uint32_t>(pos + 1);
        elem++;
      }
    }
  }
  block.sync();

  if (tid == 0) {
    uint32_t acc = 0;
    for (int e = 0; e < group_count; ++e) {
      tile_off[e] = acc;
      uint32_t start = e == 0 ? 0 : cum_histo[e - 1];
      acc += cuda::ceil_div(cum_histo[e] - start, static_cast<uint32_t>(tile_m));
    }
    *total_tiles = acc;
  }
  block.sync();

  for (int e = tid; e < group_count; e += num_threads) {
    uint32_t start = e == 0 ? 0 : cum_histo[e - 1];
    uint32_t count = cum_histo[e] - start;
    uint32_t base = tile_off[e];
    for (uint32_t j = 0, off = 0; off < count; ++j, off += tile_m) {
      tile_expert[base + j] = static_cast<uint32_t>(e);
      tile_src_base[base + j] = start + off;
      tile_src_count[base + j] =
          min(static_cast<uint32_t>(tile_m), count - off);
    }
  }

  // The tile count is bounded statically rather than read back to the host, so
  // the tail is usually short of the bound. Those tiles are launched like any
  // other and their results discarded, which costs a few percent of the
  // matmul and saves a device-to-host synchronization per projection.
  for (int t = *total_tiles + tid; t < max_tiles; t += num_threads) {
    tile_expert[t] = 0;
    tile_src_base[t] = 0;
    tile_src_count[t] = 0;
  }
}

// Copy each tile's rows out of `x` and into its slot in the padded buffer,
// zeroing the rows a partial tile does not fill.
template <typename T>
__global__ void qmm_rhs_scatter(
    const T* x,
    const uint32_t* lhs_indices,
    const uint32_t* tile_src_base,
    const uint32_t* tile_src_count,
    T* x_pad,
    int tile_m,
    int k) {
  int row = blockIdx.x;
  int tile = row / tile_m;
  int r = row - tile * tile_m;

  T* dst = x_pad + static_cast<int64_t>(row) * k;
  if (r < static_cast<int>(tile_src_count[tile])) {
    const T* src = x +
        static_cast<int64_t>(lhs_indices[tile_src_base[tile] + r]) * k;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
      dst[i] = src[i];
    }
  } else {
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
      dst[i] = T(0);
    }
  }
}

// Undo the scatter: each occupied padded row goes back to the pair it came
// from, which is its position in the sorted order and so its row of `out`.
template <typename T>
__global__ void qmm_rhs_gather(
    const T* out_pad,
    const uint32_t* tile_src_base,
    const uint32_t* tile_src_count,
    T* out,
    int tile_m,
    int n) {
  int row = blockIdx.x;
  int tile = row / tile_m;
  int r = row - tile * tile_m;
  if (r >= static_cast<int>(tile_src_count[tile])) {
    return;
  }

  const T* src = out_pad + static_cast<int64_t>(row) * n;
  T* dst = out + static_cast<int64_t>(tile_src_base[tile] + r) * n;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    dst[i] = src[i];
  }
}

} // namespace cu

namespace {

// Rows per tile. Every tile reads its expert's packed weights once, so a
// taller tile amortizes that read over more pairs; a tile taller than an
// expert's average run only pads. 64 is the largest `qmm_sm80` tiles to.
int choose_tile_m(int64_t pairs, int experts) {
  int64_t per_expert = pairs / std::max(experts, 1);
  if (per_expert >= 64) {
    return 64;
  }
  return per_expert >= 32 ? 32 : 16;
}

} // namespace

bool supports_gather_qmm_rhs(
    const array& x,
    const array& w,
    const array& out,
    bool right_sorted) {
  if (!right_sorted) {
    return false;
  }
  // Only the M == 1 case is pathological enough to be worth the round trip
  // through a padded buffer: with more rows per pair the weight read is
  // already amortized by the tiler.
  int m = out.ndim() > 1 ? out.shape(-2) : 1;
  if (m != 1) {
    return false;
  }
  if (w.ndim() != 3) {
    return false;
  }
  int64_t pairs = out.size() / out.shape(-1);
  int experts = w.shape(0);
  // Below a few pairs per expert the tiles are mostly padding and the two
  // extra passes over the activations cost more than the reuse wins.
  return pairs >= 32 && pairs / std::max(experts, 1) >= 4;
}

void gather_qmm_rhs(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::CommandEncoder& encoder) {
  int n = out.shape(-1);
  int k = x.shape(-1);
  int experts = w.shape(0);
  int64_t pairs = out.size() / n;

  int tile_m = choose_tile_m(pairs, experts);
  // Each expert wastes at most one partial tile, so this bounds the occupied
  // tile count without asking the device how many there actually are.
  int max_tiles =
      static_cast<int>(cuda::ceil_div(pairs, int64_t(tile_m))) + experts;
  int64_t padded_rows = int64_t(max_tiles) * tile_m;

  array tile_expert(
      cu::malloc_async(max_tiles * sizeof(uint32_t), encoder),
      {max_tiles},
      uint32);
  array tile_src_base(
      cu::malloc_async(max_tiles * sizeof(uint32_t), encoder),
      {max_tiles},
      uint32);
  array tile_src_count(
      cu::malloc_async(max_tiles * sizeof(uint32_t), encoder),
      {max_tiles},
      uint32);
  array x_pad(
      cu::malloc_async(padded_rows * k * x.itemsize(), encoder),
      {max_tiles, tile_m, k},
      x.dtype());
  array out_pad(
      cu::malloc_async(padded_rows * n * out.itemsize(), encoder),
      {max_tiles, tile_m, n},
      out.dtype());
  encoder.add_temporary(tile_expert);
  encoder.add_temporary(tile_src_base);
  encoder.add_temporary(tile_src_count);
  encoder.add_temporary(x_pad);
  encoder.add_temporary(out_pad);

  constexpr int N_READS = 4;
  int map_threads = std::min<int64_t>(
      1024, std::max<int64_t>(experts, cuda::ceil_div(pairs, int64_t(N_READS))));
  encoder.set_input_array(rhs_indices);
  encoder.set_output_array(tile_expert);
  encoder.set_output_array(tile_src_base);
  encoder.set_output_array(tile_src_count);
  encoder.add_kernel_node_ex(
      cu::qmm_rhs_tile_map<N_READS>,
      dim3(1),
      dim3(static_cast<uint32_t>(map_threads)),
      {},
      (2 * experts + 1) * sizeof(uint32_t),
      gpu_ptr<uint32_t>(rhs_indices),
      pairs,
      experts,
      tile_m,
      max_tiles,
      gpu_ptr<uint32_t>(tile_expert),
      gpu_ptr<uint32_t>(tile_src_base),
      gpu_ptr<uint32_t>(tile_src_count));

  dispatch_float_types(x.dtype(), "gather_qmm_rhs", [&](auto type_tag) {
    using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;

    encoder.set_input_array(x);
    encoder.set_input_array(lhs_indices);
    encoder.set_input_array(tile_src_base);
    encoder.set_input_array(tile_src_count);
    encoder.set_output_array(x_pad);
    encoder.add_kernel_node(
        cu::qmm_rhs_scatter<T>,
        dim3(static_cast<uint32_t>(padded_rows)),
        dim3(std::min(k, 1024)),
        gpu_ptr<T>(x),
        gpu_ptr<uint32_t>(lhs_indices),
        gpu_ptr<uint32_t>(tile_src_base),
        gpu_ptr<uint32_t>(tile_src_count),
        gpu_ptr<T>(x_pad),
        tile_m,
        k);

    // `x_pad` is [tiles, tile_m, k] and `tile_expert` is one expert per tile,
    // so this is the ordinary batched gather matmul with the batch dimension
    // reinterpreted as the tile index — the kernel itself needs nothing new.
    qmm_sm80(
        x_pad,
        w,
        scales,
        biases,
        std::nullopt,
        tile_expert,
        out_pad,
        bits,
        group_size,
        mode,
        encoder);

    encoder.set_input_array(out_pad);
    encoder.set_input_array(tile_src_base);
    encoder.set_input_array(tile_src_count);
    encoder.set_output_array(out);
    encoder.add_kernel_node(
        cu::qmm_rhs_gather<T>,
        dim3(static_cast<uint32_t>(padded_rows)),
        dim3(std::min(n, 1024)),
        gpu_ptr<T>(out_pad),
        gpu_ptr<uint32_t>(tile_src_base),
        gpu_ptr<uint32_t>(tile_src_count),
        gpu_ptr<T>(out),
        tile_m,
        n);
  });
}

} // namespace mlx::core
