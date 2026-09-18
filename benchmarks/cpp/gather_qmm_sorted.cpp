// What the sorted `gather_qmm` path is for, at the shapes that motivated it:
// one MoE projection of Qwen3-30B-A3B-Instruct at a training sequence length,
// where every (token, expert) pair is a one-row matmul and the expert's packed
// weights are otherwise re-read once per pair.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

#include "mlx/mlx.h"

using namespace mlx::core;

namespace {

// Sorted expert assignments for `tokens` tokens routed to `top_k` experts
// each, drawn uniformly. Real routing is not uniform, but the tile arithmetic
// only cares about run lengths, and a uniform draw is the case with the least
// reuse per tile — the pessimistic end of what routing produces.
array sorted_expert_indices(int tokens, int top_k, int experts, uint32_t seed) {
  std::vector<uint32_t> idx(static_cast<size_t>(tokens) * top_k);
  std::mt19937 rng(seed);
  std::uniform_int_distribution<uint32_t> pick(0, experts - 1);
  for (auto& v : idx) {
    v = pick(rng);
  }
  std::sort(idx.begin(), idx.end());
  return array(idx.data(), {static_cast<int>(idx.size())}, uint32);
}

double time_ms(const std::function<array()>& f, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) {
    eval(f());
  }
  synchronize();
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    eval(f());
  }
  synchronize();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  return elapsed.count() / iters;
}

void bench(const char* label, int n, int k) {
  constexpr int experts = 128;
  constexpr int tokens = 5700;
  constexpr int top_k = 8;
  constexpr int group_size = 64;
  constexpr int bits = 4;

  auto rhs_indices = sorted_expert_indices(tokens, top_k, experts, 0);
  int pairs = rhs_indices.size();

  auto w = quantize(
      astype(random::normal({experts, n, k}, float32), bfloat16),
      group_size,
      bits);
  auto x = astype(random::normal({pairs, 1, k}, float32), bfloat16);
  eval(w);
  eval(x, rhs_indices);

  auto call = [&](bool sorted) {
    return gather_qmm(
        x,
        w[0],
        w[1],
        w[2],
        /* lhs_indices = */ std::nullopt,
        rhs_indices,
        /* transpose = */ true,
        group_size,
        bits,
        "affine",
        sorted);
  };

  double unsorted = time_ms([&]() { return call(false); }, 2, 10);
  double sorted = time_ms([&]() { return call(true); }, 2, 10);
  std::printf(
      "%-12s n=%-5d k=%-5d pairs=%-6d  unsorted %8.2f ms  sorted %8.2f ms  %5.2fx\n",
      label,
      n,
      k,
      pairs,
      unsorted,
      sorted,
      unsorted / sorted);
}

} // namespace

int main() {
  // The two projection shapes of one MoE block: the gate and up projections
  // read the residual stream and write the expert hidden size, the down
  // projection reads it back.
  bench("gate/up", 768, 2048);
  bench("down", 2048, 768);
  return 0;
}
