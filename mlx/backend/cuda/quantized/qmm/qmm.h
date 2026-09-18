// Copyright © 2026 Apple Inc.

#pragma once

#include "mlx/backend/cuda/device.h"
#include "mlx/primitives.h"

#include <optional>

namespace mlx::core {

bool supports_qmm_sm90(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& out,
    bool transpose,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::Device& device);

void qmm_sm90(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    array& out,
    int bits,
    int group_size,
    cu::CommandEncoder& encoder,
    Stream s);

bool supports_qmm_sm80(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& out,
    bool transpose,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::Device& device);

void qmm_sm80(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const std::optional<array>& lhs_indices,
    const std::optional<array>& rhs_indices,
    array& out,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::CommandEncoder& encoder);

/// Whether the sorted-expert fast path below applies to this problem.
///
/// It is not a capability check on the device — `gather_qmm_rhs` runs
/// `qmm_sm80`, so `supports_qmm_sm80` still has to hold — but a check that
/// sorting the right-hand side is worth the two extra passes over the
/// activations that reordering them costs.
bool supports_gather_qmm_rhs(
    const array& x,
    const array& w,
    const array& out,
    bool right_sorted);

/// Gather matmul against sorted expert indices.
///
/// Every (row, expert) pair is its own one-row matmul when the indices are
/// taken as they come, and at 4 bits an expert's packed weights are read once
/// per pair. Sorted indices mean the pairs for one expert are contiguous, so
/// they can be packed into fixed-height tiles and run as a batch of ordinary
/// matmuls that read each expert's weights once per tile instead.
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
    cu::CommandEncoder& encoder);

bool supports_qmm_naive(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& out,
    bool transpose,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::Device& device);

void qmm_naive(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const std::optional<array>& global_scale,
    const std::optional<array>& lhs_indices,
    const std::optional<array>& rhs_indices,
    array& out,
    bool transpose,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::CommandEncoder& encoder);

bool supports_fp_qmv(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& out,
    bool transpose,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::Device& device);

void fp_qmv(
    const array& x,
    const array& w,
    const array& scales,
    array& out,
    int bits,
    int group_size,
    cu::CommandEncoder& encoder,
    Stream s);

bool supports_qmv(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& out,
    bool transpose,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::Device& device);

void qmv(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const std::optional<array>& global_scale,
    array& out,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::CommandEncoder& encoder);

void gather_qmv(
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
    cu::CommandEncoder& encoder);

} // namespace mlx::core
