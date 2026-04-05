#include "cogstate/fusion.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace cogstate {

PatchTemporalEncoder::PatchTemporalEncoder(PatchEncoderConfig config) : config_(config) {
  if (config_.patch_length <= 0 || config_.patch_stride <= 0 || config_.hidden_dim <= 0) {
    throw std::runtime_error("Invalid patch encoder configuration");
  }
}

Vector PatchTemporalEncoder::encode(const Matrix& sequence) const {
  if (sequence.empty()) {
    return {};
  }
  const std::size_t feature_dim = sequence.front().size();
  if (feature_dim == 0) {
    return {};
  }

  std::vector<Vector> patches;
  for (std::size_t start = 0; start + static_cast<std::size_t>(config_.patch_length) <= sequence.size();
       start += static_cast<std::size_t>(config_.patch_stride)) {
    Vector patch;
    patch.reserve(static_cast<std::size_t>(config_.patch_length) * feature_dim);
    for (int t = 0; t < config_.patch_length; ++t) {
      const auto& row = sequence[start + static_cast<std::size_t>(t)];
      patch.insert(patch.end(), row.begin(), row.end());
    }
    patches.push_back(std::move(patch));
  }
  if (patches.empty()) {
    return {};
  }

  std::vector<Vector> projected;
  projected.reserve(patches.size());
  for (const auto& patch : patches) {
    Vector h(static_cast<std::size_t>(config_.hidden_dim), 0.0);
    for (std::size_t i = 0; i < patch.size(); ++i) {
      for (int d = 0; d < config_.hidden_dim; ++d) {
        const double w = std::sin(static_cast<double>((i + 1) * (d + 3)) * 0.001);
        h[static_cast<std::size_t>(d)] += patch[i] * w;
      }
    }
    projected.push_back(std::move(h));
  }

  Vector query(static_cast<std::size_t>(config_.hidden_dim), 1.0 / static_cast<double>(config_.hidden_dim));
  std::vector<double> scores(projected.size(), 0.0);
  for (std::size_t i = 0; i < projected.size(); ++i) {
    scores[i] = std::inner_product(projected[i].begin(), projected[i].end(), query.begin(), 0.0);
  }
  const double max_score = *std::max_element(scores.begin(), scores.end());
  double denom = 0.0;
  for (double& s : scores) {
    s = std::exp(s - max_score);
    denom += s;
  }
  if (denom <= 1e-12) {
    return Vector(static_cast<std::size_t>(config_.hidden_dim), 0.0);
  }
  for (double& s : scores) {
    s /= denom;
  }

  Vector fused(static_cast<std::size_t>(config_.hidden_dim), 0.0);
  for (std::size_t i = 0; i < projected.size(); ++i) {
    for (int d = 0; d < config_.hidden_dim; ++d) {
      fused[static_cast<std::size_t>(d)] += scores[i] * projected[i][static_cast<std::size_t>(d)];
    }
  }
  return fused;
}

}  // namespace cogstate

