#include "cogstate/math_harmonization.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace cogstate {

double MathematicalHarmonization::polynomial_map(double x, const std::vector<double>& coefficients) {
  double acc = 0.0;
  double x_pow = 1.0;
  for (const double c : coefficients) {
    acc += c * x_pow;
    x_pow *= x;
  }
  return acc;
}

double MathematicalHarmonization::map_sdnn_to_rmssd(double sdnn_ms) {
  // Empirical cubic approximation coefficients.
  static const std::vector<double> coeff{2.731, 0.612, -0.0019, 0.000004};
  return polynomial_map(sdnn_ms, coeff);
}

double MathematicalHarmonization::map_rmssd_to_sdnn(double rmssd_ms) {
  // Empirical cubic inverse approximation coefficients.
  static const std::vector<double> coeff{-1.947, 1.381, -0.0047, 0.000008};
  return polynomial_map(rmssd_ms, coeff);
}

double MathematicalHarmonization::robust_zscore_14d(double current, const std::vector<double>& history) {
  if (history.empty()) {
    return 0.0;
  }
  std::vector<double> sorted = history;
  std::sort(sorted.begin(), sorted.end());
  const auto percentile = [&sorted](double p) {
    const double idx = p * (static_cast<double>(sorted.size()) - 1.0);
    const auto low = static_cast<std::size_t>(std::floor(idx));
    const auto high = static_cast<std::size_t>(std::ceil(idx));
    const double w = idx - static_cast<double>(low);
    return sorted[low] * (1.0 - w) + sorted[high] * w;
  };
  const double median = percentile(0.5);
  const double q1 = percentile(0.25);
  const double q3 = percentile(0.75);
  const double iqr = std::max(q3 - q1, 1e-6);
  return (current - median) / iqr;
}

std::array<double, 4> MathematicalHarmonization::smooth_sleep_probabilities(
    const std::array<double, 4>& raw_probabilities,
    const std::array<std::array<double, 4>, 4>& inverse_confusion) {
  std::array<double, 4> out{};
  for (int i = 0; i < 4; ++i) {
    double val = 0.0;
    for (int j = 0; j < 4; ++j) {
      val += inverse_confusion[i][j] * raw_probabilities[j];
    }
    out[i] = std::max(0.0, val);
  }
  const double sum = std::accumulate(out.begin(), out.end(), 0.0);
  if (sum <= 1e-12) {
    return {0.25, 0.25, 0.25, 0.25};
  }
  for (double& v : out) {
    v /= sum;
  }
  return out;
}

}  // namespace cogstate

