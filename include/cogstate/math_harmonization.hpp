#pragma once

#include <array>
#include <vector>

namespace cogstate {

class MathematicalHarmonization {
 public:
  static double polynomial_map(double x, const std::vector<double>& coefficients);
  static double map_sdnn_to_rmssd(double sdnn_ms);
  static double map_rmssd_to_sdnn(double rmssd_ms);
  static double robust_zscore_14d(double current, const std::vector<double>& history);
  static std::array<double, 4> smooth_sleep_probabilities(
      const std::array<double, 4>& raw_probabilities,
      const std::array<std::array<double, 4>, 4>& inverse_confusion);
};

}  // namespace cogstate

