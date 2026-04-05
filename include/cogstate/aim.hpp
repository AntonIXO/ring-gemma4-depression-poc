#pragma once

#include <vector>

#include "cogstate/types.hpp"

namespace cogstate {

struct AimConfig {
  double inherited_decay{0.9};
  double nmar_run_boost{1.25};
};

struct AimOutput {
  Matrix imputed;
  std::vector<std::vector<double>> adaptive_weights;
};

class AdaptiveInheritedMasking {
 public:
  explicit AdaptiveInheritedMasking(AimConfig config);
  AimOutput apply(const Matrix& signal, const std::vector<std::vector<bool>>& observed) const;

 private:
  AimConfig config_;
};

}  // namespace cogstate

