#include "cogstate/aim.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cogstate {

AdaptiveInheritedMasking::AdaptiveInheritedMasking(AimConfig config) : config_(config) {
  if (config_.inherited_decay <= 0.0 || config_.inherited_decay > 1.0) {
    throw std::runtime_error("inherited_decay must be in (0, 1]");
  }
}

AimOutput AdaptiveInheritedMasking::apply(const Matrix& signal,
                                          const std::vector<std::vector<bool>>& observed) const {
  if (signal.size() != observed.size()) {
    throw std::runtime_error("signal/observed time dimension mismatch");
  }
  if (signal.empty()) {
    return {};
  }

  const std::size_t t_len = signal.size();
  const std::size_t dim = signal.front().size();
  for (std::size_t t = 0; t < t_len; ++t) {
    if (signal[t].size() != dim || observed[t].size() != dim) {
      throw std::runtime_error("signal/observed feature dimension mismatch");
    }
  }

  Matrix imputed = signal;
  std::vector<std::vector<double>> weights(t_len, std::vector<double>(dim, 1.0));
  std::vector<double> last_value(dim, 0.0);
  std::vector<int> missing_run(dim, 0);
  std::vector<bool> initialized(dim, false);

  for (std::size_t t = 0; t < t_len; ++t) {
    for (std::size_t d = 0; d < dim; ++d) {
      if (observed[t][d]) {
        initialized[d] = true;
        last_value[d] = signal[t][d];
        missing_run[d] = 0;
        weights[t][d] = 1.0;
      } else {
        missing_run[d] += 1;
        const double decay = std::pow(config_.inherited_decay, static_cast<double>(missing_run[d]));
        const bool nmar_like = missing_run[d] >= 3;
        const double boost = nmar_like ? config_.nmar_run_boost : 1.0;
        weights[t][d] = std::min(2.0, boost * decay);
        if (initialized[d]) {
          imputed[t][d] = last_value[d] * weights[t][d];
        } else {
          imputed[t][d] = 0.0;
        }
      }
    }
  }

  return {imputed, weights};
}

}  // namespace cogstate
