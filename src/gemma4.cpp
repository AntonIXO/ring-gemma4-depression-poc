#include "cogstate/gemma4.hpp"

#include <algorithm>
#include <stdexcept>

namespace cogstate {

Gemma4Adapter::Gemma4Adapter(LoRAConfig config) : config_(std::move(config)) {
  if (config_.rank <= 0 || config_.alpha <= 0.0 || config_.dropout < 0.0 || config_.dropout >= 1.0) {
    throw std::runtime_error("Invalid LoRA configuration");
  }
}

const LoRAConfig& Gemma4Adapter::config() const { return config_; }

std::vector<double> Gemma4Adapter::lora_update(const std::vector<double>& base_weights,
                                               const std::vector<double>& gradients) const {
  if (base_weights.size() != gradients.size()) {
    throw std::runtime_error("base_weights and gradients size mismatch");
  }
  std::vector<double> updated(base_weights.size(), 0.0);
  const double scale = config_.alpha / static_cast<double>(config_.rank);
  for (std::size_t i = 0; i < base_weights.size(); ++i) {
    updated[i] = base_weights[i] - scale * (1.0 - config_.dropout) * gradients[i];
  }
  return updated;
}

std::vector<int> Gemma4Adapter::substitute_pad_token(const std::vector<int>& tokens,
                                                     int blocked_pad_token,
                                                     int replacement_token) const {
  if (blocked_pad_token == replacement_token) {
    throw std::runtime_error("blocked and replacement tokens must differ");
  }
  std::vector<int> out = tokens;
  std::replace(out.begin(), out.end(), blocked_pad_token, replacement_token);
  return out;
}

bool Gemma4Adapter::validate_cohort(const SmartRingCohort& cohort) const {
  return static_cast<int>(cohort.participant_ids.size()) == cohort.expected_size;
}

}  // namespace cogstate

