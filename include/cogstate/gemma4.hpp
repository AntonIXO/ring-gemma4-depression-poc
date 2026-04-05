#pragma once

#include <string>
#include <vector>

namespace cogstate {

struct LoRAConfig {
  int rank{16};
  double alpha{32.0};
  double dropout{0.05};
  std::vector<std::string> target_modules{"q_proj", "k_proj", "v_proj", "o_proj"};
};

struct SmartRingCohort {
  std::vector<std::string> participant_ids;
  int expected_size{60};
};

class Gemma4Adapter {
 public:
  explicit Gemma4Adapter(LoRAConfig config);
  const LoRAConfig& config() const;

  std::vector<double> lora_update(const std::vector<double>& base_weights,
                                  const std::vector<double>& gradients) const;
  std::vector<int> substitute_pad_token(const std::vector<int>& tokens, int blocked_pad_token,
                                        int replacement_token) const;
  bool validate_cohort(const SmartRingCohort& cohort) const;

 private:
  LoRAConfig config_;
};

}  // namespace cogstate

