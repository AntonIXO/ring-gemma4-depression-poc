#include <fstream>
#include <iostream>

#include "cogstate/aim.hpp"
#include "cogstate/embedding.hpp"
#include "cogstate/fusion.hpp"
#include "cogstate/gemma4.hpp"
#include "cogstate/math_harmonization.hpp"
#include "cogstate/types.hpp"

int main() {
  using namespace cogstate;

  Matrix sequence(96, Vector{70.0, 42.0, 10.0, 0.15, 36.6});
  for (std::size_t t = 0; t < sequence.size(); ++t) {
    sequence[t][0] += static_cast<double>(t % 9) * 0.4;
    sequence[t][1] += static_cast<double>(t % 5) * 0.8;
  }

  std::vector<std::vector<bool>> observed(sequence.size(), std::vector<bool>(sequence.front().size(), true));
  for (std::size_t t = 20; t < 28; ++t) {
    observed[t][1] = false;
  }

  AdaptiveInheritedMasking aim({0.92, 1.35});
  const AimOutput aim_out = aim.apply(sequence, observed);

  PatchTemporalEncoder patch_encoder({8, 4, 64});
  const Vector fused = patch_encoder.encode(aim_out.imputed);

  EpochRecord prev{"demo", 0, aim_out.imputed[0], {true, true, true, true, true}};
  EpochRecord cur{"demo", 900, aim_out.imputed[1], {true, true, true, true, true}};
  MomentVectorEncoder moment_encoder;
  const Vector moment = moment_encoder.encode(cur, &prev);

  std::vector<EpochRecord> day_epochs;
  day_epochs.reserve(96);
  for (std::size_t i = 0; i < aim_out.imputed.size(); ++i) {
    day_epochs.push_back({"demo", static_cast<std::int64_t>(i * 900), aim_out.imputed[i],
                          std::vector<bool>(aim_out.imputed[i].size(), true)});
  }
  DayVectorEncoder day_encoder;
  DayVector day_vec = day_encoder.encode("demo", 0, day_epochs);

  std::vector<DayVector> days(14, day_vec);
  WeekVectorEncoder week_encoder;
  WeekVector week_vec = week_encoder.encode("demo", 0, days);

  Gemma4Adapter gemma({16, 32.0, 0.05, {"q_proj", "k_proj", "v_proj", "o_proj"}});
  const std::vector<double> base(128, 0.01);
  const std::vector<double> grad(128, 0.0005);
  const auto updated = gemma.lora_update(base, grad);
  const auto tokens = gemma.substitute_pad_token({1, 2, 0, 0, 3}, 0, 5);

  SmartRingCohort cohort;
  cohort.expected_size = 60;
  cohort.participant_ids.resize(60, "ring_participant");
  const bool cohort_ok = gemma.validate_cohort(cohort);

  std::ofstream out("models/training_artifact.txt");
  out << "fused_dim=" << fused.size() << "\n";
  out << "moment_dim=" << moment.size() << "\n";
  out << "day_dim=" << day_vec.values.size() << "\n";
  out << "week_dim=" << week_vec.values.size() << "\n";
  out << "lora_weights_dim=" << updated.size() << "\n";
  out << "cohort_ok=" << (cohort_ok ? "true" : "false") << "\n";
  out << "pad_sub_tokens=";
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    out << tokens[i];
  }
  out << "\n";
  out.close();

  std::cout << "Training artifact written to models/training_artifact.txt\n";
  return 0;
}

