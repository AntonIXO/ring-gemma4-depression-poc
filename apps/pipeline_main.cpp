#include <array>
#include <iostream>

#include "cogstate/data_fusion.hpp"
#include "cogstate/math_harmonization.hpp"

int main() {
  using namespace cogstate;

  FusionConfig cfg;
  cfg.epoch_minutes = 15;
  cfg.steps_per_day = 96;
  cfg.lookback_days = 14;
  cfg.unified_feature_order = {"hr_bpm", "hrv_ms", "steps", "eda", "temp_c", "sleep_prob_wake",
                               "sleep_prob_light", "sleep_prob_deep", "sleep_prob_rem"};

  DataFusionPipeline pipeline(cfg);
  const std::vector<DatasetSpec> specs = {
      {"LifeSnaps", "data/lifesnaps.csv", "participant_id", "timestamp_s",
       {{"heart_rate", "hr_bpm"}, {"hrv_sdnn", "hrv_ms"}, {"steps", "steps"}}},
      {"GLOBEM", "data/globem.csv", "participant_id", "timestamp_s",
       {{"heart_rate", "hr_bpm"}, {"eda", "eda"}, {"skin_temp", "temp_c"}}},
      {"TILES", "data/tiles.csv", "participant_id", "timestamp_s",
       {{"hr", "hr_bpm"}, {"rmssd", "hrv_ms"}, {"sleep_wake", "sleep_prob_wake"},
        {"sleep_light", "sleep_prob_light"}, {"sleep_deep", "sleep_prob_deep"},
        {"sleep_rem", "sleep_prob_rem"}}},
      {"PMData", "data/pmdata.csv", "participant_id", "timestamp_s",
       {{"heart_rate", "hr_bpm"}, {"steps", "steps"}, {"temperature", "temp_c"}}},
      {"AllOfUs", "data/allofus.csv", "participant_id", "timestamp_s",
       {{"hr", "hr_bpm"}, {"sdnn", "hrv_ms"}, {"steps", "steps"}}},
  };

  for (const auto& spec : specs) {
    try {
      pipeline.ingest(spec);
    } catch (const std::exception& e) {
      std::cerr << "Skipping " << spec.name << ": " << e.what() << "\n";
    }
  }

  pipeline.export_epoch_lake_csv("data/unified_epoch_lake.csv");
  auto epochs = pipeline.build_epoch_lake();
  std::cout << "Unified epochs: " << epochs.size() << "\n";
  auto windows = pipeline.build_lookback_windows();
  std::cout << "14-day windows: " << windows.size() << "\n";

  if (!epochs.empty()) {
    // Example harmonization pass for first record.
    if (epochs.front().values.size() > 1 && epochs.front().observed_mask[1]) {
      const double harmonized = MathematicalHarmonization::map_sdnn_to_rmssd(epochs.front().values[1]);
      std::cout << "First hrv harmonized SDNN->RMSSD: " << harmonized << "\n";
    }
    if (epochs.front().values.size() > 8) {
      std::array<double, 4> sleep_raw{epochs.front().values[5], epochs.front().values[6],
                                      epochs.front().values[7], epochs.front().values[8]};
      const std::array<std::array<double, 4>, 4> inv_conf = {
          std::array<double, 4>{1.05, -0.03, -0.01, -0.01},
          std::array<double, 4>{-0.02, 1.07, -0.03, -0.02},
          std::array<double, 4>{-0.01, -0.04, 1.09, -0.04},
          std::array<double, 4>{-0.02, -0.02, -0.03, 1.07},
      };
      const auto smooth = MathematicalHarmonization::smooth_sleep_probabilities(sleep_raw, inv_conf);
      std::cout << "Sleep smoothing output: [" << smooth[0] << ", " << smooth[1] << ", " << smooth[2]
                << ", " << smooth[3] << "]\n";
    }
  }

  return 0;
}

