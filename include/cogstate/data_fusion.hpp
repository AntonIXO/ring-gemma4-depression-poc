#pragma once

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

#include "cogstate/types.hpp"

namespace cogstate {

struct DatasetSpec {
  std::string name;
  std::string csv_path;
  std::string participant_column;
  std::string timestamp_column;
  std::unordered_map<std::string, std::string> feature_column_to_unified_name;
};

struct FusionConfig {
  int epoch_minutes{15};
  int steps_per_day{96};
  int lookback_days{14};
  std::vector<std::string> unified_feature_order;
};

class DataFusionPipeline {
 public:
  explicit DataFusionPipeline(FusionConfig config);

  void ingest(const DatasetSpec& spec);
  std::vector<EpochRecord> build_epoch_lake() const;
  std::vector<std::vector<EpochRecord>> build_lookback_windows() const;
  void export_epoch_lake_csv(const std::string& path) const;

 private:
  FusionConfig config_;
  std::vector<UnifiedSample> samples_;
  mutable std::vector<EpochRecord> cached_epochs_;
  mutable bool cache_valid_{false};
};

}  // namespace cogstate
