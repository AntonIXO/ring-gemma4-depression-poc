#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace cogstate {

using FeatureMap = std::unordered_map<std::string, double>;
using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

struct UnifiedSample {
  std::string participant_id;
  std::string dataset;
  std::int64_t timestamp_s{};
  FeatureMap features;
};

struct EpochRecord {
  std::string participant_id;
  std::int64_t epoch_start_s{};
  Vector values;
  std::vector<bool> observed_mask;
};

struct DayVector {
  std::string participant_id;
  std::int64_t day_start_s{};
  Vector values;
};

struct WeekVector {
  std::string participant_id;
  std::int64_t window_start_s{};
  Vector values;
};

struct SearchResult {
  std::string id;
  double cosine_similarity{};
  std::string metadata;
};

}  // namespace cogstate

