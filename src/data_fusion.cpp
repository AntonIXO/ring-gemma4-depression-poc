#include "cogstate/data_fusion.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

#include "cogstate/csv.hpp"

namespace cogstate {

namespace {

constexpr std::int64_t kSecondsPerDay = 24 * 60 * 60;

std::int64_t floor_to_epoch(std::int64_t timestamp_s, int epoch_minutes) {
  const std::int64_t step = static_cast<std::int64_t>(epoch_minutes) * 60;
  return (timestamp_s / step) * step;
}

double parse_double(const std::string& token) {
  if (token.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  try {
    return std::stod(token);
  } catch (...) {
    return std::numeric_limits<double>::quiet_NaN();
  }
}

std::int64_t parse_int64(const std::string& token) {
  if (token.empty()) {
    throw std::runtime_error("Empty timestamp token");
  }
  return std::stoll(token);
}

std::unordered_map<std::string, std::size_t> index_header(const std::vector<std::string>& header) {
  std::unordered_map<std::string, std::size_t> idx;
  for (std::size_t i = 0; i < header.size(); ++i) {
    idx[header[i]] = i;
  }
  return idx;
}

}  // namespace

DataFusionPipeline::DataFusionPipeline(FusionConfig config) : config_(std::move(config)) {
  if (config_.steps_per_day != 96 || config_.epoch_minutes != 15) {
    throw std::runtime_error("Blueprint requires 15-minute epochs and 96 steps/day.");
  }
  if (config_.lookback_days != 14) {
    throw std::runtime_error("Blueprint requires 14-day look-back window.");
  }
}

void DataFusionPipeline::ingest(const DatasetSpec& spec) {
  CsvTable table = read_csv(spec.csv_path);
  const auto header_idx = index_header(table.header);
  if (header_idx.find(spec.participant_column) == header_idx.end() ||
      header_idx.find(spec.timestamp_column) == header_idx.end()) {
    throw std::runtime_error("Dataset spec columns missing in " + spec.csv_path);
  }

  for (const auto& row : table.rows) {
    if (row.size() < table.header.size()) {
      continue;
    }
    UnifiedSample sample;
    sample.dataset = spec.name;
    sample.participant_id = row[header_idx.at(spec.participant_column)];
    sample.timestamp_s = parse_int64(row[header_idx.at(spec.timestamp_column)]);
    for (const auto& mapping : spec.feature_column_to_unified_name) {
      const auto it = header_idx.find(mapping.first);
      if (it == header_idx.end()) {
        continue;
      }
      const double value = parse_double(row[it->second]);
      if (!std::isnan(value)) {
        sample.features[mapping.second] = value;
      }
    }
    if (!sample.features.empty() && !sample.participant_id.empty()) {
      samples_.push_back(std::move(sample));
    }
  }
  cache_valid_ = false;
}

std::vector<EpochRecord> DataFusionPipeline::build_epoch_lake() const {
  if (cache_valid_) {
    return cached_epochs_;
  }

  using Aggregation = std::pair<double, int>;
  using FeatureAgg = std::unordered_map<std::string, Aggregation>;
  std::unordered_map<std::string, std::map<std::int64_t, FeatureAgg>> grouped;

  for (const auto& sample : samples_) {
    const std::int64_t epoch_start = floor_to_epoch(sample.timestamp_s, config_.epoch_minutes);
    auto& agg = grouped[sample.participant_id][epoch_start];
    for (const auto& kv : sample.features) {
      auto& slot = agg[kv.first];
      slot.first += kv.second;
      slot.second += 1;
    }
  }

  std::vector<EpochRecord> epochs;
  const std::int64_t epoch_step_s = static_cast<std::int64_t>(config_.epoch_minutes) * 60;
  for (const auto& by_participant : grouped) {
    const auto& participant_id = by_participant.first;
    const auto& series = by_participant.second;
    if (series.empty()) {
      continue;
    }
    const std::int64_t first_epoch = series.begin()->first;
    const std::int64_t last_epoch = series.rbegin()->first;
    const std::int64_t first_day = (first_epoch / kSecondsPerDay) * kSecondsPerDay;
    const std::int64_t last_day = (last_epoch / kSecondsPerDay) * kSecondsPerDay;

    for (std::int64_t day_start = first_day; day_start <= last_day; day_start += kSecondsPerDay) {
      for (int step = 0; step < config_.steps_per_day; ++step) {
        const std::int64_t epoch_start = day_start + static_cast<std::int64_t>(step) * epoch_step_s;
        EpochRecord rec;
        rec.participant_id = participant_id;
        rec.epoch_start_s = epoch_start;
        rec.values.resize(config_.unified_feature_order.size(), 0.0);
        rec.observed_mask.resize(config_.unified_feature_order.size(), false);

        const auto eit = series.find(epoch_start);
        if (eit != series.end()) {
          for (std::size_t i = 0; i < config_.unified_feature_order.size(); ++i) {
            const auto fit = eit->second.find(config_.unified_feature_order[i]);
            if (fit == eit->second.end() || fit->second.second <= 0) {
              continue;
            }
            rec.values[i] = fit->second.first / static_cast<double>(fit->second.second);
            rec.observed_mask[i] = true;
          }
        }
        epochs.push_back(std::move(rec));
      }
    }
  }
  std::sort(epochs.begin(), epochs.end(), [](const EpochRecord& a, const EpochRecord& b) {
    return std::tie(a.participant_id, a.epoch_start_s) < std::tie(b.participant_id, b.epoch_start_s);
  });

  cached_epochs_ = epochs;
  cache_valid_ = true;
  return epochs;
}

std::vector<std::vector<EpochRecord>> DataFusionPipeline::build_lookback_windows() const {
  auto epochs = build_epoch_lake();
  std::unordered_map<std::string, std::vector<EpochRecord>> by_participant;
  for (const auto& e : epochs) {
    by_participant[e.participant_id].push_back(e);
  }

  std::vector<std::vector<EpochRecord>> windows;
  const std::int64_t span = config_.lookback_days * kSecondsPerDay;
  for (auto& kv : by_participant) {
    auto& series = kv.second;
    std::sort(series.begin(), series.end(),
              [](const EpochRecord& a, const EpochRecord& b) { return a.epoch_start_s < b.epoch_start_s; });
    for (std::size_t i = 0; i < series.size(); ++i) {
      const std::int64_t start = series[i].epoch_start_s - span;
      std::vector<EpochRecord> window;
      for (std::size_t j = 0; j <= i; ++j) {
        if (series[j].epoch_start_s >= start) {
          window.push_back(series[j]);
        }
      }
      if (!window.empty()) {
        windows.push_back(std::move(window));
      }
    }
  }
  return windows;
}

void DataFusionPipeline::export_epoch_lake_csv(const std::string& path) const {
  const auto epochs = build_epoch_lake();
  CsvTable table;
  table.header = {"participant_id", "epoch_start_s"};
  table.header.insert(table.header.end(), config_.unified_feature_order.begin(),
                      config_.unified_feature_order.end());

  for (const auto& rec : epochs) {
    std::vector<std::string> row;
    row.push_back(rec.participant_id);
    row.push_back(std::to_string(rec.epoch_start_s));
    for (std::size_t i = 0; i < rec.values.size(); ++i) {
      if (i < rec.observed_mask.size() && rec.observed_mask[i]) {
        row.push_back(std::to_string(rec.values[i]));
      } else {
        row.push_back("");
      }
    }
    table.rows.push_back(std::move(row));
  }
  write_csv(path, table);
}

}  // namespace cogstate
