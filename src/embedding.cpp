#include "cogstate/embedding.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace cogstate {

namespace {

double mean(const Vector& v) {
  if (v.empty()) {
    return 0.0;
  }
  return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

double stddev(const Vector& v, double mu) {
  if (v.size() < 2) {
    return 0.0;
  }
  double s = 0.0;
  for (double x : v) {
    const double d = x - mu;
    s += d * d;
  }
  return std::sqrt(s / static_cast<double>(v.size() - 1));
}

}  // namespace

Vector MomentVectorEncoder::encode(const EpochRecord& current, const EpochRecord* previous) const {
  Vector out;
  out.reserve(current.values.size() * 3);
  for (std::size_t i = 0; i < current.values.size(); ++i) {
    const double value = current.values[i];
    const double delta =
        (previous && i < previous->values.size()) ? (value - previous->values[i]) : 0.0;
    const double velocity = delta / 15.0;
    out.push_back(value);
    out.push_back(delta);
    out.push_back(velocity);
  }
  return out;
}

DayVector DayVectorEncoder::encode(const std::string& participant_id, std::int64_t day_start_s,
                                   const std::vector<EpochRecord>& day_epochs) const {
  DayVector out{participant_id, day_start_s, {}};
  if (day_epochs.empty()) {
    return out;
  }
  const std::size_t dims = day_epochs.front().values.size();
  out.values.reserve(dims * 4);
  for (std::size_t d = 0; d < dims; ++d) {
    Vector axis;
    axis.reserve(day_epochs.size());
    for (const auto& e : day_epochs) {
      if (d < e.values.size()) {
        axis.push_back(e.values[d]);
      }
    }
    const double mu = mean(axis);
    const double sigma = stddev(axis, mu);
    const auto mm = std::minmax_element(axis.begin(), axis.end());
    out.values.push_back(mu);
    out.values.push_back(sigma);
    out.values.push_back(mm.first == axis.end() ? 0.0 : *mm.first);
    out.values.push_back(mm.second == axis.end() ? 0.0 : *mm.second);
  }
  return out;
}

WeekVector WeekVectorEncoder::encode(const std::string& participant_id, std::int64_t window_start_s,
                                     const std::vector<DayVector>& day_vectors) const {
  WeekVector out{participant_id, window_start_s, {}};
  if (day_vectors.empty()) {
    return out;
  }
  const std::size_t dims = day_vectors.front().values.size();
  out.values.reserve(dims * 2);
  for (std::size_t d = 0; d < dims; ++d) {
    Vector axis;
    axis.reserve(day_vectors.size());
    for (const auto& day : day_vectors) {
      if (d < day.values.size()) {
        axis.push_back(day.values[d]);
      }
    }
    const double mu = mean(axis);
    double trend = 0.0;
    if (axis.size() > 1) {
      trend = (axis.back() - axis.front()) / static_cast<double>(axis.size() - 1);
    }
    out.values.push_back(mu);
    out.values.push_back(trend);
  }
  return out;
}

}  // namespace cogstate

