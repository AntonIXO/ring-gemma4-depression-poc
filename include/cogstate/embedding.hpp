#pragma once

#include <vector>

#include "cogstate/types.hpp"

namespace cogstate {

class MomentVectorEncoder {
 public:
  Vector encode(const EpochRecord& current, const EpochRecord* previous) const;
};

class DayVectorEncoder {
 public:
  DayVector encode(const std::string& participant_id, std::int64_t day_start_s,
                   const std::vector<EpochRecord>& day_epochs) const;
};

class WeekVectorEncoder {
 public:
  WeekVector encode(const std::string& participant_id, std::int64_t window_start_s,
                    const std::vector<DayVector>& day_vectors) const;
};

}  // namespace cogstate

