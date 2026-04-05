#pragma once

#include "cogstate/types.hpp"

namespace cogstate {

struct PatchEncoderConfig {
  int patch_length{8};
  int patch_stride{4};
  int hidden_dim{64};
};

class PatchTemporalEncoder {
 public:
  explicit PatchTemporalEncoder(PatchEncoderConfig config);
  Vector encode(const Matrix& sequence) const;

 private:
  PatchEncoderConfig config_;
};

}  // namespace cogstate

