export const EPOCH_MINUTES = 15;
export const STEPS_PER_DAY = 96;
export const LOOKBACK_DAYS = 14;

export const UNIFIED_FEATURE_ORDER = [
  "hr_bpm",
  "hrv_ms",
  "steps",
  "eda",
  "temp_c",
  "sleep_prob_wake",
  "sleep_prob_light",
  "sleep_prob_deep",
  "sleep_prob_rem"
] as const;

export const SECONDS_PER_DAY = 86_400;
export const EPOCH_SECONDS = EPOCH_MINUTES * 60;

