"""Preprocessing for tabular health-metric time-series from smart rings.

Input data is pre-aggregated features per time window (e.g. half-day):
  shape (T, F) where T = time steps, F = features like
  heart_rate_mean, hrv_rmssd, hrv_sdnn, spo2_mean, skin_temp_delta,
  steps, respiratory_rate, etc.

No signal processing (Butterworth, resampling, windowing) is needed here —
the ring firmware already computes these aggregates.
"""

from __future__ import annotations

import numpy as np
import torch


# Default feature names matching common smart ring exports
DEFAULT_FEATURES = [
    "heart_rate_mean",
    "hrv_rmssd",
    "hrv_sdnn",
    "spo2_mean",
    "skin_temp_delta",
    "steps",
    "respiratory_rate",
    "heart_rate_min",
    "heart_rate_max",
    "active_minutes",
]


def preprocess_tabular(
    data: np.ndarray,
    *,
    feature_means: np.ndarray | None = None,
    feature_stds: np.ndarray | None = None,
    clip_sigma: float = 5.0,
    fill_value: float = 0.0,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Preprocess a tabular health-metric time-series.

    Steps:
        1. Replace NaN / Inf with ``fill_value`` (typically 0 before z-score)
        2. Compute per-feature mean/std (or use provided stats for test-time)
        3. Z-score normalize each feature independently
        4. Clip to ±clip_sigma to bound outliers
        5. Replace any residual NaN with 0

    Args:
        data: (T, F) numpy array — T time steps, F features.
        feature_means: Optional (F,) array of pre-computed means.
        feature_stds: Optional (F,) array of pre-computed stds.
        clip_sigma: Clip z-scored values to ±this value.
        fill_value: Value used for NaN imputation before normalization.

    Returns:
        (tensor, means, stds) where tensor is (T, F) float32 torch.Tensor,
        and means/stds are the (F,) arrays used (pass them at test time).
    """
    data = np.array(data, dtype=np.float64)

    # 1. Handle missing values
    mask = ~np.isfinite(data)
    data[mask] = fill_value

    # 2. Compute or reuse stats
    if feature_means is None:
        feature_means = np.nanmean(data, axis=0)
    if feature_stds is None:
        feature_stds = np.nanstd(data, axis=0)

    # Guard against zero std
    feature_stds = np.where(feature_stds < 1e-8, 1.0, feature_stds)

    # 3. Z-score
    normed = (data - feature_means) / feature_stds

    # 4. Clip
    normed = np.clip(normed, -clip_sigma, clip_sigma)

    # 5. Final NaN sweep
    normed = np.nan_to_num(normed, nan=0.0)

    return (
        torch.tensor(normed, dtype=torch.float32),
        feature_means,
        feature_stds,
    )


def generate_synthetic_tabular(
    n_samples: int = 200,
    time_steps: int = 14,
    n_features: int = 10,
    seed: int = 42,
) -> list[tuple[np.ndarray, list[str]]]:
    """Generate synthetic tabular health data for testing.

    Produces realistic-ish ranges for common ring metrics with injected
    trends for "depressed" vs "healthy" patterns. Each sample is a
    (T, F) array + list of feature names.

    Args:
        n_samples: Number of synthetic subjects.
        time_steps: Number of time windows per subject.
        n_features: Number of features (capped at len(DEFAULT_FEATURES)).
        seed: Random seed.

    Returns:
        List of (data_array, feature_names) tuples.
    """
    rng = np.random.default_rng(seed)
    features = DEFAULT_FEATURES[:n_features]
    samples = []

    # Physiological baselines (mean, std) per feature
    baselines = {
        "heart_rate_mean": (68.0, 8.0),
        "hrv_rmssd": (42.0, 15.0),
        "hrv_sdnn": (55.0, 18.0),
        "spo2_mean": (97.5, 0.8),
        "skin_temp_delta": (0.0, 0.3),
        "steps": (5000.0, 3000.0),
        "respiratory_rate": (15.0, 2.0),
        "heart_rate_min": (52.0, 6.0),
        "heart_rate_max": (120.0, 20.0),
        "active_minutes": (45.0, 25.0),
    }

    for _ in range(n_samples):
        data = np.zeros((time_steps, n_features), dtype=np.float64)
        for j, feat in enumerate(features):
            mu, sigma = baselines.get(feat, (50.0, 10.0))
            # Random walk with mean-reversion for temporal autocorrelation
            base = mu + rng.normal(0, sigma * 0.3)
            series = [base]
            for _t in range(1, time_steps):
                step = 0.7 * (base - series[-1]) + rng.normal(0, sigma * 0.2)
                series.append(series[-1] + step)
            data[:, j] = np.array(series)

        # Inject ~5% missing values
        miss_mask = rng.random((time_steps, n_features)) < 0.05
        data[miss_mask] = np.nan

        samples.append((data, features))

    return samples
