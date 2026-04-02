"""PPG signal preprocessing: filtering, resampling, segmentation, normalization."""

import numpy as np
import torch
from scipy.signal import butter, filtfilt, resample


def _bandpass_filter(signal: np.ndarray, lowcut: float = 0.5, highcut: float = 4.0,
                     fs: float = 125.0, order: int = 4) -> np.ndarray:
    """Apply a Butterworth bandpass filter to the signal."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def preprocess_ppg(raw_signal: np.ndarray, fs_input: float,
                   fs_target: float = 125.0) -> torch.Tensor:
    """Preprocess raw PPG signal into normalized segments.

    Steps:
        1. Resample to fs_target Hz
        2. Bandpass filter 0.5–4 Hz
        3. Segment into 10-second windows with 50% overlap
        4. Z-score normalize each segment

    Args:
        raw_signal: 1-D numpy array of raw PPG values.
        fs_input: Sampling frequency of the input signal (Hz).
        fs_target: Target sampling frequency (Hz). Default 125 Hz.

    Returns:
        torch.Tensor of shape (N_segments, 1, samples_per_segment) where
        samples_per_segment = int(fs_target * 10) = 1250.
    """
    # Resample
    if fs_input != fs_target:
        n_target = int(len(raw_signal) * fs_target / fs_input)
        raw_signal = resample(raw_signal, n_target)

    # Bandpass filter
    filtered = _bandpass_filter(raw_signal, fs=fs_target)

    # Segment: 10-second windows, 50% overlap
    seg_len = int(fs_target * 10)  # 1250 samples
    hop = seg_len // 2             # 625 samples (50% overlap)

    segments = []
    start = 0
    while start + seg_len <= len(filtered):
        seg = filtered[start:start + seg_len].copy()
        # Z-score normalize
        mu = seg.mean()
        std = seg.std()
        if std > 1e-8:
            seg = (seg - mu) / std
        else:
            seg = seg - mu
        segments.append(seg)
        start += hop

    if len(segments) == 0:
        raise ValueError(
            f"Signal too short ({len(filtered)} samples) for a 10-s segment "
            f"({seg_len} samples at {fs_target} Hz)."
        )

    # Stack → (N_segments, 1, 1250)
    arr = np.stack(segments, axis=0)[:, np.newaxis, :]
    return torch.tensor(arr, dtype=torch.float32)


def generate_synthetic_ppg(n_samples: int = 100, duration: float = 60.0,
                           fs: float = 125.0) -> list[tuple[np.ndarray, float]]:
    """Generate synthetic PPG signals for testing.

    Each signal is a sum of sine waves (1 Hz fundamental + 2 Hz and 3 Hz
    harmonics) with Gaussian noise, simulating a crude PPG waveform.

    Args:
        n_samples: Number of synthetic signals to generate.
        duration: Duration of each signal in seconds.
        fs: Sampling frequency (Hz).

    Returns:
        List of (signal_array, sampling_rate) tuples.
    """
    rng = np.random.default_rng(42)
    t = np.arange(int(duration * fs)) / fs
    signals = []
    for _ in range(n_samples):
        hr = rng.uniform(0.8, 1.5)  # heart rate ~48–90 bpm fundamental
        phase = rng.uniform(0, 2 * np.pi)
        amp1 = rng.uniform(0.8, 1.2)
        amp2 = rng.uniform(0.2, 0.5)
        amp3 = rng.uniform(0.05, 0.15)
        signal = (amp1 * np.sin(2 * np.pi * hr * t + phase)
                  + amp2 * np.sin(2 * np.pi * 2 * hr * t + phase)
                  + amp3 * np.sin(2 * np.pi * 3 * hr * t + phase))
        signal += rng.normal(0, 0.05, len(t))
        signals.append((signal, fs))
    return signals
