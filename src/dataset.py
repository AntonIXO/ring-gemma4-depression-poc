"""Dataset class for depression prediction from tabular health metrics + EHR text."""

import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing import (
    generate_synthetic_tabular,
    preprocess_tabular,
)


class DepressionDataset(Dataset):
    """Dataset for multimodal depression prediction.

    Each sample contains:
        - health_ts: (T, F) preprocessed tabular health time-series
        - ehr_text:  clinical note / EHR text string
        - label:     0 (no depression) or 1 (depression)
    """

    def __init__(
        self,
        health_ts_list: list[torch.Tensor],
        ehr_texts: list[str],
        labels: list[int],
    ):
        assert len(health_ts_list) == len(ehr_texts) == len(labels)
        self.health_ts_list = health_ts_list
        self.ehr_texts = ehr_texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "health_ts": self.health_ts_list[idx],
            "ehr_text": self.ehr_texts[idx],
            "label": self.labels[idx],
        }

    # ------------------------------------------------------------------
    # Synthetic data (for --synthetic offline testing)
    # ------------------------------------------------------------------

    @classmethod
    def create_synthetic(
        cls,
        n_samples: int = 200,
        time_steps: int = 14,
        n_features: int = 10,
    ) -> "DepressionDataset":
        """Create a synthetic dataset for testing.

        Generates tabular health time-series with random mean-reverting
        walks.  "Depressed" samples get lower HRV, higher resting HR, and
        fewer steps to give the encoder a learnable signal.
        """
        rng = np.random.default_rng(123)
        raw_samples = generate_synthetic_tabular(
            n_samples=n_samples,
            time_steps=time_steps,
            n_features=n_features,
        )

        ts_list: list[torch.Tensor] = []
        ehr_list: list[str] = []
        label_list: list[int] = []

        ehr_templates_positive = [
            "Patient reports persistent low mood, fatigue, and sleep disturbances over the past two weeks.",
            "Clinical note: PHQ-9 score 15. Reports anhedonia, poor concentration, weight changes.",
            "Patient presents with depressed mood, insomnia, loss of appetite. History of MDD.",
            "Significant depressive symptoms observed. Reduced activity, social withdrawal noted.",
            "Follow-up visit: ongoing depressive episode, partial response to SSRI therapy.",
        ]
        ehr_templates_negative = [
            "Routine check-up. Patient reports feeling well, good energy, regular sleep.",
            "Annual wellness visit. No mood complaints. Active lifestyle, balanced diet.",
            "Patient in good health. No signs of depression or anxiety. PHQ-9 score 2.",
            "Follow-up: patient recovered well, mood stable, exercise routine maintained.",
            "Healthy adult, no psychiatric history. Reports good social support network.",
        ]

        for data_arr, _feat_names in raw_samples:
            label = int(rng.random() < 0.4)

            # Inject a learnable depression signal:
            # lower HRV (col 1), higher HR (col 0), fewer steps (col 5)
            if label == 1 and data_arr.shape[1] > 5:
                data_arr[:, 0] += rng.normal(8, 2)    # ↑ HR
                data_arr[:, 1] -= rng.normal(10, 3)    # ↓ HRV RMSSD
                data_arr[:, 5] *= rng.uniform(0.3, 0.6) # ↓ steps

            tensor, _m, _s = preprocess_tabular(data_arr)
            ts_list.append(tensor)
            label_list.append(label)

            if label == 1:
                text = str(rng.choice(ehr_templates_positive))
            else:
                text = str(rng.choice(ehr_templates_negative))
            ehr_list.append(text)

        return cls(ts_list, ehr_list, label_list)

    # ------------------------------------------------------------------
    # WESAD — adapted for tabular features
    # ------------------------------------------------------------------

    @classmethod
    def from_wesad(cls, data_dir: str) -> "DepressionDataset":
        """Load the WESAD dataset as tabular features.

        Instead of raw BVP waveform, we extract per-window aggregate
        features from the wrist signals: BVP (mean, std), EDA (mean),
        TEMP (mean, delta), ACC (magnitude mean), giving F=6 features.
        We split the recording into non-overlapping 5-minute windows,
        yielding T ≈ 40-120 time steps per subject depending on duration.

        Binary label: PANAS negative affect > median → 1 (distressed).

        Args:
            data_dir: Path to the WESAD root directory.

        Returns:
            A DepressionDataset instance.
        """
        data_path = Path(data_dir)
        subject_ids = sorted(
            [d.name for d in data_path.iterdir()
             if d.is_dir() and d.name.startswith("S")]
        )
        if not subject_ids:
            raise FileNotFoundError(
                f"No subject directories (S2, S3, …) found in {data_dir}"
            )

        ts_list: list[torch.Tensor] = []
        ehr_list: list[str] = []
        na_scores: list[float] = []

        for sid in subject_ids:
            pkl_path = data_path / sid / f"{sid}.pkl"
            if not pkl_path.exists():
                continue

            with open(pkl_path, "rb") as f:
                data = pickle.load(f, encoding="latin1")

            wrist = data["signal"]["wrist"]

            # BVP: (N, 1) at 64 Hz
            bvp = wrist["BVP"].flatten().astype(np.float64)
            # EDA: (N, 1) at 4 Hz
            eda = wrist["EDA"].flatten().astype(np.float64)
            # TEMP: (N, 1) at 4 Hz
            temp = wrist["TEMP"].flatten().astype(np.float64)
            # ACC: (N, 3) at 32 Hz
            acc = wrist["ACC"].astype(np.float64)

            # Aggregate into 5-minute windows
            # BVP at 64 Hz → 64*300 = 19200 samples per window
            # EDA/TEMP at 4 Hz → 4*300 = 1200 samples per window
            # ACC at 32 Hz → 32*300 = 9600 samples per window
            bvp_win = 64 * 300
            low_win = 4 * 300
            acc_win = 32 * 300

            n_windows = min(
                len(bvp) // bvp_win,
                len(eda) // low_win,
                len(temp) // low_win,
                len(acc) // acc_win,
            )

            if n_windows < 2:
                continue

            features = []
            for w in range(n_windows):
                bvp_seg = bvp[w * bvp_win : (w + 1) * bvp_win]
                eda_seg = eda[w * low_win : (w + 1) * low_win]
                temp_seg = temp[w * low_win : (w + 1) * low_win]
                acc_seg = acc[w * acc_win : (w + 1) * acc_win]

                acc_mag = np.sqrt((acc_seg ** 2).sum(axis=1))

                row = [
                    np.mean(bvp_seg),        # bvp_mean
                    np.std(bvp_seg),          # bvp_std (proxy for pulse amplitude)
                    np.mean(eda_seg),         # eda_mean
                    np.mean(temp_seg),        # temp_mean
                    temp_seg[-1] - temp_seg[0] if len(temp_seg) > 1 else 0.0,  # temp_delta
                    np.mean(acc_mag),         # acc_magnitude_mean
                ]
                features.append(row)

            data_arr = np.array(features, dtype=np.float64)  # (T, 6)

            # PANAS negative affect
            try:
                panas = data["questionnaire"]["PANAS"]
                neg_indices = [1, 3, 5, 7, 9, 11, 13]
                na_score = float(sum(panas[i] for i in neg_indices
                                     if i < len(panas)))
            except (KeyError, TypeError, IndexError):
                na_score = 0.0

            tensor, _m, _s = preprocess_tabular(data_arr)
            ts_list.append(tensor)
            na_scores.append(na_score)

            ehr_text = (
                f"Subject {sid}. Wrist wearable recording, "
                f"{n_windows} five-minute windows. "
                f"PANAS negative affect: {na_score:.0f}."
            )
            ehr_list.append(ehr_text)

        if not ts_list:
            raise FileNotFoundError(
                f"No valid WESAD subjects could be loaded from {data_dir}"
            )

        # Binary label: above-median negative affect → 1
        median_na = float(np.median(na_scores))
        labels = [int(s > median_na) for s in na_scores]

        print(f"[WESAD] Loaded {len(ts_list)} subjects, "
              f"median NA={median_na:.1f}, "
              f"class 0={labels.count(0)}, class 1={labels.count(1)}")

        return cls(ts_list, ehr_list, labels)
