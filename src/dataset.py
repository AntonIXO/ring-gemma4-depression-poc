"""Dataset class for depression prediction from PPG + EHR text."""

import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing import generate_synthetic_ppg, preprocess_ppg


class DepressionDataset(Dataset):
    """Dataset for multimodal depression prediction.

    Each sample contains:
        - ppg_segments: (N_segments, 1, 1250) preprocessed PPG windows
        - ehr_text: clinical note / EHR text string
        - label: 0 (no depression) or 1 (depression)
    """

    def __init__(self, ppg_segments_list: list[torch.Tensor],
                 ehr_texts: list[str], labels: list[int]):
        assert len(ppg_segments_list) == len(ehr_texts) == len(labels)
        self.ppg_segments_list = ppg_segments_list
        self.ehr_texts = ehr_texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "ppg_segments": self.ppg_segments_list[idx],
            "ehr_text": self.ehr_texts[idx],
            "label": self.labels[idx],
        }

    @classmethod
    def create_synthetic(cls, n_samples: int = 200) -> "DepressionDataset":
        """Create a synthetic dataset for testing.

        Generates PPG signals from sine-wave composites and random EHR texts.
        Labels are assigned randomly with ~40 % depression prevalence.
        """
        rng = np.random.default_rng(123)
        raw_signals = generate_synthetic_ppg(n_samples=n_samples, duration=60.0,
                                             fs=125.0)

        ppg_list: list[torch.Tensor] = []
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

        for i, (signal, fs) in enumerate(raw_signals):
            segments = preprocess_ppg(signal, fs_input=fs)
            ppg_list.append(segments)

            label = int(rng.random() < 0.4)
            label_list.append(label)

            if label == 1:
                text = rng.choice(ehr_templates_positive)
            else:
                text = rng.choice(ehr_templates_negative)
            ehr_list.append(text)

        return cls(ppg_list, ehr_list, label_list)

    @classmethod
    def from_wesad(cls, data_dir: str) -> "DepressionDataset":
        """Load the WESAD dataset from a local directory.

        Expects the standard WESAD layout::

            data_dir/
              S2/
                S2.pkl
              S3/
                S3.pkl
              ...

        Wrist BVP (PPG) is sampled at 64 Hz.  We use PANAS negative-affect
        scores as a proxy label: above median → 1 (distressed), else → 0.

        Args:
            data_dir: Path to the WESAD root directory.

        Returns:
            A :class:`DepressionDataset` instance.
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

        ppg_list: list[torch.Tensor] = []
        ehr_list: list[str] = []
        na_scores: list[float] = []  # collect to compute median later
        meta: list[dict] = []

        for sid in subject_ids:
            pkl_path = data_path / sid / f"{sid}.pkl"
            if not pkl_path.exists():
                continue

            with open(pkl_path, "rb") as f:
                data = pickle.load(f, encoding="latin1")

            # Wrist BVP: shape (N, 1), 64 Hz
            bvp = data["signal"]["wrist"]["BVP"].flatten().astype(np.float64)
            fs_bvp = 64.0

            # PANAS negative affect score (from questionnaire)
            # WESAD stores questionnaires under data["questionnaire"]
            try:
                panas = data["questionnaire"]["PANAS"]
                # Negative affect items are indices 1,3,5,7,9,11,13 (0-indexed)
                neg_indices = [1, 3, 5, 7, 9, 11, 13]
                na_score = float(sum(panas[i] for i in neg_indices
                                     if i < len(panas)))
            except (KeyError, TypeError, IndexError):
                na_score = 0.0

            try:
                segments = preprocess_ppg(bvp, fs_input=fs_bvp, fs_target=125.0)
            except ValueError:
                continue  # signal too short

            na_scores.append(na_score)
            ppg_list.append(segments)
            ehr_text = (
                f"Subject {sid}. Wrist PPG recording. "
                f"PANAS negative affect: {na_score:.0f}."
            )
            ehr_list.append(ehr_text)
            meta.append({"sid": sid, "na_score": na_score, "idx": len(ppg_list) - 1})

        if not ppg_list:
            raise FileNotFoundError(
                f"No valid WESAD subjects could be loaded from {data_dir}"
            )

        # Binary label: above-median negative affect → 1
        median_na = float(np.median(na_scores))
        labels = [int(s > median_na) for s in na_scores]

        print(f"[WESAD] Loaded {len(ppg_list)} subjects, "
              f"median NA={median_na:.1f}, "
              f"class 0={labels.count(0)}, class 1={labels.count(1)}")

        return cls(ppg_list, ehr_list, labels)
