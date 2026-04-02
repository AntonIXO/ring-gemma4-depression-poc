"""Dataset class for depression prediction from PPG + EHR text."""

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
