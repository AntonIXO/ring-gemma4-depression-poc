# Evaluation Report

## Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.4500 |
| F1 (macro) | 0.3103 |
| AUROC | 0.4646 |

## Confusion Matrix

```
Predicted →   0       1
Actual ↓
  0             0      11
  1             0       9
```

- True Negatives:  0
- False Positives: 11
- False Negatives: 0
- True Positives:  9

## Dataset

- Total samples: 20
- Class 0 (no depression): 11
- Class 1 (depression): 9

## Notes

- This evaluation used **synthetic data** — results are for pipeline validation only.
- For clinical relevance, train and evaluate on real wearable + EHR datasets (e.g., WESAD, GLOBEM).
