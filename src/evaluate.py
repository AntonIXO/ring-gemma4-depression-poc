"""Evaluation script: load checkpoint, compute metrics, generate report."""

import argparse
import os
import sys

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import DepressionDataset
from src.model import RingGemmaModel
from src.train import collate_fn


@torch.no_grad()
def run_evaluation(model, loader, device):
    """Run full evaluation, returning predictions and probabilities."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in loader:
        ppg = batch["ppg_segments"].to(device)
        texts = batch["ehr_text"]
        labels = batch["label"]

        out = model(ppg, texts)
        logits = out["logits"].cpu()
        probs = torch.softmax(logits, dim=-1)[:, 1].numpy()
        preds = logits.argmax(dim=-1).numpy()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())
        all_probs.extend(probs.tolist())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def generate_report(labels, preds, probs, output_path):
    """Generate a markdown evaluation report."""
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)

    try:
        auroc = roc_auc_score(labels, probs)
    except ValueError:
        auroc = float("nan")

    cm = confusion_matrix(labels, preds)

    report = f"""# Evaluation Report

## Metrics

| Metric | Value |
|--------|-------|
| Accuracy | {acc:.4f} |
| F1 (macro) | {f1:.4f} |
| AUROC | {auroc:.4f} |

## Confusion Matrix

```
Predicted →   0       1
Actual ↓
  0         {cm[0, 0]:>5d}   {cm[0, 1]:>5d}
  1         {cm[1, 0]:>5d}   {cm[1, 1]:>5d}
```

- True Negatives:  {cm[0, 0]}
- False Positives: {cm[0, 1]}
- False Negatives: {cm[1, 0]}
- True Positives:  {cm[1, 1]}

## Dataset

- Total samples: {len(labels)}
- Class 0 (no depression): {int((labels == 0).sum())}
- Class 1 (depression): {int((labels == 1).sum())}

## Notes

- This evaluation used **synthetic data** — results are for pipeline validation only.
- For clinical relevance, train and evaluate on real PPG + EHR datasets (e.g., WESAD, GLOBEM).
"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"Report saved to {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate RingGemma model")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/best_model.pt")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--use_real_llm", action="store_true")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output", type=str,
                        default="results/evaluation_report.md")
    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    if args.synthetic:
        print(f"Generating synthetic evaluation set ({args.n_samples} samples)...")
        dataset = DepressionDataset.create_synthetic(args.n_samples)
    else:
        raise NotImplementedError("Real data loading not implemented. Use --synthetic.")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn)

    # Model
    model = RingGemmaModel(use_real_llm=args.use_real_llm, device=device)
    model = model.to(device)

    # Load checkpoint if it exists
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state, strict=False)
    else:
        print(f"No checkpoint found at {args.checkpoint}, evaluating untrained model.")

    # Run evaluation
    labels, preds, probs = run_evaluation(model, loader, device)

    # Report
    report = generate_report(labels, preds, probs, args.output)
    print(report)


if __name__ == "__main__":
    main()
