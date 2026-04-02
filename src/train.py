"""Two-stage training script for RingGemma depression prediction model."""

import argparse
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, random_split

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import DepressionDataset
from src.model import RingGemmaModel


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate: pad PPG segments to the max count in the batch."""
    max_seg = max(item["ppg_segments"].size(0) for item in batch)
    seg_dim = batch[0]["ppg_segments"].size(-1)

    padded_segs = []
    for item in batch:
        segs = item["ppg_segments"]  # (N_i, 1, 1250)
        n = segs.size(0)
        if n < max_seg:
            pad = torch.zeros(max_seg - n, 1, seg_dim)
            segs = torch.cat([segs, pad], dim=0)
        padded_segs.append(segs)

    return {
        "ppg_segments": torch.stack(padded_segs),      # (B, max_seg, 1, 1250)
        "ehr_text": [item["ehr_text"] for item in batch],
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
    }


def train_one_epoch(model, loader, optimizer, device, max_norm=1.0):
    """Train for one epoch. Returns (avg_loss, accuracy, f1)."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        ppg = batch["ppg_segments"].to(device)
        texts = batch["ehr_text"]
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        out = model(ppg, texts, labels=labels)
        loss = out["loss"]
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm
        )
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = out["logits"].argmax(dim=-1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    n = len(all_labels)
    avg_loss = total_loss / max(n, 1)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model. Returns (avg_loss, accuracy, f1)."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        ppg = batch["ppg_segments"].to(device)
        texts = batch["ehr_text"]
        labels = batch["label"].to(device)

        out = model(ppg, texts, labels=labels)
        total_loss += out["loss"].item() * labels.size(0)
        preds = out["logits"].argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    n = len(all_labels)
    avg_loss = total_loss / max(n, 1)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1


def main():
    parser = argparse.ArgumentParser(description="Train RingGemma model")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data")
    parser.add_argument("--wesad_path", type=str, default=None,
                        help="Path to WESAD dataset directory (overrides --synthetic)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU mode")
    parser.add_argument("--use_real_llm", action="store_true",
                        help="Load a real Gemma model (requires download)")
    parser.add_argument("--epochs_stage1", type=int, default=5,
                        help="Epochs for stage 1 (projector only)")
    parser.add_argument("--epochs_stage2", type=int, default=10,
                        help="Epochs for stage 2 (projector + LoRA + head)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Number of synthetic samples")
    parser.add_argument("--lr_stage1", type=float, default=1e-3)
    parser.add_argument("--lr_stage2", type=float, default=2e-5)
    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data ---
    if args.wesad_path is not None:
        print(f"Loading WESAD dataset from {args.wesad_path}...")
        dataset = DepressionDataset.from_wesad(args.wesad_path)
    elif args.synthetic:
        print(f"Generating synthetic dataset ({args.n_samples} samples)...")
        dataset = DepressionDataset.create_synthetic(args.n_samples)
    else:
        raise NotImplementedError(
            "Provide --wesad_path or --synthetic."
        )

    # 80/20 train/val split
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn)

    print(f"Train: {n_train}, Val: {n_val}")

    # --- Model ---
    model = RingGemmaModel(
        use_real_llm=args.use_real_llm,
        device=device,
    )
    model = model.to(device)

    os.makedirs("checkpoints", exist_ok=True)
    best_val_f1 = -1.0
    best_path = "checkpoints/best_model.pt"

    # ========== Stage 1: Train projector only ==========
    print("\n=== Stage 1: Projector Only ===")
    # Freeze everything except projector
    for p in model.parameters():
        p.requires_grad = False
    for p in model.projector.parameters():
        p.requires_grad = True

    optimizer1 = torch.optim.AdamW(model.projector.parameters(), lr=args.lr_stage1)

    for epoch in range(1, args.epochs_stage1 + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, optimizer1,
                                                  device)
        vl_loss, vl_acc, vl_f1 = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch}/{args.epochs_stage1}  "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} train_f1={tr_f1:.3f}  "
              f"val_loss={vl_loss:.4f} val_acc={vl_acc:.3f} val_f1={vl_f1:.3f}  "
              f"({elapsed:.1f}s)")

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            torch.save(model.state_dict(), best_path)
            print(f"    -> Saved best model (val_f1={vl_f1:.3f})")

    # ========== Stage 2: Projector + LLM + classifier ==========
    print("\n=== Stage 2: Projector + LLM + Classifier ===")
    # Unfreeze LLM and classifier (projector already unfrozen)
    for p in model.llm.parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer2 = torch.optim.AdamW(trainable, lr=args.lr_stage2)

    for epoch in range(1, args.epochs_stage2 + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, optimizer2,
                                                  device)
        vl_loss, vl_acc, vl_f1 = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch}/{args.epochs_stage2}  "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} train_f1={tr_f1:.3f}  "
              f"val_loss={vl_loss:.4f} val_acc={vl_acc:.3f} val_f1={vl_f1:.3f}  "
              f"({elapsed:.1f}s)")

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            torch.save(model.state_dict(), best_path)
            print(f"    -> Saved best model (val_f1={vl_f1:.3f})")

    print(f"\nTraining complete. Best val F1: {best_val_f1:.3f}")
    print(f"Checkpoint: {best_path}")


if __name__ == "__main__":
    main()
