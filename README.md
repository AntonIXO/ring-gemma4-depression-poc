# Ring-Gemma4 Depression PoC

**Multimodal neuropsychiatric prediction from smart-ring health metrics + EHR text using learned temporal embeddings projected into LLM token space.**

This proof-of-concept implements the optiHealth 2.0 "Everything is a Vector" paradigm: pre-aggregated wearable health time-series are encoded by a custom temporal Transformer encoder, projected into virtual tokens, and fused with clinical text inside a language model backbone for binary depression classification.

## Architecture

```
     ┌──────────────────────────────────────┐
     │   Smart Ring Aggregated Metrics       │
     │   (T time steps × F features)         │
     │   HR, HRV, SpO₂, temp, steps, …      │
     └──────────────────┬───────────────────┘
                        │
                  ┌─────▼──────┐
                  │ Preprocess  │
                  │ • NaN fill  │
                  │ • Z-score   │
                  │ • Clip ±5σ  │
                  └─────┬──────┘
                        │
                     (T, F)
                        │
     ┌──────────────────▼───────────────────┐
     │  MacroTrendEncoder (FROZEN, ~2.7M)    │
     │  Transformer: 4 layers, 4 heads       │
     │  Linear(F → 256) + sinusoidal PE      │
     │  → per-step embeddings (T, 512)       │
     └──────────────────┬───────────────────┘
                        │
                     (T, 512)
                        │
     ┌──────────────────▼───────────────────┐
     │  SensorProjector (TRAINABLE, ~5.2M)   │
     │  AdaptiveAvgPool1d(T → 16 tokens)     │
     │  Linear(512 → 2048) → GELU            │
     │  LayerNorm(2048)                       │
     │  Linear(2048 → 2048)                   │
     └──────────────────┬───────────────────┘
                        │
                     (16, 2048)
                        │
   ┌──────────────┐     │
   │   EHR Text   │     │
   │   Tokenizer  │──→ (T_text, 2048)
   └──────────────┘     │        │
                        └───┬────┘
                            │ concat
                      (16 + T_text, 2048)
                            │
     ┌──────────────────────▼───────────────┐
     │   LLM Backbone (QLoRA)                │
     │   Gemma 4 / 3-1B (4-bit NF4)         │
     │   LoRA: q,k,v,o_proj  r=8, α=16      │
     └──────────────────────┬───────────────┘
                            │
                      last non-pad token
                            │
     ┌──────────────────────▼───────────────┐
     │   Classification Head                 │
     │   Linear(2048 → 2)                    │
     │   Depression: yes / no                │
     └──────────────────────────────────────┘
```

### The optiHealth 2.0 "Everything is a Vector" Paradigm

Traditional wearable health pipelines extract hand-crafted feature vectors (e.g., `[resting_hr, hrv_rmssd, sleep_score, spo2]`). The custom encoder paradigm **replaces** this with learned neural embeddings that capture non-linear physiological dynamics:

1. **Phase 1 — Custom Temporal Encoder**: Aggregated health metrics → per-step state vectors `h ∈ R^{T×d_enc}`
2. **Phase 2 — Vector Database**: Store `h` vectors for similarity search (not implemented in this PoC)
3. **Phase 3 — Cross-Modal Alignment**: Project `h` into LLM token space via `Z = MLP(pool(h))`, creating "Virtual Tokens" injected via `inputs_embeds`

### Why a Transformer Encoder for Tabular Health Data

Smart rings (Oura, RingConn, Ultrahuman) export pre-computed aggregates, not raw signals. The input is `(T, F)` where T ≈ 7-28 time steps and F ≈ 7-15 features. A lightweight Transformer is ideal because:

- **T is small** (≤28), so O(T²) self-attention is trivial
- **Features are heterogeneous** — a learned linear projection handles mixed-scale tabular data naturally
- **Self-attention captures non-local patterns** like "HRV dropped 3 days ago AND temp rose today"
- **Per-step outputs** give the projector a meaningful sequence to pool over

### Key Design Decisions

- **Last-token classification**: Causal attention means position 0 sees only itself. The classifier reads the last non-padded hidden state, ensuring full multimodal fusion.
- **Attention mask**: Explicit mask prevents padded text positions from corrupting the classifier.
- **Gemma 4 default**: Falls back to Gemma 3 1B if Gemma 4 isn't available yet.

## Setup

```bash
# Clone
git clone https://github.com/AntonIXO/ring-gemma4-depression-poc.git
cd ring-gemma4-depression-poc

# Install dependencies
pip install -r requirements.txt

# (Optional) For GPU with 4-bit quantization
pip install bitsandbytes>=0.43.0
```

### Environment

- **Python**: 3.10+
- **PyTorch**: 2.2+
- **Key deps**: transformers, peft, scipy, scikit-learn

## Quick Start

### Train with synthetic data (CPU, no downloads)

```bash
python src/train.py --synthetic --cpu --epochs_stage1 5 --epochs_stage2 10
```

### Custom data shape

```bash
# 28 daily time steps, 7 features per step
python src/train.py --synthetic --cpu --time_steps 28 --num_features 7
```

### Evaluate

```bash
python src/evaluate.py --synthetic --cpu
# → results/evaluation_report.md
```

### Train with WESAD data

```bash
# Download WESAD from https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/
# Extract to data/raw/WESAD/
python src/train.py --wesad_path data/raw/WESAD --cpu --epochs_stage1 5 --epochs_stage2 10
```

### Train with real Gemma on GPU

```bash
python src/train.py --synthetic --use_real_llm --epochs_stage1 5 --epochs_stage2 10
# Or override the model:
GEMMA_MODEL=google/gemma-4-e4b-it python src/train.py --synthetic --use_real_llm
```

## Hardware Requirements

| Mode | Device | VRAM/RAM | Notes |
|------|--------|----------|-------|
| MockLLM (default) | CPU | 4 GB RAM | For development and testing |
| Gemma 3 1B (4-bit) | GPU | ~6 GB VRAM | Minimal real LLM |
| Gemma 3 4B (4-bit) | GPU | ~12 GB VRAM | Better quality |
| Gemma 4 E4B (4-bit) | GPU | ~12 GB VRAM | Target for production PoC |

## Swapping Components

### Encoder

The encoder is a frozen feature extractor for tabular health time-series:

```python
from src.encoder import get_encoder

# Default: 10 features, 4-layer Transformer, 512-dim output
encoder = get_encoder(n_features=10)

# Customize for your ring's feature set:
encoder = get_encoder(n_features=7, d_model=128, n_layers=2, output_dim=256)
```

The encoder must satisfy:
- Input: `(batch, T, n_features)`
- Output: `(batch, T, output_dim)` — per-step embeddings
- Attribute: `self.output_dim`

### Upgrading to Gemma 4 26B A4B

For production with the larger model:

```python
model = RingGemmaModel(
    n_features=10,
    llm_dim=3072,      # 26B uses 3072 hidden dim
    use_real_llm=True,
)
```

### Using Real Datasets

Supported data formats:

- **WESAD** (Wearable Stress and Affect Detection): Wrist BVP, EDA, TEMP, ACC → tabular features
  - `ubicomp.eti.uni-siegen.de/home/datasets/icmi18/`
- **GLOBEM**: Pre-extracted phone + wearable features, daily granularity
- **Custom ring data**: Any `(T, F)` tabular health time-series

Each sample must provide:
```python
{"health_ts": Tensor(T, F), "ehr_text": str, "label": int}
```

## Project Structure

```
ring-gemma4-depression-poc/
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # Tabular z-score normalization, NaN handling
│   ├── encoder.py          # MacroTrendEncoder (Transformer, 4 layers)
│   ├── projector.py        # SensorProjector MLP (512 → 2048)
│   ├── dataset.py          # DepressionDataset + synthetic + WESAD loader
│   ├── model.py            # RingGemmaModel (encoder+projector+LLM+head)
│   ├── train.py            # Two-stage training script
│   └── evaluate.py         # Metrics + report generation
├── notebooks/
│   └── demo.ipynb          # End-to-end demo with PCA visualization
├── data/raw/               # Place raw data here
├── checkpoints/            # Saved model checkpoints
└── results/                # Evaluation reports
```

## Training Strategy

### Stage 1: Projector Warm-up (5 epochs, lr=1e-3)
- Only the SensorProjector MLP is trained
- Encoder and LLM backbone are frozen
- Goal: learn the sensor → LLM token mapping

### Stage 2: Joint Fine-tuning (10 epochs, lr=2e-5)
- Projector + LoRA adapters (q,k,v,o_proj) + classification head
- Encoder remains frozen
- Goal: adapt LLM to fuse sensor + text modalities

### Loss & Optimization
- CrossEntropyLoss with class weights (handles label imbalance)
- AdamW optimizer
- Gradient clipping: max_norm=1.0

## Citation

If you use this work, please cite:

```bibtex
@misc{ringgemma2025,
  title={Ring-Gemma4 Depression PoC: Multimodal Neuropsychiatric Prediction
         from Wearable Health Metrics and EHR Text},
  author={ITMO University},
  year={2025},
  note={Proof of concept implementing the optiHealth 2.0 custom encoder paradigm}
}
```

## License

MIT — see [LICENSE](LICENSE).
