# Ring-Gemma4 Depression PoC

**Multimodal neuropsychiatric prediction from PPG + EHR text using learned neural embeddings projected into LLM token space.**

This proof-of-concept implements the optiHealth 2.0 "Everything is a Vector" paradigm: raw wearable sensor data is encoded by a custom temporal encoder, projected into virtual tokens, and fused with clinical text inside a language model backbone for binary depression classification.

## Architecture

```
                         ┌─────────────────────────────────┐
                         │        Raw PPG Signal            │
                         │  (~25 Hz from smart ring)        │
                         └───────────────┬─────────────────┘
                                         │
                                   ┌─────▼──────┐
                                   │ Preprocess  │
                                   │ • Resample  │
                                   │   → 125 Hz  │
                                   │ • Bandpass   │
                                   │   0.5–4 Hz  │
                                   │ • Segment    │
                                   │   10s / 50%  │
                                   │ • Z-score    │
                                   └─────┬───────┘
                                         │
                                  (N, 1, 1250)
                                         │
                         ┌───────────────▼────────────────┐
                         │    Sensor Encoder (FROZEN)      │
                         │  ResNet-1D: 18 residual blocks  │
                         │  64 → 128 → 256 → 512 channels │
                         │  Global Average Pooling → 512d  │
                         │                                 │
                         │  [PaPaGei fallback available]   │
                         └───────────────┬────────────────┘
                                         │
                                  (N, 512)
                                         │
                         ┌───────────────▼────────────────┐
                         │  SensorProjector (TRAINABLE)   │
                         │  ~4M parameters                │
                         │                                │
                         │  AdaptiveAvgPool1d(16 tokens)  │
                         │  Linear(512 → 2048) → GELU    │
                         │  LayerNorm(2048)               │
                         │  Linear(2048 → 2048)           │
                         └───────────────┬────────────────┘
                                         │
                                  (16, 2048)
                                         │
       ┌──────────────┐                  │
       │   EHR Text   │                  │
       │   Tokenizer  │──→ (T, 2048)     │
       └──────────────┘         │        │
                                └───┬────┘
                                    │ concat
                              (16+T, 2048)
                                    │
                         ┌──────────▼─────────────────────┐
                         │   LLM Backbone (QLoRA)         │
                         │   Gemma 3 1B-IT (4-bit NF4)    │
                         │   LoRA: q_proj + v_proj        │
                         │         r=8, α=16              │
                         └──────────┬─────────────────────┘
                                    │
                              first token
                                    │
                         ┌──────────▼─────────────────────┐
                         │   Classification Head           │
                         │   Linear(2048 → 2)             │
                         │   Depression: yes / no          │
                         └────────────────────────────────┘
```

### The optiHealth 2.0 "Everything is a Vector" Paradigm

Traditional wearable health pipelines extract hand-crafted feature vectors (e.g., `[resting_hr, hrv_rmssd, sleep_score, spo2]`). The custom encoder paradigm **replaces** this with learned neural embeddings that capture non-linear physiological dynamics:

1. **Phase 1 — Custom Temporal Encoder**: Raw sensor data → compressed state vector `h ∈ R^d_enc`
2. **Phase 2 — Vector Database**: Store `h` vectors for similarity search (not implemented in this PoC)
3. **Phase 3 — Cross-Modal Alignment**: Project `h` into LLM token space via `Z = MLP(h)`, creating "Virtual Tokens" injected via `inputs_embeds`

This PoC implements Phases 1 and 3, demonstrating that a frozen PPG encoder + trainable projector can align sensor semantics with a language model's representation space.

## Architecture Fixes Applied

### Causal Last-Token Classification
Decoder-only transformers (Gemma, GPT) use causal attention: token at position 0 can only see itself, while the last token attends to the entire sequence. The classification head now reads the last non-padded hidden state, ensuring full multimodal fusion between sensor tokens and EHR text.

### Attention Mask for Padded Sequences
When PPG sensor tokens are concatenated with variable-length tokenized EHR text, an explicit attention mask is constructed and passed to the LLM backbone. Sensor tokens always attend (mask=1); text tokens follow the tokenizer's padding mask. This prevents padded positions from corrupting the classifier input.

### Gemma 4 Upgrade Path
The real-LLM loader now defaults to `google/gemma-4-it` and falls back to `google/gemma-3-1b-it` if Gemma 4 is not yet available on HuggingFace. Override via `GEMMA_MODEL` environment variable. LoRA targets all four attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`).

### WESAD Dataset Support
Real physiological data from the [WESAD dataset](https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/) can now be loaded directly:

```bash
# 1. Download WESAD from https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/
# 2. Extract to data/raw/WESAD/
# 3. Train:
python src/train.py --wesad_path data/raw/WESAD --cpu --epochs_stage1 5 --epochs_stage2 10
```

The loader reads wrist BVP (PPG at 64 Hz) from each subject's pickle file and uses PANAS negative-affect scores as binary labels (above median = distressed).

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

### Evaluate

```bash
python src/evaluate.py --synthetic --cpu
# → results/evaluation_report.md
```

### Train with WESAD data

```bash
python src/train.py --wesad_path data/raw/WESAD --cpu --epochs_stage1 5 --epochs_stage2 10
```

### Train with real Gemma on GPU

```bash
python src/train.py --synthetic --use_real_llm --epochs_stage1 5 --epochs_stage2 10
# Or override the model:
GEMMA_MODEL=google/gemma-4-e4b-it python src/train.py --synthetic --use_real_llm
```

### Run the demo notebook

```bash
jupyter notebook notebooks/demo.ipynb
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

The encoder is a frozen feature extractor. Swap it by modifying `src/encoder.py`:

```python
# PaPaGei (Nokia Bell Labs) — designed for PPG
encoder = get_encoder(use_papagei=True)

# Custom ResNet-1D (default fallback)
encoder = get_encoder(use_papagei=False)

# To use your own encoder:
# 1. Implement nn.Module with forward(batch, 1, 1250) → (batch, d)
# 2. Set encoder.output_dim = d
# 3. Update SensorProjector(encoder_dim=d, ...)
```

Other compatible encoders:
- **Pulse-PPG**: Specialized PPG encoder
- **MOMENT**: Time-series foundation model
- **TST**: Time Series Transformer

### Upgrading to Gemma 4 26B A4B

For production with the larger model:

```python
# In src/model.py, update:
model = RingGemmaModel(
    llm_dim=3072,      # 26B uses 3072 hidden dim
    use_real_llm=True,
)

# In src/projector.py, the MLP auto-adjusts:
# Linear(512 → 3072) → GELU → LayerNorm → Linear(3072 → 3072)
```

### Using Real Datasets

Supported datasets for depression/stress prediction from PPG:

- **WESAD** (Wearable Stress and Affect Detection): PPG + stress labels
  - `archive.ics.uci.edu/ml/datasets/WESAD`
- **GLOBEM** (Global Emotion Dataset): Wearable + mood labels
- Custom clinical data: PPG recordings + EHR notes + PHQ-9 labels

To use real data, implement a data loader in `src/dataset.py` that returns the same format:
```python
{"ppg_segments": Tensor(N, 1, 1250), "ehr_text": str, "label": int}
```

## Project Structure

```
ring-gemma4-depression-poc/
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # PPG filtering, segmentation, normalization
│   ├── encoder.py          # ResNet-1D (18 blocks) + PaPaGei wrapper
│   ├── projector.py        # SensorProjector MLP (512 → 2048)
│   ├── dataset.py          # DepressionDataset + synthetic generator
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
- Projector + LoRA adapters (q_proj, v_proj) + classification head
- Encoder remains frozen
- Goal: adapt LLM to fuse sensor + text modalities

### Loss & Optimization
- CrossEntropyLoss with class weights (handles label imbalance)
- AdamW optimizer
- Gradient clipping: max_norm=1.0

## Citation

If you use this work, please cite:

```bibtex
@misc{ringgemma2024,
  title={Ring-Gemma4 Depression PoC: Multimodal Neuropsychiatric Prediction
         from PPG and EHR Text},
  author={ITMO University},
  year={2024},
  note={Proof of concept implementing the optiHealth 2.0 custom encoder paradigm}
}
```

## License

MIT — see [LICENSE](LICENSE).
