"""Full multimodal model: PPG encoder + projector + LLM backbone + classifier."""

import math
import torch
import torch.nn as nn

from src.encoder import get_encoder
from src.projector import SensorProjector


# ---------------------------------------------------------------------------
# MockLLM — lightweight stand-in that works offline on CPU
# ---------------------------------------------------------------------------

class MockEmbedTokens(nn.Module):
    """Simple embedding layer mimicking model.embed_tokens."""

    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


class MockLLM(nn.Module):
    """Lightweight transformer that matches the Gemma interface.

    Provides:
        - embed_tokens(input_ids) → (batch, seq, hidden_dim)
        - forward(inputs_embeds) → object with .last_hidden_state
    """

    def __init__(self, vocab_size: int = 32000, hidden_dim: int = 2048,
                 n_heads: int = 8, n_layers: int = 2, max_len: int = 2048):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embed_tokens = MockEmbedTokens(vocab_size, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=n_layers)

        # Positional encoding buffer
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, dim)

    def forward(self, inputs_embeds: torch.Tensor | None = None,
                input_ids: torch.Tensor | None = None,
                **kwargs):
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Provide inputs_embeds or input_ids")
            inputs_embeds = self.embed_tokens(input_ids)

        seq_len = inputs_embeds.size(1)
        x = inputs_embeds + self.pe[:, :seq_len, :]
        hidden = self.transformer(x)

        # Return object with .last_hidden_state
        class _Out:
            pass
        out = _Out()
        out.last_hidden_state = hidden
        return out


class MockTokenizer:
    """Minimal tokenizer for offline testing."""

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, texts, return_tensors="pt", padding=True,
                 truncation=True, max_length=128, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        rng = torch.Generator().manual_seed(hash(str(texts)) % (2**31))
        batch_size = len(texts)
        # Crude: make length proportional to text length, capped at max_length
        lengths = [min(max(len(t.split()) + 2, 5), max_length) for t in texts]
        max_len = max(lengths)
        input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        for i, length in enumerate(lengths):
            ids = torch.randint(2, self.vocab_size, (length,), generator=rng)
            input_ids[i, :length] = ids
            attention_mask[i, :length] = 1

        class _Enc:
            pass
        enc = _Enc()
        enc.input_ids = input_ids
        enc.attention_mask = attention_mask
        return enc


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class RingGemmaModel(nn.Module):
    """Multimodal model: PPG encoder → projector → LLM → classification head.

    Args:
        encoder_dim: Encoder output dimension (default 512).
        llm_dim: LLM hidden dimension (default 2048).
        n_tokens: Number of virtual sensor tokens (default 16).
        use_real_llm: If True, attempt to load a real Gemma model.
        device: 'cpu' or 'cuda'.
    """

    def __init__(self, encoder_dim: int = 512, llm_dim: int = 2048,
                 n_tokens: int = 16, use_real_llm: bool = False,
                 device: str = "cpu"):
        super().__init__()
        self.device_str = device
        self.llm_dim = llm_dim
        self.use_real_llm = use_real_llm

        # Frozen PPG encoder
        self.encoder = get_encoder(use_papagei=False)
        self.encoder.output_dim = encoder_dim

        # Trainable projector
        self.projector = SensorProjector(encoder_dim=encoder_dim,
                                         llm_dim=llm_dim,
                                         n_tokens=n_tokens)

        # LLM backbone
        if use_real_llm:
            self.llm, self.tokenizer = self._load_real_llm(device)
        else:
            self.llm = MockLLM(hidden_dim=llm_dim)
            self.tokenizer = MockTokenizer()

        # Classification head on first token's hidden state
        self.classifier = nn.Linear(llm_dim, 2)

    def _load_real_llm(self, device: str):
        """Attempt to load a real Gemma model with quantization on GPU."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "google/gemma-3-1b-it"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if device == "cuda":
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float32,
            )

        # Apply LoRA
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        # Wrap so .embed_tokens and forward work consistently
        class _GemmaWrapper(nn.Module):
            def __init__(self, peft_model):
                super().__init__()
                self.model = peft_model
                base = peft_model.get_base_model()
                self.embed_tokens = base.model.embed_tokens
                self.hidden_dim = base.config.hidden_size

            def forward(self, inputs_embeds=None, input_ids=None, **kwargs):
                out = self.model(inputs_embeds=inputs_embeds,
                                 input_ids=input_ids,
                                 output_hidden_states=True, **kwargs)
                class _Out:
                    pass
                result = _Out()
                result.last_hidden_state = out.hidden_states[-1]
                return result

        wrapper = _GemmaWrapper(model)
        return wrapper, tokenizer

    def forward(self, ppg_segments: torch.Tensor, ehr_text: list[str],
                labels: torch.Tensor | None = None):
        """Forward pass.

        Args:
            ppg_segments: (batch, N_segments, 1, 1250) PPG windows.
            ehr_text: list of EHR text strings (length = batch).
            labels: (batch,) integer labels, optional.

        Returns:
            dict with 'logits' and optionally 'loss'.
        """
        batch_size = ppg_segments.size(0)
        n_seg = ppg_segments.size(1)
        device = ppg_segments.device

        # 1. Encode PPG segments
        flat = ppg_segments.view(batch_size * n_seg, 1, -1)  # (B*N, 1, 1250)
        with torch.no_grad():
            enc = self.encoder(flat)  # (B*N, 512)
        enc = enc.view(batch_size, n_seg, -1)  # (B, N, 512)

        # 2. Project to LLM token space
        sensor_tokens = self.projector(enc)  # (B, n_tokens, llm_dim)

        # 3. Tokenize EHR text
        tok = self.tokenizer(ehr_text, return_tensors="pt", padding=True,
                             truncation=True, max_length=128)
        input_ids = tok.input_ids.to(device)

        # 4. Get text embeddings
        text_embeds = self.llm.embed_tokens(input_ids)  # (B, T, llm_dim)

        # 5. Concatenate [sensor_tokens, text_embeds]
        combined = torch.cat([sensor_tokens, text_embeds], dim=1)

        # 6. Forward through LLM
        out = self.llm(inputs_embeds=combined)
        hidden = out.last_hidden_state  # (B, S, llm_dim)

        # 7. Classification on first token
        first_token = hidden[:, 0, :]
        logits = self.classifier(first_token)  # (B, 2)

        result = {"logits": logits}

        if labels is not None:
            labels = labels.to(device)
            loss_fn = nn.CrossEntropyLoss()
            result["loss"] = loss_fn(logits, labels)

        return result

    def get_trainable_params(self, stage: int = 1) -> list[nn.Parameter]:
        """Return parameters that should be trained for a given stage.

        Stage 1: projector only
        Stage 2: projector + LLM (LoRA) + classifier
        """
        params = list(self.projector.parameters())
        if stage >= 2:
            params += [p for p in self.llm.parameters() if p.requires_grad]
            params += list(self.classifier.parameters())
        return params
