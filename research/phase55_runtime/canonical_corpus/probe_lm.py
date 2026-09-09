"""Small controlled decoder-only Transformer used by Phase 4 experiments."""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LMConfig:
    vocab_size: int
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: int = 8
    ffn_size: int = 1536
    context_length: int = 512
    norm: str = "layernorm"
    activation: str = "gelu"
    positional_encoding: str = "sinusoidal"
    tie_embeddings: bool = True
    bos_id: int = 1
    eos_id: int = 2
    pad_id: int = 2

    def __post_init__(self):
        if self.hidden_size % self.num_heads or self.num_heads % self.num_kv_heads:
            raise ValueError("hidden_size must divide heads and heads must divide kv heads")
        if self.vocab_size < 4 or self.context_length < 2:
            raise ValueError("invalid vocab/context")

    @property
    def head_dim(self) -> int: return self.hidden_size // self.num_heads

    def to_dict(self): return asdict(self)


def _sinusoidal(length: int, dim: int, device: torch.device) -> torch.Tensor:
    pos = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    inv = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    out = torch.zeros(length, dim, device=device)
    out[:, 0::2], out[:, 1::2] = torch.sin(pos * inv), torch.cos(pos * inv)
    return out


class DecoderBlock(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(cfg.hidden_size), nn.LayerNorm(cfg.hidden_size)
        self.q = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.k = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.v = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.o = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.ff1 = nn.Linear(cfg.hidden_size, cfg.ffn_size, bias=False)
        self.ff2 = nn.Linear(cfg.ffn_size, cfg.hidden_size, bias=False)
        self.cfg = cfg

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        h = self.norm1(x)
        q = self.q(h).view(b, t, self.cfg.num_heads, self.cfg.head_dim).transpose(1, 2)
        k = self.k(h).view(b, t, self.cfg.num_kv_heads, self.cfg.head_dim).transpose(1, 2)
        v = self.v(h).view(b, t, self.cfg.num_kv_heads, self.cfg.head_dim).transpose(1, 2)
        if self.cfg.num_kv_heads != self.cfg.num_heads:
            repeat = self.cfg.num_heads // self.cfg.num_kv_heads
            k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        x = x + self.o(a.transpose(1, 2).reshape(b, t, -1))
        return x + self.ff2(F.gelu(self.ff1(self.norm2(x))))


class ProbeLM(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__(); self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._reset_parameters()
        if config.tie_embeddings: self.lm_head.weight = self.embedding.weight

    def _reset_parameters(self) -> None:
        # Keep tied logits numerically well-scaled at the start of a probe run.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None: nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None,
                loss_mask: torch.Tensor | None = None):
        _, t = input_ids.shape
        x = self.embedding(input_ids) + _sinusoidal(t, self.config.hidden_size, input_ids.device)
        causal = torch.ones(t, t, device=input_ids.device, dtype=torch.bool).tril()
        for block in self.blocks: x = block(x, causal)
        logits = self.lm_head(self.norm(x))
        if labels is None: return logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none")
        if loss_mask is not None: loss = loss[loss_mask.reshape(-1)]
        return logits, loss.mean() if loss.numel() else logits.sum() * 0.0

    def parameter_report(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        emb = self.embedding.weight.numel()
        return {"total": total, "embedding": emb, "lm_head": 0 if self.config.tie_embeddings else self.lm_head.weight.numel(), "non_embedding": total - emb}


def validate_tokenizer_contract(tokenizer: Any, config: LMConfig) -> dict[str, Any]:
    proc = getattr(tokenizer, "processor", tokenizer)
    size = getattr(proc, "get_piece_size", lambda: None)()
    if size is not None and int(size) != config.vocab_size: raise ValueError("tokenizer/model vocab mismatch")
    methods = {"bos_id": "bos_id", "eos_id": "eos_id", "pad_id": "pad_id"}
    for name, expected in (("bos_id", config.bos_id), ("eos_id", config.eos_id), ("pad_id", config.pad_id)):
        actual = getattr(tokenizer, name, None)
        if actual is None and callable(getattr(proc, methods[name], None)): actual = getattr(proc, methods[name])()
        if actual is None:
            raise ValueError(f"tokenizer does not expose {name}; refusing to assume alignment")
        # SentencePiece uses -1 for disabled PAD; Phase 4 intentionally aliases
        # PAD to EOS so loss masking remains well-defined.
        if name == "pad_id" and int(actual) < 0: actual = config.eos_id
        if actual is not None and int(actual) != expected: raise ValueError(f"{name} mismatch")
    return {"vocab_size": config.vocab_size, "bos_id": config.bos_id, "eos_id": config.eos_id, "pad_id": config.pad_id}


def set_seed(seed: int, *, threads: int = 2) -> None:
    random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(threads)


def save_checkpoint(path: str | Path, model: ProbeLM, optimizer: torch.optim.Optimizer | None = None, *, step: int = 0, seed: int = 0) -> None:
    payload = {"config": model.config.to_dict(), "model": model.state_dict(), "step": step, "seed": seed,
               "optimizer": optimizer.state_dict() if optimizer is not None else None}
    torch.save(payload, path)


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu", optimizer: torch.optim.Optimizer | None = None) -> tuple[ProbeLM, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model = ProbeLM(LMConfig(**payload["config"])); model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        model_param_ids = {id(p) for p in model.parameters()}
        optimizer_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        if model_param_ids != optimizer_param_ids:
            raise ValueError("optimizer is bound to a different model; create it from the restored model")
        optimizer.load_state_dict(payload["optimizer"])
    return model, payload


def config_hash(config: LMConfig) -> str:
    return hashlib.sha256(json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
