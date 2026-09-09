"""Source-accounted, resumable Phase 5.5 probe execution.

Evaluation predicts each content token once, with BOS as input framing and no
EOS target. Compute estimates are explicitly derived, never hardware counters.
This implementation does not provide KV caching or an energy profiler.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import random
import resource
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from .probe_lm import LMConfig, ProbeLM, set_seed, validate_tokenizer_contract
from .probe_lm_data import token_source_prefixes
from .tokenizer_adapters import ReversibleSentencePieceAdapter

REGIME_ALIASES = {
    "same_total": "same_total_parameters", "same_source": "same_source_characters",
    "same_tokens": "same_token_budget", "same_token": "same_token_budget",
}
REGIMES = {"same_total_parameters", "same_source_characters", "same_token_budget", "same_compute"}


def parameter_count(config: LMConfig) -> int:
    """Analytic count of unique trainable parameters in the existing ProbeLM."""
    h, f, layers = config.hidden_size, config.ffn_size, config.num_layers
    kv = config.num_kv_heads * config.head_dim
    return (config.vocab_size * h * (1 if config.tie_embeddings else 2)
            + layers * (2 * h * h + 2 * h * kv + 2 * h * f + 4 * h) + 2 * h)


def matched_config(vocab_size: int, target_parameters: int = 150_000_000,
                   hidden_size: int = 768, num_layers: int = 12,
                   num_heads: int = 12, context_length: int = 4096,
                   num_kv_heads: int | None = None, **kwargs: Any) -> LMConfig:
    """Match total capacity by integer FFN width, leaving depth/attention fixed."""
    cfg = LMConfig(vocab_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers,
                   num_heads=num_heads, num_kv_heads=num_heads if num_kv_heads is None else num_kv_heads,
                   context_length=context_length, ffn_size=1, **kwargs)
    increment = 2 * hidden_size * num_layers
    width = round((target_parameters - parameter_count(cfg)) / increment) + 1
    if width < 1:
        raise ValueError("target parameters cannot accommodate fixed attention/embedding")
    cfg = replace(cfg, ffn_size=width)
    if abs(parameter_count(cfg) - target_parameters) / target_parameters > 0.001:
        raise ValueError("integer FFN width cannot match total parameters within 0.1%")
    return cfg


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _rng_state(device: torch.device) -> dict[str, Any]:
    state = {"python": random.getstate(), "torch_cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    if device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def _restore_rng(state: dict, device: torch.device) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
    if device.type == "mps":
        torch.mps.set_rng_state(state["mps"].cpu())


def _state_digest(value: Any) -> str:
    """Stable digest without allocating a second full model or optimizer."""
    h = hashlib.sha256()
    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().contiguous().cpu()
            h.update(str(tensor.dtype).encode())
            h.update(str(tuple(tensor.shape)).encode())
            h.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                h.update(str(key).encode()); visit(item[key])
        elif isinstance(item, (tuple, list)):
            for element in item:
                visit(element)
        else:
            h.update(repr(item).encode())
    visit(value)
    return h.hexdigest()


def _validate_rows(train: list[dict], evaluation: list[dict]) -> dict:
    if not train or not evaluation:
        raise ValueError("nonempty train and evaluation rows required")
    sets = []
    unknown_eval_work_records = 0
    for split, rows in (("train", train), ("eval", evaluation)):
        docs, texts, ids = set(), set(), set()
        for row in rows:
            if not isinstance(row.get("text"), str) or not row["text"]:
                raise ValueError("each record requires nonempty text")
            for key in ("id", "category", "text_sha256"):
                if not row.get(key):
                    raise ValueError(f"each record requires {key}")
            document = row.get("document_id")
            if not document:
                identity_level = row.get("lineage", {}).get("identity_level")
                if split == "train" or identity_level not in {"synthetic", "record_only"}:
                    raise ValueError("each training/major evaluation record requires document_id")
                unknown_eval_work_records += 1
            digest = hashlib.sha256(row["text"].encode()).hexdigest()
            if digest != row["text_sha256"]:
                raise ValueError("record source text hash mismatch")
            if row["id"] in ids:
                raise ValueError("duplicate record id within split")
            if document: docs.add(str(document))
            texts.add(digest); ids.add(row["id"])
        sets.append((docs, texts, ids))
    if any(left & right for left, right in zip(*sets)):
        raise ValueError("train/eval document, record, or exact-text contamination")
    return {"train_document_count": len(sets[0][0]), "eval_document_count": len(sets[1][0]),
            "document_overlap": 0, "exact_text_overlap": 0,
            "eval_records_without_work_identity": unknown_eval_work_records,
            "scope": "known work identities and all record/exact-text hashes; diagnostic unknown work identities and upstream near-duplicates remain uncertified"}


def _source_prefix(rows: list[dict], budget: int) -> list[dict]:
    if budget < 1 or sum(len(row["text"]) for row in rows) < budget:
        raise ValueError("source character budget must fit a single input corpus pass")
    selected, remaining = [], budget
    for row in rows:
        if not remaining:
            break
        text = row["text"][:remaining]
        selected.append({**row, "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest()})
        remaining -= len(text)
    return selected


class _CompactSourcePrefixes:
    """Exact numeric source offsets, without per-token diagnostic dictionaries."""
    def __init__(self, prefixes):
        from array import array
        self.chars = array("Q", (p["source_chars"] for p in prefixes))
        self.bytes = array("Q", (p["source_bytes"] for p in prefixes))
        self.complete = bytearray(p["complete"] for p in prefixes)

    def __len__(self):
        return len(self.chars)

    def __getitem__(self, index):
        return {"source_chars": self.chars[index], "source_bytes": self.bytes[index], "complete": bool(self.complete[index])}


def _prepare(tokenizer: Any, rows: list[dict]) -> tuple[list[dict], dict]:
    prepared, exact, unknown = [], 0, 0
    for row in rows:
        ids = list(tokenizer.encode(row["text"]))
        if not ids:
            raise ValueError("nonempty source encoded to no tokens")
        decoded = tokenizer.decode(ids)
        exact += int(decoded == row["text"])
        unknown += sum(int(tokenizer.is_unk(token)) for token in ids)
        prefixes = token_source_prefixes(tokenizer, row["text"])
        if len(prefixes) != len(ids):
            raise ValueError("source prefix / token count mismatch")
        if prefixes[-1]["source_chars"] != len(row["text"]):
            raise ValueError("source prefix accounting incomplete")
        prepared.append({**row, "ids": ids, "prefixes": _CompactSourcePrefixes(prefixes)})
    if exact != len(rows) or unknown:
        raise ValueError("tokenizer exact preservation / zero unknown gate failed")
    return prepared, {"records": len(rows), "exact_preservation": 1.0,
                      "unknown_tokens": unknown, "scope": "all supplied run records"}


def _blocks(rows: list[dict], cfg: LMConfig, length: int) -> list[dict]:
    blocks = []
    for row in rows:
        ids = row["ids"]
        inputs = [cfg.bos_id] + ids[:-1]
        for start in range(0, len(ids), length):
            end = min(len(ids), start + length)
            blocks.append({"row": row, "start": start, "end": end,
                           "x": inputs[start:end], "y": ids[start:end]})
    return blocks


def _batch(blocks: list[dict], cfg: LMConfig, device: torch.device,
           remaining_tokens: int | None = None) -> tuple[tuple, dict]:
    selected = []
    completed_blocks = partial_block_tokens = 0
    for block in blocks:
        n = len(block["y"])
        if remaining_tokens is not None:
            n = min(n, remaining_tokens)
            remaining_tokens -= n
        if n:
            selected.append({**block, "x": block["x"][:n], "y": block["y"][:n],
                             "end": block["start"] + n})
            if n == len(block["y"]):
                completed_blocks += 1
            else:
                partial_block_tokens = n
        if remaining_tokens == 0:
            break
    if not selected:
        raise ValueError("empty training batch")
    width = max(len(b["y"]) for b in selected)
    inputs, labels, masks = [], [], []
    chars = bytes_ = tokens = 0
    complete = True
    documents = set()
    for b in selected:
        n = len(b["y"])
        inputs.append(b["x"] + [cfg.pad_id] * (width - n))
        labels.append(b["y"] + [cfg.pad_id] * (width - n))
        masks.append([True] * n + [False] * (width - n))
        prefixes = b["row"]["prefixes"]
        before = prefixes[b["start"] - 1] if b["start"] else {"source_chars": 0, "source_bytes": 0}
        after = prefixes[b["end"] - 1]
        chars += after["source_chars"] - before["source_chars"]
        bytes_ += after["source_bytes"] - before["source_bytes"]
        complete = complete and after["complete"]
        tokens += n; documents.add(str(b["row"]["document_id"]))
    tensors = (torch.tensor(inputs, dtype=torch.long, device=device),
               torch.tensor(labels, dtype=torch.long, device=device),
               torch.tensor(masks, dtype=torch.bool, device=device))
    accounting = {"tokens": tokens, "source_chars": chars, "source_bytes": bytes_,
                  "source_boundary_complete": complete, "documents": sorted(documents),
                  "completed_blocks": completed_blocks, "partial_block_tokens": partial_block_tokens,
                  "batch_size": len(selected), "sequence_length": width,
                  "padded_positions": len(selected) * width}
    return tensors, accounting


def _stream_selection(blocks: list[dict], progress: dict, batch_size: int) -> list[dict]:
    selected = blocks[progress["cursor"]:progress["cursor"] + batch_size]
    offset = progress["block_token_offset"]
    if offset:
        first = selected[0]
        selected[0] = {**first, "start": first["start"] + offset,
                       "x": first["x"][offset:], "y": first["y"][offset:]}
    return selected


def derived_training_flops(cfg: LMConfig, batch_size: int, sequence_length: int) -> int:
    """6P/token plus quadratic attention; excludes optimizer and kernel overhead."""
    return (6 * parameter_count(cfg) * batch_size * sequence_length
            + 12 * cfg.num_layers * cfg.hidden_size * batch_size * sequence_length ** 2)


def _learning_rate(step: int, run: dict) -> float:
    warmup, horizon = int(run["warmup_steps"]), int(run["total_steps"])
    if run.get("scheduler_axis") in {"tokens", "source_chars"}:
        warmup, horizon = ((int(run["warmup_source_chars"]), int(run["source_char_budget"])) if run["scheduler_axis"] == "source_chars" else (int(run["warmup_tokens"]), int(run["token_budget"])))
        step = int(run.get("_scheduler_position", 0))
        if warmup and step < warmup:
            return float(run["learning_rate"]) * step / warmup
    if warmup and step < warmup:
        factor = (step + 1) / warmup
    else:
        progress = min(1.0, max(0.0, (step - warmup) / max(1, horizon - warmup)))
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(run["learning_rate"]) * factor


def _step(model: ProbeLM, optimizer: Any, batch: tuple, run: dict, step: int) -> float:
    model.train(); optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group["lr"] = _learning_rate(step, run)
    _, loss = model(*batch)
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("non-finite training loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(run["gradient_clip"]), error_if_nonfinite=True)
    if not bool(torch.isfinite(norm).item()):
        raise FloatingPointError("non-finite gradient")
    optimizer.step()
    return float(loss.detach().item())


def _optimizer(model: ProbeLM, run: dict) -> Any:
    return torch.optim.AdamW(model.parameters(), lr=run["learning_rate"],
                             weight_decay=run["weight_decay"], foreach=False)


def _save(path: Path, model: ProbeLM, optimizer: Any, progress: dict, identity: str,
          run: dict, device: torch.device) -> None:
    payload = {"format": "novllm-phase55-checkpoint-v1", "identity": identity,
               "model_config": model.config.to_dict(), "model": model.state_dict(),
               "optimizer": optimizer.state_dict(), "progress": progress,
               "scheduler": {"kind": "linear_warmup_cosine", "completed_steps": progress["steps"],
                             "total_steps": run["total_steps"], "warmup_steps": run["warmup_steps"],
                             "axis": run.get("scheduler_axis", "steps"), "completed_tokens": progress["tokens"],
                             "token_budget": run.get("token_budget"), "warmup_tokens": run.get("warmup_tokens"), "source_char_budget": run.get("source_char_budget"), "warmup_source_chars": run.get("warmup_source_chars"), "completed_source_chars": progress["source_chars"]},
               "rng": _rng_state(device)}
    temp = path.with_name(path.name + ".tmp")
    torch.save(payload, temp)
    with temp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temp, path)
    digest = _file_hash(path)
    hash_path = path.with_suffix(".sha256")
    temp_hash = hash_path.with_name(hash_path.name + ".tmp")
    temp_hash.write_text(digest + "\n")
    os.replace(temp_hash, hash_path)


def _load(path: Path, model: ProbeLM, run: dict, identity: str,
          device: torch.device) -> tuple[Any, dict]:
    hash_path = path.with_suffix(".sha256")
    if not hash_path.exists() or _file_hash(path) != hash_path.read_text().strip():
        raise ValueError("checkpoint hash sidecar missing or mismatched")
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if payload["identity"] != identity or payload["model_config"] != model.config.to_dict():
        raise ValueError("checkpoint configuration / data / tokenizer identity mismatch")
    model.load_state_dict(payload["model"])
    optimizer = _optimizer(model, run)
    optimizer.load_state_dict(payload["optimizer"])
    _restore_rng(payload["rng"], device)
    return optimizer, payload["progress"]


def _normalized(nll: float, tokens: int, chars: int, bytes_: int) -> dict:
    token_nll = nll / tokens if tokens else None
    return {"nll_sum": nll, "tokens": tokens, "source_chars": chars, "source_bytes": bytes_,
            "nll_per_character": nll / chars if chars else None,
            "nll_per_utf8_byte": nll / bytes_ if bytes_ else None,
            "bits_per_byte": nll / (bytes_ * math.log(2)) if bytes_ else None,
            "token_nll": token_nll,
            "token_perplexity_reference_only": math.exp(token_nll) if token_nll is not None and token_nll < 700 else None}


@torch.no_grad()
def evaluate(model: ProbeLM, prepared: list[dict], cfg: LMConfig, device: torch.device,
             sequence_length: int) -> dict:
    model.eval(); accum: dict[str, list] = {}
    for row in prepared:
        nll, tokens = 0.0, 0
        for block in _blocks([row], cfg, sequence_length):
            batch, accounting = _batch([block], cfg, device)
            _, loss = model(*batch)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("non-finite validation loss")
            nll += float(loss.item()) * accounting["tokens"]
            tokens += accounting["tokens"]
        if tokens != len(row["ids"]):
            raise AssertionError("validation must score every source token exactly once")
        values = [nll, tokens, len(row["text"]), len(row["text"].encode())]
        for category in (str(row["category"]), "__overall__"):
            target = accum.setdefault(category, [0.0, 0, 0, 0])
            for i, value in enumerate(values):
                target[i] += value
    return {"overall": _normalized(*accum.pop("__overall__")),
            "domains": {name: _normalized(*values) for name, values in sorted(accum.items())},
            "framing": "BOS input only; EOS excluded; every content token scored exactly once",
            "context_policy": "nonoverlapping target blocks; preceding token retained as input at block boundary",
            "sequence_length": sequence_length}


@torch.no_grad()
def _inference(model: ProbeLM, tokenizer: Any, prepared: list[dict], device: torch.device,
               sequence_length: int, decode_steps: int) -> dict:
    model.eval(); row = prepared[0]
    ids = row["ids"][:max(1, sequence_length - 1)]
    prefix = [model.config.bos_id, *ids]
    tensor = torch.tensor([prefix], dtype=torch.long, device=device)
    _sync(device); start = time.perf_counter(); logits = model(tensor); _sync(device)
    elapsed = time.perf_counter() - start
    source = row["prefixes"][len(ids) - 1]
    result = {"prefill_latency_seconds": elapsed, "prefill_input_tokens": len(prefix),
              "prefill_source_tokens": len(ids), "prefill_source_chars": source["source_chars"],
              "prefill_tokens_per_second": len(prefix) / elapsed,
              "prefill_chars_per_second": source["source_chars"] / elapsed,
              "prefill_source_bytes": source["source_bytes"],
              "prefill_bytes_per_second": source["source_bytes"] / elapsed,
              "prefill_source_boundary_complete": source["complete"],
              "decode_implementation": "full context recomputation; no KV cache",
              "peak_kv_cache_bytes": None, "peak_kv_cache_bytes_reason": "model has no KV cache implementation"}
    generated = []
    _sync(device); start = time.perf_counter()
    for _ in range(decode_steps):
        # Count every decode step including its own forward pass.
        logits = model(torch.tensor([prefix[-model.config.context_length:]], dtype=torch.long, device=device))
        token = int(logits[0, -1].argmax().item())
        prefix.append(token); generated.append(token)
        if token == model.config.eos_id:
            break
    _sync(device); decode_elapsed = time.perf_counter() - start
    result.update(decode_tokens=len(generated), decode_wall_seconds=decode_elapsed,
                  decode_tokens_per_second=len(generated) / decode_elapsed if generated else None)
    try:
        text = tokenizer.decode(generated)
        result["effective_chars_per_second"] = len(text) / decode_elapsed if generated else None
        result["decoded_characters"] = len(text)
        result["effective_chars_per_second_reason"] = None if generated else "decode benchmark disabled"
    except (ValueError, RuntimeError) as exc:
        result["effective_chars_per_second"] = None
        result["effective_chars_per_second_reason"] = f"generated token sequence is not reversible: {type(exc).__name__}"
    return result


def _run_config(config: dict, cfg: LMConfig) -> dict:
    run = {"seed": 17, "device": "cpu", "stage": "smoke", "regime": "same_total_parameters",
           "sequence_length": cfg.context_length, "batch_size": 1, "threads": 2,
           "learning_rate": 3e-4, "warmup_steps": 0, "weight_decay": 0.1,
           "gradient_clip": 1.0, "checkpoint_interval": 100, "verify_resume": True,
           "resume": False, "decode_steps": 2, "curve_fractions": [],
           "deterministic_algorithms": True, **config}
    run["regime"] = REGIME_ALIASES.get(run["regime"], run["regime"])
    if run["regime"] not in REGIMES:
        raise ValueError("unknown experiment regime")
    smoke = str(run["stage"]).lower() in {"smoke", "stage0", "stage_0", "0"} or run.get("run_kind") == "smoke"
    run["is_smoke"] = smoke
    if not 2 <= int(run["sequence_length"]) <= cfg.context_length:
        raise ValueError("sequence length must be within model context")
    if not smoke and int(run["sequence_length"]) != cfg.context_length:
        raise ValueError("formal runs must use the configured full model context")
    if smoke:
        if int(run.get("max_steps", 0)) < 1:
            raise ValueError("smoke requires explicit positive max_steps")
    else:
        key = {"same_total_parameters": "token_budget", "same_token_budget": "token_budget",
               "same_source_characters": "source_char_budget", "same_compute": "compute_budget_flops"}[run["regime"]]
        if float(run.get(key, 0)) <= 0:
            raise ValueError(f"formal {run['regime']} requires explicit {key}")
        if run["regime"] == "same_compute":
            raise ValueError("formal same_compute requires a measured FLOPs/GPU compute backend; this runtime only derives FLOPs estimates")
    if int(run.get("total_steps", 0)) < 1:
        if smoke:
            run["total_steps"] = int(run["max_steps"]) + 1
        else:
            raise ValueError("formal runs require an explicit fixed scheduler total_steps")
    for key in ("batch_size", "threads", "checkpoint_interval", "total_steps"):
        if int(run[key]) < 1:
            raise ValueError(f"{key} must be positive")
    if int(run["warmup_steps"]) < 0 or int(run["warmup_steps"]) >= int(run["total_steps"]):
        raise ValueError("warmup_steps must be within scheduler horizon")
    for key in ("learning_rate", "gradient_clip"):
        if not math.isfinite(float(run[key])) or float(run[key]) <= 0:
            raise ValueError(f"{key} must be positive finite")
    for key in ("token_budget", "source_char_budget", "compute_budget_flops"):
        if run.get(key) is not None and (not math.isfinite(float(run[key])) or float(run[key]) <= 0):
            raise ValueError(f"{key} must be positive finite")
    fractions = run["curve_fractions"]
    if (not isinstance(fractions, list) or any(not isinstance(f, (int, float)) or not math.isfinite(f) or not 0 <= f <= 1 for f in fractions)
            or fractions != sorted(set(fractions))):
        raise ValueError("curve_fractions must be unique ascending fractions in [0,1]")
    if run.get("scheduler_axis", "steps") not in {"steps", "tokens", "source_chars"}:
        raise ValueError("unsupported scheduler axis")
    if run.get("scheduler_axis") == "tokens":
        if run.get("token_budget") is None or not 0 <= int(run.get("warmup_tokens", -1)) < int(run["token_budget"]):
            raise ValueError("token scheduler requires token_budget and valid warmup_tokens")
    if run.get("scheduler_axis") == "source_chars":
        if run["regime"] != "same_source_characters" or not 0 <= int(run.get("warmup_source_chars", -1)) < int(run["source_char_budget"]):
            raise ValueError("source scheduler requires same-source budget and valid warmup")
    return run


def run_probe(*, candidate: str, tokenizer_path: str | Path,
              train_records: list[dict], eval_records: list[dict], model_config: LMConfig,
              run_config: dict, output_dir: Path, dataset_hash: str) -> dict:
    """Execute a run, or resume an interrupted matching run from its checkpoint.

    ``max_steps`` is an operational stop: a main run stopped before its declared
    budget stays incomplete and can be resumed with a larger max_steps. The
    scheduler horizon, data, tokenizer and all scientific controls stay fixed.
    """
    invocation_started = time.perf_counter()
    cfg = model_config; run = _run_config(run_config, cfg)
    if len(dataset_hash) != 64 or any(c not in "0123456789abcdef" for c in dataset_hash):
        raise ValueError("dataset_hash must be a lowercase SHA-256 digest")
    contamination = _validate_rows(train_records, eval_records)
    path = Path(tokenizer_path)
    if path.is_dir(): path = path / "tokenizer.model"
    tokenizer_hash = _file_hash(path)
    original_records_hash = _json_hash({"train": train_records, "eval": eval_records})
    if run["regime"] == "same_source_characters":
        if run.get("source_char_budget") is None:
            if not run["is_smoke"]: raise ValueError("source budget required")
        else:
            train_records = _source_prefix(train_records, int(run["source_char_budget"]))
    tokenizer = ReversibleSentencePieceAdapter(path)
    validate_tokenizer_contract(tokenizer, cfg)
    train, train_gate = _prepare(tokenizer, train_records)
    evaluation, eval_gate = _prepare(tokenizer, eval_records)
    controls = {k: v for k, v in run.items() if k not in {"resume", "max_steps", "checkpoint_interval", "deadline_unix"}}
    code_hashes = {name: _file_hash(Path(__file__).with_name(name)) for name in
                   ("phase55_same_source_runtime.py", "probe_lm.py", "probe_lm_data.py", "tokenizer_adapters.py")}
    identity_record = {"candidate": candidate, "model_config": cfg.to_dict(), "controls": controls,
                       "tokenizer_sha256": tokenizer_hash, "dataset_sha256": dataset_hash,
                       "records_sha256": original_records_hash, "code_sha256": code_hashes}
    identity = _json_hash(identity_record)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    config_path, checkpoint = out / "config.json", out / "checkpoint.pt"
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if existing["identity"] != identity:
            raise ValueError("refusing to overwrite a run with different configuration/data/tokenizer/code")
        if (out / "metrics.json").exists():
            metrics = json.loads((out / "metrics.json").read_text())
            if metrics.get("complete"):
                if run["resume"]:
                    if _file_hash(checkpoint) != metrics["checkpoint_sha256"]:
                        raise ValueError("completed checkpoint hash mismatch")
                    return metrics
                raise FileExistsError("completed output exists; refusing overwrite")
        if not run["resume"]:
            raise FileExistsError("run exists; explicit resume required")
    elif any(out.iterdir()):
        raise FileExistsError("output directory contains unrecognized existing artifacts")
    else:
        _atomic_json(config_path, {**identity_record, "identity": identity, "initial_run_config": run})
        (out / "tokenizer.sha256").write_text(tokenizer_hash + "\n")
        (out / "dataset.sha256").write_text(dataset_hash + "\n")
    device = torch.device(run["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("requested MPS is unavailable")
    if device.type == "cuda" and run["deterministic_algorithms"]:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(bool(run["deterministic_algorithms"]))
    set_seed(int(run["seed"]), threads=int(run["threads"]))
    if device.type == "cuda":
        # Standalone screening starts before any tensor has initialized CUDA.
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    load_start = time.perf_counter()
    model = ProbeLM(cfg).to(device)
    if model.parameter_report()["total"] != parameter_count(cfg):
        raise AssertionError("analytic and instantiated parameter counts disagree")
    _sync(device); load_seconds = time.perf_counter() - load_start
    progress = {"steps": 0, "cursor": 0, "block_token_offset": 0, "epochs_completed": 0, "tokens": 0,
                "source_chars": 0, "source_bytes": 0, "derived_flops": 0,
                "train_wall_seconds": 0.0, "loss_weighted_sum": 0.0, "documents": [],
                "last_source_boundary_complete": True, "sampled_mps_allocated_bytes": 0,
                "learning_curves": []}
    optimizer = _optimizer(model, run)
    resumed = False
    if run["resume"] and checkpoint.exists():
        del optimizer; gc.collect()
        optimizer, progress = _load(checkpoint, model, run, identity, device); resumed = True
    blocks = _blocks(train, cfg, int(run["sequence_length"]))
    if not blocks: raise ValueError("empty training stream")
    token_budget = int(run["token_budget"]) if run.get("token_budget") else None
    source_regime = run["regime"] == "same_source_characters"
    max_steps = int(run.get("max_steps", 0)) or None
    def curve_point():
        if not run["curve_fractions"]:
            return
        if token_budget is not None:
            fraction = progress["tokens"] / token_budget
            axis = "tokens"
        elif source_regime and run.get("source_char_budget"):
            fraction = progress["source_chars"] / int(run["source_char_budget"])
            axis = "source_chars"
        elif run["is_smoke"]:
            fraction = progress["steps"] / int(run["max_steps"])
            axis = "smoke_steps"
        else:
            raise ValueError("learning curve budget axis unavailable")
        captured = {r["requested_budget_fraction"] for r in progress["learning_curves"]}
        due = [f for f in run["curve_fractions"] if f not in captured and fraction + 1e-12 >= f]
        if not due:
            return
        curve_eval = evaluate(model, evaluation, cfg, device, int(run["sequence_length"]))
        for requested in due:
            progress["learning_curves"].append({"requested_budget_fraction": requested,
                "actual_budget_fraction": fraction, "budget_axis": axis, "steps": progress["steps"],
                "tokens": progress["tokens"], "source_chars": progress["source_chars"],
                "source_bytes": progress["source_bytes"], "derived_flops": progress["derived_flops"],
                "training_hours": progress["train_wall_seconds"] / 3600,
                "compute_provenance": "derived_flops; synchronized training wall time",
                "validation": curve_eval})
        _atomic_json(out / "learning_curves.json", {"identity": identity, "points": progress["learning_curves"]})
        _save(checkpoint, model, optimizer, progress, identity, run, device)
        if run.get("retain_curve_checkpoints", False):
            for requested in due:
                retained = out / f"checkpoint-fraction-{requested:.6f}.pt"
                if not retained.exists():
                    os.link(checkpoint, retained)
                    retained.with_suffix(".sha256").write_text(_file_hash(retained) + "\n")
    curve_point()
    stop_reason = None
    while True:
        if run.get("deadline_unix") and time.time() >= run["deadline_unix"]:
            stop_reason = "operational_deadline"; break
        if token_budget is not None and progress["tokens"] >= token_budget:
            stop_reason = "token_budget_reached"; break
        if source_regime and progress["epochs_completed"] >= 1:
            stop_reason = "source_budget_reached"; break
        if max_steps is not None and progress["steps"] >= max_steps:
            stop_reason = "smoke_steps_reached" if run["is_smoke"] else "operational_step_limit"; break
        if progress["steps"] >= int(run["total_steps"]):
            stop_reason = "scheduler_horizon_exhausted"; break
        selected = _stream_selection(blocks, progress, int(run["batch_size"]))
        remaining = token_budget - progress["tokens"] if token_budget is not None else None
        batch, account = _batch(selected, cfg, device, remaining)
        flops = derived_training_flops(cfg, account["batch_size"], account["sequence_length"])
        if run["regime"] == "same_compute" and run.get("compute_budget_flops") and progress["derived_flops"] + flops > run["compute_budget_flops"]:
            stop_reason = "derived_compute_limit"; break
        _sync(device); start = time.perf_counter()
        run["_scheduler_position"] = (progress["source_chars"] + account["source_chars"]) if run.get("scheduler_axis") == "source_chars" else (progress["tokens"] + account["tokens"])
        loss = _step(model, optimizer, batch, run, progress["steps"])
        _sync(device); elapsed = time.perf_counter() - start
        progress["steps"] += 1
        old_offset = progress["block_token_offset"]
        progress["cursor"] += account["completed_blocks"]
        progress["block_token_offset"] = (account["partial_block_tokens"] +
                                           (old_offset if account["completed_blocks"] == 0 else 0)) if account["partial_block_tokens"] else 0
        if progress["cursor"] >= len(blocks):
            progress["epochs_completed"] += 1; progress["cursor"] = 0
        for key in ("tokens", "source_chars", "source_bytes"):
            progress[key] += account[key]
        progress["derived_flops"] += flops
        progress["train_wall_seconds"] += elapsed
        progress["loss_weighted_sum"] += loss * account["tokens"]
        progress["last_train_loss"] = loss
        progress["documents"] = sorted(set(progress["documents"]) | set(account["documents"]))
        progress["last_source_boundary_complete"] = account["source_boundary_complete"]
        if device.type == "mps":
            progress["sampled_mps_allocated_bytes"] = max(progress["sampled_mps_allocated_bytes"], int(torch.mps.current_allocated_memory()))
        if progress["steps"] % 10 == 0:
            _atomic_json(out / "progress.json", {k: v for k, v in progress.items() if k not in {"documents", "learning_curves"}})
        curve_point()
        if progress["steps"] % int(run["checkpoint_interval"]) == 0:
            _save(checkpoint, model, optimizer, progress, identity, run, device)
    if not progress["steps"]:
        raise ValueError("budget allowed no optimizer steps")
    _save(checkpoint, model, optimizer, progress, identity, run, device)
    checkpoint_hash = _file_hash(checkpoint)
    (out / "checkpoint.sha256").write_text(checkpoint_hash + "\n")
    check_batch, check_account = _batch(_stream_selection(blocks, progress, 1), cfg, device)
    model.eval()
    with torch.no_grad(): before_loss = float(model(*check_batch)[1].item())
    before_weights = _state_digest(model.state_dict())
    expected_weights = expected_optimizer = None
    run["_scheduler_position"] = (progress["source_chars"] + check_account["source_chars"]) if run.get("scheduler_axis") == "source_chars" else (progress["tokens"] + int(check_batch[2].sum().item()))
    if run["verify_resume"]:
        expected_loss = _step(model, optimizer, check_batch, run, progress["steps"])
        expected_weights = _state_digest(model.state_dict())
        expected_optimizer = _state_digest(optimizer.state_dict())
    del optimizer; gc.collect()
    optimizer, restored_progress = _load(checkpoint, model, run, identity, device)
    model.eval()
    with torch.no_grad(): after_loss = float(model(*check_batch)[1].item())
    reload_ok = before_weights == _state_digest(model.state_dict()) and math.isclose(before_loss, after_loss, rel_tol=1e-6, abs_tol=1e-7)
    if not reload_ok: raise RuntimeError("checkpoint reload mismatch")
    resume_ok = None
    if run["verify_resume"]:
        resumed_loss = _step(model, optimizer, check_batch, run, progress["steps"])
        resume_ok = (math.isclose(expected_loss, resumed_loss, rel_tol=1e-6, abs_tol=1e-7)
                     and expected_weights == _state_digest(model.state_dict())
                     and expected_optimizer == _state_digest(optimizer.state_dict()))
        if not resume_ok: raise RuntimeError("optimizer resume continuation differs from uninterrupted continuation")
        del optimizer; gc.collect()
        optimizer, restored_progress = _load(checkpoint, model, run, identity, device)
    if restored_progress != progress: raise RuntimeError("checkpoint cursor/counters mismatch")
    tokenizer_reloaded = ReversibleSentencePieceAdapter(path)
    if _file_hash(path) != tokenizer_hash:
        raise RuntimeError("tokenizer artifact changed during run")
    for row in train + evaluation:
        if tokenizer_reloaded.encode(row["text"]) != row["ids"] or tokenizer_reloaded.decode(row["ids"]) != row["text"]:
            raise RuntimeError("tokenizer reload behavior mismatch")
    validation = evaluate(model, evaluation, cfg, device, int(run["sequence_length"]))
    inference = _inference(model, tokenizer_reloaded, evaluation, device, int(run["sequence_length"]), int(run["decode_steps"]))
    if code_hashes != {name: _file_hash(Path(__file__).with_name(name)) for name in code_hashes}:
        raise RuntimeError("runtime code changed during run")
    wall = progress["train_wall_seconds"]
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if platform.system() == "Darwin" else 1024)
    complete = stop_reason in {"token_budget_reached", "source_budget_reached", "smoke_steps_reached"}
    if complete and source_regime and (progress["source_chars"] != int(run["source_char_budget"]) or progress["source_bytes"] != sum(len(r["text"].encode()) for r in train_records)):
        raise AssertionError("same-source completion accounting mismatch")
    peak_vram = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    metrics = {"format": "novllm-phase55-run-v1", "identity": identity, "candidate": candidate,
               "stage": run["stage"], "regime": run["regime"], "seed": run["seed"],
               "device": str(device), "complete": complete,
               "main_comparison_eligible": complete and not run["is_smoke"] and resume_ok is True,
               "stop_reason": stop_reason, "resumed_from_checkpoint": resumed,
               "model_config": cfg.to_dict(), "parameter_report": model.parameter_report(),
               "tokenizer_sha256": tokenizer_hash, "dataset_sha256": dataset_hash,
               "records_sha256": original_records_hash, "checkpoint_sha256": checkpoint_hash,
               "code_sha256": code_hashes,
               "training": {**progress, "unique_documents": len(progress["documents"]),
                            "loss_nats_per_content_token": progress["loss_weighted_sum"] / progress["tokens"],
                            "tokens_per_second": progress["tokens"] / wall,
                            "chars_per_second": progress["source_chars"] / wall,
                            "bytes_per_second": progress["source_bytes"] / wall,
                            "source_accounting": "original UTF-8 source prefixes; partial escape/byte atoms counted at completion",
                            "compute_provenance": "derived", "measured_flops": None,
                            "measured_flops_reason": "no hardware/profiler FLOPs measurement backend",
                            "derived_flops_formula": "6 * total_parameters * padded_positions + 12 * layers * hidden * batch * sequence_length^2; optimizer excluded",
                            "timing_scope": "synchronized forward/backward/optimizer only; excludes tokenization, tensor staging, checkpoints and verification",
                            "epochs_fraction": progress["tokens"] / sum(len(row["ids"]) for row in train)},
               "validation": validation, "inference": inference,
               "learning_curves": progress["learning_curves"],
               "system": {"peak_vram_bytes": peak_vram,
                          "invocation_wall_seconds": time.perf_counter() - invocation_started,
                          "peak_vram_bytes_reason": None if device.type == "cuda" else "CUDA allocator unavailable; MPS sampled allocation is separately reported",
                          "peak_vram_scope": "entire run, including resume verification/evaluation/inference" if device.type == "cuda" else None,
                          "cpu_peak_rss_bytes": rss, "cpu_peak_rss_scope": "process lifetime high watermark; not incremental run memory",
                          "gpu_utilization_percent": None, "gpu_utilization_percent_reason": "no device utilization sampler",
                          "joules": None, "joules_reason": "no energy meter", "kwh": None, "kwh_reason": "no energy meter",
                          "model_load_seconds": load_seconds, "model_load_memory_bytes": None,
                          "model_load_memory_bytes_reason": "isolated allocator/RSS delta not measured",
                          "torch_version": str(torch.__version__), "python_version": platform.python_version()},
               "checks": {"finite_loss_backward_optimizer": True, "checkpoint_reload": reload_ok,
                          "optimizer_resume_continuation": resume_ok,
                          "optimizer_resume_continuation_reason": None if run["verify_resume"] else "verification disabled",
                          "checkpoint_hash": _file_hash(checkpoint) == checkpoint_hash,
                          "tokenizer_reload": True, "tokenizer_hash": True,
                          "dataset_lineage_preserved": True, "contamination": contamination,
                          "train_preservation": train_gate, "eval_preservation": eval_gate,
                          "formal_context_training": int(run["sequence_length"]) == cfg.context_length,
                          "no_unexplained_nan_inf": True},
               "limitations": ["smoke is not evidence for candidate ranking" if run["is_smoke"] else "candidate ranking requires the full controlled cohort",
                               "FLOPs are derived estimates, not measured compute",
                               "no KV cache; decode uses full context recomputation",
                               "dataset hash is supplied lineage identity; supplied row content is separately hashed",
                               "checkpoint contains model/optimizer/RNG/cursor/scheduler; verification continuation is excluded from training counters"]}
    _atomic_json(out / "metrics.json", metrics)
    return metrics
