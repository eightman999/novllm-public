# common.py
"""デバイス/dtype自動選択の共通ヘルパー (CUDA > MPS(Mac) > CPU)。"""
import torch


def pick_device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str, prefer: str | None = None) -> torch.dtype:
    if prefer:
        return getattr(torch, prefer)
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        # MPSはbf16のカーネル対応が不完全な場合があるためfp16を既定に
        return torch.float16
    return torch.float32
