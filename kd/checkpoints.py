"""Versioned artifacts with explicit label/preprocessing compatibility."""

import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from .data import normalization


def write_json(path: Path, value: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def save_checkpoint(path: Path, state: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def fingerprint(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_state_fingerprint(state: dict) -> str:
    """Stable identity of tensor values, independent of checkpoint serialization."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}|{value.dtype}|{tuple(value.shape)}|".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def metadata(model_name: str, classes: list[str], image_size: int, source: str = "imagefolder") -> dict:
    mean, std = normalization(source)
    return {"format_version": 1, "model_name": model_name, "classes": classes,
            "preprocessing": {"image_size": image_size, "mean": list(mean), "std": list(std)}}


def load_model_checkpoint(model, path: str | Path, expected: dict) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"Incompatible checkpoint {path}: {key} does not match ({state.get(key)!r} != {value!r})")
    model.load_state_dict(state["student"], strict=True)
    return state


def capture_rng(loaders: dict) -> dict:
    np_state = np.random.get_state()
    state = {"python": random.getstate(), "numpy": [np_state[0], np_state[1].tolist(), *np_state[2:]],
             "torch": torch.get_rng_state(),
             "loaders": {name: loader.generator.get_state() for name, loader in loaders.items()}}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state: dict, loaders: dict) -> None:
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), *ns[2:]))
    torch.set_rng_state(state["torch"])
    for name, value in state["loaders"].items():
        loaders[name].generator.set_state(value)
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])
