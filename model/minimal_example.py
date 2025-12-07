"""
Minimal demonstration to instantiate an HTS-AT model and load a checkpoint.
Run this file from ./ (not from the repository's root).
Files from this directory can be copied into another repository (no dependencies outside this directory).

This script shows the smallest practical example of:
  1) Constructing the `HTSAT_Swin_Transformer` model using settings from `config.py`.
  2) Loading a checkpoint file saved from this repository's training pipeline.

Notes on checkpoints:
- Training scripts in this repo often save checkpoints with keys prefixed by
  "sed_model.". We strip that prefix automatically if present.
- The checkpoint is expected to be a PyTorch file containing a dict with a
  "state_dict" entry (the format used in this repo). If you have a raw
  `state_dict` (a plain mapping of parameter names to tensors), this script
  will try to load it as well.

Usage:
  python minimal_example.py --checkpoint ./HTSAT_AudioSet_Saved_6.ckpt [--verify-forward]

If --verify-forward is provided, a dummy 10-second waveform (zeros) will be
ran through the model to verify everything works end-to-end on CPU.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from types import SimpleNamespace
from typing import Dict, Any, Optional

import torch

from htsat import HTSAT_Swin_Transformer
import os
import yaml


def _build_model(cfg: SimpleNamespace) -> HTSAT_Swin_Transformer:
    """Construct the HTS-AT model using hyperparameters from a config-like object."""
    model = HTSAT_Swin_Transformer(
        spec_size=cfg.htsat_spec_size,
        patch_size=cfg.htsat_patch_size,
        in_chans=1,
        num_classes=cfg.classes_num,
        window_size=cfg.htsat_window_size,
        config=cfg,
        depths=cfg.htsat_depth,
        embed_dim=cfg.htsat_dim,
        patch_stride=tuple(cfg.htsat_stride),
        num_heads=cfg.htsat_num_head,
    )
    return model


def _load_config(config_yaml_path="minimal_config.yaml") -> SimpleNamespace:
    """
    Load configuration from YAML if provided/existing; otherwise fall back to config.py.
    Returns a SimpleNamespace with attribute access.
    """
    with open(os.path.join(os.path.dirname(__file__), config_yaml_path), "r") as f:
        data = yaml.safe_load(f)
    return data


def _normalize_state_dict(container: Any) -> Dict[str, torch.Tensor]:
    """
    Accept either a full checkpoint dict with a "state_dict" key or a raw
    state_dict mapping. Returns a cleaned state_dict ready for load_state_dict.
    Also strips an optional "sed_model." prefix used in this repo.
    """
    if isinstance(container, dict) and "state_dict" in container:
        state_dict = container["state_dict"]
    elif isinstance(container, dict):
        # Assume it's already a raw mapping of parameter_name -> tensor
        state_dict = container
    else:
        raise RuntimeError("Unsupported checkpoint format: expected dict or dict with 'state_dict'.")

    normalized: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        new_key = k.replace("sed_model.", "")
        normalized[new_key] = v
    return normalized


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Minimal HTS-AT checkpoint loader")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        help="Path to the HTS-AT checkpoint (.ckpt or .pth)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="minimal_config.yaml",
        help="Path to minimal YAML config (defaults to ./minimal_config.yaml if present, otherwise falls back to config.py)",
    )
    parser.add_argument(
        "--verify-forward",
        action="store_true",
        help="Run a dummy forward pass with zeros (CPU) to verify the model",
    )
    args = parser.parse_args(argv)

    # 0) Load configuration (YAML preferred)
    config_dict = _load_config(args.config)
    cfg = SimpleNamespace(**config_dict)  # To access fields

    # 1) Build model
    model = _build_model(cfg)

    if args.checkpoint is not None:
        # 2) Load checkpoint to CPU and normalize keys
        ckpt = torch.load(args.checkpoint, map_location=torch.device("cpu"))
        state_dict = _normalize_state_dict(ckpt)

        # 3) Load weights
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print("Loaded checkpoint:")
            print(f"  Missing keys: {len(missing)}")
            if missing:
                print("   - " + "\n   - ".join(missing))
            print(f"  Unexpected keys: {len(unexpected)}")
            if unexpected:
                print("   - " + "\n   - ".join(unexpected))
        except RuntimeError as e:
            warnings.warn(f"Failed to load checkpoint (Runtime Error):\n{e}")

    model.eval()
    print("Model is ready (eval mode).")

    # 4) Optional quick verification forward on a dummy waveform
    if args.verify_forward:
        with torch.no_grad():
            # Use config.sample_rate and config.clip_samples to size the dummy input
            # The model expects a 1D waveform per example; we add batch dim (1, N)
            dummy_waveform = torch.zeros((1, cfg.clip_samples), dtype=torch.float32)
            for infer_mode in [True, False]:
                print(f"---- Running dummy forward with {infer_mode=} mode -----")
                out = model(dummy_waveform, infer_mode=infer_mode)
                for k, v in out.items():
                    print(f"  out['{k}'] shape: {v.shape}")


if __name__ == "__main__":
    main(sys.argv[1:])
