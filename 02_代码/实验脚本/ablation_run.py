"""Train same-backbone modality ablations without editing the frozen v5 sources.

Inactive modalities are zeroed at the model input, including their observation
masks. The BERT tower, active branches, optimizer, training views, and validation
selection remain the v5 implementation. This wrapper must also be used for
evaluation of ablation checkpoints.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import v5_model
import v5_train

_original_forward = v5_model.V5Model.forward


def _forward_with_modality_ablation(self, batch):
    active = self.model_config.get("active_modalities")
    if active is None:
        return _original_forward(self, batch)
    active = set(active)
    if not active or not active.issubset({"text", "audio", "vision"}) or "text" not in active:
        raise ValueError("active_modalities must include text and contain only known modalities")
    current = dict(batch)
    for name in ("audio", "vision"):
        if name not in active:
            current[name] = torch.zeros_like(batch[name])
            current[name + "_mask"] = torch.zeros_like(batch[name + "_mask"])
    return _original_forward(self, current)


v5_model.V5Model.forward = _forward_with_modality_ablation


if __name__ == "__main__":
    raise SystemExit(v5_train.main())
