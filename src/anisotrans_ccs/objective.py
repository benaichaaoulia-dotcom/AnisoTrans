"""Physics anchor term of the AnisoTrans v2 objective."""

from __future__ import annotations

import torch


def anchor_loss(s: torch.Tensor, s_target: torch.Tensor) -> torch.Tensor:
    """Mean over configurations of the squared distance between ``s(theta)`` and ``s_phys(theta)``.

    Both tensors have shape (B, 3).
    """
    if s.shape != s_target.shape or s.dim() != 2 or s.shape[1] != 3:
        raise ValueError(f"expected matching (B, 3) tensors, got {tuple(s.shape)} and {tuple(s_target.shape)}")
    return torch.sum((s - s_target) ** 2, dim=1).mean()
