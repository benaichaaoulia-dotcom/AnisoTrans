import pytest
import torch

from anisotrans_ccs.objective import anchor_loss


def test_anchor_zero_when_scaler_matches_target():
    s = torch.rand(10, 3) + 0.1
    assert anchor_loss(s, s.clone()).item() == 0.0


def test_anchor_is_mean_of_squared_distance_per_config():
    s = torch.zeros(2, 3)
    target = torch.tensor([[1.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
    # per-config squared norms: 1 + 4 = 5 and 0, mean = 2.5
    assert anchor_loss(s, target).item() == pytest.approx(2.5)


def test_anchor_rejects_bad_shapes():
    with pytest.raises(ValueError):
        anchor_loss(torch.zeros(2, 3), torch.zeros(2, 2))
