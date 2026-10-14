from pathlib import Path

import torch
import yaml

from anisotrans_ccs.model import ConfigScaler, build_model, count_params, s_phys

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "v2.yaml"


def _load_cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def _graph(n: int, k: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    utm = torch.rand(n, 3, generator=g)
    idx = torch.randint(0, n, (n, k), generator=g)
    idx[:, 0] = torch.arange(n)
    valid = torch.ones(n, k, dtype=torch.bool)
    return utm, idx, valid


def test_config_scaler_has_419_parameters():
    cfg = _load_cfg()["model"]["scaler"]
    scaler = ConfigScaler(cfg["physics_init"], cfg["exp_clip"])
    assert count_params(scaler) == 419


def test_config_scaler_equals_physics_target_at_init():
    cfg = _load_cfg()["model"]["scaler"]
    scaler = ConfigScaler(cfg["physics_init"], cfg["exp_clip"])
    theta = 0.5 * torch.randn(16, 8, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        s = scaler(theta)
    target = s_phys(theta, cfg["physics_init"], cfg["exp_clip"])
    torch.testing.assert_close(s, target, rtol=1e-5, atol=1e-6)


def test_v2_parameter_count_matches_reference():
    cfg = _load_cfg()["model"]
    utm, idx, valid = _graph(64, cfg["knn_k"])
    model = build_model(cfg, utm, idx, valid)
    assert count_params(model) == 6_364_709


def test_forward_eval_shape_and_finite():
    cfg = _load_cfg()["model"]
    utm, idx, valid = _graph(64, cfg["knn_k"])
    model = build_model(cfg, utm, idx, valid).eval()
    theta = torch.randn(8)
    K = torch.rand(64, 3) + 0.1
    with torch.no_grad():
        y = model(theta, K)
    assert y.shape == (64,)
    assert torch.isfinite(y).all()


def test_forward_train_backward_gradients_finite():
    cfg = _load_cfg()["model"]
    utm, idx, valid = _graph(64, cfg["knn_k"])
    model = build_model(cfg, utm, idx, valid).train()
    theta = torch.randn(8)
    K = torch.rand(64, 3) + 0.1
    model(theta, K).pow(2).mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert model.alpha_raw.grad is not None
