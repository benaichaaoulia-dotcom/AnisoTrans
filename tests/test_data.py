import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from anisotrans_ccs.data import fit_statistics, knn_candidates, load_raw, norm_stats_from_config, verify_sha256

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "v2.yaml"
DATA_DIR = os.environ.get("ANISOTRANS_DATA_DIR")


def test_verify_sha256_detects_mismatch(tmp_path):
    f = tmp_path / "blob.bin"
    f.write_bytes(b"abc")
    verify_sha256(f, hashlib.sha256(b"abc").hexdigest())
    with pytest.raises(ValueError):
        verify_sha256(f, "0" * 64)


def test_knn_candidates_put_self_first_and_are_valid():
    utm = torch.rand(40, 3, generator=torch.Generator().manual_seed(0))
    cand, valid = knn_candidates(utm, 5, torch.device("cpu"))
    assert cand.shape == (40, 5)
    assert torch.equal(cand[:, 0], torch.arange(40))
    assert valid.all()


@pytest.mark.skipif(not DATA_DIR, reason="ANISOTRANS_DATA_DIR not set")
def test_fit_statistics_reproduce_config_constants():
    cfg = yaml.safe_load(CONFIG.read_text())["data"]
    raw = load_raw(Path(DATA_DIR), cfg)
    assert len(raw.fit_idx) == cfg["n_fit"]
    stats = fit_statistics(raw)
    ref = norm_stats_from_config(cfg)
    np.testing.assert_allclose(stats.K_mean, ref.K_mean, rtol=1e-8)
    np.testing.assert_allclose(stats.K_std, ref.K_std, rtol=1e-8)
    np.testing.assert_allclose(stats.theta_mean, ref.theta_mean, rtol=1e-8)
    np.testing.assert_allclose(stats.theta_std, ref.theta_std, rtol=1e-8)
