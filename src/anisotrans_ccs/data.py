"""Data loading for AnisoTrans v2: verified inputs, fit-split normalisation and k-NN neighbourhoods."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

N_CELLS = 13334
N_CONFIGS = 378


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing data file: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"sha256 mismatch for {path}: expected {expected}, got {actual}")


@dataclass(frozen=True)
class NormStats:
    theta_mean: np.ndarray
    theta_std: np.ndarray
    K_mean: np.ndarray
    K_std: np.ndarray


@dataclass(frozen=True)
class RawCache:
    theta: np.ndarray  # (378, 8) raw configuration vectors
    Kfield: np.ndarray  # (13334, 3) raw permeability
    YA: np.ndarray  # (378, 13334) raw pressure
    mask: np.ndarray  # (378, 13334) bool, valid cells
    fit_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    ystd_fit: float


@dataclass(frozen=True)
class Bundle:
    theta: np.ndarray  # (378, 8) float32, normalised
    K: np.ndarray  # (13334, 3) float32, normalised
    y: np.ndarray  # (378, 13334) float32, pressure / ystd_fit
    mask: np.ndarray  # (378, 13334) bool
    fit_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    ystd_fit: float
    sim_years: np.ndarray  # (378,)
    n_fit_pairs: int
    utm_norm: torch.Tensor  # (13334, 3) float32
    cand_idx: torch.Tensor  # (13334, k) long
    cand_valid: torch.Tensor  # (13334, k) bool


def norm_stats_from_config(d: Mapping) -> NormStats:
    return NormStats(
        theta_mean=np.asarray(d["theta_mean"], dtype=np.float64),
        theta_std=np.asarray(d["theta_std"], dtype=np.float64),
        K_mean=np.asarray(d["K_mean"], dtype=np.float64),
        K_std=np.asarray(d["K_std"], dtype=np.float64),
    )


def load_raw(data_dir: Path, d: Mapping) -> RawCache:
    """Load and sha256-verify the per-cell cache and the permeability field."""
    cache_path = data_dir / d["cache_file"]
    kfield_path = data_dir / d["kfield_file"]
    verify_sha256(cache_path, d["cache_sha256"])
    verify_sha256(kfield_path, d["kfield_sha256"])

    with np.load(cache_path, allow_pickle=True) as c:
        X = np.asarray(c["X"], dtype=np.float64)
        X_derived = np.asarray(c["X_derived"], dtype=np.float64)
        kfield_cache = np.asarray(c["Kfield"], dtype=np.float64)
        YA = np.asarray(c["YA_grid"], dtype=np.float64)
        mask = np.asarray(c["M"]).astype(bool)
        fit_idx = np.asarray(c["fit_idx"], dtype=np.int64)
        val_idx = np.asarray(c["val_idx"], dtype=np.int64)
        test_idx = np.asarray(c["test_idx"], dtype=np.int64)
        ystd_fit = float(np.asarray(c["ystd_fit"]).item())

    kfield = np.asarray(np.load(kfield_path, allow_pickle=True), dtype=np.float64)
    if not np.array_equal(kfield, kfield_cache):
        raise ValueError("Kfield in the cache does not match the Kfield file")

    theta = np.hstack([X, X_derived])
    if theta.shape != (N_CONFIGS, 8):
        raise ValueError(f"theta has shape {theta.shape}, expected {(N_CONFIGS, 8)}")
    if kfield.shape != (N_CELLS, 3):
        raise ValueError(f"Kfield has shape {kfield.shape}, expected {(N_CELLS, 3)}")
    if YA.shape != (N_CONFIGS, N_CELLS) or mask.shape != YA.shape:
        raise ValueError("YA_grid / M have unexpected shapes")

    return RawCache(theta, kfield, YA, mask, fit_idx, val_idx, test_idx, ystd_fit)


def fit_statistics(raw: RawCache) -> NormStats:
    """Recompute normalisation statistics from the fit split (valid cells only for K and pressure)."""
    fit = raw.fit_idx
    fit_mask = raw.mask[fit]
    K_fit = np.broadcast_to(raw.Kfield, (len(fit), N_CELLS, 3))[fit_mask]
    return NormStats(
        theta_mean=raw.theta[fit].mean(axis=0),
        theta_std=raw.theta[fit].std(axis=0),
        K_mean=K_fit.mean(axis=0),
        K_std=K_fit.std(axis=0),
    )


def load_grid_coords(data_dir: Path, d: Mapping) -> np.ndarray:
    path = data_dir / d["grid_coords_file"]
    verify_sha256(path, d["grid_coords_sha256"])
    coords = np.asarray(np.load(path, allow_pickle=True), dtype=np.float64)
    if coords.shape != (N_CELLS, 3) or not np.isfinite(coords).all():
        raise ValueError(f"grid coordinates have shape {coords.shape}, expected {(N_CELLS, 3)}")
    return coords


def knn_candidates(utm_norm: torch.Tensor, k: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the k nearest physical neighbours per cell (slot 0 is the cell itself).

    Ties on the regular lattice may be broken differently on different devices.
    """
    n = utm_norm.shape[0]
    u = utm_norm.to(device).float()
    dist = torch.cdist(u, u)
    idx = torch.arange(n, device=device)
    dist[idx, idx] = float("inf")
    _, nbr = torch.topk(dist, k - 1, dim=1, largest=False)
    cand = torch.empty(n, k, dtype=torch.long, device=device)
    cand[:, 0] = idx
    cand[:, 1:] = nbr
    valid = torch.ones(n, k, dtype=torch.bool, device=device)
    return cand.cpu(), valid.cpu()


def load_bundle(data_cfg: Mapping, data_dir: Path, knn_k: int, device: torch.device) -> Bundle:
    raw = load_raw(data_dir, data_cfg)
    if abs(raw.ystd_fit - float(data_cfg["ystd_fit"])) > 1e-9:
        raise ValueError(f"ystd_fit {raw.ystd_fit} differs from config {data_cfg['ystd_fit']}")
    if len(raw.fit_idx) != int(data_cfg["n_fit"]) or len(raw.val_idx) != int(data_cfg["n_val"]):
        raise ValueError("split sizes differ from config")
    if len(raw.test_idx) != int(data_cfg["n_test"]):
        raise ValueError("test split size differs from config")

    stats = norm_stats_from_config(data_cfg)
    theta = ((raw.theta - stats.theta_mean) / stats.theta_std).astype(np.float32)
    K = ((raw.Kfield - stats.K_mean) / stats.K_std).astype(np.float32)
    y = (raw.YA / raw.ystd_fit).astype(np.float32)

    coords = load_grid_coords(data_dir, data_cfg)
    utm_norm = torch.from_numpy(((coords - coords.mean(0)) / coords.std(0)).astype(np.float32))
    cand_idx, cand_valid = knn_candidates(utm_norm, knn_k, device)

    return Bundle(
        theta=theta,
        K=K,
        y=y,
        mask=raw.mask,
        fit_idx=raw.fit_idx,
        val_idx=raw.val_idx,
        test_idx=raw.test_idx,
        ystd_fit=raw.ystd_fit,
        sim_years=raw.theta[:, 4].copy(),
        n_fit_pairs=int(raw.mask[raw.fit_idx].sum()),
        utm_norm=utm_norm,
        cand_idx=cand_idx,
        cand_valid=cand_valid,
    )
