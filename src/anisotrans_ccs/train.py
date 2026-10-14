"""AnisoTrans v2 training, checkpointing and frozen-test evaluation.

The training step reproduces the executed objective of the R6 run (``AnisoTrans-CCS/scripts/train_v2.py``):
masked MSE over valid fit cells plus the physics anchor, AdamW with a warmup-cosine schedule, fp16
autocast with GradScaler, early stopping on pooled validation relative RMSE, and one frozen-test
evaluation of the best checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
import yaml

from .data import Bundle, load_bundle
from .model import build_model, count_params, s_phys
from .objective import anchor_loss


def set_deterministic(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(spec: str) -> torch.device:
    device = torch.device(spec)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"device {spec} requested but CUDA is unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"device {spec} requested but only {torch.cuda.device_count()} GPU(s) found")
    return device


def lr_factor(epoch: int, t: Mapping) -> float:
    """Linear warmup followed by cosine decay, as a multiplier of the base learning rate."""
    lr, lr_min = float(t["lr"]), float(t["lr_min"])
    warm, total = int(t["warmup_epochs"]), int(t["max_epochs"])
    if warm > 0 and epoch < warm:
        return (epoch + 1) / warm
    progress = min(1.0, max(0.0, (epoch - warm) / max(1, total - warm)))
    floor = lr_min / lr
    return floor + 0.5 * (1.0 + math.cos(math.pi * progress)) * (1.0 - floor)


@torch.no_grad()
def evaluate_split(model: torch.nn.Module, bd: Bundle, device: torch.device, idx: np.ndarray, kind: str) -> dict:
    """Pooled and per-config relative RMSE on valid cells, in physical pressure units / ystd_fit."""
    was_training = model.training
    model.eval()
    K = torch.from_numpy(bd.K).to(device)
    rel = []
    for ci in idx:
        ci = int(ci)
        m = bd.mask[ci]
        y_hat = model(torch.from_numpy(bd.theta[ci]).to(device), K)
        pred = y_hat[torch.from_numpy(m).to(device)] * bd.ystd_fit
        true = torch.from_numpy(bd.y[ci][m] * bd.ystd_fit).to(device)
        rmse = torch.sqrt(torch.mean((pred - true) ** 2)).item()
        rel.append(rmse / bd.ystd_fit)
    model.train(was_training)

    rel_arr = np.asarray(rel, dtype=np.float64)
    years = bd.sim_years[idx]
    median = float(np.median(years))
    early, late = rel_arr[years <= median], rel_arr[years > median]
    ratio = float("nan")
    if early.size and late.size:
        ratio = float(np.sqrt(np.mean(late**2)) / np.sqrt(np.mean(early**2)))
    return {
        f"{kind}_pooled_rel_rmse": float(np.sqrt(np.mean(rel_arr**2))),
        f"{kind}_max_per_config_rel_rmse": float(rel_arr.max()),
        f"{kind}_horizon_ratio": ratio,
        f"{kind}_n_configs": int(rel_arr.size),
    }


class Trainer:
    def __init__(self, cfg: Mapping, bd: Bundle, device: torch.device, out_dir: Path):
        self.cfg = cfg
        self.bd = bd
        self.device = device
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = out_dir / "train_log.jsonl"
        self.ckpt_path = out_dir / "best.pt"

        t = cfg["training"]
        self.t = t
        self.seed = int(cfg["seed"])
        self.lambda_anchor = float(cfg["loss"]["lambda_anchor"])
        self.accum = int(t["accum"])
        self.fp16 = t["precision"] == "fp16" and device.type == "cuda"
        self.best = math.inf
        self.bad_epochs = 0
        self.epoch = 0

        set_deterministic(self.seed)
        self.model = build_model(cfg["model"], bd.utm_norm, bd.cand_idx, bd.cand_valid).to(device)
        self.pins = cfg["model"]["scaler"]["physics_init"]
        self.exp_clip = cfg["model"]["scaler"]["exp_clip"]
        self.K = torch.from_numpy(bd.K).to(device)

        # One dropout-enabled forward pass before training, as in the source run (preflight).
        with torch.no_grad():
            self.model(torch.from_numpy(bd.theta[int(bd.fit_idx[0])]).to(device), self.K)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=float(t["lr"]), weight_decay=float(t["weight_decay"])
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda e: lr_factor(e, t))
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.fp16)

    def _step(self, cfg_ids: Sequence[int]) -> tuple[float, int, float]:
        bd, dev = self.bd, self.device
        self.optimizer.zero_grad(set_to_none=True)

        theta_fit = torch.from_numpy(bd.theta[bd.fit_idx]).to(dev)
        s_fit = self.model.scaler(theta_fit)
        anchor = anchor_loss(s_fit, s_phys(theta_fit, self.pins, self.exp_clip))
        anchor = self.lambda_anchor * anchor / len(cfg_ids)

        mse_sum, n_pairs = 0.0, 0
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=self.fp16):
            for ci in cfg_ids:
                m_np = bd.mask[ci]
                m = torch.from_numpy(m_np).to(dev)
                y_hat = self.model(torch.from_numpy(bd.theta[ci]).to(dev), self.K)
                target = torch.from_numpy(bd.y[ci][m_np]).to(dev)
                err = y_hat[m] - target
                loss = torch.sum(err**2) / bd.n_fit_pairs
                mse_sum += float(torch.sum(err.detach() ** 2).item())
                n_pairs += int(m_np.sum())
                self.grad_scaler.scale(loss).backward()
            self.grad_scaler.scale(anchor).backward()

        self.grad_scaler.unscale_(self.optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.t["grad_clip"])))
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        return mse_sum, n_pairs, grad_norm

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        order = np.arange(len(self.bd.fit_idx))
        np.random.default_rng([self.seed, epoch]).shuffle(order)

        t0 = time.time()
        mse_sum, n_pairs, grad_norms, steps = 0.0, 0, [], 0
        i = 0
        while i < len(order):
            group = order[i : i + self.accum]
            cfg_ids = [int(self.bd.fit_idx[c]) for c in group]
            try:
                s_mse, s_n, gnorm = self._step(cfg_ids)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or self.accum <= 1:
                    raise
                torch.cuda.empty_cache()
                self.accum = max(1, self.accum // 2)
                print(f"OOM: accumulation reduced to {self.accum}", flush=True)
                continue
            mse_sum += s_mse
            n_pairs += s_n
            grad_norms.append(gnorm)
            steps += 1
            i += len(group)

        return {
            "epoch": epoch,
            "train_mse": mse_sum / max(n_pairs, 1),
            "train_steps": steps,
            "accum": self.accum,
            "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else float("nan"),
            "lr": self.optimizer.param_groups[0]["lr"],
            "wall_s": time.time() - t0,
        }

    def _log(self, rec: dict) -> None:
        with self.log_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

    def fit(self) -> None:
        max_epochs = int(self.t["max_epochs"])
        patience = int(self.t["patience"])
        for epoch in range(self.epoch, max_epochs):
            rec = self._train_epoch(epoch)
            rec.update(evaluate_split(self.model, self.bd, self.device, self.bd.val_idx, "val"))
            val = rec["val_pooled_rel_rmse"]
            if val < self.best - 1e-9:
                self.best = val
                self.bad_epochs = 0
                torch.save({"model_state_dict": self.model.state_dict(), "epoch": epoch, "val": val}, self.ckpt_path)
                rec["saved_best"] = True
            else:
                self.bad_epochs += 1
                rec["saved_best"] = False
            self._log(rec)
            self.scheduler.step()
            self.epoch = epoch + 1
            if self.bad_epochs >= patience:
                print(f"early stop at epoch {epoch} (patience {patience})", flush=True)
                break
        self.load_weights(self.ckpt_path)

    def load_weights(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"], strict=True)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="anisotrans-ccs", description="Train and evaluate AnisoTrans v2.")
    p.add_argument("--config", required=True, type=Path, help="YAML configuration file")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=os.environ.get("ANISOTRANS_DATA_DIR"),
        help="directory with the R6 per-cell data files (default: $ANISOTRANS_DATA_DIR)",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="output directory (default: artifacts/<config stem>)")
    p.add_argument("--device", default=None, help="override hardware.device, e.g. cuda:0 or cpu")
    p.add_argument("--max-epochs", type=int, default=None, help="override training.max_epochs")
    p.add_argument(
        "--eval-only", type=Path, default=None, metavar="CHECKPOINT", help="skip training; evaluate a checkpoint"
    )
    args = p.parse_args(argv)

    if args.data_dir is None:
        p.error("provide --data-dir or set ANISOTRANS_DATA_DIR")

    cfg = yaml.safe_load(args.config.read_text())
    if args.max_epochs is not None:
        cfg["training"]["max_epochs"] = args.max_epochs
    device = pick_device(args.device or cfg["hardware"]["device"])
    out_dir = args.out_dir or Path("artifacts") / args.config.stem
    print(f"device={device} out_dir={out_dir}", flush=True)

    set_deterministic(int(cfg["seed"]))
    bd = load_bundle(cfg["data"], args.data_dir, int(cfg["model"]["knn_k"]), device)
    trainer = Trainer(cfg, bd, device, out_dir)
    print(f"parameters={count_params(trainer.model)}", flush=True)

    if args.eval_only is not None:
        trainer.load_weights(args.eval_only)
    else:
        trainer.fit()

    result = evaluate_split(trainer.model, bd, device, bd.test_idx, "test")
    payload = {
        "config": str(args.config),
        "parameters": count_params(trainer.model),
        "epochs_completed": trainer.epoch,
        "best_val_pooled_rel_rmse": trainer.best if math.isfinite(trainer.best) else None,
        **result,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_frozen_test.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
