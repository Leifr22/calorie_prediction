from __future__ import annotations

import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from scripts.dataset import build_dataloaders
from scripts.model import build_model

def set_seed(seed: int, deterministic: bool = True) -> None:

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
def build_loss(name):
    name=name.lower()
    if name=='smooth_l1':
        return nn.SmoothL1Loss()
    if name=='l1':
        return nn.L1Loss()
    if name=='mse':
        return nn.MSELoss()
    raise ValueError(f"Unknown loss: {name}")
def decode_predictions(raw_pred, mass, predict_density):
    if predict_density:
        return raw_pred * mass
    return raw_pred
def encode_target(calories, mass, density,
                  predict_density):
    return density if predict_density else calories
def _run_epoch(
    model, loader, criterion, device, predict_density,
    optimizer=None, scheduler=None, log_prefix: str = "", log_every: int = 20,
):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    abs_err_sum = 0.0
    sq_err_sum = 0.0
    n_samples = 0

    t0 = time.time()
    for i, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        ingr_ids = batch["ingr_ids"].to(device, non_blocking=True)
        ingr_mask = batch["ingr_mask"].to(device, non_blocking=True)
        mass_norm = batch["mass_norm"].to(device, non_blocking=True)
        mass = batch["mass"].to(device, non_blocking=True)
        calories = batch["calories"].to(device, non_blocking=True)
        density = batch["density"].to(device, non_blocking=True)
        if is_train:
            raw_pred = model(image, ingr_ids, ingr_mask, mass_norm)
            target = encode_target(calories, mass, density, predict_density)
            loss = criterion(raw_pred, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        else:
            with torch.inference_mode():
                raw_pred = model(image, ingr_ids, ingr_mask, mass_norm)
                target = encode_target(calories, mass, density, predict_density)
                loss = criterion(raw_pred, target)
        with torch.no_grad():
            pred_calories = decode_predictions(raw_pred, mass, predict_density)
            err = (pred_calories - calories).detach()

        bs = image.size(0)
        total_loss += loss.item() * bs
        abs_err_sum += err.abs().sum().item()
        sq_err_sum += (err ** 2).sum().item()
        n_samples += bs

        if is_train and log_every > 0 and (i + 1) % log_every == 0:
            running_mae = abs_err_sum / n_samples
            print(f"{log_prefix} batch {i + 1}/{len(loader)}  loss={loss.item():.4f}  MAE={running_mae:.2f}",
                  flush=True)

    elapsed = time.time() - t0
    return {
        "loss": total_loss / max(n_samples, 1),
        "mae": abs_err_sum / max(n_samples, 1),
        "rmse": (sq_err_sum / max(n_samples, 1)) ** 0.5,
        "n": n_samples,
        "time_s": elapsed,
    }
def train(cfg):
    set_seed(cfg.seed, deterministic=cfg.deterministic)

    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[train] device = {device}", flush=True)
    if device.type == "cuda":
        print(f"[train] GPU = {torch.cuda.get_device_name(0)}", flush=True)
    print("[train] preparing dataloaders...", flush=True)
    data = build_dataloaders(cfg, seed=cfg.seed)
    print(f"[train] sizes  train={data['train_size']}  val={data['val_size']}  test={data['test_size']}", flush=True)
    print(f"[train] mass stats  mean={data['mass_mean']:.2f}  std={data['mass_std']:.2f}", flush=True)
    print(f"[train] building model: {cfg.model.backbone}", flush=True)
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] params = {n_params / 1e6:.2f}M", flush=True)
    optimizer = AdamW(
        model.param_groups(cfg.train.lr_backbone, cfg.train.lr_head, cfg.train.weight_decay),
    )
    n_steps = cfg.train.epochs * len(data["train_loader"])
    scheduler = CosineAnnealingLR(optimizer, T_max=n_steps) if cfg.train.scheduler == "cosine" else None
    criterion = build_loss(cfg.train.loss)
    ckpt_dir = Path(cfg.paths.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / cfg.paths.best_model_name
    last_path = ckpt_dir / cfg.paths.last_model_name

    history = {"train_loss": [], "train_mae": [], "val_loss": [], "val_mae": [], "val_rmse": []}
    best_val_mae = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, cfg.train.epochs + 1):
        print(f"\n=== Epoch {epoch}/{cfg.train.epochs} ===", flush=True)

        train_metrics = _run_epoch(
            model, data["train_loader"], criterion, device, cfg.train.predict_density,
            optimizer=optimizer, scheduler=scheduler,
            log_prefix=f"[train e{epoch}]", log_every=cfg.train.log_every_n_batches,
        )
        val_metrics = _run_epoch(
            model, data["val_loader"], criterion, device, cfg.train.predict_density,
            log_prefix=f"[val e{epoch}]", log_every=0,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_mae"].append(train_metrics["mae"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_rmse"].append(val_metrics["rmse"])

        print(
            f"[epoch {epoch}]  "
            f"train: loss={train_metrics['loss']:.4f} MAE={train_metrics['mae']:.2f}  |  "
            f"val: loss={val_metrics['loss']:.4f} MAE={val_metrics['mae']:.2f} RMSE={val_metrics['rmse']:.2f}  |  "
            f"time train={train_metrics['time_s']:.1f}s val={val_metrics['time_s']:.1f}s",
            flush=True,
        )
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_mae": val_metrics["mae"],
            "mass_mean": data["mass_mean"],
            "mass_std": data["mass_std"],
            "config": cfg._raw,
        }
        torch.save(state, last_path)

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            epochs_no_improve = 0
            torch.save(state, best_path)
            print(f"[epoch {epoch}] >>> new best val MAE = {best_val_mae:.2f}, saved to {best_path}", flush=True)
        else:
            epochs_no_improve += 1
            print(f"[epoch {epoch}] no improvement ({epochs_no_improve}/{cfg.train.early_stopping_patience})",
                  flush=True)
            if epochs_no_improve >= cfg.train.early_stopping_patience:
                print(f"[epoch {epoch}] early stopping triggered", flush=True)
                break

    print(f"\n[train] DONE. Best val MAE = {best_val_mae:.2f}. Best ckpt: {best_path}", flush=True)
    return {
        "history": history,
        "best_val_mae": best_val_mae,
        "best_ckpt_path": str(best_path),
        "last_ckpt_path": str(last_path),
        "mass_mean": data["mass_mean"],
        "mass_std": data["mass_std"],
    }
def predict_loader(model, loader, device, predict_density: bool):
    model.eval()
    all_ids, all_preds, all_targets, all_masses = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            ingr_ids = batch["ingr_ids"].to(device, non_blocking=True)
            ingr_mask = batch["ingr_mask"].to(device, non_blocking=True)
            mass_norm = batch["mass_norm"].to(device, non_blocking=True)
            mass = batch["mass"].to(device, non_blocking=True)

            raw_pred = model(image, ingr_ids, ingr_mask, mass_norm)
            pred_calories = decode_predictions(raw_pred, mass, predict_density)

            all_ids.extend(batch["dish_id"])
            all_preds.append(pred_calories.cpu().numpy())
            all_targets.append(batch["calories"].numpy())
            all_masses.append(batch["mass"].numpy())

    return (
        all_ids,
        np.concatenate(all_preds),
        np.concatenate(all_targets),
        np.concatenate(all_masses),
    )
def load_model_from_checkpoint(ckpt_path: str, cfg, device) -> tuple:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt
