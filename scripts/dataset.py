"""Датасет блюд: фото + ингредиенты + масса -> калории.

Содержит:
- parse_ingredients(): парсинг строки 'ingr_0000000122;ingr_0000000026;...' в список int ID
- DishDataset: torch Dataset с фото, ингредиентами, массой и таргетом
- build_transforms(): train/val аугментации
- build_dataloaders(): возвращает train/val/test DataLoader-ы и mass_mean/mass_std
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ImageNet статистики — используются как для предобученных бэкбонов
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================
# Парсинг ингредиентов
# ============================================================
def parse_ingredients(s: str) -> list[int]:
    """Парсит строку 'ingr_0000000122;ingr_0000000026;...' -> [122, 26, ...].

    ID = 0 зарезервирован под padding, реальные ID начинаются с 1.
    Если в данных встретился id=0, прибавим 1 к каждому при использовании в Embedding.
    """
    if not isinstance(s, str) or not s.strip():
        return []
    ids = []
    for token in s.split(";"):
        token = token.strip()
        if not token:
            continue
        if "_" in token:
            num = token.split("_")[-1].lstrip("0") or "0"
        else:
            num = token
        try:
            ids.append(int(num))
        except ValueError:
            continue
    return ids


# ============================================================
# Dataset
# ============================================================
@dataclass
class Sample:
    dish_id: str
    image_path: Path
    ingredients: list[int]   # ID ингредиентов (без паддинга)
    mass: float              # сырое значение массы в граммах
    calories: float          # таргет — общая калорийность


class DishDataset(Dataset):
    """torch Dataset для блюд.

    Возвращает словарь:
        image:        FloatTensor [3, H, W]   — нормализованная картинка
        ingr_ids:     LongTensor  [max_ingr]  — паддинг = 0
        ingr_mask:    FloatTensor [max_ingr]  — 1 для реальных ингредиентов, 0 для паддинга
        mass_norm:    FloatTensor []          — z-score нормализованная масса
        mass:         FloatTensor []          — сырая масса в граммах (для денормализации предсказания)
        calories:     FloatTensor []          — таргет (общая калорийность)
        density:      FloatTensor []          — calories / mass (альтернативный таргет)
    """

    def __init__(
        self,
        samples: list[Sample],
        images_dir: Path,
        transform: Callable,
        max_ingredients: int,
        mass_mean: float,
        mass_std: float,
    ):
        self.samples = samples
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.max_ingredients = max_ingredients
        self.mass_mean = mass_mean
        self.mass_std = mass_std if mass_std > 1e-6 else 1.0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        # --- Картинка ---
        img = Image.open(s.image_path).convert("RGB")
        image = self.transform(img)

        # --- Ингредиенты с паддингом ---
        ids = s.ingredients[: self.max_ingredients]
        # +1, чтобы id=0 зарезервировать под паддинг
        ids_padded = ids + [-1] * (self.max_ingredients - len(ids))
        # padding token = 0, реальные ID сдвигаем на +1
        ingr_ids = torch.tensor(
            [(i + 1) if i >= 0 else 0 for i in ids_padded],
            dtype=torch.long,
        )
        ingr_mask = torch.tensor(
            [1.0 if i >= 0 else 0.0 for i in ids_padded],
            dtype=torch.float32,
        )

        # --- Масса ---
        mass = float(s.mass)
        mass_norm = (mass - self.mass_mean) / self.mass_std

        # --- Таргет ---
        calories = float(s.calories)
        density = calories / mass if mass > 0 else 0.0

        return {
            "image": image,
            "ingr_ids": ingr_ids,
            "ingr_mask": ingr_mask,
            "mass_norm": torch.tensor(mass_norm, dtype=torch.float32),
            "mass": torch.tensor(mass, dtype=torch.float32),
            "calories": torch.tensor(calories, dtype=torch.float32),
            "density": torch.tensor(density, dtype=torch.float32),
            "dish_id": s.dish_id,
        }


# ============================================================
# Трансформы / аугментации
# ============================================================
def build_transforms(image_size: int) -> tuple[Callable, Callable]:
    """Возвращает (train_transform, eval_transform).

    Train-аугментации мягкие — еда чувствительна к цвету и форме.
    """
    train_tf = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


# ============================================================
# Подготовка сэмплов
# ============================================================
def _build_samples(df: pd.DataFrame, images_dir: Path) -> list[Sample]:
    samples = []
    for _, row in df.iterrows():
        dish_id = str(row["dish_id"])
        img_path = images_dir / dish_id / "rgb.png"
        if not img_path.exists():
            continue
        samples.append(
            Sample(
                dish_id=dish_id,
                image_path=img_path,
                ingredients=parse_ingredients(row["ingredients"]),
                mass=float(row["total_mass"]),
                calories=float(row["total_calories"]),
            )
        )
    return samples


def build_dataloaders(cfg, seed: int = 42) -> dict:
    """Готовит train/val/test DataLoader-ы.

    Возвращает:
        {
            'train_loader', 'val_loader', 'test_loader',
            'mass_mean', 'mass_std',           # пригодятся для инференса
            'train_size', 'val_size', 'test_size',
        }
    """
    dish = pd.read_csv(cfg.paths.dish_csv)

    # Фильтрация аномалий
    if cfg.data.filter_zero_calories:
        dish = dish[dish["total_calories"] > 0]
    if cfg.data.filter_zero_mass:
        dish = dish[dish["total_mass"] > 0]

    train_df = dish[dish["split"] == "train"].reset_index(drop=True)
    test_df = dish[dish["split"] == "test"].reset_index(drop=True)

    # train -> train/val split (стратификация по бинам калорий, чтоб не уплыло распределение)
    rng = np.random.default_rng(seed)
    n_val = int(len(train_df) * cfg.data.val_fraction)
    perm = rng.permutation(len(train_df))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    val_df = train_df.iloc[val_idx].reset_index(drop=True)
    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)

    # Считаем статистики массы ТОЛЬКО на трейне
    mass_mean = float(tr_df["total_mass"].mean())
    mass_std = float(tr_df["total_mass"].std())

    images_dir = Path(cfg.paths.images_dir)
    train_samples = _build_samples(tr_df, images_dir)
    val_samples = _build_samples(val_df, images_dir)
    test_samples = _build_samples(test_df, images_dir)

    train_tf, eval_tf = build_transforms(cfg.data.image_size)

    train_ds = DishDataset(train_samples, images_dir, train_tf,
                           cfg.data.max_ingredients, mass_mean, mass_std)
    val_ds = DishDataset(val_samples, images_dir, eval_tf,
                         cfg.data.max_ingredients, mass_mean, mass_std)
    test_ds = DishDataset(test_samples, images_dir, eval_tf,
                          cfg.data.max_ingredients, mass_mean, mass_std)

    # generator для воспроизводимого шафла
    g = torch.Generator()
    g.manual_seed(seed)

    # pin_memory имеет смысл только с CUDA — на CPU даёт warning и не ускоряет
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=cfg.data.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, pin_memory=pin_memory,
        drop_last=True, generator=g, persistent_workers=cfg.data.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=pin_memory,
        persistent_workers=cfg.data.num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=pin_memory,
        persistent_workers=cfg.data.num_workers > 0,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "mass_mean": mass_mean,
        "mass_std": mass_std,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
    }
