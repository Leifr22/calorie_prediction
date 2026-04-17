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


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_ingredients(s):
    if not isinstance(s,str) or not s.strip():
        return []
    ids=[]
    for token in s.split(';'):
        token=token.strip()
        if not token:
            continue
        if '_' in token:
            num=token.split('_')[-1].lstrip('0') or '0'
        else:
            num=token
        try:
            ids.append(int(num))
        except ValueError:
            continue
        return ids
@dataclass
class Sample:
    dish_id: str
    image_path: Path
    ingredients: list[int]
    mass: float
    calories: float
class DishDataset(Dataset):
    def __init__(self, samples,images_dir,transform,max_ingredients,mass_mean,mass_std):
        self.samples=samples
        self.images_dir = Path(images_dir)
        self.transform=transform
        self.max_ingredients=max_ingredients
        self.mass_mean=mass_mean
        self.mass_std=mass_std if mass_std > 1e-6 else 1.0
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        s=self.samples[idx]
        img=Image.open(s.image_path).convert('RGB')
        image=self.transform(img)
        ids = s.ingredients[: self.max_ingredients]
        ids_padded = ids + [-1] * (self.max_ingredients - len(ids))
        ingr_ids = torch.tensor(
            [(i + 1) if i >= 0 else 0 for i in ids_padded],
            dtype=torch.long,
        )
        ingr_mask = torch.tensor(
            [1.0 if i >= 0 else 0.0 for i in ids_padded],
            dtype=torch.float32,
        )
        mass=float(s.mass)
        mass_norm=(mass-self.mass_mean)/self.mass_std
        calories=float(s.calories)
        density=calories/mass if mass>0 else 0.0
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
def build_transforms(image_size):
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
def build_dataloaders(cfg,seed):
        dish = pd.read_csv(cfg.paths.dish_csv)
        if cfg.data.filter_zero_calories:
            dish = dish[dish["total_calories"] > 0]
        if cfg.data.filter_zero_mass:
            dish = dish[dish["total_mass"] > 0]

        train_df = dish[dish["split"] == "train"].reset_index(drop=True)
        test_df = dish[dish["split"] == "test"].reset_index(drop=True)
        rng = np.random.default_rng(seed)
        n_val = int(len(train_df) * cfg.data.val_fraction)
        perm = rng.permutation(len(train_df))
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        val_df = train_df.iloc[val_idx].reset_index(drop=True)
        tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
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
        g = torch.Generator()
        g.manual_seed(seed)

        train_loader = DataLoader(
            train_ds, batch_size=cfg.data.batch_size, shuffle=True,
            num_workers=cfg.data.num_workers, pin_memory=True,
            drop_last=True, generator=g, persistent_workers=cfg.data.num_workers > 0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.data.batch_size, shuffle=False,
            num_workers=cfg.data.num_workers, pin_memory=True,
            persistent_workers=cfg.data.num_workers > 0,
        )
        test_loader = DataLoader(
            test_ds, batch_size=cfg.data.batch_size, shuffle=False,
            num_workers=cfg.data.num_workers, pin_memory=True,
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
