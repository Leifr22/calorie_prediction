from __future__ import annotations

import timm
import torch
import torch.nn as nn
class IngredientsEncoder(nn.Module):
    def __init__(self,vocab_size,emb_dim):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,emb_dim,padding_idx=0)
    def forward(self, ingr_ids,ingr_mask):
        emb = self.embedding(ingr_ids)
        mask = ingr_mask.unsqueeze(-1)
        summed = (emb * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom
class CalorieModel(nn.Module):
    def __init__(
            self,
            backbone,
            pretrained,
            ingr_vocab_size,
            ingr_emb_dim,
            mlp_hidden,
            dropout,
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
        img_feat_dim = self.backbone.num_features
        self.ingr_encoder = IngredientsEncoder(ingr_vocab_size, ingr_emb_dim)
        self.mass_proj = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
        )

        in_dim = img_feat_dim + ingr_emb_dim + 16
        self.head = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 2, 1),
        )

    def forward(
            self,
            image,
            ingr_ids,
            ingr_mask,
            mass_norm,
    ):
        img_feat = self.backbone(image)
        ingr_feat = self.ingr_encoder(ingr_ids, ingr_mask)
        mass_feat = self.mass_proj(mass_norm.unsqueeze(-1))
        x = torch.cat([img_feat, ingr_feat, mass_feat], dim=-1)
        out = self.head(x).squeeze(-1)
        return out
    def param_groups(self, lr_backbone: float, lr_head: float, weight_decay: float):
        backbone_params = list(self.backbone.parameters())
        other_params = [p for n, p in self.named_parameters() if not n.startswith("backbone.")]
        return [
            {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": other_params, "lr": lr_head, "weight_decay": weight_decay},
        ]

def build_model(cfg):
        return CalorieModel(
            backbone=cfg.model.backbone,
            pretrained=cfg.model.pretrained,
            ingr_vocab_size=cfg.model.ingr_vocab_size,
            ingr_emb_dim=cfg.model.ingr_emb_dim,
            mlp_hidden=cfg.model.mlp_hidden,
            dropout=cfg.model.dropout,
        )