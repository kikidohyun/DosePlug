
from typing import Tuple

import torch
import torch.nn as nn


class SafeBatchNorm1d(nn.BatchNorm1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training and input.dim() == 2 and input.size(0) == 1:
            return input
        return super().forward(input)





class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            SafeBatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            SafeBatchNorm1d(dim),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))





class DoseResNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 36,
        hidden_dim: int = 16,
        feat_dim: int = 8,
        num_res_blocks: int = 0,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            SafeBatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_res_blocks)])
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim, feat_dim), nn.GELU(), nn.Dropout(0.1),
        )
        self.head = nn.Linear(feat_dim, 1)

    def forward(
        self, x: torch.Tensor, return_dose: bool = False
    ) -> torch.Tensor:
        x = self.stem(x)
        x = self.res_blocks(x)
        feat = self.feature_extractor(x)
        if return_dose:
            dose = self.head(feat)
            return feat, dose
        return feat





class HybridDoseEstimator(nn.Module):
    def __init__(self, mi_dim: int = 36, mi_feat_dim: int = 8, context_dim: int = 8):
        super().__init__()


        self.mi_model = DoseResNet(
            input_dim=mi_dim,
            hidden_dim=16,
            feat_dim=mi_feat_dim,
            num_res_blocks=0,
        )


        self.attn_encoder = nn.Sequential(
            nn.Conv2d(1, 4, 3, stride=2, padding=1), nn.BatchNorm2d(4), nn.GELU(),
            nn.Conv2d(4, mi_feat_dim, 3, stride=2, padding=1), nn.BatchNorm2d(mi_feat_dim), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )


        self.fusion = nn.Sequential(
            nn.Linear(mi_feat_dim * 2, context_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.head_dose = nn.Linear(context_dim, 1)

    def forward(
        self, mi_vec: torch.Tensor, attn_score: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        feat_mi = self.mi_model(mi_vec)


        if attn_score.dim() == 3:
            attn_score = attn_score.unsqueeze(1)
        feat_attn = self.attn_encoder(attn_score)


        combined = torch.cat([feat_mi, feat_attn], dim=-1)
        dose_context = self.fusion(combined)
        pred_dose = self.head_dose(dose_context)

        return dose_context, pred_dose

    def load_pretrained_mi(self, pth_path: str, strict: bool = False):
        import os
        if not os.path.exists(pth_path):
            return
        pretrained = torch.load(pth_path, map_location="cpu")
        model_dict = self.mi_model.state_dict()
        matched = {k: v for k, v in pretrained.items() if k in model_dict and model_dict[k].shape == v.shape}
        self.mi_model.load_state_dict(matched, strict=strict)
