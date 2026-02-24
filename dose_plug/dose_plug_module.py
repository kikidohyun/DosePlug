
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .mi_processor import RealTimeMIProcessor
from .dose_estimator import HybridDoseEstimator
from .polynomial import DosePolynomial, DRCPolynomial
from .sinogram_library_attention import SinogramLibraryAttention2D
from .dose_loss import LogDoseSmoothL1Loss


class DosePlugModule(nn.Module):

    def __init__(
        self,
        feature_channels: int = 32,
        mi_input_len: int = 36,
        mi_bins: int = 64,
        num_sinogram_libraries: int = 31,
        sinogram_library_size: int = 16,
        sinogram_library_bank_path: Optional[str] = None,
        polynomial_type: str = "drc",
        polynomial_hidden: int = 4,
        mi_pretrained: Optional[str] = None,
    ):
        super().__init__()
        self.feature_channels = feature_channels
        self.context_dim = 8


        self.mi_processor = RealTimeMIProcessor(
            input_len=mi_input_len,
            bins=mi_bins,
        )


        self.dose_estimator = HybridDoseEstimator(
            mi_dim=mi_input_len,
            mi_feat_dim=self.context_dim,
            context_dim=self.context_dim,
        )
        if mi_pretrained and os.path.exists(mi_pretrained):
            self.dose_estimator.load_pretrained_mi(mi_pretrained)


        if sinogram_library_bank_path and os.path.exists(sinogram_library_bank_path):
            sinogram_library_data = np.load(sinogram_library_bank_path)

            if sinogram_library_data.ndim == 3:
                sinogram_library_data = sinogram_library_data[np.newaxis, ...]
            num_sinogram_libraries = sinogram_library_data.shape[1]
            sinogram_library_size = sinogram_library_data.shape[2]
            self.sinogram_library_bank = nn.Parameter(
                torch.from_numpy(sinogram_library_data).float(),
                requires_grad=True,
            )
        else:
            self.sinogram_library_bank = nn.Parameter(
                torch.randn(1, num_sinogram_libraries, sinogram_library_size, sinogram_library_size) * 0.02,
                requires_grad=True,
            )


        self.sinogram_library_attn = SinogramLibraryAttention2D(
            channels=feature_channels,
            num_sinogram_libraries=num_sinogram_libraries,
            sinogram_library_size=sinogram_library_size,
            attn_dim=8,
        )


        if polynomial_type == "drc":
            self.polynomial = DRCPolynomial(
                feature_dim=feature_channels,
                context_dim=self.context_dim,
                hidden=polynomial_hidden,
            )
        else:
            self.polynomial = DosePolynomial(
                feature_dim=feature_channels,
                context_dim=self.context_dim + 1,
            )
        self.polynomial_type = polynomial_type
        self.dose_loss_fn = LogDoseSmoothL1Loss()





    def compute_mi(self, x: torch.Tensor) -> torch.Tensor:
        return self.mi_processor(x)

    def estimate_dose(
        self,
        mi_vec: torch.Tensor,
        attn_score: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if attn_score.dim() == 3 and attn_score.shape[1] != 1:



            score_mean = attn_score.mean(dim=1)
            hw = score_mean.shape[-1]
            h = w = int(hw ** 0.5)
            if h * w != hw:

                score_2d = score_mean.unsqueeze(1).unsqueeze(1)
            else:
                score_2d = score_mean.view(-1, 1, h, w)
        else:
            score_2d = attn_score

        return self.dose_estimator(mi_vec, score_2d)

    def apply_sinogram_library_attention(
        self, feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = feat.shape[0]
        bank = self.sinogram_library_bank.expand(B, -1, -1, -1)
        return self.sinogram_library_attn(feat, bank)

    def modulate(
        self,
        feat: torch.Tensor,
        dose_context: torch.Tensor,
        pred_dose: torch.Tensor,
    ) -> torch.Tensor:
        return self.polynomial(feat, dose_context, pred_dose)

    def compute_dose_loss(self, pred_dose_log: torch.Tensor, target_dose_real: torch.Tensor) -> torch.Tensor:
        return self.dose_loss_fn(pred_dose_log, target_dose_real)

    @torch.no_grad()
    def compute_dose_mae(self, pred_dose_log: torch.Tensor, target_dose_real: torch.Tensor) -> torch.Tensor:
        return self.dose_loss_fn.mae_in_real_space(pred_dose_log, target_dose_real)





    def forward(
        self,
        x_input: torch.Tensor,
        feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        mi_vec = self.compute_mi(x_input)


        sinogram_library_feat, attn_score = self.apply_sinogram_library_attention(feat)


        dose_context, pred_dose = self.estimate_dose(mi_vec, attn_score)



        feat_combined = feat + sinogram_library_feat
        feat_mod = self.modulate(feat_combined, dose_context, pred_dose)
        feat_out = feat + feat_mod

        return feat_out, pred_dose, attn_score
