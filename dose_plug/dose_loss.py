
from typing import Optional

import torch
import torch.nn as nn


def dose_to_log1p(dose: torch.Tensor) -> torch.Tensor:
    return torch.log1p(dose.float())


def dose_log_to_real(pred_dose_log: torch.Tensor) -> torch.Tensor:
    return torch.expm1(pred_dose_log.float())


class LogDoseSmoothL1Loss(nn.Module):

    def __init__(
        self,
        beta: float = 1.0,
        reduction: str = "mean",
        clamp_min: Optional[float] = 0.0,
        clamp_max: Optional[float] = 100.0,
    ):
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss(beta=beta, reduction=reduction)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def _clamp_real_dose(self, dose: torch.Tensor) -> torch.Tensor:
        if self.clamp_min is None and self.clamp_max is None:
            return dose
        lo = self.clamp_min if self.clamp_min is not None else float("-inf")
        hi = self.clamp_max if self.clamp_max is not None else float("inf")
        return torch.clamp(dose, min=lo, max=hi)

    def forward(self, pred_dose_log: torch.Tensor, target_dose_real: torch.Tensor) -> torch.Tensor:
        pred = pred_dose_log.view(-1).float()
        target_real = self._clamp_real_dose(target_dose_real.view(-1).float())
        target_log = dose_to_log1p(target_real)
        return self.loss_fn(pred, target_log)

    @torch.no_grad()
    def mae_in_real_space(self, pred_dose_log: torch.Tensor, target_dose_real: torch.Tensor) -> torch.Tensor:
        pred_real = dose_log_to_real(pred_dose_log.view(-1))
        pred_real = self._clamp_real_dose(pred_real)
        target_real = self._clamp_real_dose(target_dose_real.view(-1).float())
        return torch.mean(torch.abs(pred_real - target_real))
