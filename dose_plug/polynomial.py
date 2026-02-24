
import torch
import torch.nn as nn





class DosePolynomial(nn.Module):
    def __init__(self, feature_dim: int, context_dim: int = 129):
        super().__init__()
        self.alpha_gen = nn.Sequential(
            nn.Linear(context_dim, feature_dim),
            nn.GELU(),
        )
        self.beta_gen = nn.Sequential(
            nn.Linear(context_dim, feature_dim),
            nn.GELU(),
        )
        self.gamma_gen = nn.Sequential(
            nn.Linear(context_dim, feature_dim),
            nn.GELU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        pred_dose: torch.Tensor,
    ) -> torch.Tensor:
        if pred_dose.dim() == 1:
            pred_dose = pred_dose.unsqueeze(-1)
        inp = torch.cat([context, pred_dose], dim=-1)
        alpha = self.alpha_gen(inp)
        beta = self.beta_gen(inp)
        gamma = self.gamma_gen(inp)

        if x.dim() == 3:
            x2 = x * x
            return alpha.unsqueeze(1) * x2 + beta.unsqueeze(1) * x + gamma.unsqueeze(1)
        elif x.dim() == 4:
            x2 = x * x
            return (
                alpha[:, :, None, None] * x2
                + beta[:, :, None, None] * x
                + gamma[:, :, None, None]
            )
        else:
            return alpha * (x * x) + beta * x + gamma





class DRCPolynomial(nn.Module):
    def __init__(self, feature_dim: int, context_dim: int = 128, hidden: int = 256):
        super().__init__()
        cond_dim = context_dim + 1

        self.alpha_net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, feature_dim),
        )

        self.beta_net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, feature_dim),
        )

        self.gamma_net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, feature_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        dose_context: torch.Tensor,
        pred_dose: torch.Tensor,
    ) -> torch.Tensor:
        if pred_dose.dim() == 1:
            pred_dose = pred_dose.unsqueeze(-1)

        c = torch.cat([dose_context, pred_dose], dim=-1)
        alpha = self.alpha_net(c)
        beta = self.beta_net(c)
        gamma = self.gamma_net(c)

        if x.dim() == 3:
            x2 = x * x
            return alpha.unsqueeze(1) * x2 + beta.unsqueeze(1) * x + gamma.unsqueeze(1)
        elif x.dim() == 4:
            x2 = x * x
            return (
                alpha[:, :, None, None] * x2
                + beta[:, :, None, None] * x
                + gamma[:, :, None, None]
            )
        else:
            return alpha * (x * x) + beta * x + gamma
