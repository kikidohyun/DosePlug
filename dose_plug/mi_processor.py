
import torch
import torch.nn as nn
import torch.nn.functional as F


class RealTimeMIProcessor(nn.Module):

    def __init__(
        self,
        input_len: int = 36,
        bins: int = 64,
        patch_h: int = 32,
        patch_w: int = 32,
        window_size: int = 5,

        window_mode: str = "minmax",
        window_min: float = None,
        window_max: float = None,
        window_percentiles: tuple = (0.01, 0.99),
    ):
        super().__init__()
        self.input_len = input_len
        self.bins = bins
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.window_size = window_size

        self.window_mode = window_mode
        self.window_min = window_min
        self.window_max = window_max
        self.window_percentiles = window_percentiles




    def compute_mi_tensor(self, x_binned: torch.Tensor, y_binned: torch.Tensor) -> torch.Tensor:
        B, N, P = x_binned.shape

        joint_idx = x_binned * self.bins + y_binned
        joint_idx_flat = joint_idx.view(B * N, P)

        joint_hist = torch.zeros(B * N, self.bins * self.bins, device=x_binned.device)
        ones = torch.ones_like(joint_idx_flat, dtype=torch.float32)
        joint_hist.scatter_add_(1, joint_idx_flat.long(), ones)

        joint_prob = joint_hist / P
        joint_prob_2d = joint_prob.view(B * N, self.bins, self.bins)

        p_x = joint_prob_2d.sum(dim=2)
        p_y = joint_prob_2d.sum(dim=1)

        eps = 1e-8
        h_x = -torch.sum(p_x * torch.log(p_x + eps), dim=1)
        h_y = -torch.sum(p_y * torch.log(p_y + eps), dim=1)
        h_xy = -torch.sum(joint_prob * torch.log(joint_prob + eps), dim=1)

        mi = h_x + h_y - h_xy
        return mi.view(B, N)

    def _window_and_normalize(self, sinogram: torch.Tensor) -> torch.Tensor:
        eps = 1e-6

        if self.window_mode == "fixed":
            low = torch.as_tensor(self.window_min, device=sinogram.device, dtype=sinogram.dtype).view(1, 1, 1, 1)
            high = torch.as_tensor(self.window_max, device=sinogram.device, dtype=sinogram.dtype).view(1, 1, 1, 1)
            sinogram = torch.clamp(sinogram, low, high)
            return ((sinogram - low) / (high - low + eps)).clamp(0.0, 1.0)

        elif self.window_mode == "percentile":
            ql, qh = self.window_percentiles
            flat = sinogram.view(sinogram.size(0), -1)
            low = torch.quantile(flat, ql, dim=1).view(-1, 1, 1, 1)
            high = torch.quantile(flat, qh, dim=1).view(-1, 1, 1, 1)
            sinogram = torch.clamp(sinogram, low, high)
            return ((sinogram - low) / (high - low + eps)).clamp(0.0, 1.0)

        else:
            s_min = sinogram.amin(dim=(2, 3), keepdim=True)
            s_max = sinogram.amax(dim=(2, 3), keepdim=True)
            return ((sinogram - s_min) / (s_max - s_min + eps)).clamp(0.0, 1.0)




    def forward(self, sinogram: torch.Tensor) -> torch.Tensor:
        if sinogram.dim() == 3:
            sinogram = sinogram.unsqueeze(1)
        B, C, H, W = sinogram.shape


        sino_norm = self._window_and_normalize(sinogram)


        sino_binned = (sino_norm * (self.bins - 1)).long().clamp(0, self.bins - 1)


        patches_all = F.unfold(
            sino_binned.float(),
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        ).transpose(1, 2)

        n_rows = H // self.patch_h
        n_cols = W // self.patch_w
        patches_grid = patches_all.view(B, n_rows, n_cols, -1)


        patches_selected = patches_grid[:, ::2, ::2, :].reshape(
            B, -1, self.patch_h * self.patch_w
        )


        win = self.window_size
        r = win // 2

        indices_1d = torch.arange(n_rows * n_cols, device=patches_all.device).view(n_rows, n_cols)
        selected_indices = indices_1d[::2, ::2].reshape(-1)

        t_row = selected_indices // n_cols
        t_col = selected_indices % n_cols

        dr = torch.arange(-r, r + 1, device=patches_all.device)
        dc = torch.arange(-r, r + 1, device=patches_all.device)
        rr = (t_row[:, None, None] + dr[None, :, None]).clamp(0, n_rows - 1)
        cc = (t_col[:, None, None] + dc[None, None, :]).clamp(0, n_cols - 1)
        cand_idx = (rr * n_cols + cc).view(-1, win * win)

        cand_patches = patches_all[:, cand_idx, :]
        p_target = patches_selected.unsqueeze(2)
        dist_local = torch.abs(p_target - cand_patches).sum(dim=-1)


        n_sel = selected_indices.shape[0]
        self_mask = (cand_idx == selected_indices.view(-1, 1))
        dist_local = dist_local + self_mask.view(1, n_sel, -1).float() * 1e9

        best_local = torch.argmin(dist_local, dim=2)
        best_match_idx = cand_idx.unsqueeze(0).expand(B, -1, -1) \
            .gather(2, best_local.unsqueeze(2)).squeeze(2)


        n_total = n_rows * n_cols
        flat_patches = patches_all.reshape(B * n_total, -1)
        batch_offsets = (torch.arange(B, device=patches_all.device) * n_total).view(-1, 1)
        final_indices = (batch_offsets + best_match_idx).view(-1)
        best_match_patches = flat_patches[final_indices].view(B, n_sel, -1)


        mi_values = self.compute_mi_tensor(
            patches_selected.long(),
            best_match_patches.long(),
        )
        mi_values[torch.isnan(mi_values)] = 0.0



        if mi_values.shape[1] != self.input_len:
            mi_values = F.interpolate(
                mi_values.unsqueeze(1), size=self.input_len,
                mode="linear", align_corners=False
            ).squeeze(1)

        return torch.log1p(mi_values)
