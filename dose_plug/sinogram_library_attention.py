
import torch
import torch.nn as nn
from einops import rearrange
from typing import Tuple





class Positional_Attention(nn.Module):
    def __init__(self, dim: int, num_sinogram_libraries: int):
        super().__init__()
        self.dim = dim
        self.num_sinogram_libraries = num_sinogram_libraries
        self.temperature = nn.Parameter(torch.ones(1, 1, 1))

        self.wq = nn.Conv2d(
            in_channels=num_sinogram_libraries,
            out_channels=num_sinogram_libraries,
            kernel_size=8, stride=8, padding=0,
            groups=num_sinogram_libraries,
        )
        self.pos_wk = nn.Conv2d(
            in_channels=1,
            out_channels=num_sinogram_libraries,
            kernel_size=8, stride=8, padding=0,
        )
        self.pos_wv = nn.Conv2d(
            in_channels=1,
            out_channels=num_sinogram_libraries,
            kernel_size=8, stride=8, padding=0,
        )
        self.upsample_layer = nn.Upsample(
            size=(dim, dim), mode='bilinear', align_corners=False,
        )
        self.project_out = nn.Conv2d(
            in_channels=num_sinogram_libraries, out_channels=1,
            kernel_size=3, stride=1, padding=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        pos_kv: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seqlen, _ = x.shape
        x_2d = x.unsqueeze(1)

        b, c, h_sinogram_library, w_sinogram_library = pos_kv.shape
        h = h_sinogram_library // 8
        w = w_sinogram_library // 8

        xq = self.wq(pos_kv)
        xk = self.pos_wk(x_2d)
        xv = self.pos_wv(x_2d)

        xq = rearrange(xq, 'b c h w -> b c (h w)')
        xk = rearrange(xk, 'b c h w -> b c (h w)')
        xv = rearrange(xv, 'b c h w -> b c (h w)')

        xq = torch.nn.functional.normalize(xq, dim=-1)
        xk = torch.nn.functional.normalize(xk, dim=-1)

        attn_score = (xq @ xk.transpose(-2, -1)) * self.temperature
        attn = attn_score.softmax(dim=-1)
        out = attn @ xv

        out = rearrange(out, 'b c (h w) -> b c h w', h=h, w=w)
        out = self.project_out(self.upsample_layer(out)).squeeze(1)

        return out, attn_score





class SinogramLibraryAttention2D(nn.Module):
    def __init__(
        self,
        channels: int,
        num_sinogram_libraries: int = 400,
        sinogram_library_size: int = 352,
        attn_dim: int = 8,
    ):
        super().__init__()
        self.channels = channels
        self.num_sinogram_libraries = num_sinogram_libraries
        self.attn_dim = int(attn_dim)



        self.sinogram_library_adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d(channels),
        )

        self.temperature = nn.Parameter(torch.ones(1))


        self.q_proj = nn.Linear(channels, self.attn_dim, bias=False)
        self.k_proj = nn.Linear(channels, self.attn_dim, bias=False)
        self.v_proj = nn.Linear(channels, self.attn_dim, bias=False)
        self.out_proj = nn.Linear(self.attn_dim, channels, bias=False)

    def forward(
        self,
        feat: torch.Tensor,
        sinogram_library_bank: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = feat.shape
        P = sinogram_library_bank.shape[1]


        sinogram_library_adapted = self.sinogram_library_adapter(sinogram_library_bank)
        if sinogram_library_adapted.shape[0] == 1:
            sinogram_library_adapted = sinogram_library_adapted.expand(B, -1, -1, -1)
        sinogram_library_tokens = sinogram_library_adapted.mean(dim=-1)


        feat_seq = feat.flatten(2).transpose(1, 2)


        Q = self.q_proj(sinogram_library_tokens)
        K = self.k_proj(feat_seq)
        V = self.v_proj(feat_seq)


        Q = torch.nn.functional.normalize(Q, dim=-1)
        K = torch.nn.functional.normalize(K, dim=-1)

        attn_score = (Q @ K.transpose(-2, -1)) * self.temperature
        attn = attn_score.softmax(dim=-1)

        context = attn @ V


        context_mean = context.mean(dim=1)


        out_seq = self.out_proj(context_mean)
        out = out_seq[:, :, None, None].expand_as(feat)

        return out, attn_score
