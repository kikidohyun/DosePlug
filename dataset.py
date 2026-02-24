
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class PetPairNPY(Dataset):
    def __init__(
        self,
        root,
        split="train",
        rec_tag="rec_1",
        normalize="minmax_0_1",
        patch_size=256,
        patch_n=1,
        center_crop_on_val=True
    ):
        self.root = Path(root)
        self.split = split
        self.rec_tag = rec_tag
        self.normalize = normalize

        self.patch_size = int(patch_size) if patch_size is not None else 0
        self.patch_n = int(patch_n) if patch_n is not None else 1
        self.center_crop_on_val = bool(center_crop_on_val)

        self.is_train = (split == "train")

        hf_dir = self.root / split / "original"
        lf_dir = self.root / split / "input"

        hf = {p.stem.replace("_image", ""): p for p in hf_dir.glob("*_image.npy")}
        lf = {p.stem.replace(f"_{rec_tag}", ""): p for p in lf_dir.glob(f"*_{rec_tag}.npy")}

        keys = sorted(set(hf.keys()) & set(lf.keys()))
        if len(keys) == 0:
            raise RuntimeError(f"No paired data found. Check paths:\nHF={hf_dir}\nLF={lf_dir}")

        self.pairs = [(lf[k], hf[k]) for k in keys]

    def __len__(self):
        return len(self.pairs)

    def _norm(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        if self.normalize == "minmax_0_1":
            mn = float(x.min())
            mx = float(x.max())
            x = (x - mn) / ((mx - mn) + 1e-8)
        elif self.normalize == "none":
            pass
        else:
            mn = float(x.min())
            mx = float(x.max())
            x = (x - mn) / ((mx - mn) + 1e-8)
        return x

    def _crop(self, arr: np.ndarray, top: int, left: int, ps: int) -> np.ndarray:

        if arr.ndim == 3:
            return arr[:, top:top+ps, left:left+ps]
        return arr[top:top+ps, left:left+ps]

    def __getitem__(self, idx):
        lf_path, hf_path = self.pairs[idx]
        lf = np.load(lf_path)
        hf = np.load(hf_path)


        if lf.size == 0 or lf.shape[-1] == 0 or lf.shape[-2] == 0:
            raise RuntimeError(f"[EMPTY LF] idx={idx}, path={lf_path}, shape={lf.shape}")
        if hf.size == 0 or hf.shape[-1] == 0 or hf.shape[-2] == 0:
            raise RuntimeError(f"[EMPTY HF] idx={idx}, path={hf_path}, shape={hf.shape}")


        if lf.ndim == 2:
            lf = lf[None, ...]
        if hf.ndim == 2:
            hf = hf[None, ...]

        lf = self._norm(lf)
        hf = self._norm(hf)

        _, H, W = lf.shape
        ps = self.patch_size




        if ps <= 0:
            return torch.from_numpy(lf), torch.from_numpy(hf)


        if ps > H or ps > W:
            return torch.from_numpy(lf), torch.from_numpy(hf)




        if self.is_train:
            if self.patch_n <= 1:
                top = np.random.randint(0, H - ps + 1)
                left = np.random.randint(0, W - ps + 1)
                lf_out = self._crop(lf, top, left, ps)
                hf_out = self._crop(hf, top, left, ps)
                return torch.from_numpy(lf_out), torch.from_numpy(hf_out)


            lfs, hfs = [], []
            for _ in range(self.patch_n):
                top = np.random.randint(0, H - ps + 1)
                left = np.random.randint(0, W - ps + 1)
                lfs.append(self._crop(lf, top, left, ps))
                hfs.append(self._crop(hf, top, left, ps))
            lf_out = np.stack(lfs, axis=0)
            hf_out = np.stack(hfs, axis=0)
            return torch.from_numpy(lf_out), torch.from_numpy(hf_out)




        if self.center_crop_on_val:
            top = (H - ps) // 2
            left = (W - ps) // 2
            lf_out = self._crop(lf, top, left, ps)
            hf_out = self._crop(hf, top, left, ps)
            return torch.from_numpy(lf_out), torch.from_numpy(hf_out)
        else:
            return torch.from_numpy(lf), torch.from_numpy(hf)
