import os
import time
import numpy as np
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

from prep import printProgressBar
from networks import DPN
from measure import compute_measure
from dose_plug import LogDoseSmoothL1Loss

import random
from glob import glob

def _unwrap_model(m):
    return m.module if isinstance(m, nn.DataParallel) else m

def _ensure_4d(x):

    if x.dim() == 3:
        x = x.unsqueeze(1)
    return x

from tensorboardX import SummaryWriter
writer = SummaryWriter("tensorboardX/DPL-Net_1e3_train_fig")

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True


seed_torch(seed=42)

class L1_Charbonnier_loss(torch.nn.Module):
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


class Solver(object):
    def __init__(self, args, data_loader, data_Valloader):
        self.mode = args.mode
        self.load_mode = args.load_mode
        self.data_loader = data_loader
        self.data_Valloader = data_Valloader

        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max
        self.data_domain = getattr(args, "data_domain", "auto")
        self.normalize_target = bool(getattr(args, "normalize_target", False))
        self._resolved_data_domain = None

        self.save_path = args.save_path
        self.multi_gpu = args.multi_gpu

        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.decay_iters = args.decay_iters
        self.save_iters = args.save_iters
        self.ckpt_path = getattr(args, "ckpt_path", "")
        self.test_tag = "ckpt"

        self.patch_size = args.patch_size
        self.dose = getattr(args, "dose", None)
        self.use_dose_plug = bool(getattr(args, "use_dose_plug", False))
        self.dose_lambda = 1.0
        self.save_recon = getattr(args, "save_recon", False)
        self.recon_root = getattr(args, "recon_root", "")
        self.recon_png = getattr(args, "recon_png", False)


        self.max_sinogram_libraries = int(getattr(args, "max_sinogram_libraries", 0) or 0)
        self.DPN = DPN(
            max_sinogram_libraries=self.max_sinogram_libraries,
            use_dose_plug=self.use_dose_plug,
        )
        if (self.multi_gpu) and (torch.cuda.device_count() > 1):
            print('Use {} GPUs'.format(torch.cuda.device_count()))
            self.DPN = nn.DataParallel(self.DPN)
        self.DPN.to(self.device)

        self.lr = args.lr
        self.resume_iters = getattr(args, "resume_iters", 0)
        self.resume_optimizer = getattr(args, "resume_optimizer", False)
        self.save_full_ckpt = getattr(args, "save_full_ckpt", False)
        self.pretrained_ckpt = getattr(args, "pretrained_ckpt", "")

        self.criterion = nn.MSELoss()
        self.criterion_dose = LogDoseSmoothL1Loss()
        self.optimizer = optim.Adam(self.DPN.parameters(), self.lr, betas=(0.5, 0.999))

        if isinstance(self.ckpt_path, str) and self.ckpt_path.strip():
            bn = os.path.basename(self.ckpt_path.strip())
            m = re.search(r"_([0-9]+)iter\.ckpt$", bn)
            self.test_tag = m.group(1) if m else os.path.splitext(bn)[0]

    @staticmethod
    def _strip_profile_keys(state_dict):
        if not isinstance(state_dict, dict):
            return state_dict
        drop_suffixes = ("total_ops", "total_params")
        cleaned = OrderedDict()
        dropped = []
        for k, v in state_dict.items():
            if any(k == s or k.endswith("." + s) for s in drop_suffixes):
                dropped.append(k)
                continue
            cleaned[k] = v
        if len(dropped) > 0:
            print(f"[load] removed profiling keys: {len(dropped)}")
        return cleaned

    @staticmethod
    def _print_key_summary(tag: str, keys, head: int = 10):
        n = len(keys)
        if n == 0:
            return
        print(f"[{tag}] {n} keys")
        preview = list(keys[:head])
        for k in preview:
            print(f"  - {k}")
        if n > head:
            print(f"  ... ({n - head} more)")

    def save_model(self, iter_, epoch=None, val_psnr=None):
        if val_psnr is None:
            name = f'DPL-Net_1e3_train_{iter_}iter.ckpt'
        else:
            psnr_tag = self._fmt_psnr_for_fname(val_psnr)
            name = f'DPL-Net_1e3_train_psnr{psnr_tag}_{iter_}iter.ckpt'
        f = os.path.join(self.save_path, name)


        if getattr(self, "save_full_ckpt", False):
            ckpt = {
                "iter": int(iter_),
                "epoch": int(epoch) if epoch is not None else None,
                "model": self.DPN.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr": float(self.optimizer.param_groups[0]["lr"]),
            }
            torch.save(ckpt, f)
        else:

            torch.save(self.DPN.state_dict(), f)
    def normalize_(self, image):

        dr = (self.trunc_max - self.trunc_min)
        image = (image - self.trunc_min) / dr
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image

    def denormalize_(self, image):

        drn = (self.norm_range_max - self.norm_range_min)
        image = (image - self.norm_range_min) / drn
        image = image * (self.trunc_max - self.trunc_min) + self.trunc_min
        return image

    def _resolve_data_domain(self, image):
        if self._resolved_data_domain is not None:
            return self._resolved_data_domain

        mode = str(self.data_domain).lower().strip()
        if mode in ("hu", "norm01"):
            self._resolved_data_domain = mode
        else:
            mn = float(image.detach().min().item())
            mx = float(image.detach().max().item())
            if mn >= -0.2 and mx <= 1.2:
                self._resolved_data_domain = "norm01"
            else:
                self._resolved_data_domain = "hu"
            print(f"[DataDomain][auto] detected={self._resolved_data_domain} from min={mn:.4f} max={mx:.4f}")

        print(f"[DataDomain] using '{self._resolved_data_domain}'")
        return self._resolved_data_domain

    def _to_model_domain(self, image):
        dom = self._resolve_data_domain(image)
        if dom == "norm01":
            return torch.clamp(image, 0.0, 1.0)
        return self.normalize_(self.trunc(image))

    def _to_model_input(self, image):
        dom = self._resolve_data_domain(image)
        if dom == "norm01":
            mn = image.amin(dim=(-2, -1), keepdim=True)
            mx = image.amax(dim=(-2, -1), keepdim=True)
            image = (image - mn) / (mx - mn + 1e-8)
            return torch.clamp(image, 0.0, 1.0)
        return self.normalize_(self.trunc(image))

    def _to_model_target(self, image):
        dom = self._resolve_data_domain(image)
        if dom == "norm01":
            if self.normalize_target:
                return self._to_model_input(image)
            return image
        return self.normalize_(self.trunc(image))

    def _to_target_domain(self, image):
        return self._to_model_target(image)

    def _from_model_domain(self, image):
        dom = self._resolved_data_domain if self._resolved_data_domain is not None else "hu"
        if dom == "norm01":
            return torch.clamp(image, 0.0, 1.0)
        return self.trunc(self.denormalize_(image))

    def _from_target_domain(self, image):
        dom = self._resolved_data_domain if self._resolved_data_domain is not None else "hu"
        if dom == "norm01" and (not self.normalize_target):
            return image
        return self._from_model_domain(image)

    def _from_output_domain(self, image):
        dom = self._resolved_data_domain if self._resolved_data_domain is not None else "hu"
        if dom == "norm01" and (not self.normalize_target):
            return image
        return self._from_model_domain(image)

    def _metric_data_range(self):
        dom = self._resolved_data_domain if self._resolved_data_domain is not None else "hu"
        return 1.0 if dom == "norm01" else float(self.trunc_max - self.trunc_min)

    def _viz_limits(self):
        dom = self._resolved_data_domain if self._resolved_data_domain is not None else "hu"
        if dom == "norm01":
            return 0.0, 1.0
        return float(self.trunc_min), float(self.trunc_max)

    def load_model(self, iter_):
        f = os.path.join(self.save_path, f'DPL-Net_1e3_train_{iter_}iter.ckpt')

        if not os.path.exists(f):
            same_iter = glob(os.path.join(self.save_path, f"DPL-Net_1e3_train*_{iter_}iter.ckpt"))
            if len(same_iter) > 0:
                f = sorted(same_iter)[-1]
                print(f"[Load] exact legacy ckpt not found -> using iter-matched: {f}")
            else:
                cand = glob(os.path.join(self.save_path, "DPL-Net_1e3_train*iter.ckpt"))
                if len(cand) == 0:
                    raise FileNotFoundError(f"No ckpt under {self.save_path}")
                def it(p):
                    m = re.search(r"_([0-9]+)iter\.ckpt$", os.path.basename(p))
                    return int(m.group(1)) if m else -1
                latest = sorted(cand, key=it)[-1]
                print(f"[Load] {f} not found -> using latest: {latest}")
                f = latest

        ckpt = torch.load(f, map_location=self.device)


        if isinstance(ckpt, dict) and ("model" in ckpt or "state_dict" in ckpt):
            state = ckpt["model"] if "model" in ckpt else ckpt["state_dict"]


            if self.multi_gpu and isinstance(self.DPN, nn.DataParallel):

                self.DPN.load_state_dict(state, strict=True)
            else:

                if any(k.startswith("module.") for k in state.keys()):
                    new_state = OrderedDict()
                    for k, v in state.items():
                        new_state[k.replace("module.", "", 1)] = v
                    state = new_state
                self.DPN.load_state_dict(state, strict=True)


            if getattr(self, "resume_optimizer", False) and "optimizer" in ckpt:
                try:
                    self.optimizer.load_state_dict(ckpt["optimizer"])
                except Exception as e:
                    print(f"[RESUME][WARN] optimizer load failed: {e}")

            loaded_iter = int(ckpt.get("iter", iter_))
            loaded_epoch = ckpt.get("epoch", None)
            return loaded_iter, loaded_epoch


        state = self._strip_profile_keys(ckpt)
        if any(k.startswith("module.") for k in state.keys()):
            new_state = OrderedDict()
            for k, v in state.items():
                new_state[k.replace("module.", "", 1)] = v
            state = new_state


        missing, unexpected = self.DPN.load_state_dict(state, strict=False)
        print(f"[load_model] missing={len(missing)} unexpected={len(unexpected)}")
        self._print_key_summary("load_model missing", missing)
        self._print_key_summary("load_model unexpected", unexpected)
        return int(iter_), None

    def load_model_from_path(self, ckpt_path: str):
        if not ckpt_path:
            raise ValueError("ckpt_path is empty")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)
        if isinstance(ckpt, dict) and ("model" in ckpt or "state_dict" in ckpt):
            state = ckpt["model"] if "model" in ckpt else ckpt["state_dict"]
        else:
            state = ckpt
        state = self._strip_profile_keys(state)

        if any(k.startswith("module.") for k in state.keys()):
            new_state = OrderedDict()
            for k, v in state.items():
                new_state[k.replace("module.", "", 1)] = v
            state = new_state

        missing, unexpected = self.DPN.load_state_dict(state, strict=False)
        print(f"[load_model_from_path] path={ckpt_path}")
        print(f"[load_model_from_path] missing={len(missing)} unexpected={len(unexpected)}")
        self._print_key_summary("load_model_from_path missing", missing)
        self._print_key_summary("load_model_from_path unexpected", unexpected)



    def lr_decay(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.5


    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat


    @staticmethod
    def _fmt_psnr_for_fname(psnr: float) -> str:
        try:
            v = float(psnr)
        except Exception:
            v = 0.0
        return f"{v:.2f}".replace("-", "m")
    @staticmethod
    def _fmt_ssim_for_fname(ssim: float) -> str:
        try:
            v = float(ssim)
        except Exception:
            v = 0.0
        return f"{v:.4f}".replace("-", "m")

    def _pred_to_uint8(self, pred2d: np.ndarray) -> np.ndarray:

        x = pred2d.astype(np.float32)
        dom = self._resolved_data_domain if self._resolved_data_domain is not None else "hu"
        if dom == "norm01":
            y = np.clip(x, 0.0, 1.0)
            return (y * 255.0).astype(np.uint8)
        dr = float(self.trunc_max - self.trunc_min)
        if dr < 1e-8:
            return np.zeros_like(x, dtype=np.uint8)
        y = (x - float(self.trunc_min)) / dr
        y = np.clip(y, 0.0, 1.0)
        return (y * 255.0).astype(np.uint8)

    def _pred_to_uint8_window(self, img_hu: np.ndarray, wl: float, ww: float) -> np.ndarray:
        x = img_hu.astype(np.float32)
        lo = wl - ww/2.0
        hi = wl + ww/2.0
        y = (x - lo) / (hi - lo + 1e-8)
        y = np.clip(y, 0.0, 1.0)
        return (y * 255.0).astype(np.uint8)

    def save_fig(self, x, y, pred, fig_name, original_result, pred_result):

        if torch.is_tensor(x):
            x = x.detach().cpu()
        if torch.is_tensor(y):
            y = y.detach().cpu()
        if torch.is_tensor(pred):
            pred = pred.detach().cpu()


        if x.ndim == 4: x = x[0, 0]
        elif x.ndim == 3: x = x[0]

        if y.ndim == 4: y = y[0, 0]
        elif y.ndim == 3: y = y[0]

        if pred.ndim == 4: pred = pred[0, 0]
        elif pred.ndim == 3: pred = pred[0]

        x, y, pred = x.numpy(), y.numpy(), pred.numpy()

        vmin, vmax = self._viz_limits()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
        ax[0].set_title('Quarter-dose', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(
            original_result[0], original_result[1], original_result[2]), fontsize=20)

        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
        ax[1].set_title('Result', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(
            pred_result[0], pred_result[1], pred_result[2]), fontsize=20)

        ax[2].imshow(y, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
        ax[2].set_title('Full-dose', fontsize=30)


        psnr_tag = self._fmt_psnr_for_fname(pred_result[0])
        out_name = f"result_{fig_name}_psnr{psnr_tag}.png"
        f.savefig(os.path.join(self.save_path, 'DPL-Net1e3trainfig', out_name))
        plt.close()

    @staticmethod
    def _unpack_batch(batch):
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 3:
                return batch[0], batch[1], batch[2]
            if len(batch) == 2:
                return batch[0], batch[1], None
        return batch, None, None

    def _get_pred_dose_log(self):
        m = _unwrap_model(self.DPN)
        fusion = getattr(m, "fusion", None)
        if fusion is None:
            return None
        return getattr(fusion, "last_pred_dose", None)



    def train(self):
        total_iters = 0
        start_time = time.time()
        start_epoch = 1
        print(
            f"[TrainConfig] data_domain={self.data_domain} normalize_target={self.normalize_target} "
            f"use_dose_plug={self.use_dose_plug} dose_lambda={self.dose_lambda}"
        )

        eval_interval = 200
        best_val_psnr = float("-inf")
        train_losses = []
        train_loss_window = []
        train_recon_loss_window = []
        train_dose_loss_window = []

        if getattr(self, "resume_iters", 0) and self.resume_iters > 0:
            print(f"[RESUME] loading ckpt at iter={self.resume_iters}")
            loaded_iter, loaded_epoch = self.load_model(self.resume_iters)
            total_iters = loaded_iter

            if loaded_epoch is not None:
                start_epoch = int(loaded_epoch) + 1
            else:
                start_epoch = (total_iters // max(1, len(self.data_loader))) + 1

            if not getattr(self, "resume_optimizer", False):
                decay_count = total_iters // max(1, self.decay_iters)
                for _ in range(decay_count):
                    self.lr_decay()

            print(f"[RESUME] resume from iter={total_iters}, start_epoch={start_epoch}, lr={self.optimizer.param_groups[0]['lr']:.3e}")
        elif isinstance(self.pretrained_ckpt, str) and self.pretrained_ckpt.strip():
            self.load_model_from_path(self.pretrained_ckpt.strip())
            print(f"[FINETUNE] initialized from pretrained_ckpt={self.pretrained_ckpt.strip()}")

        for epoch in range(start_epoch, self.num_epochs + 1):
            self.DPN.train(True)

            for iter_, batch in enumerate(self.data_loader):
                total_iters += 1
                x, y, dose_t = self._unpack_batch(batch)

                x = x.float().to(self.device)
                y = y.float().to(self.device)
                orig_batch = x.shape[0]
                if dose_t is not None:
                    dose_t = torch.as_tensor(dose_t, dtype=torch.float32, device=self.device).view(orig_batch)

                x = self._to_model_input(x)
                y = self._to_target_domain(y)

                x = _ensure_4d(x)
                y = _ensure_4d(y)

                if getattr(self, "patch_size", None) is not None and self.patch_size > 0:
                    x = x.view(-1, 1, self.patch_size, self.patch_size)
                    y = y.view(-1, 1, self.patch_size, self.patch_size)
                if dose_t is not None and x.shape[0] != orig_batch:
                    rep = int(x.shape[0] // max(1, orig_batch))
                    dose_t = dose_t.repeat_interleave(rep)

                pred = self.DPN(x)
                pred_dose = self._get_pred_dose_log()
                loss_recon = self.criterion(pred, y)
                if self.use_dose_plug and pred_dose is not None and dose_t is not None:
                    loss_dose = self.criterion_dose(pred_dose, dose_t)
                    loss = loss_recon + (self.dose_lambda * loss_dose)
                else:
                    loss_dose = torch.tensor(0.0, device=self.device)
                    loss = loss_recon

                self.DPN.zero_grad()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_item = float(loss.item())
                loss_recon_item = float(loss_recon.item())
                loss_dose_item = float(loss_dose.item())
                train_losses.append(loss_item)
                train_loss_window.append(loss_item)
                train_recon_loss_window.append(loss_recon_item)
                train_dose_loss_window.append(loss_dose_item)
                writer.add_scalar('train_loss_total', loss_item, global_step=total_iters)
                writer.add_scalar('train_loss_recon', loss_recon_item, global_step=total_iters)
                writer.add_scalar('train_loss_dose', loss_dose_item, global_step=total_iters)

                if total_iters % self.print_iters == 0:
                    print(
                        "STEP [{}], EPOCH [{}/{}], ITER [{}/{}] LOSS(total): {:.8f}, LOSS(recon): {:.8f}, LOSS(dose): {:.8f}, TIME: {:.1f}s".format(
                            total_iters, epoch, self.num_epochs, iter_ + 1, len(self.data_loader),
                            loss_item, loss_recon_item, loss_dose_item, time.time() - start_time
                        )
                    )

                if total_iters % self.decay_iters == 0:
                    self.lr_decay()

                if total_iters % eval_interval == 0:
                    self.DPN.eval()

                    val_losses = []
                    val_recon_losses = []
                    val_dose_losses = []
                    val_psnrs = []

                    with torch.no_grad():
                        for vbatch in self.data_Valloader:
                            vx, vy, vdose_t = self._unpack_batch(vbatch)
                            vx = vx.float().to(self.device)
                            vy = vy.float().to(self.device)
                            vorig_batch = vx.shape[0]
                            if vdose_t is not None:
                                vdose_t = torch.as_tensor(vdose_t, dtype=torch.float32, device=self.device).view(vorig_batch)

                            vx = self._to_model_input(vx)
                            vy = self._to_target_domain(vy)

                            vx = _ensure_4d(vx)
                            vy = _ensure_4d(vy)

                            if self.patch_size and self.patch_size > 0:
                                vx = vx.view(-1, 1, self.patch_size, self.patch_size)
                                vy = vy.view(-1, 1, self.patch_size, self.patch_size)
                            if vdose_t is not None and vx.shape[0] != vorig_batch:
                                rep = int(vx.shape[0] // max(1, vorig_batch))
                                vdose_t = vdose_t.repeat_interleave(rep)

                            vpred = self.DPN(vx)
                            vpred_dose = self._get_pred_dose_log()
                            vloss_recon = self.criterion(vpred, vy)
                            if self.use_dose_plug and vpred_dose is not None and vdose_t is not None:
                                vloss_dose = self.criterion_dose(vpred_dose, vdose_t)
                                vloss = vloss_recon + (self.dose_lambda * vloss_dose)
                            else:
                                vloss_dose = torch.tensor(0.0, device=self.device)
                                vloss = vloss_recon
                            val_losses.append(float(vloss.item()))
                            val_recon_losses.append(float(vloss_recon.item()))
                            val_dose_losses.append(float(vloss_dose.item()))

                            x2 = self._from_model_domain(vx[0, 0].detach().cpu())
                            y2 = self._from_target_domain(vy[0, 0].detach().cpu())
                            p2 = self._from_output_domain(vpred[0, 0].detach().cpu())

                            data_range = self._metric_data_range()
                            _, pred_result = compute_measure(x2, y2, p2, data_range)
                            val_psnrs.append(float(pred_result[0]))

                    mean_train_loss = float(np.mean(train_loss_window)) if len(train_loss_window) > 0 else float('nan')
                    mean_train_recon = float(np.mean(train_recon_loss_window)) if len(train_recon_loss_window) > 0 else float('nan')
                    mean_train_dose = float(np.mean(train_dose_loss_window)) if len(train_dose_loss_window) > 0 else float('nan')
                    mean_val_loss = float(np.mean(val_losses)) if len(val_losses) > 0 else float('nan')
                    mean_val_recon = float(np.mean(val_recon_losses)) if len(val_recon_losses) > 0 else float('nan')
                    mean_val_dose = float(np.mean(val_dose_losses)) if len(val_dose_losses) > 0 else float('nan')
                    mean_val_psnr = float(np.mean(val_psnrs)) if len(val_psnrs) > 0 else float('nan')

                    print(
                        f"[EVAL] iter={total_iters} "
                        f"train_total={mean_train_loss:.6f} train_recon={mean_train_recon:.6f} train_dose={mean_train_dose:.6f} "
                        f"val_total={mean_val_loss:.6f} val_recon={mean_val_recon:.6f} val_dose={mean_val_dose:.6f} "
                        f"val_psnr={mean_val_psnr:.4f}"
                    )

                    writer.add_scalar('train_loss_total_eval', mean_train_loss, global_step=total_iters)
                    writer.add_scalar('train_loss_recon_eval', mean_train_recon, global_step=total_iters)
                    writer.add_scalar('train_loss_dose_eval', mean_train_dose, global_step=total_iters)
                    writer.add_scalar('val_loss_total', mean_val_loss, global_step=total_iters)
                    writer.add_scalar('val_loss_recon', mean_val_recon, global_step=total_iters)
                    writer.add_scalar('val_loss_dose', mean_val_dose, global_step=total_iters)
                    writer.add_scalar('val_psnr', mean_val_psnr, global_step=total_iters)

                    if mean_val_psnr > best_val_psnr:
                        prev = best_val_psnr
                        best_val_psnr = mean_val_psnr
                        self.save_model(total_iters, epoch=epoch, val_psnr=mean_val_psnr)
                        np.save(
                            os.path.join(self.save_path, f'DPL-Net_1e3_train_loss_{total_iters}_iter.npy'),
                            np.array(train_losses, dtype=np.float32)
                        )
                        print(f"[CKPT] saved (best val psnr improved): {prev:.4f} -> {best_val_psnr:.4f}")

                    train_loss_window = []
                    train_recon_loss_window = []
                    train_dose_loss_window = []
                    self.DPN.train(True)

            self.save_model(total_iters, epoch=epoch)
            print(f"[CKPT] saved (end of epoch): epoch={epoch}, iter={total_iters}")

    def test(self):
        pred_psnr_std = []
        pred_ssim_std = []
        pred_rmse_std = []
        print(f"[TestConfig] data_domain={self.data_domain} normalize_target={self.normalize_target}")

        del self.DPN
        self.DPN = DPN(use_dose_plug=self.use_dose_plug).to(self.device)
        self.load_model_from_path(self.ckpt_path)

        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0


        os.makedirs(os.path.join(self.save_path, 'DPL-Net1e3trainfig'), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'val_metrics'), exist_ok=True)

        per_case_lines = []
        n_total = len(self.data_loader)
        ds = getattr(self.data_loader, "dataset", None)




        with torch.no_grad():
            for i, batch in enumerate(self.data_loader):
                x, y, _ = self._unpack_batch(batch)
                x = x.float().to(self.device)
                y = y.float().to(self.device)

                x = self._to_model_input(x)
                y = self._to_target_domain(y)

                x = _ensure_4d(x)
                y = _ensure_4d(y)

                if self.patch_size and self.patch_size > 0:
                    x = x.view(-1, 1, self.patch_size, self.patch_size)
                    y = y.view(-1, 1, self.patch_size, self.patch_size)


                pred = self.DPN(x)




                x2 = self._from_model_domain(x[0,0].detach().cpu())
                y2 = self._from_target_domain(y[0,0].detach().cpu())
                p2 = self._from_output_domain(pred[0,0].detach().cpu())










                data_range = self._metric_data_range()



                original_result, pred_result = compute_measure(x2, y2, p2, data_range)

                ori_psnr_avg += original_result[0]
                ori_ssim_avg += original_result[1]
                ori_rmse_avg += original_result[2]

                pred_psnr_avg += pred_result[0]
                pred_ssim_avg += pred_result[1]
                pred_rmse_avg += pred_result[2]

                pred_psnr_std.append(pred_result[0])
                pred_ssim_std.append(pred_result[1])
                pred_rmse_std.append(pred_result[2])









                if self.save_recon:
                    root = self.recon_root.strip() if isinstance(self.recon_root, str) else ""
                    if root == "":
                        root = os.path.join(self.save_path, "recon_outputs")

                    dose_tag = f"dose{self.dose}" if self.dose is not None else "dose_unknown"
                    out_dir = os.path.join(root, dose_tag, f"ckpt_{self.test_tag}")
                    os.makedirs(out_dir, exist_ok=True)

                    case_id = str(i)
                    try:
                        ds2 = getattr(self.data_loader, "dataset", None)
                        if ds2 is not None and hasattr(ds2, "pairs"):
                            lf_path, _ = ds2.pairs[i]
                            bn = os.path.basename(str(lf_path))
                            bn = re.sub(r"_rec_\d+\.npy$", "", bn)
                            bn = re.sub(r"\.npy$", "", bn)
                            case_id = bn
                    except Exception:
                        pass

                    safe_id = re.sub(r"[^0-9a-zA-Z._-]+", "_", case_id)
                    psnr_tag = self._fmt_psnr_for_fname(pred_result[0])
                    ssim_tag = self._fmt_ssim_for_fname(pred_result[1])

                    png_path = os.path.join(out_dir, f"{safe_id}_psnr{psnr_tag}_ssim{ssim_tag}.png")

                    from PIL import Image
                    u8 = self._pred_to_uint8(p2.numpy())
                    Image.fromarray(u8, mode="L").save(png_path)












































                case_id = str(i)
                try:
                    if ds is not None and hasattr(ds, "pairs"):
                        lf_path, _hf_path = ds.pairs[i]
                        case_id = os.path.basename(str(lf_path))
                        case_id = re.sub(r"_rec_\d+\.npy$", "", case_id)
                        case_id = re.sub(r"\.npy$", "", case_id)
                except Exception:
                    pass

                per_case_lines.append(
                    f"{case_id}\t"
                    f"ori_psnr={original_result[0]:.6f}\tori_ssim={original_result[1]:.6f}\tori_rmse={original_result[2]:.6f}\t"
                    f"pred_psnr={pred_result[0]:.6f}\tpred_ssim={pred_result[1]:.6f}\tpred_rmse={pred_result[2]:.6f}"
                )

                printProgressBar(i, len(self.data_loader),
                                prefix="Compute measurements ..",
                                suffix='Complete', length=25)

        print('\n')


        ori_psnr_mean = ori_psnr_avg / n_total
        ori_ssim_mean = ori_ssim_avg / n_total
        ori_rmse_mean = ori_rmse_avg / n_total

        pred_psnr_mean = pred_psnr_avg / n_total
        pred_ssim_mean = pred_ssim_avg / n_total
        pred_rmse_mean = pred_rmse_avg / n_total

        pred_psnr_s = float(np.std(pred_psnr_std))
        pred_ssim_s = float(np.std(pred_ssim_std))
        pred_rmse_s = float(np.std(pred_rmse_std))

        print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(
            ori_psnr_mean, ori_ssim_mean, ori_rmse_mean))
        print('\n')
        print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(
            pred_psnr_mean, pred_ssim_mean, pred_rmse_mean))
        print('Predictions === \nPSNR std: {:.4f} \nSSIM std: {:.4f} \nRMSE std: {:.4f}'.format(
            pred_psnr_s, pred_ssim_s, pred_rmse_s))

        psnr_tag = self._fmt_psnr_for_fname(pred_psnr_mean)
        metrics_name = f"val_ckpt_{self.test_tag}_psnr{psnr_tag}.txt"
        metrics_path = os.path.join(self.save_path, 'val_metrics', metrics_name)

        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(f"ckpt_path={self.ckpt_path}\n")
            f.write(f"n={n_total}\n")
            f.write(f"norm_range_min={self.norm_range_min}\n")
            f.write(f"norm_range_max={self.norm_range_max}\n")
            f.write(f"trunc_min={self.trunc_min}\n")
            f.write(f"trunc_max={self.trunc_max}\n")

            f.write("\n[MEAN]\n")
            f.write(f"ori_psnr_mean={ori_psnr_mean:.6f}\n")
            f.write(f"ori_ssim_mean={ori_ssim_mean:.6f}\n")
            f.write(f"ori_rmse_mean={ori_rmse_mean:.6f}\n")
            f.write(f"pred_psnr_mean={pred_psnr_mean:.6f}\n")
            f.write(f"pred_ssim_mean={pred_ssim_mean:.6f}\n")
            f.write(f"pred_rmse_mean={pred_rmse_mean:.6f}\n")

            f.write("\n[STD]\n")
            f.write(f"pred_psnr_std={pred_psnr_s:.6f}\n")
            f.write(f"pred_ssim_std={pred_ssim_s:.6f}\n")
            f.write(f"pred_rmse_std={pred_rmse_s:.6f}\n")

            f.write("\n[PER_CASE]\n")
            for line in per_case_lines:
                f.write(line + "\n")

        print(f"[OK] saved val metrics -> {metrics_path}")
