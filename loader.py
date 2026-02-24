import os
from glob import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch
import random
import re


def _squeeze_2d(arr: np.ndarray) -> np.ndarray:

    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    return arr

def _key_from_name(path: str, suffix: str) -> str:
    base = os.path.basename(path)
    assert base.endswith(suffix), (base, suffix)
    return base[:-len(suffix)]

def _pair_by_key(input_paths, target_paths, input_suffix, target_suffix):
    in_map = { _key_from_name(p, input_suffix): p for p in input_paths }
    tg_map = { _key_from_name(t, target_suffix): t for t in target_paths }
    keys = sorted(set(in_map.keys()) & set(tg_map.keys()))
    if len(keys) == 0:
        raise RuntimeError("No matched (input,target) pairs. Check suffix/glob paths.")
    return [in_map[k] for k in keys], [tg_map[k] for k in keys]

class PetPairNPY(Dataset):
    def __init__(self, root, split="train", dose=1, rec_id=None, patch_size=256, patch_n=2, return_dose=False):
        self.root = Path(root)
        self.split = split
        self.dose = int(dose)
        self.rec_id = None if rec_id is None else int(rec_id)
        self.patch_size = patch_size
        self.patch_n = patch_n
        self.is_train = (split == "train")
        if self.is_train:
            assert self.patch_size is not None
            assert self.patch_n is not None
        else:

            if self.patch_n is None:
                self.patch_n = 1
            if self.patch_size is None:
                self.patch_size = 256

        lf_dir = self.root / split / "input"
        hf_dir = self.root / split / "original"

        rid = self.rec_id if self.rec_id is not None else self.dose
        if self.rec_id is None:
            lf_files = list(lf_dir.glob("*_rec_*.npy"))
        else:
            lf_files = list(lf_dir.glob(f"*_rec_{rid}.npy"))
        hf_files = list(hf_dir.glob("*_image.npy"))

        hf_map = {p.stem.replace("_image", ""): p for p in hf_files}
        self.pairs = []
        rec_ids = set()

        if self.rec_id is None:
            rec_pat = re.compile(r"(.+)_rec_(\d+)$")
            for lf in lf_files:
                m = rec_pat.match(lf.stem)
                if m is None:
                    continue
                key = m.group(1)
                rec_ids.add(int(m.group(2)))
                hf = hf_map.get(key, None)
                if hf is not None:
                    self.pairs.append((lf, hf))

            self.pairs = sorted(
                self.pairs,
                key=lambda p: (
                    re.sub(r"_rec_\d+$", "", p[0].stem),
                    int(re.search(r"_rec_(\d+)$", p[0].stem).group(1))
                )
            )
            rec_info = f"all_rec_ids={sorted(rec_ids)}"
        else:
            lf_map = {p.stem.replace(f"_rec_{rid}", ""): p for p in lf_files}
            keys = sorted(set(lf_map.keys()) & set(hf_map.keys()))
            self.pairs = [(lf_map[k], hf_map[k]) for k in keys]
            rec_info = f"rec_id={rid}"

        print(f"[PetPairNPY] split={split} {rec_info} lf={len(lf_files)} hf={len(hf_files)} pairs={len(self.pairs)}")

        if len(self.pairs) == 0:
            print("  lf sample:", [p.name for p in lf_files[:3]])
            print("  hf sample:", [p.name for p in hf_files[:3]])
            raise RuntimeError("No pairs found. Check root/split/fname rules.")
    def _rand_crop(self, img, top, left, ps):
        return img[top:top+ps, left:left+ps]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lf_path, hf_path = self.pairs[idx]
        lf = np.load(lf_path).astype(np.float32)
        hf = np.load(hf_path).astype(np.float32)
        if lf.ndim == 3 and lf.shape[0] == 1: lf = lf[0]
        if hf.ndim == 3 and hf.shape[0] == 1: hf = hf[0]
        dose_t = None
        if self.return_dose:
            m = re.search(r"_rec_(\d+)\.npy$", str(lf_path))
            dose_t = np.float32(int(m.group(1))) if m else np.float32(self.dose)


        H, W = lf.shape
        ps = self.patch_size


        if (ps is None) or (int(ps) <= 0):
            if self.return_dose:
                return torch.from_numpy(lf), torch.from_numpy(hf), torch.tensor(dose_t, dtype=torch.float32)
            return torch.from_numpy(lf), torch.from_numpy(hf)

        ps = int(ps)
        if H < ps or W < ps:
            raise RuntimeError(f"[BAD PATCH] idx={idx} HxW={H}x{W} < ps={ps} | {lf_path}")


        if self.is_train:
            patches_lf, patches_hf = [], []
            for _ in range(self.patch_n):
                top = random.randint(0, H - ps)
                left = random.randint(0, W - ps)
                patches_lf.append(self._rand_crop(lf, top, left, ps))
                patches_hf.append(self._rand_crop(hf, top, left, ps))
            lf_out = np.stack(patches_lf, axis=0)
            hf_out = np.stack(patches_hf, axis=0)
            if self.return_dose:
                return torch.from_numpy(lf_out), torch.from_numpy(hf_out), torch.tensor(dose_t, dtype=torch.float32)
            return torch.from_numpy(lf_out), torch.from_numpy(hf_out)
        else:
            top = (H - ps) // 2
            left = (W - ps) // 2
            lf_out = self._rand_crop(lf, top, left, ps)
            hf_out = self._rand_crop(hf, top, left, ps)
            if self.return_dose:
                return torch.from_numpy(lf_out), torch.from_numpy(hf_out), torch.tensor(dose_t, dtype=torch.float32)
            return torch.from_numpy(lf_out), torch.from_numpy(hf_out)


class ct_dataset(Dataset):
    def __init__(self, mode, load_mode, saved_path, test_patient, patch_n=None, patch_size=None, transform=None):
        assert mode in ['train', 'test'], "mode is 'train' or 'test'"
        assert load_mode in [0,1], "load_mode is 0 or 1"

        input_path = sorted(glob(os.path.join(saved_path, '*_input.npy')))
        target_path = sorted(glob(os.path.join(saved_path, '*_target.npy')))
        self.load_mode = load_mode
        self.patch_n = patch_n
        self.patch_size = patch_size
        self.transform = transform

        if mode == 'train':
            input_ = [f for f in input_path if test_patient not in f]
            target_ = [f for f in target_path if test_patient not in f]
            if load_mode == 0:
                self.input_ = input_
                self.target_ = target_
            else:
                self.input_ = [np.load(f) for f in input_]
                self.target_ = [np.load(f) for f in target_]
        else:
            input_ = [f for f in input_path if test_patient in f]
            target_ = [f for f in target_path if test_patient in f]
            if load_mode == 0:
                self.input_ = input_
                self.target_ = target_
            else:
                self.input_ = [np.load(f) for f in input_]
                self.target_ = [np.load(f) for f in target_]

    def __len__(self):
        return len(self.target_)

    def __getitem__(self, idx):
        input_img, target_img = self.input_[idx], self.target_[idx]
        if self.load_mode == 0:
            input_img, target_img = np.load(input_img), np.load(target_img)

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        if self.patch_size:
            input_patches, target_patches = get_patch(input_img,
                                                      target_img,
                                                      self.patch_n,
                                                      self.patch_size)
            return (input_patches, target_patches)
        else:
            return (input_img, target_img)

class ct_Valdataset(Dataset):
    def __init__(self, mode, load_mode, saved_path, test_patient, patch_n=None, patch_size=None, transform=None):
        assert mode in ['train', 'test'], "mode is 'train' or 'test'"
        assert load_mode in [0,1], "load_mode is 0 or 1"

        input_path = sorted(glob(os.path.join(saved_path, '*_input.npy')))
        target_path = sorted(glob(os.path.join(saved_path, '*_target.npy')))
        self.load_mode = load_mode
        self.patch_n = patch_n
        self.patch_size = patch_size
        self.transform = transform
        input_ = [f for f in input_path if test_patient in f]
        target_ = [f for f in target_path if test_patient in f]
        if load_mode == 0:
            self.input_ = input_
            self.target_ = target_
        else:
            self.input_ = [np.load(f) for f in input_]
            self.target_ = [np.load(f) for f in target_]

    def __len__(self):
        return len(self.target_)

    def __getitem__(self, idx):
        input_img, target_img = self.input_[idx], self.target_[idx]
        if self.load_mode == 0:
            input_img, target_img = np.load(input_img), np.load(target_img)

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        if self.patch_size:
            input_patches, target_patches = get_patch(input_img,
                                                      target_img,
                                                      self.patch_n,
                                                      self.patch_size)
            return (input_patches, target_patches)
        else:
            return (input_img, target_img)


def get_patch(full_input_img, full_target_img, patch_n, patch_size):
    assert full_input_img.shape == full_target_img.shape
    patch_input_imgs = []
    patch_target_imgs = []
    h, w = full_input_img.shape
    new_h, new_w = patch_size, patch_size
    for _ in range(patch_n):
        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)
        patch_input_img = full_input_img[top:top+new_h, left:left+new_w]
        patch_target_img = full_target_img[top:top+new_h, left:left+new_w]
        patch_input_imgs.append(patch_input_img)
        patch_target_imgs.append(patch_target_img)
    return np.array(patch_input_imgs), np.array(patch_target_imgs)





def get_loader(mode='train', load_mode=0,
               saved_path=None, test_patient='049',
               patch_n=None, patch_size=None, rec_id=None,
               transform=None, batch_size=64, num_workers=6,
               train_random_ct_dose=False, train_ct_doses=None,
               return_dose=False):


    split = "train"
    ct_train_root = os.path.join(saved_path, split, "input_npy")
    ct_gt_dir = os.path.join(ct_train_root, "LD_100")
    ct_ld_dirs_any = [p for p in glob(os.path.join(ct_train_root, "LD_*")) if os.path.isdir(p)]

    pet_train_dir = os.path.join(saved_path, split, "input")

    ct_gt_dir_flat = os.path.join(saved_path, "LD_100")
    ct_ld_dirs_flat_any = [p for p in glob(os.path.join(saved_path, "LD_*")) if os.path.isdir(p)]

    if os.path.isdir(pet_train_dir):
        dataset_ = PetPairNPY(saved_path, split=split, dose=1, rec_id=rec_id,
                              patch_size=patch_size, patch_n=patch_n, return_dose=return_dose)

    elif os.path.isdir(ct_gt_dir) and (len(ct_ld_dirs_any) > 0):

        dataset_ = CTLowDosePairNPY(
            ct_train_root, split=split, dose=None, test_patient=None,
            patch_size=patch_size, patch_n=patch_n,
            random_dose=train_random_ct_dose, dose_list=train_ct_doses, return_dose=return_dose
        )

    elif os.path.isdir(ct_gt_dir_flat) and (len(ct_ld_dirs_flat_any) > 0):
        dataset_ = CTLowDosePairNPY(
            saved_path, split=split, dose=None, test_patient=test_patient,
            patch_size=patch_size, patch_n=patch_n,
            random_dose=train_random_ct_dose, dose_list=train_ct_doses, return_dose=return_dose
        )

    else:
        dataset_ = ct_dataset(mode, load_mode, saved_path, test_patient, patch_n, patch_size, transform)

    return DataLoader(dataset_, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True, drop_last=True)
def get_Valloader(mode='train', load_mode=0,
                  saved_path=None, test_patient='049',
                  patch_n=None, patch_size=None, rec_id=None,
                  transform=None, batch_size=64, num_workers=6,
                  val_ct_doses=None, return_dose=False):


    ct_val_root = os.path.join(saved_path, "val", "input_npy")
    ct_gt_dir = os.path.join(ct_val_root, "LD_100")

    pet_val_dir = os.path.join(saved_path, "val", "input")

    ct_gt_dir_flat = os.path.join(saved_path, "LD_100")
    if val_ct_doses is not None and len(val_ct_doses) > 0:
        req_doses = [int(d) for d in val_ct_doses]
    else:
        req_doses = None

    if os.path.isdir(pet_val_dir):
        dataset_ = PetPairNPY(saved_path, split="val", dose=1, rec_id=rec_id,
                              patch_size=patch_size, patch_n=patch_n, return_dose=return_dose)

    elif os.path.isdir(ct_gt_dir):

        dataset_ = CTLowDosePairNPY(ct_val_root, split="val", dose=None,
                                    test_patient=None,
                                    patch_size=patch_size, patch_n=1,
                                    random_dose=False, dose_list=req_doses, return_dose=return_dose)

    elif os.path.isdir(ct_gt_dir_flat):
        dataset_ = CTLowDosePairNPY(saved_path, split="val", dose=None,
                                    test_patient=test_patient,
                                    patch_size=patch_size, patch_n=1,
                                    random_dose=False, dose_list=req_doses, return_dose=return_dose)

    else:
        dataset_ = ct_Valdataset(mode, load_mode, saved_path, test_patient, patch_n, patch_size, transform)

    return DataLoader(dataset_, batch_size=1, shuffle=False,
                      num_workers=num_workers, pin_memory=True, drop_last=False)





























































class CTLowDosePairNPY(Dataset):
    def __init__(self, root, split="train", dose=None, test_patient="049",
                 patch_size=256, patch_n=2, random_dose=False, dose_list=None, return_dose=False):
        self.root = Path(root)
        self.split = split
        self.dose = None if dose is None else int(dose)
        self.test_patient = str(test_patient) if test_patient is not None else None
        self.patch_size = patch_size
        self.patch_n = patch_n
        self.return_dose = bool(return_dose)
        self.is_train = (split == "train")
        self.random_dose = bool(random_dose and self.is_train)

        if dose_list is None:
            self.dose_list = None
        else:
            self.dose_list = sorted({int(d) for d in dose_list})
            if len(self.dose_list) == 0:
                self.dose_list = None

        gt_dir = self.root / "LD_100"
        gt_files = sorted(list(gt_dir.rglob("*_image.npy")))
        gt_map = {}
        for f in gt_files:
            m = re.match(r"^(.+?_z\d+)", f.name)
            if m:
                gt_map[m.group(1)] = str(f)

        if len(gt_map) == 0:
            raise RuntimeError(f"No GT slices matched '*_image.npy' under: {gt_dir}")

        if self.dose_list is None:
            valid_doses = []
            for p in sorted(self.root.glob("LD_*")):
                if not p.is_dir():
                    continue
                s = p.name.replace("LD_", "")
                if s.isdigit():
                    valid_doses.append(int(s))
        else:
            valid_doses = [d for d in self.dose_list if (self.root / f"LD_{d}").is_dir()]
        if len(valid_doses) == 0:
            raise RuntimeError(
                f"Low-dose dir not found for requested doses: {self.dose_list} under {self.root}"
            )
        self.dose_list = valid_doses

        self.pairs = []
        for d in valid_doses:
            ld_dir = self.root / f"LD_{d}"
            ld_files = list(ld_dir.rglob("*.npy"))
            if len(ld_files) == 0:
                continue

            if self.random_dose:
                random.shuffle(ld_files)

            for lf in ld_files:
                m = re.match(r"^(.+?_z\d+)", lf.name)
                if not m:
                    continue
                key = m.group(1)
                if key not in gt_map:
                    continue

                patient_id = key.split("_")[0]
                if self.test_patient is not None:
                    is_test = (patient_id == self.test_patient)
                    if self.is_train and is_test:
                        continue
                    if (not self.is_train) and (not is_test):
                        continue
                self.pairs.append((str(lf), gt_map[key], int(d)))

        if self.random_dose and len(self.pairs) > 1:
            random.shuffle(self.pairs)

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No matched pairs found. root={root}, doses={self.dose_list}, split={split}"
            )

        if not self.is_train:
            self.patch_n = 1

        mode_info = "all_doses_shuffled" if self.random_dose else "fixed"
        print(f"[CTLowDosePairNPY] split={split} mode={mode_info} doses={self.dose_list} "
              f"test_patient={self.test_patient} pairs={len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def _rand_crop(self, img, top, left, ps):
        return img[top:top+ps, left:left+ps]

    def __getitem__(self, idx):
        item = self.pairs[idx]
        if len(item) == 3:
            lf_path, hf_path, dose_val = item
        else:
            lf_path, hf_path = item
            dose_val = self.dose

        lf = np.load(lf_path).astype(np.float32)
        hf = np.load(hf_path).astype(np.float32)

        if lf.ndim == 3 and lf.shape[0] == 1:
            lf = lf[0]
        if hf.ndim == 3 and hf.shape[0] == 1:
            hf = hf[0]

        if lf.shape != hf.shape:
            h = min(lf.shape[0], hf.shape[0])
            w = min(lf.shape[1], hf.shape[1])
            lf = lf[:h, :w]
            hf = hf[:h, :w]

        H, W = lf.shape
        ps = self.patch_size

        if ps is not None and ps > 0:
            if self.is_train:
                patches_lf, patches_hf = [], []
                for _ in range(self.patch_n):
                    top = random.randint(0, H - ps)
                    left = random.randint(0, W - ps)
                    patches_lf.append(self._rand_crop(lf, top, left, ps))
                    patches_hf.append(self._rand_crop(hf, top, left, ps))
                lf_out = np.stack(patches_lf, axis=0)
                hf_out = np.stack(patches_hf, axis=0)
            else:
                top = (H - ps) // 2
                left = (W - ps) // 2
                lf_out = self._rand_crop(lf, top, left, ps)
                hf_out = self._rand_crop(hf, top, left, ps)
        else:
            lf_out, hf_out = lf, hf

        if self.return_dose:
            return torch.from_numpy(lf_out), torch.from_numpy(hf_out), torch.tensor(float(dose_val), dtype=torch.float32)
        return torch.from_numpy(lf_out), torch.from_numpy(hf_out)
