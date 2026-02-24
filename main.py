






























































































































































import os
import argparse
from torch.backends import cudnn
from loader import get_loader, get_Valloader
from solver import Solver
import random
import numpy as np
import torch

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main(args):
    cudnn.benchmark = True
    seed_torch(seed=42)

    os.makedirs(args.save_path, exist_ok=True)

    if args.mode == 'train':
        train_loader = get_loader(
            mode='train',
            load_mode=args.load_mode,
            saved_path=args.saved_path,
            test_patient=args.test_patient,
            rec_id=args.rec_id,
            patch_n=args.patch_n,
            patch_size=args.patch_size,
            transform=args.transform,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_random_ct_dose=args.train_random_ct_dose,
            train_ct_doses=args.train_ct_doses,
            return_dose=args.use_dose_plug
        )

        val_loader = get_Valloader(
            mode='test',
            load_mode=args.load_mode,
            saved_path=args.saved_path,
            test_patient=args.test_patient,
            rec_id=args.rec_id,
            patch_n=args.patch_n,
            patch_size=args.patch_size,
            transform=args.transform,
            batch_size=1,
            num_workers=args.num_workers,
            val_ct_doses=args.eval_ct_doses,
            return_dose=args.use_dose_plug
        )

        solver = Solver(args, train_loader, val_loader)
        solver.train()

    elif args.mode == 'test':
        if not args.ckpt_path:
            raise ValueError("--ckpt_path is required when --mode test")
        test_loader = get_Valloader(
            mode='test',
            load_mode=args.load_mode,
            saved_path=args.saved_path,
            test_patient=args.test_patient,
            rec_id=args.rec_id,
            patch_n=args.patch_n,
            patch_size=args.patch_size,
            transform=args.transform,
            batch_size=1,
            num_workers=args.num_workers,
            val_ct_doses=args.eval_ct_doses,
            return_dose=args.use_dose_plug
        )

        solver = Solver(args, test_loader, test_loader)
        solver.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--load_mode', type=int, default=0)
    parser.add_argument('--data_path', type=str, default='./AAPM-Mayo-CT-Challenge/3mm/')
    parser.add_argument('--saved_path', type=str, default='./npy_img/')
    parser.add_argument('--save_path', type=str, default='./save/')
    parser.add_argument('--test_patient', type=str, default='L506')

    parser.add_argument('--trunc_min', type=float, default=-160.0)
    parser.add_argument('--trunc_max', type=float, default=240.0)
    parser.add_argument('--data_domain', type=str, default='auto', choices=['auto', 'hu', 'norm01'])
    parser.add_argument('--normalize_target', action='store_true')

    parser.add_argument('--transform', type=bool, default=False)

    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--print_iters', type=int, default=20)
    parser.add_argument('--decay_iters', type=int, default=3000)
    parser.add_argument('--save_iters', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--device', type=str)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--multi_gpu', action='store_true')

    parser.add_argument('--train_random_ct_dose', action='store_true')
    parser.add_argument('--train_ct_doses', type=str, default='')
    parser.add_argument('--eval_ct_doses', type=str, default='')
    parser.add_argument('--save_recon', action='store_true')
    parser.add_argument('--recon_root', type=str, default='')
    parser.add_argument('--recon_png', action='store_true')
    parser.add_argument('--resume_iters', type=int, default=0)
    parser.add_argument('--resume_latest', action='store_true')
    parser.add_argument('--resume_optimizer', action='store_true')
    parser.add_argument('--save_full_ckpt', action='store_true')
    parser.add_argument('--pretrained_ckpt', type=str, default='')
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--max_sinogram_libraries', type=int, default=0)
    parser.add_argument('--use_dose_plug', action='store_true')

    parser.add_argument('--rec_id', type=int, default=None)

    args = parser.parse_args()
    if args.mode == "test":
        args.save_recon = True
        args.recon_png = True
    if args.mode == "test" and args.ckpt_path and args.save_path == "./save/":
        args.save_path = os.path.dirname(args.ckpt_path) or "./save/"
    args.patch_n = 1
    args.patch_size = 0
    args.dose = "all"
    args.norm_range_min = 0.0
    args.norm_range_max = 1.0
    args.dose_lambda = 1.0
    if isinstance(args.train_ct_doses, str) and args.train_ct_doses.strip():
        try:
            args.train_ct_doses = [int(x.strip()) for x in args.train_ct_doses.split(',') if x.strip()]
        except ValueError as e:
            raise ValueError(f"--train_ct_doses must be comma-separated integers: {args.train_ct_doses}") from e
    else:
        args.train_ct_doses = None
    if isinstance(args.eval_ct_doses, str) and args.eval_ct_doses.strip():
        try:
            args.eval_ct_doses = [int(x.strip()) for x in args.eval_ct_doses.split(',') if x.strip()]
        except ValueError as e:
            raise ValueError(f"--eval_ct_doses must be comma-separated integers: {args.eval_ct_doses}") from e
    else:
        args.eval_ct_doses = args.train_ct_doses
    if args.mode == "train" and args.resume_latest and args.resume_iters == 0:
        import re, glob
        pat = os.path.join(args.save_path, "DPL-Net_1e3_train_*iter.ckpt")
        cands = glob.glob(pat)
        best = 0
        for p in cands:
            m = re.search(r"_([0-9]+)iter\.ckpt$", os.path.basename(p))
            if m:
                best = max(best, int(m.group(1)))
        if best > 0:
            args.resume_iters = best
            print(f"[RESUME] found latest ckpt iter={best} in {args.save_path}")
        else:
            print(f"[RESUME] no ckpt found in {args.save_path} (start fresh)")

    main(args)
