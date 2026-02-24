# DosePlug

DosePlug is a plug-in framework, and this repository is the version integrated into VisNet.

## Integration with VisNet

DosePlug is integrated into the VisNet model pipeline and can be enabled or disabled with a runtime flag:

- Enable DosePlug: add `--use_dose_plug`

## Datasets

- **CT-ICH Dataset**: https://www.physionet.org/content/ct-ich/1.3.1/
- **AutoPET Dataset**: https://www.autopet.org/fdgpetct.html

## Training

### Training with DosePlug

```bash
python main.py \
  --mode train \
  --saved_path <DATASET_ROOT> \
  --save_path <CHECKPOINT_DIR> \
  --load_mode 0 \
  --batch_size 2 \
  --num_workers 4 \
  --device cuda \
  --use_dose_plug
```

## Inference

### Inference with DosePlug

```bash
python main.py \
  --mode test \
  --saved_path <DATASET_ROOT> \
  --ckpt_path <CHECKPOINT_FILE> \
  --load_mode 0 \
  --num_workers 4 \
  --device cuda \
  --use_dose_plug \
  --recon_root <OUTPUT_DIR>
```

## Notes

- Training uses the `train` split and validation uses the `val` split.
- Checkpoints are saved at the end of every epoch.
- Normalization is fixed to the `0-1` range in the current code setup.
- Inference (`--mode test`) saves reconstruction images by default.
