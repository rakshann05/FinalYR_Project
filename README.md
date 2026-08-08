# EFDN + SR4IR — Task-Driven Super-Resolution with a Lightweight Backbone

Integrates the lightweight **EFDN** super-resolution network into the **SR4IR**
task-driven recognition framework, replacing SR4IR's heavy EDSR-baseline / SwinIR
backbones. Task: semantic segmentation on PASCAL VOC2012, scale ×8.

> **Status:** code complete, **not yet trained**. This was built on a machine
> without a CUDA GPU, so all training/eval must be run on a GPU box using the
> commands below. See [`EXPERIMENTS.md`](EXPERIMENTS.md) for the run log and the
> documented integration decisions.

## Layout

```
project-root/
├── configs/
│   ├── phase0_baseline_edsr_seg_x8.yaml   # SR4IR EDSR-baseline reproduction (Phase 0)
│   └── phase1_efdn_seg_x8.yaml            # EFDN swapped in + EG loss (Phase 1)
├── models/
│   ├── sr4ir_original/                    # cloned SR4IR (+ our integration files, below)
│   └── efdn/                              # cloned EFDN (reference code + weights)
├── checkpoints/
│   ├── phase0_edsr_baseline_seg_x8/       # weights synced here after training
│   └── phase1_efdn_seg_x8/
├── logs/                                  # eval logs (timestamped)
├── train_sr.py / train_task.py            # training entry (wrappers over SR4IR main.py)
├── test.py                                # metrics (mIoU/LPIPS/PSNR) + --cost (params/FLOPs)
├── reparameterize.py                      # fold trained EFDN -> deploy single-conv
├── EXPERIMENTS.md
└── README.md
```

### Integration files added inside SR4IR
- `models/sr4ir_original/src/archs/common/efdn_arch.py` — EFDN as an SR4IR backbone (`name: efdn`).
- `models/sr4ir_original/src/archs/common/efdn_unitv2.py` — EFDN's EDBB + reparam math, **vendored verbatim**.
- `models/sr4ir_original/src/losses/eg.py` — EG loss (**re-implementation**, see EXPERIMENTS.md #5).
- edits to `src/losses/__init__.py` (register `EGLoss`) and
  `src/models/seg/sr4ir_seg_model.py` (wire `eg_opt` into the phase-1 SR update).

## Setup (GPU machine)

```bash
# 1) Python env (SR4IR targets torch 2.0.1 / torchvision 0.15.2, CUDA 11.7)
cd models/sr4ir_original
conda env create -f assets/environment.yaml   # or: pip install -r assets/requirements.txt
conda activate SR4IR
python src/setup.py                            # makes experiments/ tb_loggers/ datasets/
```

```bash
# 2) Dataset: PASCAL VOC2012 for segmentation
#    put VOCtrainval_11-May-2012.tar in models/sr4ir_original/datasets/ then:
python preprocess/voc/main.py                  # -> datasets/VOC
```

```bash
# 3) Phase-0 pretrained SR init (EDSR-baseline ×8)
#    download edsr_baseline_x8.pt from SR4IR's pretrained-models Google Drive
#    (assets/docs/Training.md) into:
#    models/sr4ir_original/experiments/pretrained_models/edsr_baseline_x8.pt
```

All wrapper scripts are run from the **project root** (they `cd` into SR4IR for you).

## Phase 0 — reproduce the SR4IR baseline

```bash
python train_sr.py -c configs/phase0_baseline_edsr_seg_x8.yaml --gpus 0
```
```bash
python test.py -c configs/phase0_baseline_edsr_seg_x8.yaml --gpus 0
```
Target (paper Table 1, EDSR-baseline ×8 seg): **mIoU 55.0 / LPIPS 0.380 / PSNR 23.91**.
Weights land in `checkpoints/phase0_edsr_baseline_seg_x8/`.

## Phase 1 — EFDN backbone

```bash
python train_sr.py -c configs/phase1_efdn_seg_x8.yaml --gpus 0
```
```bash
# accuracy metrics
python test.py -c configs/phase1_efdn_seg_x8.yaml --gpus 0

# deploy-time efficiency: fold the multi-branch EFDN first, then benchmark
python reparameterize.py -c configs/phase1_efdn_seg_x8.yaml \
    --in  checkpoints/phase1_efdn_seg_x8/net_sr_latest.pth \
    --out checkpoints/phase1_efdn_seg_x8/net_sr_deploy.pth
python test.py -c configs/phase1_efdn_seg_x8.yaml --cost --deploy \
    --sr-ckpt checkpoints/phase1_efdn_seg_x8/net_sr_deploy.pth
```

### Gate check (record in EXPERIMENTS.md)
Compare Phase-1 EFDN mIoU against Phase-0 EDSR and the bilinear floor
(`options/seg/002_L2T_x8.yml`, "ILR→T"). If EFDN ≥ bilinear floor → integration is
sound, proceed to Phase 2. If below → likely an integration bug (check EFDN mode,
that the SR loss reaches EFDN params, and the config), not "EFDN doesn't work".

## Optional — EFDN DIV2K ×8 pre-training (recommended, not required)
EFDN has no released ×8 weights, so Phase 1 trains from scratch by default. For a
stronger init you can pre-train EFDN on DIV2K ×8 with a standalone SR objective
and point `path.network_sr` in the Phase-1 config at the result. (Not wired as a
config yet; see EXPERIMENTS.md #7.)

## Checkpoint paths are always explicit
No script hardcodes a checkpoint path. `--sr-ckpt` / `--seg-ckpt` override the
config's `path.network_sr` / `path.network_seg`; `--resume` continues a full run.
This prevents silently evaluating a stale checkpoint.
