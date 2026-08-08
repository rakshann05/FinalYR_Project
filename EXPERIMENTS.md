# EXPERIMENTS — EFDN + SR4IR integration

Running log of every experiment: config, result, checkpoint path, deviations.

## Environment note (read first)

This repository was **built and wired end-to-end**, but training/eval were **not
run here**. The build machine is an Apple **M2, 8 GB RAM, no CUDA GPU**, with
PyTorch not installed. SR4IR is CUDA-oriented and its VOC-seg ×8 runs take ~a day
of NVIDIA-GPU time and ~tens of GB of RAM — infeasible on this host. Per the
agreed plan ("Build code only"), the numbers below are **PENDING** and must be
produced on a GPU box using the exact commands in the README. Everything needed
to run is in place; only the compute is missing.

## Result template

| Date | Phase | Config | SR backbone | Scale | mIoU | LPIPS | PSNR | Params (deploy) | FLOPs (deploy) | Checkpoint | Log | Notes |
|------|-------|--------|-------------|-------|------|-------|------|-----------------|----------------|------------|-----|-------|

## Runs

| Date | Phase | Config | SR backbone | Scale | mIoU | LPIPS | PSNR | Params (deploy) | FLOPs (deploy) | Checkpoint | Log | Notes |
|------|-------|--------|-------------|-------|------|-------|------|-----------------|----------------|------------|-----|-------|
| _pending_ | Bilinear floor (ILR→T) | `models/sr4ir_original/options/seg/002_L2T_x8.yml` | bilinear | ×8 | _pending_ | — | — | 0 | — | — | — | Sanity floor for the gate check; SR4IR paper reports mIoU ≈ 40.0 for ILR→T ×8 seg. |
| _pending_ | Phase 0 baseline | `configs/phase0_baseline_edsr_seg_x8.yaml` | EDSR-baseline | ×8 | _pending_ | _pending_ | _pending_ | ~1.5M | — | `checkpoints/phase0_edsr_baseline_seg_x8/` | `logs/` | **Target (paper Table 1): mIoU 55.0 / LPIPS 0.380 / PSNR 23.91.** |
| _pending_ | Phase 1 EFDN | `configs/phase1_efdn_seg_x8.yaml` | EFDN | ×8 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | `checkpoints/phase1_efdn_seg_x8/` | `logs/` | EG+TDP loss. Reparameterize before cost. |

### Gate check (Section 5.4) — to be filled after Phase 1

- Bilinear floor mIoU: **____**
- Phase 0 (EDSR) mIoU: **____**
- Phase 1 (EFDN) mIoU: **____**
- Verdict: Phase-1 mIoU ≥ bilinear floor? **____** → (continue to Phase 2 / debug integration)

---

## Deviations & decisions (documented per the brief's instruction)

These are the judgment calls made during integration. Flagged here so they can be
reviewed/overridden.

1. **SR4IR is monolithic; the brief's separate train scripts are wrappers.**
   SR4IR runs the SR update (pixel/EG/TDP) and the task update (seg + CQMix)
   *alternately in one training loop* (`src/models/seg/sr4ir_seg_model.py::train_one_epoch`),
   driven by a single `src/main.py`. There is no separate "train SR" then "train
   task" stage. `train_sr.py` and `train_task.py` are therefore thin, documented
   wrappers that both launch the same SR4IR run; `test.py`/`reparameterize.py`
   wrap eval and folding. All expose checkpoint paths as CLI args (brief's
   "critical rule") by injecting them into a temp config — nothing hardcoded.

2. **CQMix already exists in SR4IR.** The brief describes CQMix as living in
   `train_task.py`. In fact SR4IR already implements it as the `auxce_cqmix` loss
   in `train_one_epoch` (random 8×8 Bernoulli mask upsampled ×60, mixing SR and
   HR). It is enabled via `auxce_cqmix_opt` in the config. No new CQMix code was
   added. (Adaptive/Grad-CAM-guided CQMix remains deferred per Section 8.)

3. **EFDN integrated as a native SR4IR backbone.** Added
   `src/archs/common/efdn_arch.py` exposing `build_network(**kwargs) -> EFDN`,
   matching SR4IR's contract (`forward(LR[0,1]) -> SR[0,1]`, `scale` injected).
   EFDN's forward already clamps to [0,1], consistent with how SR4IR feeds
   `img_lr = quantize(interpolate(...))` and compares to `img_hr` in [0,1].
   `network_sr.name: efdn` selects it.

4. **Reparameterization math reused verbatim, not re-derived.** EFDN's
   `models/unitv2.py` (EDBB + `switch_to_deploy`/`rep_params`) is vendored
   unchanged as `src/archs/common/efdn_unitv2.py`. The training model uses the
   multi-branch `EDBB`; `EFDN.reparameterize()` walks the modules and calls the
   authors' `switch_to_deploy()`. `reparameterize.py` additionally asserts the
   folded model is numerically identical to the multi-branch model on random
   input (tolerance 1e-4).

5. **EG loss is a RE-IMPLEMENTATION (verify).** EFDN's repo ships only model +
   inference code — the training "EG" (edge-enhanced gradient-variance) loss is
   *not released*. `src/losses/eg.py` re-implements it from the described
   components: a Sobel edge-magnitude L1 term + the Gradient-Variance loss
   (Abrahamyan et al., ICASSP 2022; per-patch variance of Sobel gradients matched
   with L2). This is the least-certain piece of the integration and is called out
   for verification. It is fully config-driven (`eg_opt`) and can be ablated by
   deleting that block. Combined Phase-1 SR loss = `L_pixel + L_EG + L_TDP`.

6. **EG loss weight starts small (0.1).** The EG term's raw scale (esp. the
   variance term) differs from L1 pixel loss; `loss_weight: 0.1` is a conservative
   start to avoid it dominating. Sweep later (brief 5.1.3/5.1.4).

7. **EFDN ×8 trains from scratch.** EFDN releases only x2/x4 *folded* weights
   (`model_zoo/EFDN_gv.pth`), which (a) are the deploy single-conv topology,
   incompatible with the multi-branch training model, and (b) are not ×8. So
   `configs/phase1_efdn_seg_x8.yaml` sets `path.network_sr: ~` (scratch). This
   deviates from EFDN's "start from a DIV2K-pretrained SR" recipe; the optional
   pre-step is documented in the README (train EFDN on DIV2K ×8 first, then point
   `path.network_sr` at it). Phase 0 (EDSR) keeps its released DIV2K ×8 init.

8. **EFDN width n_feats=48.** EFDN paper/reference default. EDSR baseline uses
   n_feats=64 / 16 blocks (SR4IR default). Kept each backbone at its native width
   so neither is handicapped.

9. **`task: seg` set explicitly in the project configs.** SR4IR normally infers
   the task from the options path (`options/seg/...`); since these configs live in
   `configs/`, `task: seg` is set explicitly so inference isn't needed.
