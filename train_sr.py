#!/usr/bin/env python3
"""
Phase-1 training entry (trains the SR network within SR4IR's alternating loop).

NOTE on SR4IR's design: for the `sr4ir_seg` model the SR update and the task
update happen alternately inside ONE training loop (see
models/sr4ir_original/src/models/seg/sr4ir_seg_model.py::train_one_epoch).
So this script and train_task.py both launch the same SR4IR run -- they are
provided to match the brief's file layout and to expose checkpoint paths as CLI
args. The SR-side losses (pixel + EG + TDP) are configured under `train:` in the
YAML config.

Usage:
  python train_sr.py -c configs/phase1_efdn_seg_x8.yaml --gpus 0
  python train_sr.py -c configs/phase0_baseline_edsr_seg_x8.yaml --gpus 0 \
        --sr-ckpt experiments/pretrained_models/edsr_baseline_x8.pt
  # resume:
  python train_sr.py -c configs/phase1_efdn_seg_x8.yaml --gpus 0 \
        --resume experiments/seg/phase1_efdn_seg_x8/checkpoints/checkpoint_latest.pth

After training, weights are synced into checkpoints/<name>/ automatically.
"""
import os.path as osp
import sys

import _sr4ir_common as C


def main():
    parser = C.add_common_args(
        __import__("argparse").ArgumentParser(description=__doc__,
                                              formatter_class=__import__("argparse").RawDescriptionHelpFormatter))
    parser.add_argument("--resume", default=None, help="Path to a checkpoint_*.pth for resuming.")
    parser.add_argument("--no-sync", action="store_true", help="Do not copy weights into checkpoints/<name>/.")
    args = parser.parse_args()

    cfg = C.load_config(args.config)
    cfg = C.apply_path_overrides(cfg, sr_ckpt=args.sr_ckpt, seg_ckpt=args.seg_ckpt)
    if args.resume:
        cfg["resume"] = osp.abspath(args.resume)

    if not cfg.get("train"):
        print("ERROR: config has no `train:` section; nothing to train.", file=sys.stderr)
        sys.exit(2)

    rc = C.run_main(cfg, gpus=args.gpus)
    if rc != 0:
        print(f"[train_sr] SR4IR main.py exited with code {rc}", file=sys.stderr)
        sys.exit(rc)

    if not args.no_sync:
        dest = osp.join(C.PROJECT_ROOT, "checkpoints", cfg["name"])
        copied = C.sync_checkpoints(cfg, dest)
        print("[train_sr] synced:")
        for p in copied:
            print("   ", p)


if __name__ == "__main__":
    main()
