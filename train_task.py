#!/usr/bin/env python3
"""
Phase-2 training entry (trains the task/segmentation network).

IMPORTANT: In SR4IR's `sr4ir_seg` model, the task update -- including CQMix
(the `auxce_cqmix` loss, already implemented in
models/sr4ir_original/src/models/seg/sr4ir_seg_model.py::train_one_epoch) --
runs in the SAME alternating loop as the SR update. There is no separate
task-only training stage in the SR4IR pipeline. This script therefore launches
the same SR4IR run as train_sr.py; both are kept to match the brief's layout and
to expose checkpoint paths as CLI arguments.

CQMix is NOT something added here -- it already ships in SR4IR and is enabled via
`auxce_cqmix_opt` in the config. (Adaptive / Grad-CAM-guided CQMix is a Phase-2+
research item explicitly deferred by the brief, Section 8.)

Usage:
  python train_task.py -c configs/phase1_efdn_seg_x8.yaml --gpus 0
"""
import os.path as osp
import sys

import _sr4ir_common as C


def main():
    argparse = __import__("argparse")
    parser = C.add_common_args(
        argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter))
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
        print(f"[train_task] SR4IR main.py exited with code {rc}", file=sys.stderr)
        sys.exit(rc)

    if not args.no_sync:
        dest = osp.join(C.PROJECT_ROOT, "checkpoints", cfg["name"])
        copied = C.sync_checkpoints(cfg, dest)
        print("[train_task] synced:")
        for p in copied:
            print("   ", p)


if __name__ == "__main__":
    main()
