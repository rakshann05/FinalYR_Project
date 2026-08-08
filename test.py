#!/usr/bin/env python3
"""
Evaluation entry for the EFDN+SR4IR pipeline.

Two modes:

  (1) METRICS (default): mIoU + PSNR (+ LPIPS) on the VOC val set.
      SR4IR's --test_only loads weights from
      experiments/<task>/<name>/models/net_{sr,seg}_latest.pth, so this wrapper
      copies the CLI-provided checkpoints into a dedicated `<name>_eval`
      experiment folder (never clobbering a training run) and then evaluates.

        python test.py -c configs/phase1_efdn_seg_x8.yaml --gpus 0 \
            --sr-ckpt  checkpoints/phase1_efdn_seg_x8/net_sr_latest.pth \
            --seg-ckpt checkpoints/phase1_efdn_seg_x8/net_seg_latest.pth

      If --sr-ckpt/--seg-ckpt are omitted, they default to
      checkpoints/<name>/net_{sr,seg}_latest.pth.

  (2) COST (--cost): params + MACs/FLOPs of the SR network (and seg network).
      For EFDN, pass --deploy together with a *reparameterized* SR checkpoint
      (produced by reparameterize.py) so the numbers reflect deploy-time compute,
      not the multi-branch training graph.

        python test.py -c configs/phase1_efdn_seg_x8.yaml --cost --deploy \
            --sr-ckpt checkpoints/phase1_efdn_seg_x8/net_sr_deploy.pth

Console output is also tee'd to logs/<name>_<mode>_<timestamp>.log.
"""
import datetime
import os
import os.path as osp
import shutil
import subprocess
import sys

import yaml
import _sr4ir_common as C


def _tee_run(cfg, extra_args, gpus, log_tag):
    rel_opt = C._write_temp_config(cfg)
    env = os.environ.copy()
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpus)
    cmd = [sys.executable, C.MAIN, "-opt", rel_opt] + extra_args
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = osp.join(C.PROJECT_ROOT, "logs", f"{cfg['name']}_{log_tag}_{ts}.log")
    os.makedirs(osp.dirname(log_path), exist_ok=True)
    print(f"[test] cmd={' '.join(cmd)} (cwd={C.SR4IR_ROOT})")
    print(f"[test] logging to {log_path}")
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, cwd=C.SR4IR_ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        proc.wait()
    return proc.returncode, log_path


def run_metrics(args):
    cfg = C.load_config(args.config)
    name = cfg["name"]
    sr_ckpt = args.sr_ckpt or osp.join(C.PROJECT_ROOT, "checkpoints", name, "net_sr_latest.pth")
    seg_ckpt = args.seg_ckpt or osp.join(C.PROJECT_ROOT, "checkpoints", name, "net_seg_latest.pth")
    for p in (sr_ckpt, seg_ckpt):
        if not osp.exists(p):
            print(f"ERROR: checkpoint not found: {p}", file=sys.stderr)
            sys.exit(2)

    # Evaluate under a dedicated experiment name so we never overwrite a train run.
    eval_name = f"{name}_eval"
    cfg["name"] = eval_name
    models_dir = osp.join(C.SR4IR_ROOT, "experiments", cfg["task"], eval_name, "models")
    os.makedirs(models_dir, exist_ok=True)
    shutil.copy2(sr_ckpt, osp.join(models_dir, "net_sr_latest.pth"))
    shutil.copy2(seg_ckpt, osp.join(models_dir, "net_seg_latest.pth"))
    print(f"[test] evaluating\n    SR : {sr_ckpt}\n    seg: {seg_ckpt}")

    cfg.setdefault("test", {})
    cfg["test"]["calculate_lpips"] = True  # report LPIPS in metrics mode
    rc, log_path = _tee_run(cfg, extra_args=["--test_only"], gpus=args.gpus, log_tag="metrics")
    if rc != 0:
        sys.exit(rc)
    print(f"\n[test] done. Metrics (PSNR / LPIPS / mIoU-SR) are in the 'Test:' line above and in {log_path}")


def run_cost(args):
    cfg = C.load_config(args.config)
    cfg.pop("train", None)          # calculate_cost branch requires no train section
    cfg["calculate_cost"] = True
    cfg.setdefault("test", {})
    cfg["test"]["calculate_lpips"] = False
    if args.deploy:
        cfg.setdefault("network_sr", {})
        cfg["network_sr"]["deploy"] = True  # build folded EFDN topology
    if args.sr_ckpt:
        cfg = C.apply_path_overrides(cfg, sr_ckpt=args.sr_ckpt, seg_ckpt=args.seg_ckpt)
    rc, log_path = _tee_run(cfg, extra_args=[], gpus=args.gpus, log_tag="cost")
    if rc != 0:
        sys.exit(rc)
    print(f"\n[test] cost report (params + MACs) in {log_path}")
    if args.deploy:
        print("[test] NOTE: reported with EFDN in DEPLOY mode (reparameterized).")
    else:
        print("[test] NOTE: SR reported AS-BUILT. For EFDN deploy-time cost, rerun with "
              "--deploy and a reparameterized --sr-ckpt (see reparameterize.py).")


def main():
    argparse = __import__("argparse")
    parser = C.add_common_args(
        argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument("--cost", action="store_true", help="Report params/FLOPs instead of accuracy metrics.")
    parser.add_argument("--deploy", action="store_true", help="(cost mode) build folded EFDN + load reparameterized weights.")
    args = parser.parse_args()

    if args.cost:
        run_cost(args)
    else:
        run_metrics(args)


if __name__ == "__main__":
    main()
