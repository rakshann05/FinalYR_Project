"""
Shared plumbing for the top-level EFDN+SR4IR wrapper scripts
(train_sr.py / train_task.py / test.py / reparameterize.py).

Why wrappers exist
------------------
SR4IR is a single, YAML-driven entry point (`models/sr4ir_original/src/main.py`).
For the `sr4ir_seg` model it performs the SR update (phase-1: pixel/EG/TDP losses)
and the task update (phase-2: seg + CQMix losses) *alternately inside one training
loop* -- there is no separate "train SR" then "train task" executable. These
wrappers therefore dispatch to the same `main.py`; the brief's train_sr.py /
train_task.py distinction is documented here rather than enforced by separate
training code (see EXPERIMENTS.md, "Deviations").

Every wrapper takes the config and any checkpoint paths as CLI arguments and
injects them into a temporary copy of the config -- checkpoint paths are never
hardcoded inside the scripts.
"""
import argparse
import os
import os.path as osp
import shutil
import subprocess
import sys
import tempfile

import yaml

PROJECT_ROOT = osp.dirname(osp.abspath(__file__))
SR4IR_ROOT = osp.join(PROJECT_ROOT, "models", "sr4ir_original")
MAIN = osp.join("src", "main.py")  # relative to SR4IR_ROOT


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _write_temp_config(cfg):
    """Write cfg to a temp YAML inside SR4IR_ROOT so `-opt` can reach it and
    copy_opt_file() can archive it. Returns the path (relative to SR4IR_ROOT)."""
    tmp_dir = osp.join(SR4IR_ROOT, "options", "_generated")
    os.makedirs(tmp_dir, exist_ok=True)
    fd, abspath = tempfile.mkstemp(suffix=".yml", dir=tmp_dir)
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    return osp.relpath(abspath, SR4IR_ROOT)


def apply_path_overrides(cfg, sr_ckpt=None, seg_ckpt=None):
    """Override the pretrained/checkpoint paths from CLI args, if given.
    Paths are resolved to absolute so they work from SR4IR_ROOT cwd."""
    cfg.setdefault("path", {})
    if sr_ckpt is not None:
        cfg["path"]["network_sr"] = osp.abspath(sr_ckpt)
    if seg_ckpt is not None:
        cfg["path"]["network_seg"] = osp.abspath(seg_ckpt)
    return cfg


def experiment_dir(cfg):
    """Where SR4IR writes this run: models/sr4ir_original/experiments/<task>/<name>/"""
    return osp.join(SR4IR_ROOT, "experiments", cfg["task"], cfg["name"])


def run_main(cfg, extra_args=None, gpus=None):
    """Invoke SR4IR's src/main.py with cfg, from SR4IR_ROOT. Returns exit code."""
    rel_opt = _write_temp_config(cfg)
    env = os.environ.copy()
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpus)
    cmd = [sys.executable, MAIN, "-opt", rel_opt] + (extra_args or [])
    print(f"[wrapper] cwd={SR4IR_ROOT}")
    print(f"[wrapper] CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '(unset)')}")
    print(f"[wrapper] cmd={' '.join(cmd)}")
    return subprocess.call(cmd, cwd=SR4IR_ROOT, env=env)


def sync_checkpoints(cfg, dest_dir):
    """Copy the run's saved weights + archived config into the brief's
    checkpoints/<...>/ folder. Returns list of copied files."""
    exp = experiment_dir(cfg)
    models_dir = osp.join(exp, "models")
    os.makedirs(dest_dir, exist_ok=True)
    copied = []
    for fname in ("net_sr_latest.pth", "net_seg_latest.pth"):
        src = osp.join(models_dir, fname)
        if osp.exists(src):
            dst = osp.join(dest_dir, fname)
            shutil.copy2(src, dst)
            copied.append(dst)
    # archive the exact config used
    cfg_dump = osp.join(dest_dir, "config_used.yaml")
    with open(cfg_dump, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    copied.append(cfg_dump)
    return copied


def add_common_args(parser):
    parser.add_argument("-c", "--config", required=True, help="Path to experiment YAML config.")
    parser.add_argument("--sr-ckpt", default=None, help="Override path.network_sr (SR weights).")
    parser.add_argument("--seg-ckpt", default=None, help="Override path.network_seg (task weights).")
    parser.add_argument("--gpus", default=None, help="Value for CUDA_VISIBLE_DEVICES, e.g. '0' or '0,1'.")
    return parser
