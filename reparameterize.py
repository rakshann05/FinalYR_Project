#!/usr/bin/env python3
"""
Fold a trained (multi-branch) EFDN SR checkpoint into its deploy form.

EFDN trains with each EDBB block running 5 parallel branches. Before you measure
params / FLOPs / runtime, every EDBB must be collapsed into a single 3x3 conv via
EFDN's own `switch_to_deploy()` (vendored unmodified in
models/sr4ir_original/src/archs/common/efdn_unitv2.py). This script:

  1. builds the training-mode EFDN (deploy=False) with the config's scale/n_feats,
  2. loads the trained net_sr weights,
  3. calls model.reparameterize() (folds all EDBB blocks), and
  4. saves the folded state_dict, which loads into the deploy EFDN
     (network_sr.deploy: true) for cost benchmarking.

It also sanity-checks that the folded model produces numerically identical output
to the multi-branch model on a random input (this is the whole point of
reparameterization -- same function, cheaper compute).

Usage:
  python reparameterize.py -c configs/phase1_efdn_seg_x8.yaml \
      --in  checkpoints/phase1_efdn_seg_x8/net_sr_latest.pth \
      --out checkpoints/phase1_efdn_seg_x8/net_sr_deploy.pth

Then benchmark:
  python test.py -c configs/phase1_efdn_seg_x8.yaml --cost --deploy \
      --sr-ckpt checkpoints/phase1_efdn_seg_x8/net_sr_deploy.pth
"""
import argparse
import copy
import os.path as osp
import sys

import torch
import yaml

PROJECT_ROOT = osp.dirname(osp.abspath(__file__))
SR4IR_SRC = osp.join(PROJECT_ROOT, "models", "sr4ir_original", "src")
sys.path.insert(0, SR4IR_SRC)

from archs.common.efdn_arch import EFDN  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--in", dest="inp", required=True, help="trained net_sr .pth (multi-branch).")
    parser.add_argument("--out", required=True, help="output folded net_sr .pth.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ns = dict(cfg["network_sr"])
    assert ns.get("name") == "efdn", f"config network_sr.name is '{ns.get('name')}', expected 'efdn'"
    scale = cfg["scale"]
    n_feats = ns.get("n_feats", 48)

    # 1-2. build training model + load trained weights
    model = EFDN(scale=scale, n_feats=n_feats, deploy=False)
    state = torch.load(args.inp, map_location="cpu")
    if isinstance(state, dict) and "net_sr" in state:
        state = state["net_sr"]
    model.load_state_dict(state, strict=True)
    model.eval()

    # numeric-equivalence check
    x = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        y_before = model(x)
    folded = copy.deepcopy(model).reparameterize().eval()
    with torch.no_grad():
        y_after = folded(x)
    max_diff = (y_before - y_after).abs().max().item()
    print(f"[reparam] max |train - deploy| output diff on random input: {max_diff:.3e}")
    if max_diff > 1e-4:
        print("[reparam] WARNING: outputs differ more than expected; inspect the fold.", file=sys.stderr)

    # 4. save folded weights
    torch.save(folded.state_dict(), args.out)
    n_train = sum(p.numel() for p in model.parameters())
    n_deploy = sum(p.numel() for p in folded.parameters())
    print(f"[reparam] params  train-mode: {n_train:,}   deploy-mode: {n_deploy:,}")
    print(f"[reparam] saved folded weights -> {args.out}")


if __name__ == "__main__":
    main()
