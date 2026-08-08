Phase 1 (EFDN) weights land here after:
  python train_sr.py -c configs/phase1_efdn_seg_x8.yaml --gpus 0
Expected files: net_sr_latest.pth, net_seg_latest.pth, config_used.yaml
After reparameterize.py: net_sr_deploy.pth
