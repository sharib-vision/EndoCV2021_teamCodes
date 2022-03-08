#!/usr/bin/env bash
python test_check.py --model_name fpnet_dpn92_W  --ckpt_path checkpoints/F20/fpnet_dpn92_W --round 2 --tta 1
zip -r fpnet_dpn92_W_F20_tta_round2_NEWCHECK.zip EndoCV2021/
rm -r EndoCV2021/
