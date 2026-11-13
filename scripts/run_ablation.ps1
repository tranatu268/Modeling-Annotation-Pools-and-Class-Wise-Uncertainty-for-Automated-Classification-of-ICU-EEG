# Authors: Tran Anh Tu, Huynh Hoang Nhut
# Ablations asked for by the internal review (P1-06). Seed 42, 5 folds each, with locked-test predictions.
# 6 runs: cnn1d ~40 min/run (pool-0-only is much shorter), spec2d ~55 min/run -> about 4 h. Each line can be re-run alone.
#   powershell -ExecutionPolicy Bypass -File scripts\run_ablation.ps1
$C1 = "--mixup 0.2 --ema 0.999 --aug_mask --select last --predict_test --seed 42"
$S2 = "--backbone spec2d:tf_efficientnetv2_b0 --lr 3e-4 --mixup 0.4 --drop_rate 0.3 --drop_path 0.2 --ema 0.999 --select last --predict_test --seed 42"

# 1) >=10-vote events only (soft-label KL, one stage)
Invoke-Expression "python scripts\04_train.py --loss kl --stage2_epochs 0 --pools 0 --epochs 30 $C1"
Invoke-Expression "python scripts\04_train.py --loss kl --stage2_epochs 0 --pools 0 --epochs 20 $S2"
# 2) all events, one stage, unweighted soft-label KL (= two-stage without its second stage)
Invoke-Expression "python scripts\04_train.py --loss kl --stage2_epochs 0 --epochs 30 --name cnn1d_kl1stage_e30_last_s42 $C1"
Invoke-Expression "python scripts\04_train.py --loss kl --stage2_epochs 0 --epochs 20 --name spec2d_kl1stage_e20_last_s42 $S2"
# 3) pool-heads with the 1-2-vote pool merged into the >=10-vote pool
Invoke-Expression "python scripts\04_train.py --loss pheads --kappa_fixed 1000 --pools merge2 --epochs 30 $C1"
Invoke-Expression "python scripts\04_train.py --loss pheads --kappa_fixed 1000 --pools merge2 --epochs 20 $S2"
Get-ChildItem work\results -Directory | Where-Object { $_.Name -match "pool0|poolmerge2|kl1stage" } | Select-Object Name
