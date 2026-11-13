# Authors: Tran Anh Tu, Huynh Hoang Nhut
# Final, frozen configurations for Table II (run from the repo root:  powershell -File scripts\run_final.ps1)
# Every run: 5 folds, EMA, last checkpoint (no validation peeking), equal epoch budget per backbone,
# and --predict_test so the locked internal test is predicted at training time (do NOT look at it until all runs end).
# Seed 42 of the main rows already exists; its test predictions come from 08_predict_test.py (block C).
# Approximate time: cnn1d ~40 min/run, spec2d ~55 min/run -> blocks A+B ~13 h. Each line can be re-run alone.

$C1 = "--mixup 0.2 --ema 0.999 --aug_mask --select last --predict_test"
$S2 = "--backbone spec2d:tf_efficientnetv2_b0 --lr 3e-4 --mixup 0.4 --drop_rate 0.3 --drop_path 0.2 --ema 0.999 --select last --predict_test"

# ---- A. main rows, two extra seeds (two-stage, RaTE, pool-heads) x 2 backbones = 12 runs -----------------------
foreach ($s in 1, 2) {
  Invoke-Expression "python scripts\04_train.py --loss kl --epochs 25 --stage2_epochs 5 $C1 --seed $s"
  Invoke-Expression "python scripts\04_train.py --loss rate --kappa_fixed 1000 --rate_lr 1e-3 --epochs 30 $C1 --seed $s"
  Invoke-Expression "python scripts\04_train.py --loss pheads --kappa_fixed 1000 --epochs 30 $C1 --seed $s"
  Invoke-Expression "python scripts\04_train.py --loss kl --epochs 15 --stage2_epochs 5 $S2 --seed $s"
  Invoke-Expression "python scripts\04_train.py --loss rate --kappa_fixed 1000 --rate_lr 1e-3 --epochs 20 $S2 --seed $s"
  Invoke-Expression "python scripts\04_train.py --loss pheads --kappa_fixed 1000 --epochs 20 $S2 --seed $s"
}

# ---- B. ablation rows, seed 42 (vote-weighted KL = DiVo limit; DiVo with learned kappa) x 2 backbones = 4 runs --
Invoke-Expression "python scripts\04_train.py --loss kl_w --epochs 30 $C1"
Invoke-Expression "python scripts\04_train.py --loss divo --epochs 30 $C1"
Invoke-Expression "python scripts\04_train.py --loss kl_w --epochs 20 $S2"
Invoke-Expression "python scripts\04_train.py --loss divo --epochs 20 $S2"

# ---- C. locked-test predictions for the existing seed-42 main rows (no retraining, a few minutes) --------------
python scripts\08_predict_test.py --runs `
  cnn1d_kl_mx0.2_ema0.999_am_e25_last_s42 `
  cnn1d_rate_k1000_mx0.2_rlr0.001_ema0.999_am_e30_last_s42 `
  cnn1d_pheads_k1000_mx0.2_ema0.999_am_e30_last_s42 `
  spec2d-tf_efficientnetv2_b0_kl_mx0.4_ema0.999_last_s42 `
  spec2d-tf_efficientnetv2_b0_rate_k1000_mx0.4_rlr0.001_ema0.999_e20_last_s42 `
  spec2d-tf_efficientnetv2_b0_pheads_k1000_mx0.4_ema0.999_e20_last_s42
