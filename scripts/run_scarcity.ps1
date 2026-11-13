# Authors: Tran Anh Tu, Huynh Hoang Nhut
# Stress test: fraction of >=10-vote PATIENTS kept for training (all low-vote events kept). 1-D CNN, dev folds only
# (no --predict_test: the locked test set is not used). Run from the repo root:  powershell -File scripts\run_scarcity.ps1
# Block A (~3.5-4 h): seed 42 at 25%, 10% and 50%.  Block B (optional, same time): seed 1, for a seed SD.
$C1 = "--mixup 0.2 --ema 0.999 --aug_mask --select last"

# ---- A. seed 42 ---------------------------------------------------------------------------------------------
foreach ($f in 0.25, 0.1, 0.5) {
  Invoke-Expression "python scripts\04_train.py --loss kl --epochs 25 --stage2_epochs 5 $C1 --hi_frac $f --seed 42"
  Invoke-Expression "python scripts\04_train.py --loss pheads --kappa_fixed 1000 --epochs 30 $C1 --hi_frac $f --seed 42"
}
python scripts\16_scarcity.py

# ---- B. seed 1 (optional; uncomment) --------------------------------------------------------------------------
# foreach ($f in 0.25, 0.1, 0.5) {
#   Invoke-Expression "python scripts\04_train.py --loss kl --epochs 25 --stage2_epochs 5 $C1 --hi_frac $f --seed 1"
#   Invoke-Expression "python scripts\04_train.py --loss pheads --kappa_fixed 1000 --epochs 30 $C1 --hi_frac $f --seed 1"
# }
# python scripts\16_scarcity.py
