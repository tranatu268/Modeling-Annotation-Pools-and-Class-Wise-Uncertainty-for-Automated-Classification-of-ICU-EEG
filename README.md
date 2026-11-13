# Annotation pools and class-wise uncertainty for harmful brain activity in ICU EEG

Code for the manuscript *Modeling Annotation Pools and Class-Wise Uncertainty for Automated Classification of
Harmful Brain Activity in ICU EEG* (submitted to IEEE JBHI).

**Authors:** Tran Anh Tu, Huynh Hoang Nhut

The votes behind each label of the HMS Harmful Brain Activity Classification benchmark come from annotation
efforts of different size (>=10, 3-7 and 1-2 votes). Instead of filtering or a two-stage curriculum, the models
score each event's vote counts with a Dirichlet-multinomial likelihood under the distribution of the pool that
produced them, all pools sharing one consensus encoder:

- **RaTE** (random-threshold experts): a pool-specific logit shift and random thresholds (mixed logit);
- **pool-heads**: an input-dependent residual shift per pool;
- label-conditional, patient-weighted **conformal prediction sets** with deferral;
- evaluation on de-duplicated events with patient-grouped folds, a locked test set, paired patient bootstrap,
  Bayes decision cost and exact model-vs-expert agreement from anonymous vote counts;
- a technical shift test on post-cardiac-arrest EEG (I-CARE).

## Installation

```bash
# 1) install PyTorch for your CUDA version: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
python -m pytest -q        # synthetic data, about 20 s on CPU
```

## Data

- **HMS**: download the competition files from
  [Kaggle](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification) into
  `data/hms-harmful-brain-activity-classification/` (or edit `data_dir` in `config.yaml`).
- **I-CARE** (optional, Section on dataset shift): `scripts/09_icare_fetch.py` downloads the selected records from
  [PhysioNet](https://physionet.org/content/i-care/2.1/) (open access, CC BY-NC-SA 4.0).

No data, checkpoints or predictions are stored in this repository.

## Pipeline

Run every command from the repo root.

| Step | Command | Output |
|---|---|---|
| Data audit | `python scripts/01_audit.py` | `work/audit/` |
| Events, folds, locked test, cache | `python scripts/02_prepare.py --workers 4` | `work/events.parquet`, `work/windows.npy` |
| Noise floor | `python scripts/03_noise_floor.py` | `work/noise_floor/` |
| Smoke test | `python scripts/04_train.py --loss pheads --folds 0 --epochs 1 --max_steps 20 --name smoke` | |
| Main runs (3 seeds) | `powershell -File scripts/run_final.ps1` | `work/results/<run>/fold<k>/` |
| Ablations | `powershell -File scripts/run_ablation.ps1` | |
| Scarcity stress test | `powershell -File scripts/run_scarcity.ps1` | |
| Evaluation | `python scripts/05_evaluate.py --runs <run> ... --ref <baseline> [--split test]` | KL, cost, CIs, paired p |
| Conformal sets | `python scripts/06_conformal.py --run <run>` | coverage, set size, deferral |
| Expert agreement | `python scripts/07_expert_agreement.py` | `work/expert_agreement/` |
| Locked-test predictions | `python scripts/08_predict_test.py --runs <run> ...` | |
| I-CARE | `09_icare_fetch.py`, `10_amplitude_check.py`, `13_icare_infer.py` | `work/icare/` |
| Held-out pool likelihood | `python scripts/15_pool_likelihood.py` | |
| Tables | `11_tables.py`, `16_scarcity.py`, `17_supp_tables.py`, `18_review_analyses.py` | `work/results/` |
| Figure data | `python scripts/14_export_figdata.py` | `work/figdata/*.csv` |
| Figures (MATLAB R2020b+) | `cd matlab; make_all_figures` | `paper/figs/*.pdf` |

Main configurations (seed 42 shown; `run_final.ps1` adds seeds 1 and 2):

```bash
C1="--mixup 0.2 --ema 0.999 --aug_mask --select last --predict_test"
python scripts/04_train.py --loss kl     --epochs 25 --stage2_epochs 5        $C1   # two-stage curriculum
python scripts/04_train.py --loss rate   --kappa_fixed 1000 --rate_lr 1e-3 --epochs 30 $C1   # RaTE
python scripts/04_train.py --loss pheads --kappa_fixed 1000 --epochs 30       $C1   # pool-heads
# spectrogram backbone: --backbone spec2d:tf_efficientnetv2_b0 --lr 3e-4 --mixup 0.4 --drop_rate 0.3 --drop_path 0.2
```

The locked internal test set (about 20% of patients) is predicted only for the frozen configurations.

## Layout

```
hms/data.py         events (rows merged by recording and vote vector), patient-grouped folds + locked test,
                    leakage check, 18-channel double-banana montage, 0.5-40 Hz filter, memmap cache
hms/losses.py       soft-label KL, Dirichlet-multinomial, RaTE, Sinkhorn cost
hms/models.py       cnn1d (shared-channel CNN + BiGRU), spec2d (STFT + timm CNN), per-pool heads
hms/train.py        one run = backbone x loss x fold (losses: kl, kl_w, divo, rate, pheads)
hms/metrics.py      Kaggle KL, calibration, decision cost, patient bootstrap
hms/conformal.py    weighted split conformal for vote-distribution labels, label-conditional sets, deferral
hms/agreement.py    model-vs-expert and expert-vs-expert agreement from vote counts
hms/noise_floor.py  irreducible KL of a perfect model given finite panels
hms/external.py, hms/icare.py   I-CARE download and conversion to the HMS input
scripts/            numbered pipeline steps and PowerShell run lists
matlab/             figure scripts (read work/figdata/*.csv)
tests/              unit and end-to-end tests on synthetic data
```

Design rules: the vote count and pool are never model inputs (they only select the pool in the likelihood);
statistics used for training are computed on training folds only; every model is evaluated on one central window
per >=10-vote event and on the same set of events.

## Citation

If you use this code, please cite the paper (reference will be added on publication).
