# Authors: Tran Anh Tu, Huynh Hoang Nhut
"""End-to-end on the synthetic dataset: events -> folds -> cache -> train (every loss) -> predictions."""
import numpy as np
import pandas as pd
import pytest

from hms import load_config
from hms.data import VOTES, build_cache, build_events, check_no_leakage, extract_window, make_folds, read_bipolar
from hms.train import LOSSES, run


@pytest.fixture(scope="module")
def prepared(fake_data):
    root, cfg_path = fake_data
    cfg = load_config(cfg_path)
    cfg["work_dir"].mkdir(parents=True, exist_ok=True)
    ev = build_events(pd.read_csv(root / "train.csv"))
    ev["fold"] = make_folds(ev, 5, 42)
    ev.to_parquet(cfg["work_dir"] / "events.parquet", index=False)
    build_cache(ev, root / "train_eegs", cfg["work_dir"] / "windows.npy", 60, 100, workers=2)
    return cfg, ev


def test_events_deduplicate(prepared):
    _, ev = prepared
    assert len(ev) == 80 and ev.n_rows.sum() == 40 * 5
    assert set(ev.stratum) == {0, 1}


def test_folds_have_no_leakage(prepared):
    _, ev = prepared
    check_no_leakage(ev, ev.fold.values)
    assert (ev.fold == -1).any() and set(ev.fold[ev.fold >= 0]) == set(range(5))
    leaky = ev.fold.values.copy()
    i = np.flatnonzero(ev.patient_id == ev.patient_id.iloc[0])
    leaky[i[0]] = (leaky[i[1]] + 1) % 5
    with pytest.raises(AssertionError):
        check_no_leakage(ev, leaky)


def test_window_shape_and_padding(fake_data):
    root, _ = fake_data
    x = read_bipolar(root / "train_eegs" / "1000.parquet")
    w = extract_window(x, 0.0, 60, 100)  # starts at t=0, so the first 5 s are outside the recording
    assert w.shape == (18, 6000) and w.dtype == np.float16 and np.isfinite(w.astype(float)).all()
    w = w.astype(float)
    assert (w[:, :300] ** 2).mean() < 0.05 * (w[:, 1000:] ** 2).mean()  # padded region stays near zero


def test_cache_matches_direct_extraction(prepared, fake_data):
    cfg, ev = prepared
    root, _ = fake_data
    X = np.load(cfg["work_dir"] / "windows.npy", mmap_mode="r")
    e = ev.iloc[3]
    direct = extract_window(read_bipolar(root / "train_eegs" / f"{e.eeg_id}.parquet"), e.center_offset, 60, 100)
    assert np.array_equal(X[e.event_id], direct)


@pytest.mark.parametrize("loss,backbone,ot,kfix", [(l, "cnn1d", 0.0, None) for l in LOSSES]
                         + [("rate", "cnn1d", 0.3, None), ("divo", "cnn1d", 0.0, 50.0),
                            ("divo", "spec2d:resnet18", 0.0, None)])
def test_train_smoke(prepared, tmp_path, loss, backbone, ot, kfix):
    cfg, ev = prepared
    out = tmp_path / f"{loss}_{backbone.replace(':', '-')}"
    best = run(cfg["work_dir"], out, fold=0, backbone=backbone, loss=loss, ot_lambda=ot, epochs=1,
               stage2_epochs=1, batch_size=4, workers=0, pretrained=False, M_eval=16, max_steps=2,
               predict_test=True, kappa_fixed=kfix, device="cpu")
    assert len(pd.read_csv(out / "log.csv").phase.unique()) == 2  # stage 2 ran for every loss
    assert np.isfinite(best)
    pr = pd.read_parquet(out / "preds.parquet")
    P = pr.filter(like="p_").values
    assert np.allclose(P.sum(1), 1, atol=1e-4) and (P >= 0).all()
    assert set(pr.split) == {"val", "test"}
    assert set(pr.event_id[pr.split == "val"]) == set(ev.event_id[ev.fold == 0])
    assert len(VOTES) == P.shape[1]


def test_rate_param_group_and_within_pool_mixup(prepared, tmp_path):
    import torch
    cfg, _ = prepared
    out = tmp_path / "rlr"
    best = run(cfg["work_dir"], out, fold=0, backbone="cnn1d", loss="rate", epochs=1, batch_size=8, workers=0,
               pretrained=False, M_eval=16, max_steps=3, mixup=0.4, mixup_within=True, rate_lr=1e-2,
               weight_decay=0.5, device="cpu")
    assert np.isfinite(best)
    mu = torch.load(out / "rate_params.pt")["mu"]
    assert torch.isfinite(mu).all() and mu[1:].abs().sum() > 0  # pools 1-2 moved; pool 0 is pinned at 0


@pytest.mark.parametrize("loss", ["kl", "rate", "pheads"])
def test_predict_test_from_checkpoint_matches_training(prepared, tmp_path, loss):
    from hms.train import predict_test_from_checkpoint
    cfg, _ = prepared
    out = tmp_path / loss
    run(cfg["work_dir"], out, fold=1, backbone="cnn1d", loss=loss, epochs=1, stage2_epochs=0, batch_size=4,
        workers=0, pretrained=False, M_eval=16, max_steps=2, predict_test=True, kappa_fixed=1000.0, device="cpu")
    pr = pd.read_parquet(out / "preds.parquet")
    expected = pr[pr.split == "test"].sort_values("event_id").filter(like="p_").values
    pr[pr.split == "val"].to_parquet(out / "preds.parquet", index=False)  # as if trained without predict_test
    n = predict_test_from_checkpoint(out, cfg["work_dir"], device="cpu")
    got = pd.read_parquet(out / "preds.parquet")
    assert n == len(expected) and (got.split == "val").sum() == (pr.split == "val").sum()
    assert np.allclose(got[got.split == "test"].sort_values("event_id").filter(like="p_").values, expected, atol=1e-6)


def test_select_last_uses_final_epoch(prepared, tmp_path):
    cfg, _ = prepared
    out = tmp_path / "last"
    last = run(cfg["work_dir"], out, fold=0, backbone="cnn1d", loss="kl", epochs=2, stage2_epochs=2, batch_size=4,
               workers=0, pretrained=False, max_steps=2, select="last", device="cpu")
    log = pd.read_csv(out / "log.csv")
    assert np.isclose(last, log.val_kl_hi.iloc[-1])


def test_ema_tracks_buffers_and_caps_decay():
    import torch

    from hms.train import EMA
    m = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3))
    ema = EMA([m], decay=0.999)
    ema.reset(steps=100)
    assert abs(ema.decay - 0.95) < 1e-9          # horizon capped at 20% of the phase
    m.train()
    m(torch.randn(16, 3) + 5)                     # moves BatchNorm running_mean
    ema.update()
    live = m[1].running_mean.clone()
    ema.swap()
    assert not torch.allclose(m[1].running_mean, live)  # evaluation sees averaged BN statistics
    ema.swap()
    assert torch.allclose(m[1].running_mean, live)


def test_train_smoke_recipe_v2(prepared, tmp_path):
    """mixup + EMA + time masking together, with RaTE (pool index follows the dominant mixup sample)."""
    cfg, _ = prepared
    best = run(cfg["work_dir"], tmp_path / "v2", fold=0, backbone="cnn1d", loss="rate", epochs=2, stage2_epochs=0,
               batch_size=4, workers=0, pretrained=False, M_eval=16, max_steps=3, mixup=0.2, ema_decay=0.9,
               aug_mask=True, device="cpu")
    assert np.isfinite(best)
    P = pd.read_parquet(tmp_path / "v2" / "preds.parquet").filter(like="p_").values
    assert np.allclose(P.sum(1), 1, atol=1e-4)


def test_icare_infer_reproduces_hms_predictions(prepared, tmp_path):
    """HMS cached windows packed as fake I-CARE blocks must get exactly the fold model's own val predictions,
    and the script must run without an outcome column being used."""
    import json
    import subprocess
    import sys
    from pathlib import Path
    cfg, ev = prepared
    work = cfg["work_dir"]
    name = "cnn1d_pheads_smoke"
    run(work, work / "results" / name / "fold0", fold=0, backbone="cnn1d", loss="pheads", epochs=1, batch_size=4,
        workers=0, pretrained=False, max_steps=2, kappa_fixed=1000.0, device="cpu")
    val = ev[ev.fold == 0].sort_values("event_id")
    X = np.load(work / "windows.npy", mmap_mode="r")
    s = (X.shape[2] - 5000) // 2
    ic = tmp_path / "icare"
    (ic / "windows").mkdir(parents=True)
    rows, ids = [], []
    for j, (pid, g) in enumerate(val.groupby("patient_id")):
        np.save(ic / "windows" / f"{pid}_{12 + j:03d}.npy", np.stack([X[e, :, s:s + 5000] for e in g.event_id]))
        rows.append({"patient": str(pid), "hour": 12 + j, "n_windows": len(g)})
        ids += list(g.event_id)
    pd.DataFrame(rows).to_csv(ic / "blocks.csv", index=False)
    pd.DataFrame({"Patient": [r["patient"] for r in rows], "Hospital": "A", "Outcome": "Good"}).to_csv(
        ic / "patients.csv", index=False)
    cfg_path = Path(work).parent / "config.yaml"
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, str(root / "scripts" / "13_icare_infer.py"), "--config", str(cfg_path),
                        "--icare", str(ic), "--runs", name, "--device", "cpu", "--chunk", "3", "--batch", "2"],
                       capture_output=True, text=True, cwd=root)
    assert r.returncode == 0, r.stderr[-2000:]
    got = pd.read_parquet(ic / f"preds_{name}.parquet")
    pr = pd.read_parquet(work / "results" / name / "fold0" / "preds.parquet").set_index("event_id")
    assert len(got) == len(ids)
    assert np.allclose(got.filter(regex="^p_").values, pr.loc[ids].filter(regex="^p_").values, atol=1e-5)
    assert "Outcome" not in got.columns and got.Hospital.eq("A").all() and got.qc_pass.all()
    summ = pd.read_csv(ic / "icare_summary.csv")
    assert {"all", "hospital", "hour"} <= set(summ.by) and summ.defer.between(0, 1).all()
    json.dumps(summ.iloc[0].to_dict(), default=str)


def test_export_figdata_examples_and_noise_floor(prepared, tmp_path):
    """The figure-data exporter selects EEG examples by rule, recovers the pool-0 probabilities from the
    checkpoint, and bins the noise floor; run on a tiny pool-heads model with the run names patched."""
    import importlib.util
    from pathlib import Path
    cfg, ev = prepared
    work = cfg["work_dir"]
    spec = importlib.util.spec_from_file_location(
        "ex", Path(__file__).resolve().parents[1] / "scripts" / "14_export_figdata.py")
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    name = "figdata_pheads_s{}"
    for fold in range(5):
        run(work, work / "results" / name.format(42) / f"fold{fold}", fold=fold, backbone="cnn1d", loss="pheads",
            epochs=1, batch_size=4, workers=0, pretrained=False, max_steps=1, kappa_fixed=1000.0, device="cpu")
    ex.RUNS = {k: name for k in ex.RUNS}
    ex.SEEDS = (42,)
    (work / "figdata").mkdir(exist_ok=True)
    meta = ex.eeg_examples(cfg, work / "results", device="cpu")
    assert set(meta.example) <= {"A", "B", "C"} and len(meta) >= 1
    pr = ex.load_preds(work / "results", name.format(42), "val")
    for r in meta.itertuples():
        got = np.array([getattr(r, f"p0_{c}") for c in ex.CLASSES])
        assert np.allclose(got, pr.loc[r.event_id, ex.PCOLS].values.astype(float))
        sig = pd.read_csv(work / "figdata" / f"eeg_case_{r.example}.csv")
        assert sig.shape == (5000, 19)
    # noise floor per bin (fake per-event floor file)
    dev_hi = ev[(ev.stratum == 0) & (ev.fold >= 0)]
    (work / "noise_floor").mkdir(exist_ok=True)
    pd.DataFrame({"event_id": dev_hi.event_id, "n": dev_hi.n, "empirical_bayes": 0.1, "asymptotic": 0.05}).to_csv(
        work / "noise_floor" / "per_event.csv", index=False)
    nf = ex.noise_floor_by_n(cfg, work / "results", B=20)
    assert {"kl", "kl_lo", "kl_hi", "floor_eb"} <= set(nf.columns) and (nf.kl_lo <= nf.kl_hi).all()
    assert nf.groupby("method").events.sum().nunique() == 1


@pytest.mark.parametrize("loss", ["pheads", "rate", "kl"])
def test_predict_pools_heads(prepared, tmp_path, loss):
    """Pool-0 output of predict_pools reproduces the stored predictions; single-pool models repeat pi_0."""
    import importlib.util
    from pathlib import Path
    cfg, ev = prepared
    work = cfg["work_dir"]
    name = f"poolll_{loss}_s42"
    run(work, work / "results" / name / "fold0", fold=0, backbone="cnn1d", loss=loss, epochs=1, batch_size=4,
        workers=0, pretrained=False, max_steps=2, kappa_fixed=1000.0, M_eval=64, device="cpu")
    spec = importlib.util.spec_from_file_location(
        "pl", Path(__file__).resolve().parents[1] / "scripts" / "15_pool_likelihood.py")
    pl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pl)
    s = pl.run_pool_preds(work / "results", work, ev, name, "val", "cpu", 0)
    P = np.stack(s.values)
    assert P.shape[1:] == (3, 6) and np.allclose(P.sum(-1), 1, atol=1e-4)
    stored = pd.read_parquet(work / "results" / name / "fold0" / "preds.parquet").set_index("event_id")
    p0 = stored.loc[s.index].filter(regex="^p_").values
    assert np.allclose(P[:, 0], p0, atol=1e-5 if loss != "rate" else 0.05)
    if loss == "kl":
        assert np.allclose(P[:, 0], P[:, 1]) and np.allclose(P[:, 0], P[:, 2])
    assert (work / "results" / name / "fold0" / "pool_preds_val.npz").exists()  # cached


def test_scarcity_run_names():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "sc", Path(__file__).resolve().parents[1] / "scripts" / "16_scarcity.py")
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    assert sc.run_name(sc.BASE["cnn1d"]["two-stage"], 0.25, 42) == "cnn1d_kl_mx0.2_ema0.999_am_e25_last_hi0.25_s42"
    assert sc.run_name(sc.BASE["cnn1d"]["pool-heads"], 1, 1) == "cnn1d_pheads_k1000_mx0.2_ema0.999_am_e30_last_s1"
    assert sc.run_name(sc.BASE["spec2d"]["two-stage"], 0.1, 2) == \
        "spec2d-tf_efficientnetv2_b0_kl_mx0.4_ema0.999_last_hi0.1_s2"


def test_pool_likelihood_and_scarcity_main(prepared, tmp_path, monkeypatch, capsys):
    """End-to-end smoke test of 15_pool_likelihood.py and 16_scarcity.py on tiny models (all five folds)."""
    import importlib.util
    import sys
    from pathlib import Path
    cfg, ev = prepared
    work = cfg["work_dir"]
    root = Path(__file__).resolve().parents[1]
    names = {}
    for loss in ("kl", "rate", "pheads"):
        names[loss] = f"e2e_{loss}_s42"
        for fold in range(5):
            run(work, work / "results" / names[loss] / f"fold{fold}", fold=fold, backbone="cnn1d", loss=loss,
                epochs=1, batch_size=4, workers=0, pretrained=False, max_steps=1, kappa_fixed=1000.0, M_eval=16,
                device="cpu", predict_test=True)

    def load(fname):
        spec = importlib.util.spec_from_file_location(fname, root / "scripts" / fname)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    pl = load("15_pool_likelihood.py")
    pl.FROZEN = {"cnn1d": {"two-stage": [names["kl"]], "RaTE": [names["rate"]], "pool-heads": [names["pheads"]]}}
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(work.parent / "config.yaml"), "--backbones", "cnn1d",
                                      "--B", "20"])
    pl.main()
    tab = pd.read_csv(work / "results" / "pool_likelihood_val.csv")
    own = tab[tab["head"] == "own head"]
    assert 1 in set(own.pool) and set(own.method) == {"RaTE", "pool-heads"}  # fake data have no 1-2-vote events
    assert own.p_holm_vs_two_stage.notna().all() and (tab.kl > 0).all()

    sc = load("16_scarcity.py")
    sc.BASE = {"cnn1d": {"two-stage": "e2e_kl_s{}", "pool-heads": "e2e_pheads_s{}"}}
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(work.parent / "config.yaml"), "--fracs", "1", "--B", "20"])
    sc.main()
    out = pd.read_csv(work / "results" / "scarcity.csv")
    assert set(out.method) == {"two-stage", "pool-heads"} and out.d_kl.notna().sum() == 1

    cf = load("06_conformal.py")
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(work.parent / "config.yaml"), "--run", names["pheads"],
                                      "--repeats", "5"])
    cf.main()
    st = load("17_supp_tables.py")
    st.FROZEN = {"1-D CNN": {"Two-stage": [names["kl"]], "RaTE": [names["rate"]], "Pool-heads": [names["pheads"]]}}
    st.CONFORMAL_RUNS = {"1-D CNN pool-heads": names["pheads"]}
    tex = tmp_path / "supp.tex"
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(work.parent / "config.yaml"), "--out", str(tex)])
    st.main()
    body = tex.read_text()
    for lab in ("tab:s_val", "tab:s_test", "tab:s_perclass", "tab:s_conformal", "tab:s_heldout_val",
                "tab:s_scarcity"):
        assert f"\\label{{{lab}}}" in body, lab
    per = pd.read_csv(work / "results" / "supp" / "per_run_val.csv")
    assert len(per) == 3 and per.kl.gt(0).all() and "auprc_seizure" in per
