# Authors: Tran Anh Tu, Huynh Hoang Nhut
import importlib.util
from pathlib import Path

import numpy as np

from hms import icare
from hms.data import BIPOLAR, extract_window
from hms.external import ALIASES, decode_block, parse_header, standardize_block, to_bipolar

# a real I-CARE header (patient 0582, hour 14)
HEADER = """0582_012_014_EEG 20 256 921600
0582_012_014_EEG.mat 16+24 665.2587876369603(191246)/nu 16 0 4358 17440 0 C3
0582_012_014_EEG.mat 16+24 526.3292909007921(-60297)/nu 16 0 5816 -22012 0 F3
0582_012_014_EEG.mat 16+24 648.7920987100422(21607)/nu 16 0 3504 3421 0 Cz
0582_012_014_EEG.mat 16+24 48.636533197947614(-10747)/nu 16 0 -6345 17541 0 Fpz
0582_012_014_EEG.mat 16+24 713.0359076868846(-52235)/nu 16 0 1390 -611 0 C4
0582_012_014_EEG.mat 16+24 535.3594864441528(-321)/nu 16 0 8888 -14676 0 F8
0582_012_014_EEG.mat 16+24 624.0762201218928(20659)/nu 16 0 -749 16886 0 T3
0582_012_014_EEG.mat 16+24 527.6006273343111(52343)/nu 16 0 3641 -22300 0 F7
0582_012_014_EEG.mat 16+24 452.85416941308944(-48437)/nu 16 0 5095 16172 0 Fp1
0582_012_014_EEG.mat 16+24 516.3750763358009(-64992)/nu 16 0 6171 16133 0 Fz
0582_012_014_EEG.mat 16+24 506.39875724180786(117698)/nu 16 0 6027 19860 0 Fp2
0582_012_014_EEG.mat 16+24 602.8335272439913(114157)/nu 16 0 8471 14030 0 F4
0582_012_014_EEG.mat 16+24 737.9278854596031(106936)/nu 16 0 10628 6264 0 T6
0582_012_014_EEG.mat 16+24 586.6428117721642(69377)/nu 16 0 10003 -1450 0 T4
0582_012_014_EEG.mat 16+24 804.0245488632494(-140597)/nu 16 0 -1327 2237 0 T5
0582_012_014_EEG.mat 16+24 770.0117870978754(-219974)/nu 16 0 -2734 30232 0 O1
0582_012_014_EEG.mat 16+24 801.075818515741(77391)/nu 16 0 4647 7360 0 P3
0582_012_014_EEG.mat 16+24 710.7158166217905(19404)/nu 16 0 4122 20373 0 Pz
0582_012_014_EEG.mat 16+24 608.4308245501212(4655)/nu 16 0 -5081 -21329 0 O2
0582_012_014_EEG.mat 16+24 730.5240266523017(91945)/nu 16 0 6904 -27260 0 P4
#Utility frequency: 50
#Start time: 14:00:00
#End time: 14:59:59
"""
HMS_NAMES = ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz",
             "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2"]
rng = np.random.default_rng(0)


def test_parse_real_icare_header():
    h = parse_header(HEADER)
    assert (h["nsig"], h["fs"], h["nsamp"], h["utility_freq"]) == (20, 256.0, 921600, 50)
    s = h["signals"][0]
    assert (s["file"], s["offset"], s["baseline"], s["units"], s["name"]) == \
        ("0582_012_014_EEG.mat", 24, 191246, "nu", "C3")
    assert np.isclose(s["gain"], 665.2587876369603)
    names = [x["name"] for x in h["signals"]]
    _, missing = to_bipolar(np.zeros((20, 10)), names)
    assert missing == []  # all 19 double-banana electrodes present (Fpz is extra)


def test_decode_interleaved_int16():
    h = parse_header(HEADER)
    adc = rng.integers(-30000, 30000, size=(7, 20)).astype("<i2")  # 7 samples x 20 channels, interleaved
    x = decode_block(adc.tobytes(), h)
    g = np.array([s["gain"] for s in h["signals"]])
    b = np.array([s["baseline"] for s in h["signals"]])
    assert x.shape == (20, 7) and np.allclose(x, ((adc.astype(float) - b) / g).T)


def test_standardize_matches_hms_pipeline_at_200hz():
    T = 200 * 160
    x = np.cumsum(rng.normal(0, 3, (19, T)), 1) + rng.normal(0, 20, (19, T))  # EEG-like, uV, with drift
    wins, missing, _ = standardize_block(x, HMS_NAMES, 200)
    bip = np.stack([x[HMS_NAMES.index(a)] - x[HMS_NAMES.index(c)] for a, c in BIPOLAR])
    ref = extract_window(bip, center_offset=55.0, span_s=50, fs_out=100)  # window 1 covers 55-105 s
    w = wins[1].astype(float)
    assert missing == [] and wins.shape == (3, 18, 5000)
    assert np.corrcoef(w.ravel(), ref.astype(float).ravel())[0, 1] > 0.999
    assert np.abs(w - ref.astype(float)).max() < 0.1  # in /32 units, i.e. < 3.2 uV


def test_resampling_from_256hz_keeps_amplitude_and_frequency():
    fs, T = 256, 256 * 120
    t = np.arange(T) / fs
    x = np.zeros((19, T))
    x[HMS_NAMES.index("Fp1")] = 64 * np.sin(2 * np.pi * 10 * t)
    wins, _, sd = standardize_block(x, HMS_NAMES, fs)
    ch = wins[0, BIPOLAR.index(("Fp1", "F7"))].astype(float)
    f = np.fft.rfftfreq(len(ch), 1 / 100)[np.argmax(np.abs(np.fft.rfft(ch)))]
    assert abs(f - 10) < 0.1 and abs(ch.max() - 64 / 32) < 0.1
    assert wins[0, BIPOLAR.index(("T3", "T5"))].astype(float).std() < 1e-3  # untouched chain stays flat


def test_aliases_and_missing_electrodes():
    names = [ALIASES.get(n, n) for n in HMS_NAMES]
    names = [{"T3": "T7", "T4": "T8"}.get(n, n) for n in names]  # 10-10 names from another hospital
    _, missing = to_bipolar(np.zeros((19, 5)), names)
    assert missing == []
    drop = [n for n in HMS_NAMES if n != "O2"]
    x = rng.normal(size=(18, 5))
    bip, missing = to_bipolar(x, drop)
    assert missing == ["O2"] and np.all(bip[BIPOLAR.index(("T6", "O2"))] == 0)


def _fake_physionet(monkeypatch, supports_range):
    """In-memory PhysioNet: one patient, hours 10..15, fs 256, 2 minutes per hour, 20 channels."""
    names = HMS_NAMES + ["Fpz"]
    fs, n = 256, 256 * 120
    files = {icare.BASE + "RECORDS": b"training/0001/\n", icare.BASE + "0001/0001.txt": b"Patient: 0001\nHospital: B\n"
             b"Age: 60\nSex: Male\nOutcome: Poor\nCPC: 5\n"}
    recs, truth = [], {}
    for seg, h in enumerate(range(10, 16), 1):
        rec = f"0001_{seg:03d}_{h:03d}_EEG"
        recs.append(rec)
        adc = rng.integers(-2000, 2000, size=(n, len(names))).astype("<i2")
        truth[h] = adc
        hea = f"{rec} {len(names)} {fs} {n}\n" + "".join(
            f"{rec}.mat 16+24 100(0)/uV 16 0 0 0 0 {nm}\n" for nm in names) + "#Utility frequency: 60\n"
        files[f"{icare.BASE}0001/{rec}.hea"] = hea.encode()
        files[f"{icare.BASE}0001/{rec}.mat"] = b"\0" * 24 + adc.tobytes()
    files[icare.BASE + "0001/RECORDS"] = "\n".join(recs).encode()

    def fake_get(url, start=None, end=None, **_):
        if url not in files:
            import urllib.error
            raise urllib.error.HTTPError(url, 404, "nf", None, None)
        data = files[url]
        if start is None:
            return data, False
        return data[start:end + 1], supports_range

    monkeypatch.setattr(icare, "http_get", fake_get)
    return truth, fs, n


def test_fetch_block_reads_the_middle_of_the_hour(monkeypatch):
    for supports in (True, False):
        truth, fs, n = _fake_physionet(monkeypatch, supports)
        x, header, ranged = icare.fetch_block("0001", "0001_003_012_EEG", minutes=1)
        need = 60 * fs
        s0 = (n - need) // 2
        assert ranged is supports and x.shape == (20, need)
        assert np.allclose(x, truth[12][s0:s0 + need].T / 100.0)


def test_fetch_script_end_to_end(monkeypatch, tmp_path):
    _fake_physionet(monkeypatch, True)
    spec = importlib.util.spec_from_file_location("fetch", Path(__file__).resolve().parents[1] / "scripts" /
                                                  "09_icare_fetch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = type("A", (), {"hours_from": 12, "hours_to": 72, "max_hours": 6, "minutes": 1})()
    meta, n = mod.do_patient("0001", args, tmp_path)
    assert meta["Outcome"] == "Poor" and n == 4  # hours 12..15
    w = np.load(tmp_path / "0001_012.npy")
    assert w.shape == (1, 18, 5000) and w.dtype == np.float16  # 60 s - 2 x 5 s margin -> one 50 s window
    assert mod.do_patient("0001", args, tmp_path)[1] == 4  # second run skips finished blocks


def test_listing_parsers_and_hour_selection():
    assert icare.patients_from_records("training/0284/\n0286/\n0296\n") == ["0284", "0286", "0296"]
    html = '<a href="0284/">0284/</a> <a href="0286/">0286/</a> Published Dec. 14, 2023'
    assert icare.patients_from_html(html) == ["0284", "0286"]
    recs = icare.eeg_records_from_text("0582_001_003_EEG\n0582_002_004_EEG\n0582_002_004_ECG\n", "0582")
    assert recs == ["0582_001_003_EEG", "0582_002_004_EEG"]
    spec = importlib.util.spec_from_file_location("fetch", Path(__file__).resolve().parents[1] / "scripts" /
                                                  "09_icare_fetch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    records = [f"0582_{i:03d}_{h:03d}_EEG" for i, h in enumerate([3, 12, 13, 13, 20, 30, 40, 50, 71, 72], 1)]
    sel = mod.choose_hours(records, 12, 72, 3)
    assert [h for h, _ in sel] == [12, 30, 71] and sel[1][1] == "0582_006_030_EEG"
