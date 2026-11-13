# Authors: Tran Anh Tu, Huynh Hoang Nhut
"""Selective download of the I-CARE database (PhysioNet, open access, CC BY-NC-SA 4.0).

Records are named <patient>_<segment>_<hour>_EEG (.hea/.mat), one per recording hour; the third field is the hour
used by the PhysioNet 2023 Challenge (time since return of spontaneous circulation). WFDB .mat files store int16
samples interleaved across channels after a small header, so a contiguous stretch of time is a contiguous byte
range and can be fetched with an HTTP Range request instead of downloading the whole hour."""
import http.client
import re
import threading
import time
import urllib.error
import urllib.request

from .external import decode_block, parse_header

BASE = "https://physionet.org/files/i-care/2.1/training/"
_UA = {"User-Agent": "hms-jbhi-research/1.0 (python urllib)"}
_downloaded, _lock = 0, threading.Lock()


def bytes_downloaded():
    """Total bytes received by http_get in this process (all threads)."""
    return _downloaded


def _count(n):
    global _downloaded
    with _lock:
        _downloaded += n


def http_get(url, start=None, end=None, retries=6, timeout=120):
    """GET url, optionally bytes [start, end] inclusive. If the server ignores Range (status 200), the needed
    bytes are cut out of the stream. Returns (bytes, range_supported)."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=dict(_UA))
            if start is not None:
                req.add_header("Range", f"bytes={start}-{end}")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if start is not None and r.status == 200:
                    data = r.read(end + 1)
                    _count(len(data))
                    return data[start:end + 1], False
                data = r.read()
                _count(len(data))
                return data, start is not None
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as e:
            # HTTPException covers IncompleteRead (connection dropped mid-transfer) -> retry the same range
            if isinstance(e, urllib.error.HTTPError) and e.code == 404:
                raise
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def patients_from_records(text):
    return sorted(set(re.findall(r"(?m)^\s*(?:training/)?(\d{4})/?\s*$", text)))


def patients_from_html(text):
    return sorted(set(re.findall(r'href="(\d{4})/"', text)))


def eeg_records_from_text(text, pid):  # works for a RECORDS file and for an HTML listing
    return sorted(set(re.findall(rf"({pid}_\d{{3}}_\d{{3}}_EEG)", text)))


def _listing(url, from_records, from_html):
    """Parse the RECORDS file of a folder; fall back to its HTML directory listing."""
    for target, parse in ((url + "RECORDS", from_records), (url, from_html)):
        try:
            found = parse(http_get(target)[0].decode("utf-8", "replace"))
        except urllib.error.HTTPError:
            continue
        if found:
            return found
    return []


def list_patients():
    return _listing(BASE, patients_from_records, patients_from_html)


def list_eeg_records(pid):
    f = lambda t: eeg_records_from_text(t, pid)  # noqa: E731
    return _listing(f"{BASE}{pid}/", f, f)


def record_hour(rec):
    return int(rec.split("_")[2])


def read_metadata(pid):
    text = http_get(f"{BASE}{pid}/{pid}.txt")[0].decode("utf-8", "replace")
    meta = {}
    for ln in text.splitlines():
        if ":" in ln:
            k, v = ln.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta


def fetch_block(pid, rec, minutes):
    """Middle `minutes` of one recording hour. Returns (physical uV array nsig x n, header, range_supported),
    or (None, header, None) when the recording is shorter than half the requested block."""
    header = parse_header(http_get(f"{BASE}{pid}/{rec}.hea")[0].decode())
    fs, nsig, nsamp = header["fs"], header["nsig"], header["nsamp"]
    need = int(minutes * 60 * fs)
    if nsamp < need // 2:
        return None, header, None
    need = min(need, nsamp)
    s0 = (nsamp - need) // 2
    sig0 = header["signals"][0]
    assert all(s["file"] == sig0["file"] and s["offset"] == sig0["offset"] for s in header["signals"])
    a = sig0["offset"] + 2 * nsig * s0
    raw, ranged = http_get(f"{BASE}{pid}/{sig0['file']}", a, a + 2 * nsig * need - 1)
    return decode_block(raw, header), header, ranged
