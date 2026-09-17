"""One browser dashboard for everything in 06-14, served over plain HTTP
polling -- the same BrainFlow-to-Python-to-browser pipeline as before, just
without a WebSocket.

    python examples/15_web_dashboard.py --board UNICORN_BOARD --serial-number UN-2022.04.52
    python examples/15_web_dashboard.py --config configs/alpha_attention_unicorn.yaml

Real hardware only -- the headset connects here, at backend start, from
``--board``/``--serial-number`` or ``--config``, exactly like every other
script in this directory.

Two ways to get this data: run the matplotlib scripts 06-14 directly at the
terminal (unedited, unchanged), or run this script and open the printed
``http://127.0.0.1:<port>`` URL. Both take the same ``--board``/
``--serial-number``/``--config`` arguments.

Why HTTP polling instead of a WebSocket
----------------------------------------
The first version of this script used this project's existing
``neurobridge.transport.websocket.WebSocketServer``. On at least one real
deployment machine, every WebSocket upgrade handshake through that server --
and through an independent, from-scratch RFC 6455 implementation written
purely to rule out a library bug -- was corrupted somewhere between the
browser and the Python process: the server computed a value that, byte for
byte, matched what a captured real request required, and the browser (or
.NET's ClientWebSocket, tested independently) still rejected it as invalid.
Plain HTTP GET/POST to the same machine, same port range, same process,
never showed this problem. That points at something inspecting/rewriting
WebSocket-upgrade traffic specifically at the OS network layer (a security
product's traffic-inspection module, in the one case this was diagnosed on)
rather than a bug in either WebSocket implementation. Since a page cannot
reliably tell in advance whether a given machine has that problem, this
version sidesteps the entire class of failure: it never sends an `Upgrade:
websocket` header at all. The browser polls ``GET /api/state`` roughly every
150ms and posts commands to ``POST /api/command`` -- both are exactly the
kind of request that was already proven to work.

Two recording tabs, on purpose
-------------------------------
"Record (Focus)" is 19_record_focus_states.py: wall-clock spans, the whole
session held in memory until you stop. "Record (7-class)" is
33_record_four_classes.py (still named for the protocol it was written for):
the same seven classes and the same ``Session_Classification`` layout, so
20_train_focus_classifier.py reads both without knowing which made them, but
labels go into the stream's own marker row (so a boundary sits on the exact
sample), the EEG is stored exactly as it arrived, and every chunk is flushed
to a CSV so a crash loses nothing. The 7-class tab is the better one to record
with; the focus tab stays because the recordings already made with it are
still valid.

The seven classes are two eye baselines and five motor states: Eye Open (also
the resting baseline), Eye Close, Clenching Left Hand, Clenching Right Hand,
Moving Left Leg, Moving Right Leg, Moving Tongue. Internally the 33-derived
tab, its command names and its ``--four-*`` options all still say "four"; the
names are kept so existing command lines and the browser's tab ids keep
working, and only what the operator reads says seven.

Resource design
----------------
Every tab's underlying feature is *cheap* except two: ICA (a fresh FastICA
fit) and connectivity (28 pairwise coherence + PLV computations on an 8
channel device). Recomputing those every 50ms regardless of which tab is
open would waste CPU for no one looking at the result, so this script tracks
which view the browser currently has selected (via the ``view`` query
parameter on each poll) and only runs the expensive analyses when that view
is active. The trace, quality, focus-game, time-domain and frequency-domain
features are cheap enough to always run, so switching tabs feels instant
rather than waiting for the first computation.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np
from scipy import signal as sp_signal
from scipy import stats
from sklearn.decomposition import FastICA

from neurobridge.core.buffer import RingBuffer
from neurobridge.core.montage import Montage, parse_label
from neurobridge.io.recording import (END_OFFSET, Recorder, StreamingRecorder,
                                      epochs_from_recording, load_recording,
                                      spans_from_markers)
from neurobridge.process.dropouts import DropoutRepair
from neurobridge.process.features import integrate_band, welch_psd
from neurobridge.process.filters import StreamFilter
from neurobridge.process.quality import QualityMonitor, spectral_slope
# badge_for, not a reimplementation of it: the browser badges then say exactly
# what 32/33's matplotlib badges say. neurobridge.viz imports matplotlib only
# inside ChannelStackView, so this is safe in a headless server process.
from neurobridge.viz import badge_for

BANDS = {
    "delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0), "gamma": (30.0, 50.0),
}

#: 06_live_scope.py's own band table for its alpha-bar panel -- delta starts at
#: 1 Hz, gamma stops at 45 Hz, and every band is a fraction of the 1-45 Hz
#: total. Deliberately separate from BANDS above, which the band-filter, sync
#: and topomap tabs use with 08/12/24's edges: the scope tab should agree with
#: the script it mirrors, not with its neighbouring tabs.
SCOPE_BANDS = {
    "delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0), "gamma": (30.0, 45.0),
}
SCOPE_TOTAL = (1.0, 45.0)

#: Points per channel on the wire for the live trace, as min/max pairs -- so
#: half this many bins. The old 300 was sized for the 700px-wide canvas the
#: scope tab used to have; the scope and record tabs now size their canvas to
#: the window, and at 1100-1700px a 300-point trace is drawn as visible
#: zig-zag rather than as a waveform. 600 points is 300 bins over the 5s trace
#: window -- 60 bins a second, about 4 samples a bin at 250 Hz -- which is the
#: same resolution the 4-class tab already asks for.
TRACE_POINTS = 600

# ---------------------------------------------------------------------------
# 19/20/21_..._focus_*.py integration: record the seven focus/motor states
# (Eye Open -- which doubles as the resting baseline -- Eye Close, Clenching
# Left/Right Hand, Moving Left/Right Leg, Moving Tongue) with participant/session
# management, train via 20 (as a subprocess -- its
# cross-validation and model-comparison logic has no live-acquisition
# component, so running it unmodified guarantees the dashboard agrees with
# the terminal tool instead of risking a hand-copied drift in the trickiest
# part of that file), and classify live via the identical feature pipeline
# duplicated (by hand, like everywhere else in this file) from 20/21.
# ---------------------------------------------------------------------------

FOCUS_TRAIN_SCRIPT = Path(__file__).resolve().with_name("20_train_focus_classifier.py")
FOCUS_BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
FOCUS_ROLES = ("theta_frontal", "alpha_posterior")
FOCUS_LABELS = (
    "Eye Open", "Eye Close",
    "Clenching Left Hand", "Clenching Right Hand",
    "Moving Left Leg", "Moving Right Leg", "Moving Tongue",
)

#: Same pseudonymous-code pattern 19_record_focus_states.py uses.
_PARTICIPANT_RE = re.compile(r"^[A-Z]{2,4}-\d{3,5}$")

# ---------------------------------------------------------------------------
# 33_record_four_classes.py integration -- the "Record (7-class)" tab. Same
# seven classes, participant codes and Session_Classification layout as
# 19_record_focus_states.py, so 20_train_focus_classifier.py trains on
# recordings from all three without knowing which made them. What it takes
# from 33 rather than 19 is the *recording* logic: labels written into the
# stream's own marker row so they sit on the exact sample, a CSV flushed every
# chunk so a crash loses nothing, and samples stored exactly as they arrived
# with no dropout repair and no filtering.
# ---------------------------------------------------------------------------

#: Marker code per label, as 33 -- and identical to 33's CODES, which is the
#: point: the two tabs and the terminal script must agree on what a code in a
#: recorded marker row means. Codes 1-4 are frozen for the sessions already on
#: disk; the leg and tongue classes were appended as 5-7 rather than renumbered.
FOUR_CODES = {1: "Eye Open", 2: "Eye Close",
              3: "Clenching Left Hand", 4: "Clenching Right Hand",
              5: "Moving Left Leg", 6: "Moving Right Leg", 7: "Moving Tongue"}
FOUR_CODE_OF = {label: code for code, label in FOUR_CODES.items()}
FOUR_SHORT = {"Eye Open": "eyes open", "Eye Close": "eyes closed",
              "Clenching Left Hand": "L hand", "Clenching Right Hand": "R hand",
              "Moving Left Leg": "L leg", "Moving Right Leg": "R leg",
              "Moving Tongue": "tongue"}

FOCUS_FEATURE_NAMES = [
    # frequency domain
    "posterior_alpha_rel", "posterior_beta_rel", "posterior_theta_rel",
    "frontal_alpha_rel", "frontal_engagement", "frontal_theta_beta",
    "global_spectral_entropy", "global_alpha_rel",
    "frontal_posterior_alpha_ratio",
    # time domain
    "posterior_amplitude", "posterior_zcr", "frontal_amplitude", "frontal_zcr",
    # synchronicity
    "frontal_posterior_coherence_alpha", "frontal_posterior_plv_alpha",
]


def next_focus_session(sessions_root: Path, participant: str) -> int:
    """Identical to 33_record_four_classes.py's next_session.

    33 counts a session that crashed and left only its CSV, which 19's version
    (globbing ``.npz`` alone) does not. Both recording tabs write into the same
    participant folder, so they have to agree: numbering off the NPZs alone
    would hand a new session the number of one that died mid-recording, and
    StreamingRecorder then refuses to start rather than overwrite it.
    """
    pdir = sessions_root / participant
    pdir.mkdir(parents=True, exist_ok=True)
    nums = []
    for p in pdir.glob(f"{participant}_session*_*"):
        m = re.search(r"_session(\d+)_", p.name)
        if m and p.suffix in (".npz", ".csv"):
            nums.append(int(m.group(1)))
    return max(nums, default=0) + 1


def append_focus_registry(sessions_root: Path, record: dict) -> None:
    """Identical to 19_record_focus_states.py's append_registry."""
    path = sessions_root / "registry.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def frontal_posterior_connectivity(window: np.ndarray, sf: float, f_idx: list[int],
                                   p_idx: list[int], lo: float = 8.0, hi: float = 13.0
                                   ) -> tuple[float, float]:
    """Identical to 20/21_..._focus_*.py's frontal_posterior_connectivity."""
    nperseg = min(window.shape[1], int(2 * sf))
    nyq = sf / 2.0
    sos = sp_signal.butter(4, [lo / nyq, min(hi, nyq * 0.99) / nyq],
                           btype="bandpass", output="sos")
    band_limited = sp_signal.sosfiltfilt(sos, window, axis=-1)
    phase = np.angle(sp_signal.hilbert(band_limited, axis=-1))

    coh_vals, plv_vals = [], []
    for fi in f_idx:
        for pi in p_idx:
            freqs, cxy = sp_signal.coherence(window[fi], window[pi], fs=sf, nperseg=nperseg)
            band = (freqs >= lo) & (freqs <= hi)
            coh_vals.append(float(np.mean(cxy[band])) if np.any(band) else 0.0)
            diff = phase[fi] - phase[pi]
            plv_vals.append(float(np.abs(np.mean(np.exp(1j * diff)))))
    return float(np.mean(coh_vals)), float(np.mean(plv_vals))


def extract_focus_features(window: np.ndarray, sf: float,
                           role_idx: dict[str, list[int]]) -> np.ndarray:
    """Identical to 20/21_..._focus_*.py's extract_features, FOCUS_FEATURE_NAMES order."""
    freqs, psd = welch_psd(window, sf, nperseg=window.shape[1])
    total = np.maximum(integrate_band(freqs, psd, 0.5, min(50.0, sf / 2 - 1)), 1e-12)
    p_idx = role_idx["alpha_posterior"]
    f_idx = role_idx["theta_frontal"]

    def rel(lo, hi, idx):
        return float(np.mean(integrate_band(freqs, psd, lo, hi)[idx] / total[idx]))

    posterior_alpha = rel(*FOCUS_BANDS["alpha"], p_idx)
    posterior_beta = rel(*FOCUS_BANDS["beta"], p_idx)
    posterior_theta = rel(*FOCUS_BANDS["theta"], p_idx)
    frontal_alpha = rel(*FOCUS_BANDS["alpha"], f_idx)
    frontal_theta = rel(*FOCUS_BANDS["theta"], f_idx)
    frontal_beta = rel(*FOCUS_BANDS["beta"], f_idx)
    frontal_engagement = frontal_beta / max(frontal_alpha + frontal_theta, 1e-12)
    frontal_theta_beta = frontal_theta / max(frontal_beta, 1e-12)

    all_idx = list(range(window.shape[0]))
    global_alpha = rel(*FOCUS_BANDS["alpha"], all_idx)
    mask = freqs >= 1
    entropies = []
    for i in all_idx:
        p = psd[i][mask] / max(np.sum(psd[i][mask]), 1e-12)
        p = p[p > 0]
        entropies.append(float(-np.sum(p * np.log2(p)) / np.log2(p.size)) if p.size > 1 else 0.0)
    global_entropy = float(np.mean(entropies))

    ratio = frontal_alpha / max(posterior_alpha, 1e-12)

    centred = window - window.mean(axis=1, keepdims=True)
    amp_per_ch = np.sqrt(np.mean(centred ** 2, axis=1))
    signs = np.sign(centred)
    signs[signs == 0] = 1.0
    crossings = np.sum(signs[:, :-1] != signs[:, 1:], axis=1)
    zcr_per_ch = crossings / max((window.shape[1] - 1) / sf, 1e-12)
    posterior_amplitude = float(np.mean(amp_per_ch[p_idx]))
    posterior_zcr = float(np.mean(zcr_per_ch[p_idx]))
    frontal_amplitude = float(np.mean(amp_per_ch[f_idx]))
    frontal_zcr = float(np.mean(zcr_per_ch[f_idx]))

    coherence_alpha, plv_alpha = frontal_posterior_connectivity(window, sf, f_idx, p_idx, 8.0, 13.0)

    return np.array([
        posterior_alpha, posterior_beta, posterior_theta,
        frontal_alpha, frontal_engagement, frontal_theta_beta,
        global_entropy, global_alpha, ratio,
        posterior_amplitude, posterior_zcr, frontal_amplitude, frontal_zcr,
        coherence_alpha, plv_alpha,
    ], dtype=float)


class RunningStatsVec:
    """Welford's algorithm over a feature vector -- identical to
    21_realtime_focus_classifier.py's RunningStats.
    """

    def __init__(self, n_features: int) -> None:
        self.n = 0
        self.mean = np.zeros(n_features)
        self._m2 = np.zeros(n_features)

    def update(self, x: np.ndarray) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self._m2 += d * (x - self.mean)

    @property
    def sd(self) -> np.ndarray:
        if self.n > 1:
            sd = np.sqrt(self._m2 / (self.n - 1))
        else:
            sd = np.zeros_like(self.mean)
        sd[sd < 1e-8] = 1.0
        return sd

    def zscore(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.sd


# ---------------------------------------------------------------------------
# Reused, self-contained analysis functions -- kept identical in spirit to
# 07_focus_game_V1.py / 09_ica_artifact_removal.py / 10-12, on purpose: this
# dashboard should agree with the terminal tools, not drift from them.
# ---------------------------------------------------------------------------

def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class RunningBaseline:
    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self._m2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self._m2 += d * (x - self.mean)

    @property
    def sd(self) -> float:
        return math.sqrt(self._m2 / (self.n - 1)) if self.n > 1 else 0.0

    def z(self, x: float) -> float:
        return (x - self.mean) / self.sd if self.sd > 0 else float("nan")


def time_domain_features(window: np.ndarray, sf: float):
    centred = window - window.mean(axis=1, keepdims=True)
    amplitude = np.sqrt(np.mean(centred ** 2, axis=1))
    mad = np.mean(np.abs(centred), axis=1)
    signs = np.sign(centred)
    signs[signs == 0] = 1.0
    crossings = np.sum(signs[:, :-1] != signs[:, 1:], axis=1)
    duration = (window.shape[1] - 1) / sf
    zcr = crossings / max(duration, 1e-12)
    return amplitude, mad, zcr


def spectral_entropy(psd_row: np.ndarray) -> float:
    p = psd_row / max(np.sum(psd_row), 1e-12)
    p = p[p > 0]
    if p.size < 2:
        return 0.0
    h = -np.sum(p * np.log2(p))
    return float(h / np.log2(p.size))


def classify_components(sources, mixing, sf, frontal_idx, kurtosis_thresh,
                        muscle_slope_thresh=-0.7):
    """Identical to the classifier in 09_ica_artifact_removal.py, kept in sync by hand."""
    n_comp = sources.shape[0]
    bad, reasons = [], {}
    for k in range(n_comp):
        s = sources[k]
        kurt = float(stats.kurtosis(s, fisher=True))
        topo = np.abs(mixing[:, k])
        frontal_frac = float(topo[frontal_idx].sum() / max(topo.sum(), 1e-12))
        is_blink = kurt > kurtosis_thresh and frontal_frac > 0.35

        freqs, psd = welch_psd(s[None, :], sf, nperseg=min(len(s), int(4 * sf)))
        slope = spectral_slope(freqs, psd[0])
        is_muscle = math.isfinite(slope) and slope > muscle_slope_thresh

        if is_blink:
            bad.append(k)
            reasons[k] = "blink/saccade"
        elif is_muscle:
            bad.append(k)
            reasons[k] = "muscle/jaw"
    return bad, reasons


def clean_epoch_focus(window, sf, frontal_idx, kurtosis_thresh, muscle_slope_thresh):
    """Identical to 20/21_..._focus_*.py's clean_epoch (reuses this file's own
    classify_components, already kept identical to those two files' copy).
    """
    n_comp = window.shape[0]
    ica = FastICA(n_components=n_comp, whiten="unit-variance", max_iter=500, random_state=0)
    try:
        sources = ica.fit_transform(window.T).T
    except Exception:
        return window, []
    bad, reasons = classify_components(sources, ica.mixing_, sf, frontal_idx,
                                       kurtosis_thresh, muscle_slope_thresh)
    sources_clean = sources.copy()
    sources_clean[bad] = 0.0
    cleaned = ica.mixing_ @ sources_clean
    if getattr(ica, "mean_", None) is not None:
        cleaned = cleaned + ica.mean_[:, None]
    return cleaned, [reasons[k] for k in bad]


def coherence_matrix(window: np.ndarray, sf: float, lo: float, hi: float) -> np.ndarray:
    n_ch = window.shape[0]
    mat = np.eye(n_ch)
    nperseg = min(window.shape[1], int(2 * sf))
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            freqs, cxy = sp_signal.coherence(window[i], window[j], fs=sf, nperseg=nperseg)
            band = (freqs >= lo) & (freqs <= hi)
            mat[i, j] = mat[j, i] = float(np.mean(cxy[band])) if np.any(band) else 0.0
    return mat


def plv_matrix(window: np.ndarray, sf: float, lo: float, hi: float) -> np.ndarray:
    n_ch, _ = window.shape
    nyq = sf / 2.0
    sos = sp_signal.butter(4, [lo / nyq, min(hi, nyq * 0.99) / nyq],
                           btype="bandpass", output="sos")
    band_limited = sp_signal.sosfiltfilt(sos, window, axis=-1)
    phase = np.angle(sp_signal.hilbert(band_limited, axis=-1))
    mat = np.eye(n_ch)
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            diff = phase[i] - phase[j]
            mat[i, j] = mat[j, i] = float(np.abs(np.mean(np.exp(1j * diff))))
    return mat


def decimate(row: np.ndarray, target: int = 150) -> list:
    """Stride-sample down to ~target points. Fine for a spectrum (PSD is
    already a smooth curve, not a fast waveform), wrong for a time-domain
    trace -- see decimate_minmax for why.
    """
    n = row.shape[-1]
    if n <= target:
        return np.round(row, 2).tolist()
    step = max(1, n // target)
    return np.round(row[..., ::step], 2).tolist()


def decimate_minmax(row: np.ndarray, target: int = 300) -> list:
    """Downsample a waveform for the wire without flattening it.

    Picking every Nth sample (``decimate``) silently drops peaks -- exactly
    what made the dashboard's live trace look smoother and lower-amplitude
    than the same signal plotted at full resolution in the matplotlib
    scripts. Each output bin instead keeps both the min and the max of the
    samples that fall in it, so a fast fluctuation still shows up as a fast
    fluctuation (a wide swing between consecutive output points), just at a
    coarser time resolution -- the standard technique for downsampling a
    waveform for display, used by every audio/EEG viewer that has to plot
    more samples than there are pixels.
    """
    n = row.shape[-1]
    if n <= target:
        return np.round(row, 2).tolist()
    n_bins = max(1, target // 2)
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    out: list[float] = []
    for i in range(n_bins):
        lo, hi = edges[i], max(edges[i] + 1, edges[i + 1])
        chunk = row[..., lo:hi]
        out.append(round(float(np.min(chunk)), 2))
        out.append(round(float(np.max(chunk)), 2))
    return out


#: Sources that carry a live signal off a person's head. Anything else -- a
#: replay of a recording, a synthetic generator -- is a development aid.
#: ReplaySource in particular defaults to loop=True and restarts the file from
#: sample 0 when it runs out, which on screen is indistinguishable from a
#: working headset: the trace moves, the quality score is whatever the
#: recording had, and every reading is meaningless. This dashboard presents its
#: trace as a live brain signal, so it refuses to start on anything else unless
#: asked in so many words.
LIVE_SOURCES = ("BrainFlowSource", "LSLSource")


def decimate_psd(row: np.ndarray, target: int = 200, digits: int = 4) -> list:
    """Stride-sample a PSD for the wire, keeping small values small.

    ``decimate`` finishes with ``np.round(row, 2)``. That is right for
    microvolts and destructive for a power spectral density: a real Oz spectrum
    from this headset runs from 1.25 down to 1e-6 uV^2/Hz, so two decimal
    places flatten 152 of 249 bins -- 7 of them inside 1-45 Hz -- to exactly
    0.0. The browser then plots log10(max(0, 1e-9)) = -9 and the curve spikes
    to the floor of a nine-decade axis at frequencies where there is nothing
    wrong with the signal at all. Significant figures keep the shape at every
    magnitude.

    Uses the same stride rule as ``decimate``, so the frequency axis published
    beside it still lines up bin for bin.
    """
    n = row.shape[-1]
    step = max(1, n // target) if n > target else 1
    a = np.asarray(row[..., ::step], dtype=float)
    out = np.zeros_like(a)
    nz = a != 0
    if nz.any():
        scale = 10.0 ** (digits - 1 - np.floor(np.log10(np.abs(a[nz]))))
        out[nz] = np.round(a[nz] * scale) / scale
    return out.tolist()


def build_source(args):
    if args.config:
        from neurobridge.config import Config

        # Whatever the YAML says, including whether the source repairs dropouts
        # internally -- with --config the 4-class tab's "stored exactly as it
        # arrived" guarantee is only as good as the config. --board below is
        # explicit about it.
        cfg = Config.from_file(args.config)
        return cfg.build_source()
    from neurobridge import BrainFlowSource

    params = {}
    if args.serial_port:
        params["serial_port"] = args.serial_port
    if args.serial_number:
        params["serial_number"] = args.serial_number
    # No repair inside the source, following 32/33: every recorder here then
    # writes samples exactly as they arrived, and load_recording() repairs
    # Bluetooth dropouts on load. Repairing before recording would bake an
    # estimate into the archive with no way to recover what the headset
    # actually sent. The live copies are repaired in the processing loop.
    return BrainFlowSource(args.board, mains=args.mains, repair_dropouts=False, **params)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, help="use the source from a config file")
    ap.add_argument("--board", help="BrainFlow board name, e.g. UNICORN_BOARD")
    ap.add_argument("--serial-port", help="COM3 on Windows, /dev/tty... elsewhere")
    ap.add_argument("--serial-number", help="e.g. UN-2022.04.52")
    ap.add_argument("--mains", type=float, default=60.0, choices=[50.0, 60.0])
    ap.add_argument("--display-low", type=float, default=0.5,
                    help="high-pass for the DISPLAYED trace only, in Hz (default 0.5). "
                         "Recording is unaffected -- what is written to disk is always "
                         "what the headset sent. Raise to 1.0 to match the band the "
                         "quality score is computed on, which removes most of the slow "
                         "electrode drift that makes traces wander out of their lanes")
    ap.add_argument("--allow-simulated", action="store_true",
                    help="permit a synthetic or replayed source. Off by default: "
                         "this dashboard presents its trace as a live brain signal, "
                         "and a looping ReplaySource is indistinguishable on screen "
                         "from a working headset")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090, help="single HTTP port for page + API")
    ap.add_argument("--window", type=float, default=4.0,
                    help="seconds of data kept for the heaviest views (ICA, sync)")
    ap.add_argument("--scale", type=float, default=50.0,
                    help="microvolts between channel baselines on the scope and "
                         "focus-recording traces -- same fixed-scale convention as "
                         "06_live_scope.py and 19_record_focus_states.py, not auto-scaled")
    ap.add_argument("--refit", type=float, default=0.5,
                    help="seconds between ICA/connectivity recomputations")
    ap.add_argument("--kurtosis-thresh", type=float, default=5.0)
    ap.add_argument("--muscle-slope-thresh", type=float, default=-0.7)
    ap.add_argument("--reject-uv", type=float, default=100.0)
    ap.add_argument("--sessions-root", type=Path, default=Path("Session_Classification"),
                    help="root folder for focus-state recordings (participant/session managed)")
    ap.add_argument("--calibrate", type=float, default=20.0,
                    help="seconds of live baseline before focus classification starts")
    ap.add_argument("--focus-refit", type=float, default=1.0,
                    help="seconds between focus-classifier predictions")
    # The 7-class tab mirrors 33_record_four_classes.py, whose display settings
    # are Unicorn Recorder's and differ from this dashboard's shared ones;
    # --window/--scale above already mean something else here, hence --four-*.
    ap.add_argument("--four-low", type=float, default=0.1,
                    help="7-class tab: display band-pass low edge, Hz")
    ap.add_argument("--four-high", type=float, default=30.0,
                    help="7-class tab: display band-pass high edge, Hz")
    ap.add_argument("--four-notch", type=float, default=60.0, choices=[0.0, 50.0, 60.0],
                    help="7-class tab: display notch, 0 for none")
    ap.add_argument("--four-scale", type=float, default=50.0,
                    help="7-class tab: fixed display half-range, µV")
    ap.add_argument("--four-window", type=float, default=8.0,
                    help="7-class tab: seconds on screen")
    ap.add_argument("--four-max-minutes", type=float, default=30.0,
                    help="7-class tab: hard cap on recording length")
    ap.add_argument("--quality-window", type=float, default=5.0,
                    help="seconds scored by the quality check, on a separate "
                         "1-45 Hz copy of the stream (32/33's LiveMonitor rule)")
    args = ap.parse_args()

    if args.four_scale <= 0 or args.four_window <= 0 or not 0 < args.four_low < args.four_high:
        ap.error("need a positive --four-scale and --four-window, and "
                 "0 < --four-low < --four-high")

    if not args.config and not args.board:
        ap.error("this dashboard only runs on real hardware: pass --board "
                 "(e.g. --board UNICORN_BOARD --serial-number UN-2022.04.52) "
                 "or --config pointing at a brainflow config")

    source = build_source(args)
    source_kind = type(source).__name__
    source_is_live = source_kind in LIVE_SOURCES
    if not source_is_live and not args.allow_simulated:
        ap.error(
            f"the source built here is a {source_kind}, which does not deliver a "
            f"live signal from a headset"
            + (f" (--config {args.config})" if args.config else "")
            + ". Every trace, spectrum and quality score on this dashboard would "
              "describe simulated data while looking exactly like a real "
              "recording -- a ReplaySource loops the file back to its first "
              "sample when it ends, so even the movement on screen is a lie. "
              "Use --board UNICORN_BOARD --serial-number <UN-....> for the real "
              "headset, or pass --allow-simulated if you genuinely want to drive "
              "the UI from simulated data.")
    if not source_is_live:
        print(f"!! {source_kind}: SIMULATED DATA, NOT A LIVE BRAIN SIGNAL "
              f"(--allow-simulated was passed)")
    source.start()
    device = source.device
    sf = device.sfreq
    names = list(device.ch_names)
    n_ch = len(names)

    frontal = Montage(names).resolve("theta_frontal")
    frontal_idx = list(frontal.indices) if frontal.ok else list(range(n_ch))

    # Spectrum channel for the scope tab: posterior by default, resolved the
    # same way 06_live_scope.py resolves it, falling back to the first channel.
    # That is where alpha lives, and opening the tab on whatever channel the
    # device happens to list first is what makes the eyes-closed demonstration
    # -- the most convincing thing the whole system does -- show nothing.
    posterior = Montage(names).resolve("alpha_posterior")
    spec_idx = posterior.indices[0] if posterior.ok else 0

    # Topomap positions: derived from each channel's 10-20 label (see
    # neurobridge.core.montage.parse_label), so this works on whatever
    # montage the connected device reports rather than one hard-coded layout.
    topo_idx = [i for i, nm in enumerate(names) if parse_label(nm) is not None]
    topo_names = [names[i] for i in topo_idx]
    topo_positions = [list(parse_label(names[i])[1:]) for i in topo_idx]

    print(f"device       {device.name} @ {sf} Hz")
    print(f"channels     {', '.join(names)}")
    print(f"spectrum     {names[spec_idx]}")
    if len(topo_idx) < 3:
        print(f"topomap      only {len(topo_idx)} channel(s) with a resolvable scalp "
             f"position; the topomap tab will stay empty")

    # -- shared, always-on infrastructure -----------------------------------
    # Four copies of the stream, the split 32/33's LiveMonitor uses: what
    # arrived (handed straight to the recorders, never touched), the repaired
    # broadband copy every analysis tab reads, a 1-45 Hz copy for the quality
    # check, and the 4-class tab's own Unicorn-Recorder-style display copy.
    repair = DropoutRepair(n_ch, sf, args.mains)
    disp_hi = min(50.0, sf / 2 - 5)
    broadband = StreamFilter(sf, n_ch, l_freq=args.display_low, h_freq=disp_hi,
                             notch=args.mains)
    # Never the display filter: the quality metrics are defined on 1-45 Hz
    # data, and scoring whatever a tab happens to be displaying is how the
    # badge ends up describing a filter rather than an electrode.
    quality_filter = StreamFilter(sf, n_ch, l_freq=1.0, h_freq=min(45.0, sf / 2 - 5),
                                  notch=args.mains)
    quality = QualityMonitor(mains=args.mains)
    quality_ring = RingBuffer(n_ch, max(int(round(args.quality_window * sf)), int(2 * sf)))
    n_show = int(args.window * sf)
    ring = RingBuffer(n_ch, n_show)
    n_trace = int(min(args.window, 5.0) * sf)

    # -- 7-class tab (33_record_four_classes.py) ----------------------------
    four_filter = StreamFilter(sf, n_ch, l_freq=args.four_low, h_freq=args.four_high,
                               notch=args.four_notch or None)
    four_show = max(1, int(round(args.four_window * sf)))
    four_ring = RingBuffer(n_ch, four_show)
    # How long the display filter's start-up transient may still be on screen:
    # 33's conservative rule of thumb, not a proof that it has gone.
    four_settle_s = max(8.0, 5.0 / args.four_low)
    four_headline = (f"{device.name}  |  {sf:g} Hz  |  {args.four_low:g}-{args.four_high:g} Hz"
                     + (f", {args.four_notch:g} Hz notch" if args.four_notch else ", no notch")
                     + f"  |  ±{args.four_scale:g} µV  |  no OSCAR  |  "
                     f"Good/Bad: 1-45 Hz, last {args.quality_window:g} s")
    # Markers go through the one thread that reads the source, exactly as
    # AcquisitionThread does it for 33: a click only queues a code, the reader
    # writes it into the board's own marker row, so the board is touched from
    # one thread and the label lands on the sample it belongs to.
    marker_q: queue.SimpleQueue = queue.SimpleQueue()
    native_markers = callable(getattr(source, "insert_marker", None))
    soft_markers: list[float] = []

    band_filters = {name: StreamFilter(sf, n_ch, l_freq=lo, h_freq=min(hi, sf / 2 - 5),
                                       notch=None)
                    for name, (lo, hi) in BANDS.items()}
    band_rings = {name: RingBuffer(n_ch, n_trace) for name in BANDS}

    entropy_hist = [float("nan")] * 60

    state = {
        "view": "scope",
        "channel": names[spec_idx],
        "band": "alpha",
        "topo_band": "alpha",
        "since_quality": 0,
        "quality_msg": {},
        # focus game
        "baseline": RunningBaseline(),
        "calib_start": time.time(),
        "calibrate_s": 8.0,
        "ema_focus": 0.0,
        "charge": 0.0,
        "below_since": None,
        "threshold": 0.60,
        "round": 1,
        "flash_until": 0.0,
        # ICA
        "last_ica_refit": 0.0,
        "n_ica_windows": 0,
        "n_ica_rejected": 0,
        # sync
        "last_sync_refit": 0.0,
        # recording
        "recorder": None,
        "rec_t0": None,
        "rec_path": None,
        "active_label": None,
        "active_label_start": None,
        "spans": [],
        "last_saved": None,
        # focus recording (19_record_focus_states.py equivalent)
        "focus_recorder": None,
        "focus_participant": None,
        "focus_session_n": None,
        "focus_rec_t0": None,
        "focus_path": None,
        "focus_spans": [],
        "focus_active_label": None,
        "focus_active_label_start": None,
        "focus_last_saved": None,
        # Sticky, because focus_record_status is republished every loop tick:
        # a note published once from a command handler is overwritten within
        # 50ms and the browser never renders it. 19_record_focus_states.py
        # puts both of these in front of the operator unmissably (argparse
        # refuses to start on a bad code; the ignored keypress prints), so
        # they have to survive here too rather than flash for one frame.
        "focus_note": "",
        # 7-class recording (33_record_four_classes.py equivalent)
        "four_recorder": None,
        "four_participant": None,
        "four_session_n": None,
        "four_rec_t0": None,
        "four_active": None,
        "four_since": 0.0,
        "four_count": {label: 0 for label in FOUR_CODES.values()},
        "four_secs": {label: 0.0 for label in FOUR_CODES.values()},
        "four_problems": [],
        "four_note": "",
        "four_last_saved": None,
        "four_stop_requested": False,
        # Badges and the last quality scores, kept so the 4-class trace can
        # label every channel between the once-a-second quality runs.
        "quality_badges": None,
        "last_chunk_at": None,
        "total_samples": 0,
        # focus training (20_train_focus_classifier.py, run as a subprocess)
        "train_status": "idle",
        "train_log": [],
        # focus classification (21_realtime_focus_classifier.py equivalent)
        "clf_bundle": None,
        "clf_role_idx": None,
        "clf_baseline": None,
        "clf_calib_start": None,
        "clf_last_refit": 0.0,
    }
    # RLock, not Lock: apply_command already holds this while dispatching, and
    # a couple of the focus command handlers (clf_load, run_focus_training)
    # need to publish() -- which also acquires this lock -- synchronously, so
    # the browser sees success/failure immediately instead of waiting for the
    # next periodic loop() tick. A plain Lock would deadlock on that reentry.
    lock = threading.RLock()

    #: The latest known value of every message "type" -- what /api/state
    #: returns each poll. Simple last-value-wins, which fits a dashboard of
    #: current readings (not a discrete event log) exactly.
    latest: dict[str, dict] = {
        "hello": {"device": device.name, "sfreq": sf, "channels": names,
                  # So the page can say, permanently and without being asked,
                  # whether what it is drawing came off someone's head.
                  "source": source_kind, "live": source_is_live,
                  "bands": {k: list(v) for k, v in BANDS.items()}, "window": args.window,
                  "default_channel": names[spec_idx], "scale_uv": args.scale,
                  "alpha_band": list(SCOPE_BANDS["alpha"]),
                  "topomap": {"channels": topo_names, "positions": topo_positions}},
    }

    def publish(msg_type: str, payload: dict) -> None:
        with lock:
            latest[msg_type] = payload

    # -- command handling -----------------------------------------------------
    def start_recording(cmd: dict) -> None:
        if state["recorder"] is not None:
            return
        path = cmd.get("path") or f"sessions/web_{datetime.now():%Y%m%d_%H%M%S}.npz"
        state["recorder"] = Recorder(device, path, max_seconds=1200.0)
        state["rec_t0"] = time.time()
        state["rec_path"] = path
        state["spans"] = []
        state["active_label"] = None
        print(f"  recording -> {path}")

    def stop_recording() -> None:
        rec = state["recorder"]
        if rec is None:
            return
        if state["active_label"] is not None:
            print(f"  warning: '{state['active_label']}' was still open at stop; discarding")
        saved = rec.close(annotations=state["spans"])
        print(f"  saved {saved} ({rec.n_samples} samples, {len(state['spans'])} span(s))")
        state["last_saved"] = {"path": str(saved), "n_samples": rec.n_samples,
                               "n_spans": len(state["spans"])}
        state["recorder"] = None
        state["rec_t0"] = None
        state["active_label"] = None

    def handle_label(cmd: dict) -> None:
        label = cmd.get("label")
        if not label or state["rec_t0"] is None:
            return
        now = time.time() - state["rec_t0"]
        if state["active_label"] is None:
            state["active_label"] = label
            state["active_label_start"] = now
        elif state["active_label"] == label:
            state["spans"].append({"label": label, "t_start": state["active_label_start"],
                                   "t_end": now})
            print(f"  logged {label}: {state['active_label_start']:.2f}s-{now:.2f}s")
            state["active_label"] = None
            state["active_label_start"] = None

    def run_validate(cmd: dict) -> None:
        def worker():
            path = cmd.get("path")
            try:
                data, dev, meta = load_recording(path)
            except Exception as exc:
                publish("validate_result", {"path": path, "error": str(exc)})
                return
            annotations = meta.get("annotations") or []
            if not annotations:
                publish("validate_result", {"path": path,
                                            "error": "no labelled spans in this recording"})
                return
            labels = [(a["t_start"], a["t_end"], a["label"]) for a in annotations]
            X, y = epochs_from_recording(data, dev, 2.0, 1.0, labels)
            if len(y) == 0:
                publish("validate_result", {"path": path,
                                            "error": "no window fit entirely inside a labelled span"})
                return
            f_idx = frontal_idx if dev.n_channels == n_ch else list(range(dev.n_channels))
            counts: dict[str, list[int]] = {}
            for window, label in zip(X, y):
                ica = FastICA(n_components=dev.n_channels, whiten="unit-variance",
                              max_iter=500, random_state=0)
                try:
                    sources = ica.fit_transform(window.T).T
                except Exception:
                    continue
                bad, _ = classify_components(sources, ica.mixing_, dev.sfreq, f_idx,
                                             args.kurtosis_thresh, args.muscle_slope_thresh)
                c = counts.setdefault(str(label), [0, 0])
                c[1] += 1
                if bad:
                    c[0] += 1
            rows = [{"label": lab, "flagged": v[0], "total": v[1]}
                   for lab, v in sorted(counts.items())]
            publish("validate_result", {"path": path, "rows": rows})

        threading.Thread(target=worker, daemon=True, name="validate").start()

    # -- focus recording (19_record_focus_states.py equivalent) -------------
    def focus_start_recording(cmd: dict) -> None:
        if state["focus_recorder"] is not None:
            return
        participant = (cmd.get("participant") or "").strip()
        if not _PARTICIPANT_RE.match(participant):
            state["focus_note"] = (
                f"'{participant}' is not a valid pseudonymous code (expected e.g. "
                f"PT-001: 2-4 uppercase letters, a dash, 3-5 digits). Never use a "
                f"real name.")
            print(f"  refused to start: {state['focus_note']}")
            return
        state["focus_note"] = ""
        session_n = next_focus_session(args.sessions_root, participant)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = (args.sessions_root / participant /
               f"{participant}_session{session_n:02d}_{stamp}.npz")
        state["focus_recorder"] = Recorder(device, path, max_seconds=1800.0)
        state["focus_participant"] = participant
        state["focus_session_n"] = session_n
        state["focus_rec_t0"] = time.time()
        state["focus_path"] = str(path)
        state["focus_spans"] = []
        state["focus_active_label"] = None
        state["focus_active_label_start"] = None
        print(f"  focus recording -> {path}")

    def focus_stop_recording() -> None:
        rec = state["focus_recorder"]
        if rec is None:
            return
        if state["focus_active_label"] is not None:
            state["focus_note"] = (f"'{state['focus_active_label']}' was still open at stop "
                                   f"and was discarded -- a span with no end is not a span")
            print(f"  warning: '{state['focus_active_label']}' was still open at stop; discarding")
        saved = rec.close(annotations=state["focus_spans"])
        by_label: dict[str, int] = {}
        for s in state["focus_spans"]:
            by_label[s["label"]] = by_label.get(s["label"], 0) + 1
        append_focus_registry(args.sessions_root, {
            "participant": state["focus_participant"], "session": state["focus_session_n"],
            "path": str(saved), "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_spans": len(state["focus_spans"]), "spans_by_label": by_label,
        })
        print(f"  saved {saved} ({rec.n_samples} samples, {len(state['focus_spans'])} span(s))")
        state["focus_last_saved"] = {"path": str(saved), "n_samples": rec.n_samples,
                                     "n_spans": len(state["focus_spans"])}
        state["focus_recorder"] = None
        state["focus_rec_t0"] = None
        state["focus_active_label"] = None
        state["focus_active_label_start"] = None

    def focus_handle_label(cmd: dict) -> None:
        label = cmd.get("label")
        if label not in FOCUS_LABELS or state["focus_rec_t0"] is None:
            return
        now = time.time() - state["focus_rec_t0"]
        if state["focus_active_label"] is None:
            state["focus_active_label"] = label
            state["focus_active_label_start"] = now
            state["focus_note"] = ""
        elif state["focus_active_label"] == label:
            state["focus_spans"].append({
                "label": label, "t_start": state["focus_active_label_start"], "t_end": now,
                "participant": state["focus_participant"], "session": state["focus_session_n"],
            })
            print(f"  logged {label}: {state['focus_active_label_start']:.2f}s-{now:.2f}s "
                 f"({now - state['focus_active_label_start']:.2f}s)")
            state["focus_active_label"] = None
            state["focus_active_label_start"] = None
            state["focus_note"] = ""
        else:
            # 19_record_focus_states.py prints exactly this and drops the press
            # rather than silently closing one span and opening another, which
            # would mislabel the boundary. Silence here was worse than in the
            # terminal: the button simply did nothing.
            state["focus_note"] = (f"'{label}' ignored: '{state['focus_active_label']}' is "
                                   f"still open -- press its button again to close it first")
            print(f"  ignored '{label}': '{state['focus_active_label']}' is still open, "
                 f"press its button again to close it first")

    # -- 7-class recording (33_record_four_classes.py equivalent) -----------
    def four_start_recording(cmd: dict) -> None:
        if state["four_recorder"] is not None:
            return
        participant = (cmd.get("participant") or "").strip()
        if not _PARTICIPANT_RE.match(participant):
            state["four_note"] = (
                f"'{participant}' is not a valid pseudonymous code (expected e.g. "
                f"PT-001: 2-4 uppercase letters, a dash, 3-5 digits). Never use a "
                f"real name.")
            print(f"  refused to start: {state['four_note']}")
            return
        session_n = next_focus_session(args.sessions_root, participant)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = (args.sessions_root / participant /
               f"{participant}_session{session_n:02d}_{stamp}.npz")
        try:
            recorder = StreamingRecorder(
                device, path, max_seconds=args.four_max_minutes * 60,
                extra_meta={"participant": participant, "session": session_n,
                            "mains": args.mains, "marker_codes": FOUR_CODES,
                            "marker_end_offset": END_OFFSET,
                            "script": Path(__file__).name + " (4-class tab)",
                            "markers": "board marker row" if native_markers
                                       else "written into the chunk's marker row",
                            "display_filter": {"low": args.four_low, "high": args.four_high,
                                               "notch": args.four_notch}})
        except FileExistsError as exc:
            state["four_note"] = f"could not start: {exc}"
            print(f"  refused to start: {exc}")
            return
        state["four_recorder"] = recorder
        state["four_participant"] = participant
        state["four_session_n"] = session_n
        state["four_rec_t0"] = time.time()
        state["four_active"] = None
        state["four_count"] = {label: 0 for label in FOUR_CODES.values()}
        state["four_secs"] = {label: 0.0 for label in FOUR_CODES.values()}
        state["four_problems"] = []
        state["four_note"] = ""
        print(f"  7-class recording -> {path}  (+ {recorder.csv_path.name} as you go)")

    def four_label(cmd: dict) -> None:
        """33's on_key: a press only queues a marker code; the reader places it."""
        label = cmd.get("label")
        if label not in FOUR_CODE_OF or state["four_recorder"] is None:
            return
        if state["four_active"] is None:
            marker_q.put(float(FOUR_CODE_OF[label]))
            state["four_active"], state["four_since"] = label, time.monotonic()
            state["four_note"] = ""
            print(f"  start {label}")
        elif state["four_active"] == label:
            marker_q.put(float(FOUR_CODE_OF[label] + END_OFFSET))
            dur = time.monotonic() - state["four_since"]
            state["four_count"][label] += 1
            state["four_secs"][label] += dur
            state["four_active"] = None
            state["four_note"] = ""
            print(f"  end   {label}  ({dur:.1f} s)")
        else:
            state["four_note"] = (f"'{label}' ignored: '{state['four_active']}' is still "
                                  f"open -- press its button again to close it first")
            print(f"  ignored '{label}': '{state['four_active']}' is still open -- "
                 f"press its button again to close it first")

    def four_stop_recording() -> None:
        """Finish the session. Runs on the loop thread, right after a drain.

        33 closes acquisition first, so every sample and marker the board had
        already delivered is in the recorder before the marker row is read.
        Stopping straight from the HTTP thread would instead read the markers
        while chunks were still arriving, and a span whose closing marker had
        not landed yet would be reported as never closed and thrown away.
        """
        rec = state["four_recorder"]
        if rec is None:
            return
        state["four_recorder"] = None
        # Spans come from the recorded marker row, not from wall-clock times:
        # every boundary is the sample the marker landed on. A span left open
        # is reported as a problem and dropped, never guessed at.
        spans, problems = spans_from_markers(
            rec.markers, sf, FOUR_CODES,
            participant=state["four_participant"], session=state["four_session_n"])
        problems = list(state["four_problems"]) + problems
        path = rec.close(annotations=spans, extra_meta={"marker_problems": problems})
        by_label: dict[str, int] = {}
        for s in spans:
            by_label[s["label"]] = by_label.get(s["label"], 0) + 1
        append_focus_registry(args.sessions_root, {
            "participant": state["four_participant"], "session": state["four_session_n"],
            "path": str(path), "csv": str(rec.csv_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_spans": len(spans), "spans_by_label": by_label,
            "script": Path(__file__).name + " (4-class tab)",
        })
        held = getattr(source, "held_total", 0)
        gaps = getattr(source, "counter_missing_total", 0)
        print(f"  saved {path}  ({rec.n_samples} samples, {len(spans)} labelled span(s); "
             f"{held} held samples, {gaps} packet-counter gaps)")
        for p in problems:
            print(f"    note: {p}")
        state["four_last_saved"] = {
            "path": str(path), "csv": str(rec.csv_path), "n_samples": rec.n_samples,
            "n_spans": len(spans), "spans_by_label": by_label, "problems": problems,
        }
        state["four_note"] = ""
        state["four_rec_t0"] = None
        state["four_active"] = None

    # -- focus training (20_train_focus_classifier.py, unmodified, as a
    # subprocess -- see the module-level comment above FOCUS_TRAIN_SCRIPT) ---
    def run_focus_training(cmd: dict) -> None:
        if state["train_status"] == "running":
            return
        sessions_root = Path(cmd.get("sessions_root") or args.sessions_root)
        out_path = Path(cmd.get("out") or (sessions_root / "focus_classifier.joblib"))
        state["train_status"] = "running"
        state["train_log"] = []
        publish("train_status", {"status": "running", "log": [], "out_path": str(out_path)})

        def worker():
            cmd_list = [sys.executable, str(FOCUS_TRAIN_SCRIPT),
                       "--sessions-root", str(sessions_root), "--out", str(out_path),
                       "--mains", str(args.mains),
                       "--kurtosis-thresh", str(args.kurtosis_thresh),
                       "--muscle-slope-thresh", str(args.muscle_slope_thresh)]
            try:
                proc = subprocess.Popen(cmd_list, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in proc.stdout:
                    state["train_log"].append(line.rstrip("\n"))
                    publish("train_status", {"status": "running",
                                             "log": state["train_log"][-300:],
                                             "out_path": str(out_path)})
                proc.wait()
                state["train_status"] = "done" if proc.returncode == 0 else "error"
            except Exception as exc:
                state["train_log"].append(f"error launching training: {exc}")
                state["train_status"] = "error"
            publish("train_status", {"status": state["train_status"],
                                     "log": state["train_log"][-300:],
                                     "out_path": str(out_path)})

        threading.Thread(target=worker, daemon=True, name="focus-train").start()

    # -- focus classification (21_realtime_focus_classifier.py equivalent) --
    def clf_load(cmd: dict) -> None:
        path = cmd.get("path") or "models/focus_classifier.joblib"
        try:
            bundle = joblib.load(path)
        except Exception as exc:
            publish("clf_status", {"loaded": False, "error": f"could not load {path}: {exc}"})
            return
        if list(bundle["channel_names"]) != names or float(bundle["sfreq"]) != sf:
            publish("clf_status", {"loaded": False,
                "error": (f"model trained on {list(bundle['channel_names'])} @ "
                         f"{bundle['sfreq']} Hz, this device reports {names} @ {sf} Hz -- "
                         f"retrain (20_train_focus_classifier.py) on this device first.")})
            return
        if bundle["window"] > args.window:
            publish("clf_status", {"loaded": False,
                "error": (f"model needs a {bundle['window']:g}s window but the dashboard "
                         f"was started with --window {args.window:g}; restart it with "
                         f"--window {bundle['window']:g} or larger.")})
            return
        montage = Montage(names)
        role_idx = {}
        for role in FOCUS_ROLES:
            m = montage.resolve(role)
            role_idx[role] = list(m.indices) if m.ok else list(range(n_ch))
        state["clf_bundle"] = bundle
        state["clf_role_idx"] = role_idx
        state["clf_baseline"] = RunningStatsVec(len(FOCUS_FEATURE_NAMES))
        state["clf_calib_start"] = time.time()
        state["clf_last_refit"] = 0.0
        publish("clf_status", {"loaded": True, "path": str(path),
            "model_name": bundle.get("model_name"), "classes": list(bundle["classes"]),
            "n_participants_trained_on": bundle.get("n_participants_trained_on")})
        publish("clf_result", {"phase": "calibrating", "remaining": round(args.calibrate, 1)})

    def clf_stop() -> None:
        state["clf_bundle"] = None
        state["clf_role_idx"] = None
        state["clf_baseline"] = None
        publish("clf_status", {"loaded": False})
        publish("clf_result", {})

    def apply_command(cmd: dict) -> None:
        action = cmd.get("cmd")
        with lock:
            if action == "set_channel" and cmd.get("channel") in names:
                state["channel"] = cmd["channel"]
            elif action == "set_band" and cmd.get("band") in BANDS:
                state["band"] = cmd["band"]
            elif action == "set_topo_band" and cmd.get("band") in BANDS:
                state["topo_band"] = cmd["band"]
            elif action == "start_recording":
                start_recording(cmd)
            elif action == "stop_recording":
                stop_recording()
            elif action == "label":
                handle_label(cmd)
            elif action == "validate":
                run_validate(cmd)
            elif action == "focus_start_recording":
                focus_start_recording(cmd)
            elif action == "focus_stop_recording":
                focus_stop_recording()
            elif action == "focus_label":
                focus_handle_label(cmd)
            elif action == "four_start_recording":
                four_start_recording(cmd)
            elif action == "four_stop_recording":
                # Handed to the loop thread rather than done here; see
                # four_stop_recording's docstring.
                state["four_stop_requested"] = True
            elif action == "four_label":
                four_label(cmd)
            elif action == "focus_train":
                run_focus_training(cmd)
            elif action == "clf_load":
                clf_load(cmd)
            elif action == "clf_stop":
                clf_stop()

    # -- HTTP server ------------------------------------------------------------
    web_dir = Path(__file__).resolve().parents[1] / "web" / "dashboard"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send_json(self, obj: dict, status: int = 200) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, rel_path: str) -> None:
            path = (web_dir / rel_path.lstrip("/")).resolve()
            if web_dir not in path.parents and path != web_dir:
                self.send_error(403)
                return
            if not path.is_file():
                self.send_error(404)
                return
            ctype = "text/html" if path.suffix == ".html" else (
                "application/javascript" if path.suffix == ".js" else
                "text/css" if path.suffix == ".css" else "application/octet-stream")
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/api/state":
                qs = parse_qs(parsed.query)
                view = (qs.get("view") or ["scope"])[0]
                # NOTE: "record" and "validate" were missing here originally,
                # which meant state["view"] never actually became "record" or
                # "validate" -- the ICA-on-the-record-tab badge only worked by
                # accident, if you happened to visit the ICA tab first in the
                # same run. Fixed while adding the three focus_* views below,
                # since the same gap would have silently broken them too.
                if view in ("scope", "focus", "bands", "ica", "time", "freq", "sync",
                           "topomap", "record", "validate", "record_focus", "train_focus",
                           "classify_focus", "record_four"):
                    with lock:
                        state["view"] = view
                with lock:
                    snapshot = dict(latest)
                self._send_json(snapshot)
                return
            rel = parsed.path if parsed.path != "/" else "/index.html"
            self._send_file(rel)

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != "/api/command":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b"{}"
            try:
                cmd = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json({"error": "invalid JSON"}, status=400)
                return
            apply_command(cmd)
            self._send_json({"ok": True})

    print(f"7-class      {four_headline}")
    if not native_markers:
        print("             this source has no marker row of its own; label codes are "
              "written onto the first sample of the next chunk instead")

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True, name="dashboard-http").start()
    print(f"dashboard    http://{args.host}:{args.port}/index.html")
    print("\nCtrl-C to stop.\n")

    # -- processing loop ------------------------------------------------------
    stop_event = threading.Event()

    def place_markers() -> None:
        """AcquisitionThread._step's marker handling, on this thread."""
        while True:
            try:
                code = marker_q.get_nowait()
            except queue.Empty:
                return
            if native_markers:
                try:
                    source.insert_marker(code)
                except Exception as exc:      # e.g. the stream already stopped
                    state["four_problems"].append(f"marker {code:g} was not placed: {exc}")
            else:
                soft_markers.append(code)

    def apply_soft_markers(chunk) -> None:
        """Sources with no marker row of their own: put the code on the first
        sample of the next chunk, as AcquisitionThread does."""
        if not soft_markers:
            return
        aux = dict(chunk.aux or {})
        marker = np.array(aux.get("marker", np.zeros(chunk.n_samples)), dtype=np.float64)
        k = min(len(soft_markers), chunk.n_samples)
        marker[:k] = soft_markers[:k]
        del soft_markers[:k]
        aux["marker"] = marker
        chunk.aux = aux

    def loop() -> None:
        last_pub = {"trace": 0.0, "scope": 0.0, "time": 0.0, "freq": 0.0, "bands": 0.0,
                   "topomap": 0.0, "four": 0.0}
        while not stop_event.is_set():
            for _ in range(50):
                # Before every read, as AcquisitionThread._step does it: a code
                # queued by the browser is placed on the next sample the board
                # delivers, not up to a poll later.
                place_markers()
                chunk = source.read()
                if chunk is None:
                    break
                apply_soft_markers(chunk)
                # Recorders first, and on the chunk as delivered: what is
                # written is exactly what arrived.
                if state["recorder"] is not None:
                    state["recorder"].add(chunk)
                if state["focus_recorder"] is not None:
                    state["focus_recorder"].add(chunk)
                rec4 = state["four_recorder"]
                if rec4 is not None:
                    try:
                        rec4.add(chunk)
                    except RuntimeError:
                        pass          # stopped from the browser mid-chunk
                state["total_samples"] += chunk.n_samples
                state["last_chunk_at"] = time.monotonic()
                # Everything on screen works from the repaired copy.
                repaired = repair(chunk.data)
                clean = broadband(repaired)
                ring.push(clean)
                for name, bf in band_filters.items():
                    band_rings[name].push(bf(clean))
                quality_ring.push(quality_filter(repaired))
                four_ring.push(four_filter(repaired))

            # Everything the board had already delivered is now in the
            # recorder, so this is the safe point to close it out.
            if state["four_stop_requested"]:
                state["four_stop_requested"] = False
                four_stop_recording()

            have = min(ring.total_written, n_show)
            now = time.time()
            if have < int(0.5 * sf):
                time.sleep(0.05)
                continue

            with lock:
                view = state["view"]
                channel = state["channel"]
                band = state["band"]
            ch_idx = names.index(channel)

            # -- trace (always, throttled) --------------------------------
            if now - last_pub["trace"] > 0.1:
                last_pub["trace"] = now
                have_trace = min(have, n_trace)
                window = ring.latest(have_trace)
                publish("trace", {"channels": names,
                                  "data": [decimate_minmax(row, TRACE_POINTS)
                                           for row in window]})

            # -- quality (~1 Hz) --------------------------------------------
            # Scored on the dedicated 1-45 Hz copy over the last
            # --quality-window seconds, which is what 32/33's LiveMonitor does
            # and the same data and window 06_live_scope.py scores. It used to
            # clamp to the last 2 s of the broadband copy: too short (one blink
            # or knock flips the readout) and the wrong band. The 2 s below is
            # a floor, not a window -- under it the spectral metrics have
            # nothing to fit.
            nq = min(quality_ring.total_written, quality_ring.capacity)
            state["since_quality"] += 1
            if nq >= int(2 * sf) and state["since_quality"] >= 20:
                state["since_quality"] = 0
                qs = quality(quality_ring.latest(nq), sf, names)
                # 33's per-channel badge, from neurobridge.viz.badge_for, so the
                # browser says "Bad: mains hum" where the matplotlib viewer does.
                state["quality_badges"] = [
                    {"text": t, "color": c} for t, c in (badge_for(q) for q in qs)]
                summary = quality.summary(qs)
                state["quality_msg"] = summary
                worst = [q for q in qs if not q.usable]
                lost = getattr(source, "held_fraction", 0.0) or 0.0
                publish("quality", {
                    "mean_score": summary["mean_score"],
                    "n_usable": summary["n_usable"], "n_channels": summary["n_channels"],
                    "advice": [q.advice() for q in worst],
                    # Per-channel usability, so the trace panels can colour each
                    # channel's label green or red the way 06_live_scope.py
                    # colours its y-tick labels: a lifting electrode shows up in
                    # the label before it is obvious in the trace, which is the
                    # whole point of having the readout during a recording.
                    "channels": names,
                    "per_channel": [q.score for q in qs],
                    "usable": [bool(q.usable) for q in qs],
                    # 06/19 both surface this: held Bluetooth samples inflate
                    # RMS and peak-to-peak far beyond their share of the data.
                    "held_fraction": round(float(lost), 4),
                })

            n_feat = int(2.0 * sf)
            have_feat = min(have, n_feat)
            if have_feat >= n_feat:
                feat_window = ring.latest(n_feat)
                freqs, psd = welch_psd(feat_window, sf, nperseg=n_feat)
                total_full = np.maximum(integrate_band(freqs, psd, 0.5, min(50.0, sf / 2 - 1)),
                                        1e-12)

                # -- scope (cheap, always) -----------------------------------
                if now - last_pub["scope"] > 0.2:
                    last_pub["scope"] = now
                    # Stop where the display filter stops. Publishing all the
                    # way to Nyquist put 150 of 249 bins into the Butterworth
                    # stopband -- 60% of the plotted curve was roll-off rather
                    # than signal, and it set the floor of the log axis, which
                    # squashed the real 1-45 Hz spectrum into a sliver at the
                    # top of the panel.
                    mask = (freqs >= 1) & (freqs <= disp_hi)
                    # The selected channel alone, against a 1-45 Hz total --
                    # 06_live_scope.py's alpha bar exactly. This used to average
                    # the relative powers across every channel, which buries the
                    # one thing the panel exists to show: posterior alpha rising
                    # when the eyes close. Frontal channels have no alpha to
                    # rise, and averaging them in cancels most of the change.
                    one = psd[ch_idx:ch_idx + 1]
                    nyq_hi = sf / 2 - 1
                    scope_total = max(
                        float(integrate_band(freqs, one, SCOPE_TOTAL[0],
                                             min(SCOPE_TOTAL[1], nyq_hi))[0]), 1e-12)
                    rp = {b: float(integrate_band(freqs, one, lo, min(hi, nyq_hi))[0]
                                   / scope_total)
                         for b, (lo, hi) in SCOPE_BANDS.items()}
                    publish("scope", {
                        "channel": channel,
                        "psd_freqs": decimate(freqs[mask], 200),
                        "psd_values": decimate_psd(np.maximum(psd[ch_idx][mask], 1e-12), 200),
                        "relative_power": rp,
                    })

                # -- time-domain (cheap, always) ------------------------------
                if now - last_pub["time"] > 0.2:
                    last_pub["time"] = now
                    amp, mad, zcr = time_domain_features(feat_window, sf)
                    publish("time_features", {"channels": names,
                                              "amplitude": np.round(amp, 3).tolist(),
                                              "mad": np.round(mad, 3).tolist(),
                                              "zcr": np.round(zcr, 3).tolist()})

                # -- frequency-domain (cheap, always) -------------------------
                if now - last_pub["freq"] > 0.2:
                    last_pub["freq"] = now
                    mask = freqs >= 1
                    ent = spectral_entropy(psd[ch_idx][mask])
                    entropy_hist.append(ent)
                    del entropy_hist[: len(entropy_hist) - 60]
                    publish("freq_features", {
                        "channel": channel, "entropy": round(ent, 4),
                        "entropy_history": [round(e, 4) if math.isfinite(e) else None
                                            for e in entropy_hist]})

                # -- topomap (only while that tab is open) ----------------------
                if (view == "topomap" and topo_idx
                       and now - last_pub["topomap"] > 0.2):
                    last_pub["topomap"] = now
                    lo, hi = BANDS[state["topo_band"]]
                    hi = min(hi, sf / 2 - 1)
                    band_power = integrate_band(freqs, psd, lo, hi) / total_full
                    publish("topomap", {
                        "band": state["topo_band"],
                        "power": np.round(band_power[topo_idx], 4).tolist(),
                    })

                # -- focus game (cheap, always) --------------------------------
                total_f = np.maximum(integrate_band(freqs, psd, 0.5, min(50.0, sf / 2 - 1)),
                                     1e-12)[frontal_idx]
                theta_rel = np.mean(integrate_band(freqs, psd, 4, 8)[frontal_idx] / total_f)
                alpha_rel = np.mean(integrate_band(freqs, psd, 8, 13)[frontal_idx] / total_f)
                beta_rel = np.mean(integrate_band(freqs, psd, 13, 30)[frontal_idx] / total_f)
                engagement = float(beta_rel / max(alpha_rel + theta_rel, 1e-12))

                elapsed = now - state["calib_start"]
                if elapsed < state["calibrate_s"]:
                    state["baseline"].update(engagement)
                    publish("focus", {"phase": "calibrating",
                                      "remaining": round(state["calibrate_s"] - elapsed, 1)})
                else:
                    z = state["baseline"].z(engagement)
                    score = sigmoid(z) if math.isfinite(z) else 0.5
                    state["ema_focus"] = 0.25 * score + 0.75 * state["ema_focus"]
                    focus = state["ema_focus"]
                    q_ok = state["quality_msg"].get("mean_score", 1.0) >= 0.6
                    engaged = focus >= state["threshold"] and q_ok
                    dt = 0.05
                    if engaged:
                        state["below_since"] = None
                        state["charge"] = min(2.5, state["charge"] + dt)
                    elif q_ok:
                        if state["below_since"] is None:
                            state["below_since"] = now
                        elif now - state["below_since"] > 1.0:
                            state["charge"] = 0.0
                    cleared = state["charge"] >= 2.5
                    if cleared:
                        state["round"] += 1
                        state["threshold"] = min(0.9, state["threshold"] + 0.03)
                        state["charge"] = 0.0
                        state["flash_until"] = now + 1.5
                    publish("focus", {
                        "phase": "playing", "round": state["round"],
                        "threshold": round(state["threshold"], 3), "focus": round(focus, 3),
                        "charge": round(state["charge"], 3), "hold": 2.5,
                        "cleared": now < state["flash_until"],
                    })

            # -- bands (only while that tab is open) --------------------------
            if view == "bands" and now - last_pub["bands"] > 0.1:
                last_pub["bands"] = now
                have_b = min(band_rings["alpha"].total_written, n_trace)
                if have_b >= int(0.5 * sf):
                    series = {name: decimate_minmax(band_rings[name].latest(have_b)[ch_idx])
                             for name in BANDS}
                    publish("bands", {"channel": channel, "series": series})

            # -- ICA (while that tab OR the record tab is open, refit cadence) --
            # Also runs on "record" so labelling a live span shows, in real
            # time, whether the automatic classifier agrees with what you're
            # doing -- closing the loop with 14_validate_ica_thresholds.py's
            # offline check instead of only reporting after the fact.
            if view in ("ica", "record") and have >= n_show and now - state["last_ica_refit"] > args.refit:
                state["last_ica_refit"] = now
                window = ring.latest(n_show)
                ica = FastICA(n_components=n_ch, whiten="unit-variance", max_iter=500,
                             random_state=0)
                try:
                    sources = ica.fit_transform(window.T).T
                    bad, reasons = classify_components(
                        sources, ica.mixing_, sf, frontal_idx, args.kurtosis_thresh,
                        args.muscle_slope_thresh)
                    sources_clean = sources.copy()
                    sources_clean[bad] = 0.0
                    cleaned = ica.mixing_ @ sources_clean
                    if getattr(ica, "mean_", None) is not None:
                        cleaned = cleaned + ica.mean_[:, None]
                    state["n_ica_windows"] += 1
                    rejected = bool(np.max(np.abs(cleaned)) > args.reject_uv)
                    if rejected:
                        state["n_ica_rejected"] += 1
                    flagged = [{"component": k, "reason": reasons[k]} for k in bad]
                    publish("ica", {
                        "channels": names,
                        "raw": [decimate_minmax(row) for row in window],
                        "cleaned": None if rejected else [decimate_minmax(row) for row in cleaned],
                        "flagged": flagged,
                        "rejected": rejected,
                        "n_windows": state["n_ica_windows"],
                        "n_rejected": state["n_ica_rejected"],
                        "live_label": "amplitude too high" if rejected
                                      else (", ".join(sorted({f["reason"] for f in flagged}))
                                           if flagged else "clean"),
                    })
                except Exception as exc:
                    publish("ica", {"error": str(exc)})

            # -- synchronicity (only while that tab is open, refit cadence) ----
            if view == "sync" and have >= n_show and now - state["last_sync_refit"] > args.refit:
                state["last_sync_refit"] = now
                window = ring.latest(n_show)
                lo, hi = BANDS[band]
                hi = min(hi, sf / 2 - 1)
                coh = coherence_matrix(window, sf, lo, hi)
                plv = plv_matrix(window, sf, lo, hi)
                publish("sync", {"band": band, "channels": names,
                                 "coherence": np.round(coh, 3).tolist(),
                                 "plv": np.round(plv, 3).tolist()})

            # -- focus classifier (only while that tab is open, model loaded) --
            bundle = state["clf_bundle"]
            if view == "classify_focus" and bundle is not None:
                n_clf = int(bundle["window"] * sf)
                if have >= n_clf and now - state["clf_last_refit"] > args.focus_refit:
                    state["clf_last_refit"] = now
                    clf_window = ring.latest(n_clf)
                    cleaned, artifact_reasons = clean_epoch_focus(
                        clf_window, sf, frontal_idx, bundle["kurtosis_thresh"],
                        bundle["muscle_slope_thresh"])
                    raw_feats = extract_focus_features(cleaned, sf, state["clf_role_idx"])
                    elapsed = now - state["clf_calib_start"]
                    if elapsed < args.calibrate:
                        state["clf_baseline"].update(raw_feats)
                        publish("clf_result", {"phase": "calibrating",
                                               "remaining": round(args.calibrate - elapsed, 1)})
                    else:
                        feats = state["clf_baseline"].zscore(raw_feats).reshape(1, -1)
                        model = bundle["model"]
                        classes = list(bundle["classes"])
                        pred = model.predict(feats)[0]
                        if hasattr(model, "predict_proba"):
                            proba = model.predict_proba(feats)[0]
                            class_order = list(model.classes_)
                        else:
                            proba = np.array([1.0 if c == pred else 0.0 for c in classes])
                            class_order = classes
                        probs = {c: float(proba[class_order.index(c)]) if c in class_order else 0.0
                                for c in classes}
                        publish("clf_result", {
                            "phase": "predicting", "prediction": str(pred),
                            "probabilities": probs,
                            "artifact_note": (", ".join(sorted(set(artifact_reasons)))
                                             if artifact_reasons
                                             else "no artifacts flagged this window"),
                        })

            # -- 7-class live view (only while that tab is open) ----------------
            # ChannelStackView's readout, computed here on the full-resolution
            # window: the trace is min/max decimated for the wire, but "outside
            # ±scale" and peak-to-peak are measured on every sample, so the two
            # numbers that say how much of a trace you are not seeing stay
            # exact even though the drawing is coarser than the data.
            if view == "record_four" and now - last_pub["four"] > 0.1:
                last_pub["four"] = now
                fh = min(four_ring.total_written, four_show)
                if fh >= int(0.25 * sf):
                    fw = four_ring.latest(fh)
                    stale = (state["last_chunk_at"] is None
                             or time.monotonic() - state["last_chunk_at"] > 2.0)
                    received = state["total_samples"] / sf
                    parts = ["NO RECENT DATA" if stale else "receiving"]
                    if received < four_settle_s:
                        parts.append(f"filter settling {received:.0f}/{four_settle_s:.0f} s")
                    if state["quality_msg"]:
                        parts.append(f"{state['quality_msg']['n_usable']}/"
                                     f"{state['quality_msg']['n_channels']} channels Good")
                    held = getattr(source, "held_fraction", None)
                    if held is not None:
                        parts.append(f"Bluetooth held {100 * held:.1f}% (10 s)")
                    gaps = getattr(source, "counter_missing_total", None)
                    if gaps is not None:
                        parts.append(f"counter gaps {gaps}")
                    publish("four_trace", {
                        "channels": names,
                        "data": [decimate_minmax(row, 600) for row in fw],
                        "outside": [round(float(100.0 * np.mean(np.abs(row) > args.four_scale)), 1)
                                    for row in fw],
                        "ptp": [round(float(np.ptp(row)), 1) for row in fw],
                        "badges": state["quality_badges"],
                        "scale": args.four_scale,
                        "seconds": args.four_window,
                        "headline": four_headline,
                        "status": "  |  ".join(parts),
                        "stale": bool(stale),
                    })

            # -- 7-class recording status (always, cheap) -----------------------
            rec4 = state["four_recorder"]
            publish("four_status", {
                "recording": rec4 is not None,
                "participant": state["four_participant"],
                "session": state["four_session_n"],
                "path": str(rec4.path) if rec4 is not None else None,
                "csv": str(rec4.csv_path) if rec4 is not None else None,
                "elapsed": round(now - state["four_rec_t0"], 1) if state["four_rec_t0"] else 0.0,
                "saved_seconds": round(rec4.n_samples / sf, 1) if rec4 is not None else 0.0,
                "truncated": bool(rec4.truncated) if rec4 is not None else False,
                "active_label": state["four_active"],
                "active_seconds": (round(time.monotonic() - state["four_since"], 1)
                                   if state["four_active"] else 0.0),
                # 33 keeps these on screen to keep the classes balanced, which
                # matters more than it sounds: an unbalanced set quietly costs
                # macro-F1 in 20_train_focus_classifier.py.
                "counts": [{"label": label, "short": FOUR_SHORT[label],
                            "n": state["four_count"][label],
                            "seconds": round(state["four_secs"][label], 0)}
                           for label in FOUR_CODES.values()],
                "note": state["four_note"],
                "last_saved": state["four_last_saved"],
            })

            # -- focus recording status (always, cheap) -------------------------
            publish("focus_record_status", {
                "recording": state["focus_recorder"] is not None,
                "participant": state["focus_participant"], "session": state["focus_session_n"],
                "path": state["focus_path"], "n_spans": len(state["focus_spans"]),
                "active_label": state["focus_active_label"],
                "elapsed": round(now - state["focus_rec_t0"], 1) if state["focus_rec_t0"] else 0.0,
                "last_saved": state["focus_last_saved"],
                "error": state["focus_note"],
            })

            # -- recording status (always, cheap) -------------------------------
            publish("record_status", {
                "recording": state["recorder"] is not None,
                "path": state["rec_path"], "n_spans": len(state["spans"]),
                "active_label": state["active_label"],
                "elapsed": round(now - state["rec_t0"], 1) if state["rec_t0"] else 0.0,
                "last_saved": state["last_saved"],
            })

            time.sleep(0.05)

    threading.Thread(target=loop, daemon=True, name="dashboard-loop").start()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        stop_event.set()
        if state["recorder"] is not None:
            stop_recording()
        # 19_record_focus_states.py closes its recorder and appends the registry
        # line in its own finally, so Ctrl-C there still leaves a usable session
        # on disk. Without this, Ctrl-C during a focus recording here threw the
        # whole session and its registry entry away.
        if state["focus_recorder"] is not None:
            focus_stop_recording()
        if state["four_recorder"] is not None:
            four_stop_recording()
        httpd.shutdown()
        source.stop()
        total = getattr(source, "samples_total", 0)
        if total:
            print(f"Bluetooth: {source.held_total} of {total} samples held "
                 f"({100 * source.held_total / total:.2f}%); "
                 f"packet-counter gaps: {source.counter_missing_total}")
        print("stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
