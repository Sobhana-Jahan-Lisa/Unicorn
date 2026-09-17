"""Train and compare classifiers for four states: Eye Open, Eye Close,
Clenching Left Hand, Clenching Right Hand.

Eye Open is the resting baseline -- there is deliberately no separate "Rest"
class, since eyes-open rest and Eye Open are the same physiological state.

Pipeline:

1. Load recordings made by 19_record_focus_states.py, either every
   participant under --sessions-root, or an explicit --recording list. The
   participant code is read from the parent folder name (the convention
   19_record_focus_states.py itself writes to), with a fallback to the
   "participant" key already stored in each labelled span for safety if a
   file ever gets moved out of its participant folder.
2. Re-apply the 0.5-50 Hz + 60 Hz notch filter (recordings are raw by
   design), cut into overlapping epochs tracking which labelled span each
   epoch came from, and clean each epoch with the same FastICA-based
   artifact removal as 09/16-18 before extracting features.
3. Extract a small, deliberately chosen feature set originally built for a
   three-way eye-state distinction and reused unchanged for the current
   four-way task (see FEATURE NOTES below) rather than reusing
   17_train_classifiers.py's broader 75-feature set built for five harder,
   more diffuse classes.
4. Normalise each participant's features against their *own* mean/sd before
   pooling across people -- standard practice for cross-subject EEG
   classification, since absolute band power varies enormously between
   people (skull thickness, impedance, electrode placement) in a way that
   has nothing to do with their mental state. This uses only each
   participant's own unlabelled feature distribution, so it does not leak
   any label information across the train/test boundary.
5. Evaluate with leave-one-participant-out cross-validation when >= 2
   participants are available -- the honest test of whether this
   generalises to someone the model has never seen -- falling back to
   span-grouped cross-validation (as in 17_train_classifiers.py) with a
   clear note if only one participant has been recorded so far.
6. Train the same classical model family as 17 (Logistic Regression, LDA,
   SVM, Random Forest, sklearn's MLPClassifier), report per-class
   performance for the winner, and save it plus everything
   21_realtime_focus_classifier.py needs to reproduce the exact
   preprocessing and run the equivalent per-session calibration.

FEATURE NOTES -- three domains, chosen deliberately, not exhaustively
------------------------------------------------------------------------
This feature set was originally designed for a three-way eye_close /
eye_open / reading distinction and is reused here unchanged for the current
four-way task (two eye baselines and left/right hand clenching, with Eye Open
serving as the resting reference). It is generic band-power, engagement, and synchrony
instrumentation rather than something retuned per class, so the rationale
below (written for the original three states) still explains what each
feature measures, even though the two clench classes have no dedicated
feature of their own yet -- a real gap, because telling left- from
right-hand clenching apart is exactly what lateralised sensorimotor mu/beta
ERD over C3/C4 encodes, and nothing in the list below is lateralised at all
(every feature averages over a whole frontal or posterior role). Expect
left-vs-right to be the weakest pair in the confusion matrix until a
lateralised feature is added.

**Frequency domain.** Eyes-closed is dominated by posterior alpha (Berger
rhythm); eyes-open (relaxed) is the baseline; reading (focused) was expected
to show both frontal engagement *and* further posterior alpha suppression
relative to the eyes-open baseline (visual-attention alpha
desynchronisation), which is why an anterior/posterior alpha *ratio* is its
own engineered feature rather than left for the model to learn from the two
halves separately::

    posterior_alpha_rel, posterior_beta_rel, posterior_theta_rel  (alpha_posterior role)
    frontal_alpha_rel, frontal_engagement (beta/(alpha+theta)),
        frontal_theta_beta (theta/beta)                            (theta_frontal role)
    global_spectral_entropy, global_alpha_rel                      (all channels)
    frontal_posterior_alpha_ratio = frontal_alpha_rel / posterior_alpha_rel

**Time domain.** Amplitude and zero-crossing rate, aggregated per role (not
per raw channel -- that would mostly duplicate the band-power features
above with noisier proxies). MAD is deliberately left out: it is a robust
twin of RMS amplitude and would add little beyond it::

    posterior_amplitude, posterior_zcr, frontal_amplitude, frontal_zcr

**Synchronicity.** Eyes-open vs. eyes-closed resting states are documented
in the literature to differ in alpha-band connectivity, not just alpha
power -- this is the one domain with a real case for inclusion beyond band
power alone. Averaged coherence and phase-locking value across every
frontal x posterior channel pair, in the alpha band, using the same
formulas as 12_synchronicity_features.py::

    frontal_posterior_coherence_alpha, frontal_posterior_plv_alpha

Coherence needs more data than a single window to stabilise, which is why
the epoch window default is 4.0s here, not the 2.0s files 10/11/17 use for
purely spectral features.

    python examples/20_train_focus_classifier.py --sessions-root Session_Classification --out models/focus_classifier.joblib
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from scipy import signal as sp_signal
from scipy import stats
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from neurobridge.core.montage import Montage
from neurobridge.io.recording import load_recording
from neurobridge.process.features import integrate_band, welch_psd
from neurobridge.process.filters import StreamFilter
from neurobridge.process.quality import QualityMonitor, spectral_slope

ROLES = ("theta_frontal", "alpha_posterior")
BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}


# ---------------------------------------------------------------------------
# Kept identical (by hand) to 09/16-18's artifact classifier.
# ---------------------------------------------------------------------------

def classify_components(sources, mixing, sf, frontal_idx, kurtosis_thresh,
                        muscle_slope_thresh=-0.7):
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
            bad.append(k); reasons[k] = "blink/saccade"
        elif is_muscle:
            bad.append(k); reasons[k] = "muscle/jaw"
    return bad, reasons


def clean_epoch(window, sf, frontal_idx, kurtosis_thresh, muscle_slope_thresh):
    n_ch = window.shape[0]
    ica = FastICA(n_components=n_ch, whiten="unit-variance", max_iter=500, random_state=0)
    try:
        sources = ica.fit_transform(window.T).T
    except Exception:
        return window
    bad, _ = classify_components(sources, ica.mixing_, sf, frontal_idx,
                                 kurtosis_thresh, muscle_slope_thresh)
    sources_clean = sources.copy()
    sources_clean[bad] = 0.0
    cleaned = ica.mixing_ @ sources_clean
    if getattr(ica, "mean_", None) is not None:
        cleaned = cleaned + ica.mean_[:, None]
    return cleaned


FEATURE_NAMES = [
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


def frontal_posterior_connectivity(window: np.ndarray, sf: float, f_idx: list[int],
                                   p_idx: list[int], lo: float = 8.0, hi: float = 13.0
                                   ) -> tuple[float, float]:
    """Mean coherence and PLV, in [lo, hi] Hz, averaged over every frontal x
    posterior channel pair -- same formulas as 12_synchronicity_features.py.
    """
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


def extract_features(window: np.ndarray, sf: float, role_idx: dict[str, list[int]]) -> np.ndarray:
    """The features documented in this file's module docstring, in
    FEATURE_NAMES order. ``window`` is (n_channels, n_samples), ICA-cleaned.
    """
    freqs, psd = welch_psd(window, sf, nperseg=window.shape[1])
    total = np.maximum(integrate_band(freqs, psd, 0.5, min(50.0, sf / 2 - 1)), 1e-12)

    p_idx = role_idx["alpha_posterior"]
    f_idx = role_idx["theta_frontal"]

    def rel(lo, hi, idx):
        return float(np.mean(integrate_band(freqs, psd, lo, hi)[idx] / total[idx]))

    posterior_alpha = rel(*BANDS["alpha"], p_idx)
    posterior_beta = rel(*BANDS["beta"], p_idx)
    posterior_theta = rel(*BANDS["theta"], p_idx)
    frontal_alpha = rel(*BANDS["alpha"], f_idx)
    frontal_theta = rel(*BANDS["theta"], f_idx)
    frontal_beta = rel(*BANDS["beta"], f_idx)
    frontal_engagement = frontal_beta / max(frontal_alpha + frontal_theta, 1e-12)
    frontal_theta_beta = frontal_theta / max(frontal_beta, 1e-12)

    all_idx = list(range(window.shape[0]))
    global_alpha = rel(*BANDS["alpha"], all_idx)
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


def epochs_with_groups(data, device, window, step, labels, group_offset=0):
    n_win = int(round(window * device.sfreq))
    n_step = int(round(step * device.sfreq))
    X, y, groups = [], [], []
    for start in range(0, data.shape[1] - n_win + 1, n_step):
        t0 = start / device.sfreq
        t1 = (start + n_win) / device.sfreq
        for gi, (a, b, lab) in enumerate(labels):
            if a <= t0 and t1 <= b:
                X.append(data[:, start:start + n_win])
                y.append(lab)
                groups.append(group_offset + gi)
                break
    X = np.stack(X) if X else np.empty((0, data.shape[0], n_win))
    return X, np.array(y), np.array(groups)


def find_recordings(args) -> list[tuple[Path, str]]:
    """Return [(path, participant_id), ...]."""
    out = []
    if args.recording:
        for p in args.recording:
            out.append((p, p.parent.name))
    else:
        for p in sorted(args.sessions_root.glob("*/*.npz")):
            out.append((p, p.parent.name))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sessions-root", type=Path, default=Path("Session_Classification"),
                    help="scan every participant's recordings under this root")
    ap.add_argument("--recording", type=Path, nargs="+",
                    help="explicit .npz files instead of scanning --sessions-root")
    ap.add_argument("--out", type=Path, required=True,
                    help="where to save the winning model, e.g. models/focus_classifier.joblib")
    ap.add_argument("--window", type=float, default=4.0,
                    help="epoch length, seconds -- 4s (not 2s) because the "
                         "coherence/PLV features need more data to stabilise")
    ap.add_argument("--step", type=float, default=1.0, help="epoch hop, seconds")
    ap.add_argument("--mains", type=float, default=60.0, choices=[50.0, 60.0])
    ap.add_argument("--kurtosis-thresh", type=float, default=5.0)
    ap.add_argument("--muscle-slope-thresh", type=float, default=-0.7)
    ap.add_argument("--folds", type=int, default=5,
                    help="max folds for the single-participant fallback CV")
    args = ap.parse_args()

    recordings = find_recordings(args)
    if not recordings:
        print("no recordings found. Record some with 19_record_focus_states.py first.")
        return 1

    all_X, all_y, all_groups, all_participants = [], [], [], []
    names_ref, sf_ref = None, None
    group_base = 0
    for path, pid_from_dir in recordings:
        data, device, meta = load_recording(path)
        if names_ref is None:
            names_ref, sf_ref = list(device.ch_names), device.sfreq
        elif list(device.ch_names) != names_ref or device.sfreq != sf_ref:
            ap.error(f"{path} was recorded on a different device/montage than the first "
                     f"recording; train on one consistent device at a time")

        n_ch = len(names_ref)
        filt = StreamFilter(sf_ref, n_ch, l_freq=0.5, h_freq=min(50.0, sf_ref / 2 - 5),
                            notch=args.mains)
        data = filt(data)

        # This script never streams live EEG -- it only ever reads back
        # already-recorded files -- so there is no "live" quality reading to
        # show. The closest honest equivalent: a per-recording quality
        # summary computed on a representative window (the middle 2s) of
        # this file's own filtered data, so a bad recording is visible
        # before its epochs get trained on rather than only showing up as
        # unexplained poor accuracy later.
        quality = QualityMonitor(mains=args.mains)
        mid = data.shape[1] // 2
        q_win = data[:, max(0, mid - int(sf_ref)):mid + int(sf_ref)]
        if q_win.shape[1] >= int(sf_ref):
            qs = quality(q_win, sf_ref, names_ref)
            q_summary = quality.summary(qs)
            worst = [q for q in qs if not q.usable]
            quality_note = (f"signal {q_summary['mean_score']:.2f}  "
                           f"{q_summary['n_usable']}/{q_summary['n_channels']} usable")
            if worst:
                quality_note += "   |   " + worst[0].advice()
        else:
            quality_note = "recording too short for a quality check"

        annotations = meta.get("annotations") or []
        if not annotations:
            print(f"warning: {path} has no labelled spans; skipping")
            continue
        pid = pid_from_dir
        if annotations and "participant" in annotations[0]:
            pid = annotations[0]["participant"]
        labels = [(a["t_start"], a["t_end"], a["label"]) for a in annotations]
        X, y, groups = epochs_with_groups(data, device, args.window, args.step,
                                          labels, group_offset=group_base)
        group_base += len(labels)
        print(f"{path} [{pid}]: {len(y)} epoch(s) from {len(labels)} span(s)  |  {quality_note}")
        if len(y):
            all_X.append(X); all_y.append(y); all_groups.append(groups)
            all_participants.append(np.array([pid] * len(y)))

    if not all_X:
        print("no labelled epochs found.")
        return 1

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    groups = np.concatenate(all_groups, axis=0)
    participants = np.concatenate(all_participants, axis=0)
    unique_participants = sorted(set(participants))
    print(f"\ntotal: {len(y)} epochs from {len(unique_participants)} participant(s): "
         f"{unique_participants}")
    print(f"classes: {dict(Counter(y))}")

    montage = Montage(names_ref)
    role_idx = {}
    for role in ROLES:
        m = montage.resolve(role)
        role_idx[role] = list(m.indices) if m.ok else list(range(len(names_ref)))
        print(f"  role {role:<16} -> {', '.join(names_ref[i] for i in role_idx[role])}"
             + ("" if m.exact else f"  ({m.note})"))

    print(f"\ncleaning epochs with ICA and extracting features ({len(y)} epochs)...")
    feats = np.zeros((len(y), len(FEATURE_NAMES)))
    for i in range(len(y)):
        cleaned = clean_epoch(X[i], sf_ref, role_idx["theta_frontal"],
                              args.kurtosis_thresh, args.muscle_slope_thresh)
        feats[i] = extract_features(cleaned, sf_ref, role_idx)

    # Per-participant normalisation: each person's own mean/sd, label-blind.
    feats_norm = feats.copy()
    for pid in unique_participants:
        mask = participants == pid
        mu = feats[mask].mean(axis=0)
        sd = feats[mask].std(axis=0)
        sd[sd < 1e-8] = 1.0
        feats_norm[mask] = (feats[mask] - mu) / sd

    models = {
        "logistic_regression": Pipeline([("scale", StandardScaler()),
                                         ("clf", LogisticRegression(max_iter=2000))]),
        "lda": Pipeline([("scale", StandardScaler()),
                        ("clf", LinearDiscriminantAnalysis())]),
        "svm_rbf": Pipeline([("scale", StandardScaler()),
                            ("clf", SVC(kernel="rbf", probability=True))]),
        "random_forest": Pipeline([("clf", RandomForestClassifier(
            n_estimators=300, random_state=0))]),
        "mlp": Pipeline([("scale", StandardScaler()),
                        ("clf", MLPClassifier(hidden_layer_sizes=(32, 16),
                                              max_iter=2000, random_state=0))]),
    }

    if len(unique_participants) >= 2:
        print(f"\ncross-validating {len(models)} model(s), leave-one-participant-out "
             f"({len(unique_participants)} participant(s)):\n")
        cv_splits = list(LeaveOneGroupOut().split(feats_norm, y, participants))
        split_label = "participant"
    else:
        print(f"\nonly one participant recorded so far -- cannot test generalisation to a "
             f"new person yet. Falling back to span-grouped cross-validation (as in "
             f"17_train_classifiers.py) within this one participant's data:\n")
        class_counts = Counter(y)
        min_groups = min(len(set(groups[y == c])) for c in class_counts)
        if min_groups < 2:
            print("not enough distinct labelled spans per class (need >= 2 in your "
                 "smallest class). Record more spans and try again.")
            return 1
        n_splits = max(2, min(args.folds, min_groups))
        cv_splits = list(StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=0).split(feats_norm, y, groups))
        split_label = "fold"

    scores: dict[str, list[float]] = {}
    for name, pipe in models.items():
        fold_f1 = []
        for train_idx, test_idx in cv_splits:
            pipe.fit(feats_norm[train_idx], y[train_idx])
            pred = pipe.predict(feats_norm[test_idx])
            fold_f1.append(f1_score(y[test_idx], pred, average="macro", zero_division=0))
        scores[name] = fold_f1
        print(f"  {name:<20} macro-F1 = {np.mean(fold_f1):.3f} +/- {np.std(fold_f1):.3f}  "
             f"({split_label}s: {[round(f, 2) for f in fold_f1]})")

    best_name = max(scores, key=lambda k: np.mean(scores[k]))
    print(f"\nbest model: {best_name} (macro-F1 = {np.mean(scores[best_name]):.3f})")

    print(f"\nper-class report for {best_name}, held-out predictions across all {split_label}s:")
    all_true, all_pred = [], []
    for train_idx, test_idx in cv_splits:
        models[best_name].fit(feats_norm[train_idx], y[train_idx])
        all_pred.append(models[best_name].predict(feats_norm[test_idx]))
        all_true.append(y[test_idx])
    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)
    print(classification_report(all_true, all_pred, zero_division=0))
    classes = sorted(set(y))
    w = max(10, max(len(c) for c in classes))
    print("confusion matrix (rows=true, cols=predicted):")
    print(" " * (w + 2) + "  ".join(f"{c:>{w}}" for c in classes))
    cm = confusion_matrix(all_true, all_pred, labels=classes)
    for c, row in zip(classes, cm):
        print(f"{c:>{w}}  " + "  ".join(f"{v:>{w}d}" for v in row))

    best_pipe = models[best_name]
    best_pipe.fit(feats_norm, y)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": best_pipe,
        "model_name": best_name,
        "classes": classes,
        "feature_names": FEATURE_NAMES,
        "channel_names": names_ref,
        "sfreq": sf_ref,
        "window": args.window,
        "mains": args.mains,
        "kurtosis_thresh": args.kurtosis_thresh,
        "muscle_slope_thresh": args.muscle_slope_thresh,
        "cv_macro_f1": {k: float(np.mean(v)) for k, v in scores.items()},
        "n_participants_trained_on": len(unique_participants),
    }, args.out)
    print(f"\nsaved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
