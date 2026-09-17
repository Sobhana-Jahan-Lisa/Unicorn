"""Record live EEG with real-time labels for seven states -- two eye baselines
and five motor states -- organized by participant and session for a
multi-participant study.

Eye Open doubles as the resting baseline for ERD/ERS analysis: a separate
"Rest" class would be the same physiological state (sitting still, eyes open,
not moving), and two indistinguishable classes only depress macro-F1 and
obscure the contrasts that matter -- left vs right hand, left vs right leg,
and the limbs against tongue.

    o   Eye Open
    c   Eye Close
    1   Clenching Left Hand
    2   Clenching Right Hand
    3   Moving Left Leg
    4   Moving Right Leg
    5   Moving Tongue
    q   quit and save

A caution about the two leg classes, since the recording is the expensive part
and nothing downstream can fix a distinction that was never on the scalp: the
foot/leg area of motor cortex sits on the medial wall of the central sulcus,
so left and right leg project to electrodes millimetres apart near the vertex
and are largely picked up by the same one (Cz on a Unicorn montage). Hand
left-vs-right, and legs-vs-hands-vs-tongue, are the contrasts an 8-channel dry
montage can realistically separate; left leg vs right leg may well not
separate at all. Recording them as two classes keeps that option open -- they
can always be merged into one "leg" class at training time, which is what the
standard four-class motor-imagery datasets do -- but do not read a low
left-leg/right-leg F1 as a problem with the session.

Click the plot window first so it has keyboard focus, then press a label key
once to START a labelled span and press the SAME key again to END it.

    python examples/19_record_focus_states.py --participant PT-001 --board UNICORN_BOARD --serial-number UN-2022.04.52

Real hardware only.

Participant and session management
-----------------------------------
``--participant`` must be a pseudonymous code, not a name -- validated
against the same pattern this project's own ``neurobridge.education.events``
module already enforces for exactly this reason: ``LL-999`` (2-4 uppercase
letters, a dash, 3-5 digits), e.g. ``PT-001``. The mapping from code to
person stays outside this system entirely, on paper, in a locked drawer.

The session number is never typed by hand -- it is the next integer after
whatever ``.npz`` files already exist in
``Session_Classification/<participant>/``, so running this script again for
the same participant just continues the sequence. Every completed recording
also appends one line to ``Session_Classification/registry.jsonl``
(participant, session, path, timestamp, span
counts) -- a single append-only, human-readable index across every
participant and session, without needing to scan the filesystem to find out
who has been recorded and how much.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from neurobridge.core.buffer import RingBuffer
from neurobridge.io.recording import Recorder
from neurobridge.process.filters import StreamFilter
from neurobridge.process.quality import QualityMonitor

LABELS = {
    "o": "Eye Open",
    "c": "Eye Close",
    "1": "Clenching Left Hand",
    "2": "Clenching Right Hand",
    "3": "Moving Left Leg",
    "4": "Moving Right Leg",
    "5": "Moving Tongue",
}

#: Same pseudonymous-code pattern neurobridge.education.events.EventLog uses,
#: kept in sync by hand rather than imported (that name is module-private).
_PARTICIPANT_RE = re.compile(r"^[A-Z]{2,4}-\d{3,5}$")


def build_source(args):
    if args.config:
        from neurobridge.config import Config

        cfg = Config.from_file(args.config)
        return cfg.build_source()
    from neurobridge import BrainFlowSource

    params = {}
    if args.serial_port:
        params["serial_port"] = args.serial_port
    if args.serial_number:
        params["serial_number"] = args.serial_number
    return BrainFlowSource(args.board, mains=args.mains, **params)


def next_session(sessions_root: Path, participant: str) -> int:
    pdir = sessions_root / participant
    pdir.mkdir(parents=True, exist_ok=True)
    existing = list(pdir.glob(f"{participant}_session*_*.npz"))
    nums = []
    for p in existing:
        m = re.search(r"_session(\d+)_", p.name)
        if m:
            nums.append(int(m.group(1)))
    return (max(nums) + 1) if nums else 1


def append_registry(sessions_root: Path, record: dict) -> None:
    path = sessions_root / "registry.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--participant", required=True,
                    help="pseudonymous code, e.g. PT-001 -- never a name")
    ap.add_argument("--sessions-root", type=Path, default=Path("Session_Classification"),
                    help="root folder for all participants' recordings")
    ap.add_argument("--config", type=Path, help="use the source from a config file")
    ap.add_argument("--board", help="BrainFlow board name, e.g. UNICORN_BOARD")
    ap.add_argument("--serial-port", help="COM3 on Windows, /dev/tty... elsewhere")
    ap.add_argument("--serial-number", help="e.g. UN-2022.04.52")
    ap.add_argument("--mains", type=float, default=60.0, choices=[50.0, 60.0])
    ap.add_argument("--scale", type=float, default=50.0,
                    help="microvolts between channel baselines -- same fixed-scale "
                         "convention as 06_live_scope.py, not auto-scaled")
    ap.add_argument("--seconds", type=float, default=5.0, help="width of the live view")
    ap.add_argument("--max-minutes", type=float, default=30.0,
                    help="hard cap on recording length")
    args = ap.parse_args()

    if not _PARTICIPANT_RE.match(args.participant):
        ap.error(f"--participant {args.participant!r} does not look like a pseudonymous "
                 f"code (expected e.g. 'PT-001': 2-4 uppercase letters, a dash, 3-5 "
                 f"digits). Never use a real name.")
    if not args.config and not args.board:
        ap.error("this script only runs on real hardware: pass --board "
                 "(e.g. --board UNICORN_BOARD --serial-number UN-2022.04.52) "
                 "or --config pointing at a brainflow config")

    session_n = next_session(args.sessions_root, args.participant)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = (args.sessions_root / args.participant /
               f"{args.participant}_session{session_n:02d}_{stamp}.npz")

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    source = build_source(args)
    source.start()
    device = source.device
    sf = device.sfreq
    names = list(device.ch_names)
    n_ch = len(names)

    print(f"participant  {args.participant}   session {session_n:02d}")
    print(f"device       {device.name} @ {sf} Hz")
    print(f"channels     {', '.join(names)}")
    print(f"output       {out_path}")
    print("\nclick the plot window so it has keyboard focus, then:")
    for k, v in LABELS.items():
        print(f"  {k}   start/stop '{v}'")
    print("  q   quit and save\n")

    recorder = Recorder(device, out_path, max_seconds=args.max_minutes * 60)
    display_filter = StreamFilter(sf, n_ch, l_freq=0.5,
                                  h_freq=min(50.0, sf / 2 - 5), notch=args.mains)
    quality = QualityMonitor(mains=args.mains)

    n_show = int(args.seconds * sf)
    ring = RingBuffer(n_ch, n_show)
    t_axis = np.arange(n_show) / sf - args.seconds
    # Fixed scale, same convention as 06_live_scope.py: a constant spacing
    # between channel baselines, not auto-scaled to the live signal. Auto-
    # scaling to whatever the loudest channel is doing means one railing
    # electrode inflates the scale for every channel and makes the six good
    # ones look artificially flat -- a real signal should look like a real
    # signal, and a bad channel should look bad, not flatten its neighbours.
    offsets = np.arange(n_ch)[::-1] * args.scale

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.canvas.manager.set_window_title(
        f"neurobridge record focus states — {args.participant} session {session_n:02d}")
    ax.set_xlim(t_axis[0], 0)
    ax.set_ylim(-args.scale, offsets[0] + args.scale)
    ax.set_yticks(offsets)
    ax.set_yticklabels(names)
    ax.set_xlabel("seconds before now")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    lines = [ax.plot(t_axis, np.zeros(n_show), linewidth=0.8)[0] for _ in range(n_ch)]

    # Microvolt scale bar, same as 06_live_scope.py: without it the trace
    # amplitude has no absolute reference.
    ax.plot([t_axis[0] + 0.15] * 2, [-args.scale * 0.8, -args.scale * 0.8 + 20],
           color="black", linewidth=2.5)
    ax.text(t_axis[0] + 0.25, -args.scale * 0.8 + 6, "20 µV", fontsize=9)
    status = fig.text(0.5, 0.965, "no active label -- press a key to start one",
                      ha="center", fontsize=11)
    quality_text = fig.text(0.5, 0.935, "", ha="center", fontsize=9, color="#555555")

    state = {
        "t0": time.time(),
        "spans": [],
        "active_label": None,
        "active_start": None,
        "since_quality": 0,
        "quality_msg": "",
    }

    def on_key(event):
        key = event.key
        if key == "q":
            plt.close(fig)
            return
        label = LABELS.get(key)
        if label is None:
            return
        now = time.time() - state["t0"]
        if state["active_label"] is None:
            state["active_label"] = label
            state["active_start"] = now
            status.set_text(f"recording '{label}'... press '{key}' again to end it")
        elif state["active_label"] == label:
            state["spans"].append({
                "label": label, "t_start": state["active_start"], "t_end": now,
                "participant": args.participant, "session": session_n,
            })
            print(f"  logged {label}: {state['active_start']:.2f}s - {now:.2f}s "
                 f"({now - state['active_start']:.2f}s)")
            state["active_label"] = None
            state["active_start"] = None
            status.set_text("no active label -- press a key to start one")
        else:
            print(f"  ignored '{key}': '{state['active_label']}' is still open, "
                 f"press its key again to close it first")

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_):
        for _ in range(50):
            chunk = source.read()
            if chunk is None:
                break
            recorder.add(chunk)
            ring.push(display_filter(chunk.data))

        have = min(ring.total_written, n_show)
        if have < int(0.5 * sf):
            return
        window = np.full((n_ch, n_show), np.nan)
        window[:, -have:] = ring.latest(have)
        for i, line in enumerate(lines):
            line.set_ydata(window[i] - np.nanmean(window[i]) + offsets[i])

        state["since_quality"] += 1
        if have >= int(2 * sf) and state["since_quality"] >= 20:
            state["since_quality"] = 0
            # The whole on-screen window (5 s by default), as 06_live_scope.py
            # does: a 2-s window lets one blink or knock flip the readout.
            qs = quality(ring.latest(min(have, n_show)), sf, names)
            summary = quality.summary(qs)
            worst = [q for q in qs if not q.usable]
            state["quality_msg"] = (f"signal {summary['mean_score']:.2f}  "
                                    f"{summary['n_usable']}/{summary['n_channels']} usable")
            if worst:
                state["quality_msg"] += "   |   " + worst[0].advice()
            lost = getattr(source, "held_fraction", 0.0) or 0.0
            if lost >= 0.01:
                state["quality_msg"] += f"   |   Bluetooth: {100 * lost:.0f}% of samples lost"
            quality_text.set_color("#d6412b" if worst else "#1f7a5c")
        quality_text.set_text(state["quality_msg"])

    anim = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
    try:
        plt.show()
    finally:
        if state["active_label"] is not None:
            print(f"  warning: '{state['active_label']}' was still open at quit; discarding it")
        source.stop()
        path = recorder.close(annotations=state["spans"])
        print(f"\nsaved {path}  ({recorder.n_samples} samples, "
             f"{len(state['spans'])} labelled span(s))")
        by_label: dict[str, int] = {}
        for s in state["spans"]:
            by_label[s["label"]] = by_label.get(s["label"], 0) + 1
        append_registry(args.sessions_root, {
            "participant": args.participant, "session": session_n,
            "path": str(path), "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_spans": len(state["spans"]), "spans_by_label": by_label,
        })
    return 0


if __name__ == "__main__":
    sys.exit(main())
