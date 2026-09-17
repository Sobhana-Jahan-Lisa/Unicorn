"""Live EEG scope: the actual waveform, the spectrum, and contact quality.

This is the view the browser client deliberately does not give you. The game
page draws the *decoded control value*, and in the default `derived` privacy
tier no raw sample ever crosses the socket. This script runs inside the same
process as acquisition, so nothing leaves the machine and there is no
transport, no browser and no privacy question to answer.

    python examples/06_live_scope.py                        # synthetic
    python examples/06_live_scope.py --config configs/alpha_attention.yaml
    python examples/06_live_scope.py --board UNICORN_BOARD
    python examples/06_live_scope.py --board UNICORN_BOARD --raw

Three panels:

* **Traces** -- filtered channels, stacked, with a microvolt scale bar. Each
  label is coloured by that channel's quality score, so a lifting electrode is
  visible before you start wondering why the game feels wrong.
* **Spectrum** -- Welch PSD of one channel on log axes. Close your eyes and
  the alpha peak near 10 Hz should rise out of the 1/f background within a
  second or two. That single moment is the most convincing demonstration in
  the whole system, and it is worth rehearsing before a class sees it.
* **Alpha bar** -- relative alpha in the posterior role, updated per window.

``--raw`` skips the band-pass so you can see drift, blinks and mains hum as
they really are. Useful for teaching what filtering removes; useless for
judging alpha.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from neurobridge.core.buffer import RingBuffer
from neurobridge.core.montage import Montage
from neurobridge.process.features import integrate_band, welch_psd
from neurobridge.process.filters import StreamFilter
from neurobridge.process.quality import QualityMonitor


def build_source(args):
    """Construct the source from a config file, a board name, or the default."""
    if args.config:
        from neurobridge.config import Config

        cfg = Config.from_file(args.config)
        return cfg.build_source(), cfg
    if args.board:
        from neurobridge import BrainFlowSource

        params = {}
        if args.serial_port:
            params["serial_port"] = args.serial_port
        if args.serial_number:
            params["serial_number"] = args.serial_number
        return BrainFlowSource(args.board, mains=args.mains, **params), None

    from neurobridge import SyntheticSource

    return SyntheticSource(alpha_amp=args.synthetic_alpha), None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, help="use the source from a config file")
    ap.add_argument("--board", help="BrainFlow board name, e.g. UNICORN_BOARD")
    ap.add_argument("--serial-port", help="COM3 on Windows, /dev/tty... elsewhere")
    ap.add_argument("--serial-number", help="e.g. UN-2022.04.52")
    ap.add_argument("--seconds", type=float, default=5.0,
                    help="width of the time window on screen")
    ap.add_argument("--scale", type=float, default=50.0,
                    help="microvolts between channel baselines")
    ap.add_argument("--mains", type=float, default=60.0, choices=[50.0, 60.0])
    ap.add_argument("--raw", action="store_true",
                    help="skip the band-pass and show unfiltered voltage")
    ap.add_argument("--spectrum-channel", default=None,
                    help="channel for the spectrum panel; default is posterior")
    ap.add_argument("--synthetic-alpha", type=float, default=12.0)
    ap.add_argument("--save", type=Path,
                    help="render one frame to this PNG and exit (no window)")
    args = ap.parse_args()

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    source, _ = build_source(args)
    source.start()
    device = source.device
    sf = device.sfreq
    names = list(device.ch_names)
    n_ch = len(names)

    print(f"device   {device.name} @ {sf} Hz")
    print(f"channels {', '.join(names)}")

    # Pick the channel for the spectrum: posterior by default, since that is
    # where alpha lives and where the demonstration actually works.
    if args.spectrum_channel and args.spectrum_channel in names:
        spec_idx = names.index(args.spectrum_channel)
    else:
        match = Montage(device.ch_names).resolve("alpha_posterior")
        spec_idx = match.indices[0] if match.ok else 0
    print(f"spectrum {names[spec_idx]}")
    print(f"filter   {'OFF (raw voltage)' if args.raw else f'1-45 Hz, {args.mains:g} Hz notch'}")
    print("\nclose the window to stop.\n")

    stream_filter = None if args.raw else StreamFilter(
        sf, n_ch, l_freq=1.0, h_freq=min(45.0, sf / 2 - 5), notch=args.mains)
    quality = QualityMonitor(mains=args.mains)

    n_show = int(args.seconds * sf)
    ring = RingBuffer(n_ch, n_show)

    fig = plt.figure(figsize=(13, 7.5))
    fig.canvas.manager.set_window_title(f"neurobridge scope — {device.name}")
    gs = fig.add_gridspec(2, 2, width_ratios=[2.6, 1], height_ratios=[1, 1],
                          hspace=0.28, wspace=0.22)
    ax_t = fig.add_subplot(gs[:, 0])
    ax_f = fig.add_subplot(gs[0, 1])
    ax_a = fig.add_subplot(gs[1, 1])

    t_axis = np.arange(n_show) / sf - args.seconds
    offsets = np.arange(n_ch)[::-1] * args.scale

    ax_t.set_xlim(t_axis[0], 0)
    ax_t.set_ylim(-args.scale, offsets[0] + args.scale)
    ax_t.set_xlabel("seconds before now")
    ax_t.set_yticks(offsets)
    ax_t.set_yticklabels(names)
    ax_t.grid(True, alpha=0.25, linewidth=0.6)
    ax_t.set_title("live EEG" + (" (unfiltered)" if args.raw else ""))
    lines = [ax_t.plot(t_axis, np.zeros(n_show), linewidth=0.8)[0]
             for _ in range(n_ch)]

    # Microvolt scale bar: without it the trace amplitude is unreadable.
    ax_t.plot([t_axis[0] + 0.15] * 2, [-args.scale * 0.8, -args.scale * 0.8 + 20],
              color="black", linewidth=2.5)
    ax_t.text(t_axis[0] + 0.25, -args.scale * 0.8 + 6, "20 µV", fontsize=9)

    ax_f.set_xscale("log")
    ax_f.set_yscale("log")
    ax_f.set_xlim(1, min(60, sf / 2))
    ax_f.set_xlabel("Hz")
    ax_f.set_ylabel("µV²/Hz")
    ax_f.grid(True, which="both", alpha=0.25, linewidth=0.6)
    ax_f.axvspan(8, 13, color="gold", alpha=0.28)
    ax_f.text(0.5, 0.93, "alpha", fontsize=9, color="#8a6d00",
              transform=ax_f.transAxes, ha="center")
    (spec_line,) = ax_f.plot([], [], linewidth=1.2)
    ax_f.set_title(f"spectrum — {names[spec_idx]}")

    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    ranges = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
    bars = ax_a.bar(bands, [0] * 5,
                    color=["#c8d2d0", "#c8d2d0", "#f0c419", "#c8d2d0", "#c8d2d0"])
    ax_a.set_ylim(0, 1)
    ax_a.set_ylabel("relative power")
    ax_a.set_title("posterior bands")
    ax_a.grid(True, axis="y", alpha=0.25, linewidth=0.6)

    status = fig.text(0.5, 0.965, "", ha="center", fontsize=10)
    state = {"since_quality": 0}

    def update(_):
        # Drain everything the device has produced since the last frame.
        for _ in range(50):
            chunk = source.read()
            if chunk is None:
                break
            data = chunk.data
            if stream_filter is not None:
                data = stream_filter(data)
            ring.push(data)

        have = min(ring.total_written, n_show)
        if have < int(0.5 * sf):
            status.set_text("filling the buffer...")
            return

        recent = ring.latest(have)
        # Pad the unrecorded part with NaN so matplotlib leaves it blank
        # instead of drawing a flat line that looks like a dead electrode.
        window = np.full((n_ch, n_show), np.nan)
        window[:, -have:] = recent
        for i, line in enumerate(lines):
            trace = window[i] - np.nanmean(window[i])
            line.set_ydata(trace + offsets[i])

        # Spectral panels need a window long enough to resolve the narrowest
        # band. Delta is 3 Hz wide, so anything under ~2 s would be rejected by
        # integrate_band -- correctly, since the number would be meaningless.
        n_spec = int(2 * sf)
        if have < n_spec:
            status.set_text(
                f"filling the buffer... {have / sf:.1f}s of {n_spec / sf:.0f}s "
                f"needed for the spectrum")
            return
        window = recent

        freqs, psd = welch_psd(window, sf, nperseg=min(window.shape[1], n_spec))
        mask = freqs >= 1
        spec_line.set_data(freqs[mask], np.maximum(psd[spec_idx][mask], 1e-6))
        ax_f.set_ylim(max(1e-4, psd[spec_idx][mask].min() * 0.5),
                      psd[spec_idx][mask].max() * 3)

        total = integrate_band(freqs, psd[spec_idx:spec_idx + 1], 1, 45)[0]
        for bar, (lo, hi) in zip(bars, ranges):
            p = integrate_band(freqs, psd[spec_idx:spec_idx + 1], lo, hi)[0]
            bar.set_height(float(p / max(total, 1e-12)))

        # Quality is expensive; once a second is plenty.
        state["since_quality"] += 1
        if state["since_quality"] >= 20:
            state["since_quality"] = 0
            qs = quality(window, sf, names)
            for label, q in zip(ax_t.get_yticklabels(), qs):
                label.set_color("#1f7a5c" if q.usable else "#d6412b")
            summary = quality.summary(qs)
            worst = [q for q in qs if not q.usable]
            lost = getattr(source, "held_fraction", 0.0) or 0.0
            status.set_text(
                f"signal {summary['mean_score']:.2f}   "
                f"{summary['n_usable']}/{summary['n_channels']} usable"
                + (f"   |   {worst[0].advice()}" if worst else "")
                + (f"   |   Bluetooth: {100 * lost:.0f}% of samples lost"
                   if lost >= 0.01 else "")
            )
            status.set_color("#d6412b" if worst else "#1f7a5c")

    if args.save:
        import time as _time

        deadline = _time.time() + args.seconds + 5
        while ring.total_written < n_show and _time.time() < deadline:
            update(0)
            _time.sleep(0.02)
        for _ in range(25):            # let the quality panel populate
            update(0)
            _time.sleep(0.02)
        fig.savefig(args.save, dpi=110, bbox_inches="tight")
        source.stop()
        print(f"saved {args.save}")
        return 0

    anim = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
    try:
        plt.show()
    finally:
        source.stop()
        print("stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
