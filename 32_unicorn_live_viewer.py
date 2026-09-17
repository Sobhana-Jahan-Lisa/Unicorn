"""Unicorn-Recorder-style live EEG viewer with a per-channel quality check.

Rebuilds the standalone Unicorn Live Viewer (``unicorn_live.py``) on
neurobridge, and adds the channel-quality check it deliberately left out.

* Acquisition runs on its own thread (``neurobridge.io.acquisition``), so a
  stalled window never stalls reading or recording.
* Display defaults match Unicorn Recorder's visible settings: one axis per
  channel, fixed ±50 µV, 0.1-30 Hz band-pass with a 60 Hz notch, 8-second
  window, every sample drawn. Beside each trace: the share of samples outside
  the view and the peak-to-peak amplitude, so a big artifact is never hidden
  just because it left the axis.
* Status line: "filter settling" for max(8 s, 5 / low cutoff), "NO RECENT
  DATA" after 2 s without samples, Bluetooth loss (held samples over the last
  10 s) and packet-counter gaps.
* Channel quality (new here): neurobridge's QualityMonitor on a separate
  1-45 Hz copy of the stream, after Bluetooth-dropout repair, over the last
  5 s, once a second -- the same check 06_live_scope.py makes -- shown as a
  Good/Bad badge on each channel. The 0.1-30 Hz display copy is not used for
  it: the quality metrics are defined on 1-45 Hz data.
* ``--record``: unfiltered EEG exactly as delivered, with BrainFlow's
  timestamp, packet counter and marker row, appended to a CSV and flushed
  every chunk, plus an NPZ and a JSON sidecar on close. Never overwrites.

There is no OSCAR here, so even with matching settings the traces will not
look like Unicorn Recorder's with OSCAR on. Good/Bad is not Unicorn Suite's
contact measurement, and "Good" does not mean the window is free of
artifacts: read it together with the peak-to-peak beside it.

    python examples/32_unicorn_live_viewer.py --serial-number UN-2022.04.52
    python examples/32_unicorn_live_viewer.py --demo
    python examples/32_unicorn_live_viewer.py --serial-number UN-2022.04.52 --record recordings/viewer_01.npz
    python examples/32_unicorn_live_viewer.py --serial-number UN-2022.04.52 --raw --scale 500

Close Unicorn Recorder first; the headset streams to one program at a time.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from neurobridge.io.acquisition import AcquisitionThread
from neurobridge.io.recording import StreamingRecorder
from neurobridge.viz import LiveMonitor


def build_source(args):
    if args.demo:
        from neurobridge import SyntheticSource

        return SyntheticSource()
    from neurobridge import BrainFlowSource

    params = {}
    if args.serial_port:
        params["serial_port"] = args.serial_port
    if args.serial_number:
        params["serial_number"] = args.serial_number
    # No repair inside the source: a recording keeps every sample exactly as
    # it arrived. LiveMonitor repairs its own display and quality copies.
    return BrainFlowSource(args.board, mains=args.mains, repair_dropouts=False, **params)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", default="UNICORN_BOARD", help="BrainFlow board name")
    ap.add_argument("--serial-number", help="e.g. UN-2022.04.52")
    ap.add_argument("--serial-port", help="only for boards that need one")
    ap.add_argument("--demo", action="store_true",
                    help="neurobridge's synthetic EEG instead of a headset")
    ap.add_argument("--record", type=Path,
                    help="new .npz path; the .csv (written as you go) and .json sit "
                         "next to it. Never overwrites")
    ap.add_argument("--low", type=float, default=0.1, help="display band-pass low edge, Hz")
    ap.add_argument("--high", type=float, default=30.0, help="display band-pass high edge, Hz")
    ap.add_argument("--notch", type=float, default=60.0, choices=[0.0, 50.0, 60.0],
                    help="display notch, 0 for none")
    ap.add_argument("--mains", type=float, default=60.0, choices=[50.0, 60.0],
                    help="mains frequency for the quality check and dropout repair")
    ap.add_argument("--scale", type=float, default=50.0, help="fixed display half-range, µV")
    ap.add_argument("--window", type=float, default=8.0, help="seconds on screen")
    ap.add_argument("--raw", action="store_true",
                    help="show samples as delivered, median-centred for display only")
    ap.add_argument("--max-minutes", type=float, default=60.0, help="cap on --record length")
    ap.add_argument("--save", type=Path,
                    help="run headless for --save-after seconds, write one frame to "
                         "this PNG and exit")
    ap.add_argument("--save-after", type=float, default=10.0)
    args = ap.parse_args()
    if args.scale <= 0 or args.window <= 0 or not 0 < args.low < args.high:
        ap.error("need a positive --scale and --window, and 0 < --low < --high")
    if args.record is not None and args.record.suffix != ".npz":
        ap.error("--record takes a .npz path; the .csv and .json are written next to it")

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    source = build_source(args)
    source.start()
    acq = recorder = None
    try:
        device = source.device
        monitor = LiveMonitor(device, low=args.low, high=args.high, notch=args.notch,
                              mains=args.mains, seconds=args.window, scale=args.scale,
                              raw=args.raw,
                              window_title=f"neurobridge live viewer — {device.name}")
        if args.record:
            recorder = StreamingRecorder(
                device, args.record, max_seconds=args.max_minutes * 60,
                extra_meta={"mains": args.mains, "script": Path(__file__).name,
                            "display_filter": {"low": args.low, "high": args.high,
                                               "notch": args.notch}})
        acq = AcquisitionThread(source, recorder=recorder, consumers=[monitor])
        print(f"device   {device.name} @ {device.sfreq:g} Hz")
        print(f"channels {', '.join(device.ch_names)}")
        print(f"display  {monitor.headline}")
        if recorder is not None:
            print(f"record   {recorder.path}  (+ {recorder.csv_path.name} written as you go, "
                  f"+ {recorder.json_path.name})")
        print("\nclose the window to stop.\n")
        acq.start()

        def update(_=None):
            extra = []
            if recorder is not None:
                extra.append(f"REC {recorder.n_samples / device.sfreq:.0f} s"
                             + (" (cap reached)" if recorder.truncated else ""))
            monitor.draw(acq, source, extra_parts=extra)

        if args.save:
            end = time.monotonic() + args.save_after
            while time.monotonic() < end:
                update()
                time.sleep(0.05)
            monitor.view.fig.savefig(args.save, dpi=100)
            print(f"saved {args.save}")
        else:
            anim = FuncAnimation(monitor.view.fig, update, interval=50,  # noqa: F841
                                 cache_frame_data=False)
            plt.show()
    finally:
        if acq is not None:
            acq.close()
        else:
            source.stop()
        if recorder is not None:
            path = recorder.close()
            print(f"recording {path}: {recorder.n_samples} samples")
    total = getattr(source, "samples_total", 0)
    if total:
        print(f"Bluetooth: {source.held_total} of {total} samples held "
              f"({100 * source.held_total / total:.2f}%); "
              f"packet-counter gaps: {source.counter_missing_total}")
    if acq is not None and acq.error:
        print(f"acquisition stopped: {acq.error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
