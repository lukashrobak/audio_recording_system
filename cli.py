"""
Cli app v4
Bachelor's Project — Design and Implementation of a System for Long-Term Audio Recording (Lukas Hrobak)

Commands:
  py cli.py -i rec.wav -a                          # Analyze single file to stdout
  py cli.py -i recordings/ -s                      # Analyze folder, write summary txt
  py cli.py -i rec.wav -p waveform                 # Save waveform PNG
  py cli.py -i rec.wav -p spectrogram -f 500 4000  # Save spectrogram (filtered)
  py cli.py -i recordings/ -p rms                  # Save RMS timeline (needs rms_log.csv)
  py cli.py -i recordings/ --all -o results/       # Full analysis + all plots

  # For Linux/MacOS use "python3" instead of "py"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from audio_processing import (
    load_wav, load_csv, analyze_wav,
    apply_bandpass, compute_spectrogram,
    get_wav_duration, rms_to_db,
    TRIGGER_DB, NOISE_DB,
    BAND_LABELS,
)


def plot_waveform(wav_path: Path, out_dir: Path, low_hz: float = 20, high_hz: float = 20000, analysis: dict = None) -> Path:
    """Saves a waveform + RMS envelope plot."""
    samples, sr = load_wav(wav_path)
    if low_hz > 20 or high_hz < 20000:
        samples = apply_bandpass(samples, sr, low_hz, high_hz)

    n = len(samples)
    duration = n / sr
    t_axis = np.linspace(0, duration, n)

    fig, ax = plt.subplots(figsize=(14, 4))

    step = max(1, n // 10000)
    ax.plot(t_axis[::step], samples[::step], linewidth=0.6, label="Waveform")

    win = int(sr * 0.05)
    rms_env, rms_t = [], []
    for i in range(0, n - win, win):
        rms_env.append(float(np.sqrt(np.mean(samples[i:i+win] ** 2))))
        rms_t.append(i / sr)
    ax.fill_between(rms_t, np.array(rms_env), -np.array(rms_env), alpha=0.25, label="RMS envelope")

    trig_amp = 10 ** (TRIGGER_DB / 20.0)
    ax.axhline( trig_amp, color="red", linestyle="--", linewidth=0.8, label=f"Trigger ({TRIGGER_DB} dBFS)")
    ax.axhline(-trig_amp, color="red", linestyle="--", linewidth=0.8)
    peak = float(np.max(np.abs(samples)))
    y_lim = max(peak * 2, trig_amp * 1.5)
    ax.set_ylim(-y_lim, y_lim)
    ax.set_title(wav_path.name)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = out_dir / f"{wav_path.stem}_waveform.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_spectrogram(wav_path: Path, out_dir: Path, low_hz: float = 20, high_hz: float = 20000) -> Path:
    """Saves a spectrogram plot."""
    samples, sr = load_wav(wav_path)
    if low_hz > 20 or high_hz < 20000:
        samples = apply_bandpass(samples, sr, low_hz, high_hz)

    f, t, Sxx_db = compute_spectrogram(samples, sr)
    duration = len(samples) / sr

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.pcolormesh(t, f, Sxx_db, shading="gouraud", cmap="inferno", vmin=-120, vmax=0)
    ax.set_ylim(0, min(float(f[-1]), 20000))
    ax.set_xlim(0, duration)
    ax.set_title(wav_path.name)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(vmin=-120, vmax=0)), ax=ax, pad=0.01)
    cbar.set_label("dB")

    if low_hz > 20:
        ax.axhline(low_hz, color="green", linestyle="--", linewidth=1, label=f"Low cut {low_hz} Hz")
    if high_hz < 20000:
        ax.axhline(high_hz, color="orange", linestyle="--", linewidth=1, label=f"High cut {high_hz} Hz")
    if low_hz > 20 or high_hz < 20000:
        ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = out_dir / f"{wav_path.stem}_spectrogram.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_rms(folder: Path, out_dir: Path) -> Path:
    """Saves an RMS timeline plot from rms_log.csv."""
    csv_path = folder / "rms_log.csv"
    if not csv_path.exists():
        print(f"  [!] rms_log.csv not found in {folder}")
        return None

    timestamps, rms_vals, triggered, recording = load_csv(csv_path)
    if not timestamps:
        print("  [!] rms_log.csv is empty.")
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.4})

    ax1.plot(timestamps, rms_vals, linewidth=0.5, label="RMS")

    for i, rec in enumerate(recording):
        if rec and i + 1 < len(timestamps):
            ax1.axvspan(timestamps[i], timestamps[i+1], alpha=0.15, color="orange", linewidth=0)

    ax1.axhline(TRIGGER_DB, color="red",   linestyle="--", linewidth=1, label=f"Trigger ({TRIGGER_DB} dBFS)")
    ax1.axhline(NOISE_DB,   color="green", linestyle="--", linewidth=1, label=f"Noise floor ({NOISE_DB} dBFS)")

    event_times = [timestamps[i] for i in range(1, len(triggered)) if triggered[i] == 1 and triggered[i-1] == 0]
    for et in event_times:
        ax1.axvline(et, color="red", linewidth=0.6, alpha=0.5, zorder=3)
    if event_times:
        ax1.axvline(event_times[0], color="red", linewidth=0.6, alpha=0.5,
                    label=f"Trigger events ({len(event_times)})", zorder=3)

    ax1.set_ylim(-90, 0)
    ax1.set_ylabel("dBFS")
    ax1.set_title("RMS Level Over Time")
    ax1.legend(fontsize=8)

    ax2.fill_between(timestamps, np.array(recording, dtype=float), 0, alpha=0.7, step="post")
    ax2.set_ylim(0, 1.5)
    ax2.set_yticks([])
    ax2.set_title("Recording Activity")
    ax2.set_xlabel("Time")

    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    for ax in [ax1, ax2]:
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    out_path = out_dir / "rms_timeline.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def analyse_file(wav_path: Path, verbose: bool = True) -> dict:
    """Runs quality analysis on a WAV file and optionally prints results."""
    samples, sr = load_wav(wav_path)
    result = analyze_wav(samples, sr)
    dur = len(samples) / sr
    peak = rms_to_db(float(np.max(np.abs(samples))))

    if verbose:
        snr, dc, clip = result["snr_db"], result["dc_offset_db"], result["clipping_pct"]
        flag = "  CLIPPING DETECTED" if clip > 0.01 else ""
        print(f"\n  File     : {wav_path.name}")
        print(f"  Duration : {dur:.1f} s")
        print(f"  Peak     : {peak:.1f} dBFS")
        print(f"  SNR      : {snr:.1f} dB")
        print(f"  DC offset: {dc:.0f} dB")
        print(f"  Clipping : {clip:.3f} %{flag}")
        bands = result.get("band_energies_db", [])
        if bands:
            print("  Band energies:")
            for label, val in zip(BAND_LABELS, bands):
                print(f"    {label.split(chr(10))[0]:<8}: {val:.1f} dBFS")

    result["duration"], result["peak_db"] = dur, peak
    return result


def write_summary(wav_files: list, analyses: dict, folder: Path, out_dir: Path) -> Path:
    """Writes a plain-text session summary to out_dir."""
    from datetime import datetime

    total_size = sum(f.stat().st_size for f in wav_files)
    total_dur = sum(get_wav_duration(f) for f in wav_files)
    snr_vals = [analyses[f]["snr_db"] for f in wav_files if f in analyses]
    clip_count = sum(1 for f in wav_files if f in analyses and analyses[f]["clipping_pct"] > 0.01)

    csv_line = ""
    csv_path = folder / "rms_log.csv"
    if csv_path.exists():
        ts, _, trg, _ = load_csv(csv_path)
        if ts:
            n_trig = sum(1 for i in range(1, len(trg)) if trg[i] == 1 and trg[i-1] == 0)
            csv_line = (
                f"Session start  : {ts[0].strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Session end    : {ts[-1].strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Trigger events : {n_trig}"
            )

    lines = [
        "Session Summary:",
        f"Generated      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Source folder  : {folder}",
        "",
        "── RECORDINGS ───────────────────────────────────────",
        f"Total files    : {len(wav_files)}",
        f"Total size     : {total_size / (1024**2):.1f} MB",
        f"Total duration : {total_dur/3600:.2f} h  ({total_dur:.0f} s)",
        "",
        "── SESSION ──────────────────────────────────────────",
    ]
    if csv_line:
        lines.append(csv_line)

    lines += [
        "",
        "── QUALITY ANALYSIS ─────────────────────────────────",
        f"Files analysed : {len(snr_vals)} / {len(wav_files)}",
    ]
    if snr_vals:
        lines += [
            f"Mean SNR       : {np.mean(snr_vals):.1f} dB",
            f"Min  SNR       : {np.min(snr_vals):.1f} dB",
            f"Max  SNR       : {np.max(snr_vals):.1f} dB",
            f"Clipped files  : {clip_count}",
        ]

    lines += ["", "── PER-FILE DETAILS ─────────────────────────────────"]
    for f in wav_files:
        dur = get_wav_duration(f)
        a = analyses.get(f)
        if a:
            flag = "  CLIP" if a["clipping_pct"] > 0.01 else ""
            lines.append(
                f"  {f.name:<42}  {dur:6.1f}s  "
                f"SNR:{a['snr_db']:5.1f}dB  "
                f"DC:{a['dc_offset_db']:5.0f}dB  "
                f"Clip:{a['clipping_pct']:.3f}%{flag}"
            )
        else:
            lines.append(f"  {f.name:<42}  {dur:6.1f}s  (not analysed)")
    lines += ["", "=" * 56]

    out_path = out_dir / "session_summary.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="console",
        description="Audio Recording CLI — analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("-i", "--input",   required=True, metavar="PATH",
                   help="Path to a .wav file or folder.")
    p.add_argument("-o", "--output",  default="outputs", metavar="PATH",
                   help="Output folder (default: ./outputs/).")
    p.add_argument("-a", "--analyze", action="store_true",
                   help="Print quality metrics.")
    p.add_argument("-s", "--summary", action="store_true",
                   help="Write a summary .txt file.")
    p.add_argument("-p", "--plot",    action="append", metavar="TYPE",
                   choices=["waveform", "spectrogram", "rms"],
                   help="Save plot: waveform | spectrogram | rms.")
    p.add_argument("-f", "--filter",  nargs=2, type=float, metavar=("LOW", "HIGH"),
                   help="Bandpass filter (e.g., -f 500 4000).")
    p.add_argument("--all", action="store_true",
                   help="Run full analysis and generate all plots.")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"[error] Input path not found: {input_path}")

    if input_path.is_dir():
        wav_files = sorted(input_path.glob("*.wav"), key=lambda f: f.stat().st_mtime)
        folder = input_path
    elif input_path.suffix.lower() == ".wav":
        wav_files, folder = [input_path], input_path.parent
    else:
        sys.exit("[error] Input must be a .wav file or a folder.")

    if not wav_files:
        sys.exit(f"[error] No .wav files found in {input_path}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        args.analyze = True
        args.summary = True
        args.plot = ["waveform", "spectrogram", "rms"]

    plot_types = list(set(args.plot)) if args.plot else []
    low_hz, high_hz = args.filter if args.filter else (20, 20000)

    analyses = {}
    needs_analysis = args.analyze or args.summary or "waveform" in plot_types

    if needs_analysis:
        print(f"\nAnalysing {len(wav_files)} file(s)...")
        for wav in wav_files:
            try:
                analyses[wav] = analyse_file(wav, verbose=args.analyze)
            except Exception as e:
                print(f"  [!] Analysis failed for {wav.name}: {e}")

    if plot_types:
        print(f"\nSaving plots to {out_dir}/")

    for wav in wav_files:
        if "waveform" in plot_types:
            try:
                p = plot_waveform(wav, out_dir, low_hz, high_hz, analyses.get(wav))
                print(f"  [waveform]     {p.name}")
            except Exception as e:
                print(f"  [!] Waveform failed for {wav.name}: {e}")

        if "spectrogram" in plot_types:
            try:
                p = plot_spectrogram(wav, out_dir, low_hz, high_hz)
                print(f"  [spectrogram]  {p.name}")
            except Exception as e:
                print(f"  [!] Spectrogram failed for {wav.name}: {e}")

    if "rms" in plot_types:
        try:
            p = plot_rms(folder, out_dir)
            if p:
                print(f"  [rms]          {p.name}")
        except Exception as e:
            print(f"  [!] RMS plot failed: {e}")

    if args.summary:
        try:
            p = write_summary(wav_files, analyses, folder, out_dir)
            print(f"\nSummary written: {p}")
        except Exception as e:
            print(f"  [!] Summary failed: {e}")

    if not (args.analyze or args.summary or plot_types):
        parser.print_help()

    print()


if __name__ == "__main__":
    main()
