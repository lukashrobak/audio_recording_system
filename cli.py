"""
Cli app v4
Bachelor's Project — Design and Implementation of a System for Long-Term Audio Recording (Lukas Hrobak)

Commands:
  python cli.py -i rec.wav -a                          # Analyze single file to stdout
  python cli.py -i recordings/ -s                      # Analyze folder, write summary txt
  python cli.py -i rec.wav -p waveform                 # Save waveform PNG
  python cli.py -i rec.wav -p spectrogram -f 500 4000  # Save spectrogram (filtered)
  python cli.py -i recordings/ -p rms                  # Save RMS timeline (needs rms_log.csv)
  python cli.py -i recordings/ --all -o results/       # Full analysis + all plots
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

from audio_processing import (
    load_wav, load_csv, analyze_wav,
    apply_bandpass, compute_spectrogram,
    get_wav_duration, rms_to_db,
    parse_hour_from_filename,
    TRIGGER_DB, NOISE_DB,
    BAND_LABELS, BAND_COLORS, BAND_RANGES,
)

# Dark Theme Constants
BG, FG, DIM, GRID = "#13161f", "#e8eaf6", "#6b7280", "#1e2235"
ACCENT, ACCENT2 = "#4fc3f7", "#81d4fa"
GREEN, ORANGE, RED, PURPLE = "#66bb6a", "#ffa726", "#ef5350", "#ce93d8"


def _style_ax(ax, title="", xlabel="", ylabel=""):
    """Applies dark theme to a matplotlib Axes."""
    ax.set_facecolor(BG)
    ax.tick_params(colors=DIM, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    if title:  ax.set_title(title, color=FG, fontsize=10, pad=6, fontweight="bold")
    if xlabel: ax.set_xlabel(xlabel, color=DIM, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=DIM, fontsize=8)


def _new_fig(w=14, h=6):
    return plt.figure(figsize=(w, h), facecolor=BG)


def plot_waveform(wav_path: Path, out_dir: Path, low_hz: float = 20, high_hz: float = 20000, analysis: dict = None) -> Path:
    """Saves a waveform + RMS envelope + band-energy plot."""
    samples, sr = load_wav(wav_path)
    if low_hz > 20 or high_hz < 20000:
        samples = apply_bandpass(samples, sr, low_hz, high_hz)

    n = len(samples)
    duration = n / sr
    t_axis = np.linspace(0, duration, n)

    fig = _new_fig(14, 7)
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.08, height_ratios=[5, 2], width_ratios=[4, 1])
    ax_wav = fig.add_subplot(gs[0, :])
    ax_bnd = fig.add_subplot(gs[1, 0])
    ax_sta = fig.add_subplot(gs[1, 1])

    # Waveform
    _style_ax(ax_wav, title=f"Waveform — {wav_path.name}", xlabel="Time (s)", ylabel="Amplitude")
    step = max(1, n // 10000)
    ax_wav.plot(t_axis[::step], samples[::step], color=ACCENT, linewidth=0.6, alpha=0.9, label="Waveform")

    # RMS envelope (50 ms windows)
    win = int(sr * 0.05)
    rms_env, rms_t = [], []
    for i in range(0, n - win, win):
        rms_env.append(float(np.sqrt(np.mean(samples[i:i+win] ** 2))))
        rms_t.append(i / sr)
    ax_wav.fill_between(rms_t, np.array(rms_env), -np.array(rms_env), color=ACCENT2, alpha=0.2, label="RMS envelope")

    trig_amp = 10 ** (TRIGGER_DB / 20.0)
    ax_wav.axhline(trig_amp, color=RED, linestyle="--", linewidth=0.8, alpha=0.7, label=f"Trigger ({TRIGGER_DB} dBFS)")
    ax_wav.axhline(-trig_amp, color=RED, linestyle="--", linewidth=0.8, alpha=0.7)
    ax_wav.set_ylim(-1.05, 1.05)
    ax_wav.legend(loc="upper right", fontsize=7, facecolor="#22263a", edgecolor="#22263a", labelcolor=FG)

    # Info panel
    ax_sta.set_facecolor(BG)
    ax_sta.axis("off")
    peak_db = rms_to_db(float(np.max(np.abs(samples))))

    def row(y, label, value, color=ACCENT2):
        ax_sta.text(0.05, y, label, transform=ax_sta.transAxes, fontsize=8, color=DIM, fontfamily="Courier New", va="top")
        ax_sta.text(0.95, y, value, transform=ax_sta.transAxes, fontsize=8, color=color, fontfamily="Courier New", va="top", ha="right", fontweight="bold")

    ax_sta.text(0.5, 1.04, "Info", transform=ax_sta.transAxes, ha="center", va="top", fontsize=9, color=ACCENT, fontfamily="Courier New", fontweight="bold")
    row(0.88, "Duration", f"{duration:.1f} s")
    row(0.74, "Peak", f"{peak_db:.1f} dB")

    if analysis:
        snr, dc, clip = analysis["snr_db"], analysis["dc_offset_db"], analysis["clipping_pct"]
        row(0.58, "SNR", f"{snr:.1f} dB", GREEN if snr > 30 else (ORANGE if snr > 15 else RED))
        row(0.44, "DC", f"{dc:.0f} dB")
        row(0.30, "Clip", f"{clip:.3f} %", RED if clip > 0.01 else GREEN)
    else:
        row(0.58, "Analysis", "not run", DIM)

    # Band energy bars
    ax_bnd.set_facecolor(BG)
    ax_bnd.tick_params(colors=DIM, labelsize=7)
    for spine in ax_bnd.spines.values(): spine.set_color(GRID)
    ax_bnd.set_title("Frequency Band Energy", color=FG, fontsize=8, pad=4, fontweight="bold")
    ax_bnd.set_xlabel("dBFS", color=DIM, fontsize=7)

    if analysis and analysis.get("band_energies_db"):
        vals = [max(-120.0, min(0.0, v)) for v in analysis["band_energies_db"]]
        y_pos = np.arange(len(BAND_LABELS))
        bars = ax_bnd.barh(y_pos, vals, color=BAND_COLORS, alpha=0.85, height=0.55)
        ax_bnd.set_yticks(y_pos)
        ax_bnd.set_yticklabels(BAND_LABELS, fontsize=7, color=DIM)
        ax_bnd.set_xlim(-120, 0)
        ax_bnd.axvline(TRIGGER_DB, color=RED, linewidth=0.8, linestyle="--", alpha=0.6)
        ax_bnd.axvline(NOISE_DB, color=GREEN, linewidth=0.8, linestyle="--", alpha=0.6)
        for bar, val in zip(bars, vals):
            ax_bnd.text(max(val + 1, -118), bar.get_y() + bar.get_height() / 2,f"{val:.0f} dB", va="center", ha="left", fontsize=7, color=DIM, fontfamily="Courier New")
    else:
        ax_bnd.text(0.5, 0.5, "Run --analyze to see band energies", ha="center", va="center", transform=ax_bnd.transAxes, color=DIM, fontsize=8)
        ax_bnd.set_yticks([])
        ax_bnd.set_xticks([])

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

    fig = _new_fig(14, 6)
    ax = fig.add_subplot(111)
    _style_ax(ax, title=f"Spectrogram — {wav_path.name}", xlabel="Time (s)", ylabel="Frequency (Hz)")

    ax.pcolormesh(t, f, Sxx_db, shading="gouraud", cmap="inferno", vmin=-120, vmax=0)
    ax.set_ylim(0, min(float(f[-1]), 20000))
    ax.set_xlim(0, duration)

    sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(vmin=-120, vmax=0))
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("dB", color=DIM, fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=DIM, labelcolor=DIM)

    if low_hz > 20:
        ax.axhline(low_hz, color=GREEN, linewidth=1, linestyle="--", alpha=0.8, label=f"Low cut {low_hz} Hz")
    if high_hz < 20000:
        ax.axhline(high_hz, color=ORANGE, linewidth=1, linestyle="--", alpha=0.8, label=f"High cut {high_hz} Hz")
    if low_hz > 20 or high_hz < 20000:
        ax.legend(loc="upper right", fontsize=7, facecolor="#22263a", edgecolor="#22263a", labelcolor=FG)

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

    fig = _new_fig(14, 6)
    gs = GridSpec(2, 1, figure=fig, hspace=0.4, height_ratios=[3, 1])
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    _style_ax(ax1, title="RMS Level Over Time", ylabel="dBFS")
    _style_ax(ax2, title="Recording Activity", xlabel="Time")

    ax1.plot(timestamps, rms_vals, color=ACCENT, linewidth=0.5, alpha=0.8, label="RMS")

    for i, rec in enumerate(recording):
        if rec and i + 1 < len(timestamps):
            ax1.axvspan(timestamps[i], timestamps[i+1], alpha=0.15, color=ORANGE, linewidth=0)

    ax1.axhline(TRIGGER_DB, color=RED, linestyle="--", linewidth=1, label=f"Trigger ({TRIGGER_DB} dBFS)")
    ax1.axhline(NOISE_DB, color=GREEN, linestyle="--", linewidth=1, label=f"Noise floor ({NOISE_DB} dBFS)")

    event_times = [timestamps[i] for i in range(1, len(triggered)) if triggered[i] == 1 and triggered[i-1] == 0]
    for et in event_times:
        ax1.axvline(et, color=RED, linewidth=0.6, alpha=0.5, zorder=3)
    if event_times:
        ax1.axvline(event_times[0], color=RED, linewidth=0.6, alpha=0.5,
                    label=f"Trigger events ({len(event_times)})", zorder=3)

    ax1.set_ylim(-90, 0)
    ax1.legend(loc="upper right", fontsize=7, facecolor="#22263a", edgecolor="#22263a", labelcolor=FG)

    ax2.fill_between(timestamps, np.array(recording, dtype=float), 0, color=ORANGE, alpha=0.7, step="post")
    ax2.set_ylim(0, 1.5)
    ax2.set_yticks([])

    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
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