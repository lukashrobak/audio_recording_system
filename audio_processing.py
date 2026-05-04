"""
Audio processing software v3
Bachelor's Project — Design and Implementation of a System for Long-Term Audio Recording (Lukas Hrobak)
"""

import csv
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import signal as sp_signal

# Recording parameters
SAMPLE_RATE    = 48_000   # Hz 
INT32_MAX      = 2 ** 31  # Normalisation divisor (int32 full-scale)
TRIGGER_DB     = -35.0    # dBFS level when the recording starts
NOISE_DB       = -45.0    # dBFS estimated background noise floor

# Frequency bands used for spectral energy analysis
BAND_RANGES = [
    (20,   500),    # Low
    (500,  4_000),  # Mid
    (4_000, 10_000),# Hi-Mid
    (10_000, 20_000)# High
]
BAND_LABELS = [
    "Low\n20–500 Hz",
    "Mid\n500–4k Hz",
    "Hi-Mid\n4–10k Hz",
    "High\n10–20k Hz",
]
BAND_COLORS = ["#4fc3f7", "#66bb6a", "#ffa726", "#ce93d8"]

RAM_WARN_MB = 200


def load_wav(path: Path):
    """Reads raw int32 .wav and normalises it to float32 in range [-1.0, 1.0]."""
    with wave.open(str(path), "r") as wf:
        raw = wf.readframes(wf.getnframes())
        sr  = wf.getframerate()

    samples = (np.frombuffer(raw, dtype=np.int32).astype(np.float32) 
               / np.float32(INT32_MAX))
    return samples, sr


def load_csv(path: Path):
    """Parses rms_log.csv into timestamps, rms levels, and trigger states."""
    timestamps, rms_vals, triggered, recording = [], [], [], []
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    timestamps.append(datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S.%f"))
                    rms_vals.append(float(row["rms_db"]))
                    triggered.append(int(row["triggered"]))
                    recording.append(int(row["recording"]))
                except Exception:
                    continue  # Skip malformed rows silently
    except Exception:
        pass
    return timestamps, rms_vals, triggered, recording


def rms_to_db(rms: float) -> float:
    """Converts normalised RMS to dBFS. Returns -120dB for silence."""
    if rms <= 0.0:
        return -120.0
    return float(20.0 * np.log10(float(rms)))


def apply_bandpass(samples: np.ndarray, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    """Applies a 4th-order Butterworth bandpass filter. Bypasses if range is 20-20k."""
    nyq = sr / 2.0
    
    if low_hz <= 20 and high_hz >= 20_000:
        return samples
        
    low  = max(low_hz  / nyq, 0.001)
    high = min(high_hz / nyq, 0.999)
    
    if low >= high:
        return samples
        
    try:
        b, a = sp_signal.butter(4, [low, high], btype="band")
        return sp_signal.filtfilt(b, a, samples).astype(samples.dtype)
    except Exception:
        return samples


def compute_spectrogram(samples: np.ndarray, sr: int):
    """Computes a power spectrogram in dB using a 1024-point STFT with 50% overlap."""
    f, t, Sxx = sp_signal.spectrogram(
        samples, fs=sr, nperseg=1024, noverlap=512, scaling="spectrum")
    Sxx_db = 10.0 * np.log10(Sxx + 1e-12)
    return f, t, Sxx_db


def analyze_wav(samples: np.ndarray, sr: int) -> dict:
    """Computes recording quality metrics: SNR, clipping %, DC offset, and band energies."""
    n = len(samples)

    # Frame-based noise floor estimation (using quietest 10% of 2048-sample frames)
    frame_size = 2048
    frames     = [samples[i : i + frame_size] for i in range(0, n - frame_size, frame_size)]
    
    if len(frames) >= 5:
        frame_rms = np.array([np.sqrt(np.mean(f ** 2)) for f in frames], dtype=np.float64)
        frame_rms = frame_rms[frame_rms > 0]
        n_quiet   = max(1, len(frame_rms) // 10)
        noise_rms = float(np.mean(np.sort(frame_rms)[:n_quiet]))
        sig_rms   = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
        snr_db    = float(20.0 * np.log10((sig_rms + 1e-12) / (noise_rms + 1e-12)))
    else:
        snr_db = 0.0

    # Clipping detection (threshold at 99% of full scale)
    clipping_pct = float(100.0 * np.sum(np.abs(samples) >= 0.99) / n)

    # DC offset
    dc           = float(np.mean(samples))
    dc_offset_db = float(20.0 * np.log10(abs(dc) + 1e-12))

    # Spectral band energies via FFT (capped at 2**18 samples for speed)
    fft_n  = min(n, 2 ** 18)
    freqs  = np.fft.rfftfreq(fft_n, d=1.0 / sr)
    power  = (np.abs(np.fft.rfft(samples[:fft_n].astype(np.float64))) ** 2) / fft_n
    
    band_db = []
    for lo, hi in BAND_RANGES:
        mask = (freqs >= lo) & (freqs < hi)
        bp   = float(np.mean(power[mask])) if mask.any() else 0.0
        band_db.append(float(10.0 * np.log10(bp + 1e-12)))

    return {
        "snr_db":           snr_db,
        "clipping_pct":     clipping_pct,
        "dc_offset_db":     dc_offset_db,
        "band_energies_db": band_db,
    }


def estimate_load_ram_mb(wav_path: Path) -> float:
    """Estimates peak RAM needed to process the file (roughly 2x file size)."""
    size_bytes = wav_path.stat().st_size
    return (size_bytes * 2) / (1024 ** 2)


def parse_hour_from_filename(name: str):
    """Extracts the recording hour (0-23) from a filename like rec_YYYYMMDD_HHMMSS.wav."""
    try:
        stem  = Path(name).stem
        parts = stem.split("_")
        if len(parts) >= 3:
            return int(parts[2][:2])
    except Exception:
        pass
    return None


def get_wav_duration(path: Path) -> float:
    """Returns WAV duration in seconds using only the header data."""
    try:
        with wave.open(str(path), "r") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0