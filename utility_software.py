"""
Utility Software v4
Bachelor's Project — Design and Implementation of a System for Long-Term Audio Recording (Lukas Hrobak)
RPi5 + PCM1807 ADC

Commands for ssh:
ssh pi@pitester.local               # Connect to Rpi
tail -f /home/pi/recorder.log       # Real time recording analysis
sudo umount /media/pi/RECDATA       # Unmount usb

"""

import shutil
import time
import wave
import subprocess
import threading
from collections import deque
import numpy as np
from datetime import datetime
from pathlib import Path

# Configuration
AUDIO_DEVICE    = "monomic"
SAMPLE_RATE     = 48000
CHANNELS        = 1
FORMAT          = "S32_LE"
BIT_DEPTH       = 32

# Dynamic threshold configuration
DYNAMIC_THRESHOLD       = True    # Enable adaptive noise floor estimation
NOISE_EST_WINDOW_SEC    = 60.0    # Rolling window for noise estimation (seconds)
TRIGGER_MARGIN_DB       = 12.0    # dB above estimated noise floor to trigger
NOISE_MARGIN_DB         = 5.0     # dB above estimated noise floor to stop recording
ABSOLUTE_MIN_TRIGGER_DB = -55.0   # Never trigger below this level (true silence guard)

# Static fallback thresholds (used when DYNAMIC_THRESHOLD = False,
# or during the initial warm-up period before the noise estimate stabilises)
TRIGGER_DB      = -35.0           # dBFS — recording starts above this
NOISE_FLOOR_DB  = -42.0           # dBFS — recording stops below this

PRE_RECORD_SEC  = 1.0             # Seconds to keep before trigger
POST_RECORD_SEC = 3.0             # Seconds to keep after silence
MAX_RECORD_SEC  = 600.0           # Maximum single recording length
                                  # Note: the entire recording is buffered in RAM before saving.
                                  # At 48kHz/32-bit mono, 600 s ≈ 115 MB — safe for RPi 5 (4 GB).
                                  # For longer limits, consider a streaming WAV writer instead.

USB_LABEL       = "RECDATA"
USB_MOUNT_BASE  = "/media/pi"
USB_CHECK_SEC   = 5.0

SAVE_DIR        = None            # Set automatically from USB
LOG_FILE        = None            # Set automatically from USB

MIN_FREE_GB     = 2.0             # Keep at least 2 GB free
FREE_TARGET_GB  = 3.0             # Stop deleting once this much free space is restored

CHUNK_SEC       = 0.1
CHUNK_SAMPLES   = int(SAMPLE_RATE * CHUNK_SEC)
BYTES_PER_CHUNK = CHUNK_SAMPLES * CHANNELS * (BIT_DEPTH // 8)

PRE_BUFFER_CHUNKS  = int(PRE_RECORD_SEC / CHUNK_SEC)
NOISE_EST_CHUNKS   = int(NOISE_EST_WINDOW_SEC / CHUNK_SEC)  # chunks in estimation window
NOISE_EST_PERCENTILE = 30   # use 30th percentile of recent RMS values as noise estimate

CSV_BATCH_SIZE  = 100             # Flush to disk every N chunks


def find_usb():
    """Searches for the USB drive by label."""
    mount_path = Path(USB_MOUNT_BASE) / USB_LABEL
    if mount_path.exists() and mount_path.is_dir():
        return mount_path
    base = Path(USB_MOUNT_BASE)
    if base.exists():
        for child in base.iterdir():
            candidate = child / USB_LABEL
            if candidate.exists():
                return candidate
    return None


def wait_for_usb():
    """Blocks until the USB drive is connected and sets global paths."""
    global SAVE_DIR, LOG_FILE
    print(f"Looking for USB stick '{USB_LABEL}'...")
    while True:
        mount = find_usb()
        if mount:
            SAVE_DIR = mount / "recordings"
            LOG_FILE = mount / "recordings" / "rms_log.csv"
            print(f"USB found at: {mount}")
            print(f"Saving to: {SAVE_DIR}")
            return mount
        print(f"USB not found — retrying in {USB_CHECK_SEC:.0f}s... (plug in '{USB_LABEL}')")
        time.sleep(USB_CHECK_SEC)


def check_usb_still_present():
    """Checks if the USB mount still exists."""
    return find_usb() is not None


def setup_dirs():
    """Creates save directories and initializes the CSV log file if missing."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_FILE.exists():
        with open(LOG_FILE, "w") as f:
            f.write("timestamp,rms_db,triggered,recording\n")
    print(f"Save directory ready: {SAVE_DIR}")


def estimate_noise_floor(noise_buffer):
    """
    Estimates the ambient noise floor from a rolling buffer of recent RMS dB values.
    Uses the NOISE_EST_PERCENTILE-th percentile to represent background noise
    while ignoring transient acoustic events in the window.
    """
    if len(noise_buffer) < NOISE_EST_CHUNKS // 2:
        return None   # not enough data yet — fall back to static thresholds
    return float(np.percentile(list(noise_buffer), NOISE_EST_PERCENTILE))


def rms_to_db(rms):
    """Converts RMS to dBFS. Uses 2**31 divisor for 24-bit left-justified ALSA samples."""
    if rms == 0:
        return -120.0
    return 20 * np.log10(rms / (2 ** (BIT_DEPTH - 1)))


def read_chunk(process):
    """Reads exactly BYTES_PER_CHUNK from the pipe. Returns None if stream dies."""
    data = b""
    remaining = BYTES_PER_CHUNK
    while remaining > 0:
        chunk = process.stdout.read(remaining)
        if not chunk:
            return None
        data += chunk
        remaining -= len(chunk)
    return np.frombuffer(data, dtype=np.int32).astype(np.float64)


def compute_rms(samples):
    """Calculates RMS of a sample array."""
    return np.sqrt(np.mean(samples ** 2))


def flush_csv_batch(batch):
    """Appends accumulated CSV rows to the log file."""
    if not batch:
        return
    try:
        with open(LOG_FILE, "a") as f:
            f.write("".join(batch))
    except Exception as e:
        print(f"\nCSV write error: {e}")


def check_storage():
    """Deletes oldest recordings if free space drops below MIN_FREE_GB."""
    try:
        usage   = shutil.disk_usage(SAVE_DIR)
        free_gb = usage.free / (1024 ** 3)

        if free_gb < MIN_FREE_GB:
            print(f"\nLow storage! {free_gb:.1f} GB free — cleaning up...")
            files = sorted(SAVE_DIR.glob("*.wav"), key=lambda f: f.stat().st_mtime)

            for f in files:
                f.unlink(missing_ok=True)
                print(f"Deleted: {f.name}")

                new_free = shutil.disk_usage(SAVE_DIR).free / (1024 ** 3)
                if new_free > FREE_TARGET_GB:
                    print(f"Storage OK — {new_free:.1f} GB now free.")
                    break

    except Exception as e:
        print(f"\nStorage check error: {e}")


def save_wav(buf, name):
    """Saves the sample buffer to a WAV file."""
    try:
        all_samples = np.concatenate(buf).astype(np.int32)
        filepath = SAVE_DIR / name
        with wave.open(str(filepath), "w") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(BIT_DEPTH // 8)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(all_samples.tobytes())
        size_kb = filepath.stat().st_size // 1024
        print(f"\nSaved: {filepath.name} ({size_kb} KB)")
    except Exception as e:
        print(f"\nSave error: {e}")


def save_async(buf, name):
    """Background thread target to check storage and save a WAV file."""
    check_storage()
    save_wav(buf, name)


def start_arecord():
    """Spawns the ALSA arecord process and returns the Popen object."""
    cmd = [
        "arecord",
        "-D", AUDIO_DEVICE,
        "-f", FORMAT,
        "-r", str(SAMPLE_RATE),
        "-c", str(CHANNELS),
        "-t", "raw",
        "--buffer-size=8192",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def monitor():
    """Main loop: monitors audio stream, handles triggers, and manages state."""

    print("Utility Software v4")
    print("System for Long-Term Audio Recording (Lukas Hrobak)")
    print(f"Threshold mode    : {'dynamic' if DYNAMIC_THRESHOLD else 'static'}")
    if DYNAMIC_THRESHOLD:
        print(f"  Trigger margin  : +{TRIGGER_MARGIN_DB} dB above noise estimate")
        print(f"  Noise margin    : +{NOISE_MARGIN_DB} dB above noise estimate")
        print(f"  Estimation win  : {NOISE_EST_WINDOW_SEC:.0f}s ({NOISE_EST_PERCENTILE}th percentile)")
        print(f"  Absolute min    : {ABSOLUTE_MIN_TRIGGER_DB} dBFS")
    else:
        print(f"  Trigger         : {TRIGGER_DB} dBFS")
        print(f"  Noise floor     : {NOISE_FLOOR_DB} dBFS")
    print(f"Max clip length   : {MAX_RECORD_SEC}s")
    print(f"Min free space    : {MIN_FREE_GB} GB")
    print(f"USB label         : {USB_LABEL}")
    print(f"Pre-buffer        : {PRE_BUFFER_CHUNKS} chunks ({PRE_RECORD_SEC}s)")
    print(f"CSV batch size    : {CSV_BATCH_SIZE} chunks ({CSV_BATCH_SIZE * CHUNK_SEC:.0f}s)")
    print("Press Ctrl+C to stop.\n")

    wait_for_usb()
    setup_dirs()

    pre_buffer   = deque(maxlen=PRE_BUFFER_CHUNKS)
    noise_buffer = deque(maxlen=NOISE_EST_CHUNKS)   # rolling RMS history for noise estimation

    recording      = False
    record_buffer  = []
    record_chunks  = 0
    silence_chunks = 0
    rms_db_peak    = -120.0

    csv_batch      = []
    usb_missing_warned = False

    process = start_arecord()
    print("Listening...\n")

    try:
        while True:

            # USB presence check
            if not check_usb_still_present():
                if not usb_missing_warned:
                    print("\nUSB removed! Pausing saves — plug back in to resume...")
                    usb_missing_warned = True
                if read_chunk(process) is None:
                    break
                time.sleep(0.05)
                continue
            else:
                if usb_missing_warned:
                    print("USB detected — resuming.")
                    wait_for_usb()
                    setup_dirs()
                    usb_missing_warned = False

            samples = read_chunk(process)

            # Restart arecord if stream genuinely dies
            if samples is None:
                print("\nAudio stream ended — restarting...")
                process.kill()
                time.sleep(1)
                process = start_arecord()
                continue

            rms    = compute_rms(samples)
            rms_db = rms_to_db(rms)
            now    = datetime.now()
            ts     = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            # Update noise estimation buffer (only during silence to avoid
            # polluting the estimate with loud acoustic events)
            if not recording:
                noise_buffer.append(rms_db)

            # Compute effective thresholds
            if DYNAMIC_THRESHOLD:
                noise_est = estimate_noise_floor(noise_buffer)
                if noise_est is not None:
                    eff_trigger     = max(noise_est + TRIGGER_MARGIN_DB, ABSOLUTE_MIN_TRIGGER_DB)
                    eff_noise_floor = noise_est + NOISE_MARGIN_DB
                else:
                    # Warm-up: fall back to static thresholds
                    eff_trigger     = TRIGGER_DB
                    eff_noise_floor = NOISE_FLOOR_DB
                    noise_est       = float("nan")
            else:
                eff_trigger     = TRIGGER_DB
                eff_noise_floor = NOISE_FLOOR_DB
                noise_est       = float("nan")

            above_trigger = rms_db > eff_trigger
            above_noise   = rms_db > eff_noise_floor

            # Batch CSV accumulation
            csv_batch.append(f"{ts},{rms_db:.2f},{int(above_trigger)},{int(recording)}\n")
            if len(csv_batch) >= CSV_BATCH_SIZE:
                flush_csv_batch(csv_batch)
                csv_batch = []

            # Live level meter (show effective trigger threshold)
            bar_len = int(max(0, min(40, (rms_db + 80) / 2)))
            bar     = "#" * bar_len + "-" * (40 - bar_len)
            status  = "REC" if recording else "   "
            thr_str = f"{eff_trigger:.1f}" if not np.isnan(noise_est) else f"{eff_trigger:.1f}*"
            print(f"\r[{status}] {rms_db:6.1f} dBFS [{bar}] thr:{thr_str}", end="", flush=True)

            # Pre-trigger rolling buffer
            if not recording:
                pre_buffer.append(samples.copy())

            # Trigger detection
            if above_trigger and not recording:
                print(f"\nTRIGGER {now.strftime('%H:%M:%S')} — {rms_db:.1f} dBFS  (thr: {eff_trigger:.1f} dBFS)")
                recording      = True
                record_buffer  = list(pre_buffer)
                record_chunks  = len(pre_buffer)
                silence_chunks = 0
                rms_db_peak    = rms_db

            # Active recording state
            if recording:
                record_buffer.append(samples.copy())
                record_chunks += 1
                if rms_db > rms_db_peak:
                    rms_db_peak = rms_db

                if above_noise:
                    silence_chunks = 0
                else:
                    silence_chunks += 1

                record_sec  = record_chunks  * CHUNK_SEC
                silence_sec = silence_chunks * CHUNK_SEC

                stop_reason = None
                if silence_sec >= POST_RECORD_SEC:
                    stop_reason = f"silence ({POST_RECORD_SEC}s below noise floor)"
                elif record_sec >= MAX_RECORD_SEC:
                    stop_reason = f"max length ({MAX_RECORD_SEC}s)"

                if stop_reason:
                    print(f"\nSTOP  {now.strftime('%H:%M:%S')} — {stop_reason}")
                    print(f"      Duration: {record_sec:.1f}s | Peak: {rms_db_peak:.1f} dBFS")
                    recording = False

                    fname    = now.strftime("rec_%Y%m%d_%H%M%S.wav")
                    buf_copy = list(record_buffer)

                    threading.Thread(
                        target=save_async,
                        args=(buf_copy, fname),
                        daemon=True
                    ).start()

                    record_buffer  = []
                    record_chunks  = 0
                    silence_chunks = 0
                    rms_db_peak    = -120.0
                    pre_buffer.clear()

    except KeyboardInterrupt:
        print("\n\nStopping...")
        flush_csv_batch(csv_batch)
        if recording and record_buffer:
            print("Saving current recording before exit...")
            fname = datetime.now().strftime("rec_%Y%m%d_%H%M%S_final.wav")
            save_wav(record_buffer, fname)
    finally:
        process.kill()
        print("Firmware stopped.")


if __name__ == "__main__":
    monitor()
