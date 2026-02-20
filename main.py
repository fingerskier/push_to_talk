"""
Push-to-Talk Speech-to-Text for Windows
========================================
Hold INSERT to record, release to transcribe and type into active window.

Requirements:
    pip install faster-whisper sounddevice numpy keyboard pyperclip

First run will download the Whisper model (~150MB for 'base', ~75MB for 'tiny').

Usage:
    python push_to_talk.py [--model tiny|base|small|medium|large-v3]
                           [--key insert]
                           [--language en]
                           [--paste]         # use clipboard paste instead of typing
                           [--no-beep]       # disable audio feedback

Run as Administrator for global hotkey capture in all apps.
"""

import argparse
import io
import logging
import os
import queue
import sys
import threading
import time
import wave
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path

import keyboard
import numpy as np
import pyperclip
import sounddevice as sd

# --- Audio feedback (optional beeps) ---

def generate_beep(freq=800, duration=0.08, volume=0.3, sample_rate=16000):
    """Generate a short beep as numpy array."""
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    # Apply fade in/out to avoid clicks
    fade = int(sample_rate * 0.01)
    envelope = np.ones_like(t)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    return (np.sin(2 * np.pi * freq * t) * volume * envelope).astype(np.float32)

BEEP_START = generate_beep(freq=600, duration=0.06)
BEEP_STOP = generate_beep(freq=800, duration=0.06)
BEEP_ERROR = generate_beep(freq=300, duration=0.15)

HOOK_REFRESH_INTERVAL = 300  # seconds between keyboard hook re-registration


def _setup_cuda_dll_paths():
    """Register CUDA DLL directories on Windows so ctranslate2 can find them."""
    if sys.platform != "win32":
        return

    # Add pip-installed NVIDIA package DLL paths (e.g. nvidia-cublas-cu12)
    import importlib.util
    spec = importlib.util.find_spec("nvidia")
    if spec and spec.submodule_search_locations:
        for nvidia_dir in spec.submodule_search_locations:
            nvidia_path = Path(nvidia_dir)
            for bin_dir in nvidia_path.glob("*/bin"):
                if bin_dir.is_dir():
                    os.add_dll_directory(str(bin_dir))

    # Add system CUDA Toolkit path if available
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        cuda_bin = Path(cuda_path) / "bin"
        if cuda_bin.is_dir():
            os.add_dll_directory(str(cuda_bin))


def play_beep(beep_data, sample_rate=16000):
    """Play a beep non-blocking."""
    try:
        sd.play(beep_data, samplerate=sample_rate, blocksize=1024)
    except Exception:
        pass


# --- Logging ---

def setup_logging(enabled=True):
    """Configure rotating file logger in ~/.push_to_talk/."""
    logger = logging.getLogger("push_to_talk")
    logger.setLevel(logging.DEBUG)

    if not enabled:
        logger.addHandler(logging.NullHandler())
        return logger

    log_dir = Path.home() / ".push_to_talk"
    log_dir.mkdir(exist_ok=True)

    handler = RotatingFileHandler(
        log_dir / "push_to_talk.log",
        maxBytes=1_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


# --- Core push-to-talk engine ---

class PushToTalk:
    def __init__(self, model_size="base", hotkey="insert", language="en",
                 use_paste=False, beep=True, device=None, force_cpu=False,
                 beam_size=1, logger=None):
        self.hotkey = hotkey
        self.language = language
        self.use_paste = use_paste
        self.beep = beep
        self.device = device
        self.beam_size = beam_size
        self.sample_rate = 16000
        self.channels = 1
        self.recording = False
        self.audio_chunks = []
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self._transcription_queue = queue.Queue()
        self.log = logger or logging.getLogger("push_to_talk")

        self.log.info("Starting push_to_talk (model=%s, key=%s, lang=%s, paste=%s, cpu=%s, beam=%d)",
                      model_size, hotkey, language, use_paste, force_cpu, beam_size)

        print(f"Loading Whisper model '{model_size}'... ", end="", flush=True)
        _setup_cuda_dll_paths()
        from faster_whisper import WhisperModel

        if force_cpu:
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
            print("done (CPU).")
        else:
            try:
                import ctranslate2
                cuda_count = ctranslate2.get_cuda_device_count()
                self.log.info("CUDA devices detected: %d", cuda_count)
                if cuda_count == 0:
                    print("no CUDA devices found, using CPU... ", end="", flush=True)
                    self.log.warning("No CUDA devices found, using CPU")
                    self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
                    print("done (CPU).")
                else:
                    self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
                    print("done (CUDA).")
            except Exception as e:
                print(f"GPU unavailable ({e}), falling back to CPU... ", end="", flush=True)
                self.log.warning("GPU unavailable (%s), falling back to CPU", e)
                self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
                print("done (CPU).")
        self.log.info("Model '%s' loaded", model_size)

        # Warm up the model with a short silence
        silence = np.zeros(self.sample_rate, dtype=np.float32)
        segments, _ = self.model.transcribe(silence, language=self.language)
        list(segments)  # consume generator
        print("Model warmed up and ready.\n")
        self.log.info("Model warmed up and ready")

        # Start persistent audio input stream (avoids open/close overhead per recording)
        self._audio_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=1024,
            device=self.device,
            callback=self._audio_callback,
        )
        self._audio_stream.start()
        self.log.info("Persistent audio stream started")

        # Start persistent transcription worker thread
        self._transcription_worker = threading.Thread(
            target=self._transcription_loop, daemon=True
        )
        self._transcription_worker.start()

    def _audio_callback(self, indata, frames, time_info, status):
        """Persistent audio stream callback — only buffers data while recording."""
        if self.recording:
            self.audio_chunks.append(indata.copy())

    def start_recording(self):
        """Called when hotkey is pressed."""
        with self.lock:
            if self.recording:
                return
            self.recording = True
            self.audio_chunks = []

        self.log.info("Recording started")
        if self.beep:
            play_beep(BEEP_START)

    def stop_recording(self):
        """Called when hotkey is released."""
        with self.lock:
            if not self.recording:
                return
            self.recording = False
            chunks = self.audio_chunks
            self.audio_chunks = []

        self.log.info("Recording stopped")
        if self.beep:
            play_beep(BEEP_STOP)

        # Enqueue for the persistent transcription worker
        self._transcription_queue.put(chunks)

    def _transcription_loop(self):
        """Persistent worker that processes transcription jobs from the queue."""
        while True:
            chunks = self._transcription_queue.get()
            try:
                self._transcribe(chunks)
            except Exception as e:
                print(f"\n  [!] Transcription error: {e}")
                self.log.exception("Transcription error")
                if self.beep:
                    play_beep(BEEP_ERROR)

    def _transcribe(self, chunks):
        """Transcribe recorded audio and type the result."""
        if not chunks:
            return

        # Concatenate directly to a flat array (avoids extra .flatten() copy)
        audio = np.concatenate(chunks, axis=None)
        duration = len(audio) / self.sample_rate

        if duration < 0.3:
            # Too short, probably accidental
            return

        print(f"  Transcribing {duration:.1f}s of audio... ", end="", flush=True)
        self.log.info("Transcribing %.1fs of audio", duration)

        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
        )

        text = " ".join(seg.text.strip() for seg in segments).strip()

        if not text:
            print("(no speech detected)")
            self.log.info("No speech detected")
            return

        print(f'"{text}"')
        self.log.info("Transcribed: %s", text)
        self._type_text(text)

    def _type_text(self, text):
        """Insert text into the active window."""
        if self.use_paste:
            # Save and restore clipboard
            try:
                old_clip = pyperclip.paste()
            except Exception:
                old_clip = ""
            pyperclip.copy(text)
            time.sleep(0.05)
            keyboard.send("ctrl+v")
            time.sleep(0.1)
            try:
                pyperclip.copy(old_clip)
            except Exception:
                pass
        else:
            # Type character by character (works everywhere but slower)
            keyboard.write(text, delay=0.01)

    def _register_hooks(self):
        """Register (or re-register) all keyboard hooks."""
        keyboard.unhook_all()
        keyboard.on_press_key(self.hotkey, lambda e: e.is_keypad or self.start_recording(), suppress=True)
        keyboard.on_release_key(self.hotkey, lambda e: e.is_keypad or self.stop_recording(), suppress=True)

    def run(self):
        """Main loop."""
        print("=" * 55)
        print(f"  Push-to-Talk active!")
        print(f"  Hold [{self.hotkey.upper()}] to speak, release to transcribe")
        print(f"  Input method: {'clipboard paste' if self.use_paste else 'keyboard typing'}")
        print(f"  Press Ctrl+C to quit")
        print("=" * 55)
        print()

        self._register_hooks()
        self.log.info("Keyboard hooks registered")

        try:
            while not self._stop_event.wait(timeout=HOOK_REFRESH_INTERVAL):
                with self.lock:
                    busy = self.recording
                if busy:
                    self.log.debug("Hook refresh skipped (busy)")
                    continue
                try:
                    self._register_hooks()
                    self.log.debug("Keyboard hooks refreshed")
                except Exception:
                    self.log.warning("Hook refresh failed", exc_info=True)
            self.log.info("Shutdown reason: stop event was set")
        except KeyboardInterrupt:
            self.log.info("Shutdown reason: Ctrl+C")
        except Exception:
            self.log.exception("Shutdown reason: unexpected error")
        finally:
            print("\nShutting down.")
            try:
                self._audio_stream.stop()
                self._audio_stream.close()
            except Exception:
                pass
            self.log.info("Shutting down (cleanup complete)")
            keyboard.unhook_all()


def list_audio_devices():
    """Print available audio input devices."""
    print("\nAvailable audio input devices:")
    print("-" * 50)
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            marker = " <-- default" if i == sd.default.device[0] else ""
            print(f"  [{i}] {d['name']}{marker}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Push-to-Talk Speech-to-Text for Windows"
    )
    parser.add_argument(
        "--model", default="base", 
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "--key", default="insert",
        help="Push-to-talk key (default: insert)"
    )
    parser.add_argument(
        "--language", default="en",
        help="Language code (default: en)"
    )
    parser.add_argument(
        "--paste", action="store_true",
        help="Use clipboard paste instead of keyboard typing"
    )
    parser.add_argument(
        "--no-beep", action="store_true",
        help="Disable audio feedback beeps"
    )
    parser.add_argument(
        "--device", type=int, default=None,
        help="Audio input device index (use --list-devices to see options)"
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="List audio input devices and exit"
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU mode (skip CUDA/GPU)"
    )
    parser.add_argument(
        "--beam-size", type=int, default=1,
        help="Beam size for decoding (default: 1 greedy, higher=slower but may improve accuracy)"
    )
    parser.add_argument(
        "--no-logs", action="store_true",
        help="Disable file logging (enabled by default, logs to ~/.push_to_talk/)"
    )

    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    logger = setup_logging(enabled=not args.no_logs)

    ptt = PushToTalk(
        model_size=args.model,
        hotkey=args.key,
        language=args.language,
        use_paste=args.paste,
        beep=not args.no_beep,
        device=args.device,
        force_cpu=args.cpu,
        beam_size=args.beam_size,
        logger=logger,
    )
    ptt.run()


if __name__ == "__main__":
    main()
