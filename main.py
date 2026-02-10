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
import queue
import sys
import threading
import time
import wave
from collections import deque

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


def play_beep(beep_data, sample_rate=16000):
    """Play a beep non-blocking."""
    try:
        sd.play(beep_data, samplerate=sample_rate, blocksize=1024)
    except Exception:
        pass


# --- Core push-to-talk engine ---

class PushToTalk:
    def __init__(self, model_size="base", hotkey="insert", language="en",
                 use_paste=False, beep=True, device=None, force_cpu=False):
        self.hotkey = hotkey
        self.language = language
        self.use_paste = use_paste
        self.beep = beep
        self.device = device
        self.sample_rate = 16000
        self.channels = 1
        self.recording = False
        self.audio_chunks = []
        self.lock = threading.Lock()
        self.transcribing = False

        print(f"Loading Whisper model '{model_size}'... ", end="", flush=True)
        from faster_whisper import WhisperModel

        if force_cpu:
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        else:
            try:
                self.model = WhisperModel(model_size, device="auto", compute_type="auto")
            except RuntimeError:
                print("GPU unavailable, falling back to CPU... ", end="", flush=True)
                self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("done.")

        # Warm up the model with a short silence
        silence = np.zeros(self.sample_rate, dtype=np.float32)
        segments, _ = self.model.transcribe(silence, language=self.language)
        list(segments)  # consume generator
        print("Model warmed up and ready.\n")

    def start_recording(self):
        """Called when hotkey is pressed."""
        with self.lock:
            if self.recording or self.transcribing:
                return
            self.recording = True
            self.audio_chunks = []

        if self.beep:
            play_beep(BEEP_START)

        def audio_callback(indata, frames, time_info, status):
            if self.recording:
                self.audio_chunks.append(indata.copy())

        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=1024,
                device=self.device,
                callback=audio_callback,
            )
            self.stream.start()
        except Exception as e:
            print(f"  [!] Audio error: {e}")
            self.recording = False

    def stop_recording(self):
        """Called when hotkey is released."""
        with self.lock:
            if not self.recording:
                return
            self.recording = False
            self.transcribing = True

        if self.beep:
            play_beep(BEEP_STOP)

        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass

        chunks = self.audio_chunks
        self.audio_chunks = []

        # Transcribe in background thread to not block hotkey listener
        threading.Thread(target=self._transcribe, args=(chunks,), daemon=True).start()

    def _transcribe(self, chunks):
        """Transcribe recorded audio and type the result."""
        try:
            if not chunks:
                return

            audio = np.concatenate(chunks, axis=0).flatten()
            duration = len(audio) / self.sample_rate

            if duration < 0.3:
                # Too short, probably accidental
                return

            print(f"  Transcribing {duration:.1f}s of audio... ", end="", flush=True)

            segments, info = self.model.transcribe(
                audio,
                language=self.language,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=200,
                ),
            )

            text = " ".join(seg.text.strip() for seg in segments).strip()

            if not text:
                print("(no speech detected)")
                return

            print(f'"{text}"')
            self._type_text(text)

        except Exception as e:
            print(f"\n  [!] Transcription error: {e}")
            if self.beep:
                play_beep(BEEP_ERROR)
        finally:
            self.transcribing = False

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

    def run(self):
        """Main loop."""
        print("=" * 55)
        print(f"  Push-to-Talk active!")
        print(f"  Hold [{self.hotkey.upper()}] to speak, release to transcribe")
        print(f"  Input method: {'clipboard paste' if self.use_paste else 'keyboard typing'}")
        print(f"  Press Ctrl+C or ESC to quit")
        print("=" * 55)
        print()

        # Register hotkey handlers
        keyboard.on_press_key(self.hotkey, lambda _: self.start_recording(), suppress=True)
        keyboard.on_release_key(self.hotkey, lambda _: self.stop_recording(), suppress=True)

        try:
            keyboard.wait("esc")
        except KeyboardInterrupt:
            pass
        finally:
            print("\nShutting down.")
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

    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    ptt = PushToTalk(
        model_size=args.model,
        hotkey=args.key,
        language=args.language,
        use_paste=args.paste,
        beep=not args.no_beep,
        device=args.device,
        force_cpu=args.cpu,
    )
    ptt.run()


if __name__ == "__main__":
    main()
