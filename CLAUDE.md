# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application (requires admin/root for global hotkey capture)
python main.py [options]

# Build standalone executable (Windows)
build.bat
# Output: dist\push-to-talk.exe

# Build standalone executable (Linux)
./build.sh
# Output: dist/push-to-talk
```

There are no tests in this project.

## Architecture

Single-file Python application (`main.py`, ~355 lines) with one main class:

**`PushToTalk`** — Core engine that:
- Loads a `faster-whisper` model on init (GPU with CUDA fallback to CPU int8)
- Registers global hotkey press/release handlers via `keyboard` library
- Records audio with `sounddevice` (16kHz, mono, float32) while hotkey is held
- Spawns daemon threads for transcription on hotkey release (non-blocking)
- Outputs text via `keyboard.write()` or clipboard paste (`pyperclip` + Ctrl+V)

**Supporting systems:**
- Audio feedback: sine wave beeps at 600Hz (start), 800Hz (stop), 300Hz (error)
- Logging: rotating file handler at `~/.push_to_talk/push_to_talk.log` (1MB, 5 backups)
- CLI: argparse with model size, hotkey, language, paste mode, device selection, CPU override

**Threading model:** Recording state protected by `threading.Lock()`. Transcription runs in daemon threads so the hotkey listener never blocks.

## CI/CD

- `.github/workflows/build.yml` — Builds on push to main (Windows + Linux matrix, Python 3.11)
- `.github/workflows/release.yml` — Creates GitHub Release with binaries on `v*` tags

Release workflow: `git tag v1.2.3 && git push origin --tags`

## Key Dependencies

| Package | Purpose |
|---------|---------|
| faster-whisper | Speech-to-text (optimized Whisper) |
| sounddevice | Audio capture and playback |
| numpy | Audio buffer manipulation |
| keyboard | Global hotkey capture (needs admin) |
| pyperclip | Clipboard for paste mode |
| PyInstaller | Build-time only, creates standalone binaries |
