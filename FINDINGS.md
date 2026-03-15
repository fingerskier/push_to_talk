# Performance Review Findings

Review of `main.py` (~448 lines) for performance issues in the push-to-talk speech-to-text application.

---

## P0 - Critical Performance Issues

### ~~1. `keyboard.write()` character-by-character typing is extremely slow~~ ✅ RESOLVED

Paste mode is now the default (`--type` flag opts into keyboard typing). The `keyboard.write()` delay was reduced from `0.01` to `0.002`. CLI flag `--paste` was replaced by `--type` (opt-in slower fallback).

---

### ~~2. Unbounded `audio_chunks` list growth during recording~~ ✅ RESOLVED

`audio_chunks` now uses `collections.deque` instead of a plain list. `deque` provides O(1) append without reallocation-induced GC pauses.

---

### ~~3. `np.concatenate(..., axis=None)` forces a flatten + copy~~ ✅ RESOLVED

Chunks are now stored as 1D mono arrays (`indata[:, 0].copy()`) in the callback, and `np.concatenate(chunks)` is called without `axis=None`. No intermediate flatten operations.

---

## P1 - Moderate Performance Issues

### ~~4. Audio callback copies full frame even when not recording~~ ✅ RESOLVED

The callback now stores `indata[:, 0].copy()` (1D mono slice) instead of `indata.copy()` (2D), halving per-callback allocation.

### ~~5. Global hook teardown on refresh~~ ✅ RESOLVED

`_register_hooks()` now tracks individual hook handles and removes only those via `keyboard.unhook(handle)` instead of calling `keyboard.unhook_all()`. Hook refresh is also skipped when recording is in progress.

### ~~6. Clipboard save/restore race condition adds latency~~ ✅ RESOLVED

Both hardcoded sleeps were removed. Clipboard restore is now deferred to a 500ms `threading.Timer` so the target application has time to read the clipboard before restoration, eliminating the paste race condition.

### ~~7. Transcription queue has no backpressure~~ ✅ RESOLVED

Queue now uses `maxsize=3`. When full, recordings are dropped with a user-visible warning and log entry.

---

## P2 - Minor / Optimization Opportunities

### ~~8. Synchronous model warm-up~~ ✅ RESOLVED

Warm-up now runs in a background daemon thread. A `threading.Event` (`_warmup_done`) gates the first real transcription until warm-up completes.

### ~~9. `keyboard.write()` delay parameter is per-character, not per-word~~ ✅ RESOLVED

Delay reduced from `0.01` (10ms) to `0.002` (2ms) per character. Paste mode is now the default, making this a fallback concern only.

### ~~10. `generate_beep()` is called at module import time~~ ✅ RESOLVED

Beep arrays are now generated lazily inside `PushToTalk.__init__`, gated on the `beep` flag. No work is done if `--no-beep` is passed.

### ~~11. Unused `deque` import~~ ✅ RESOLVED

`deque` is now actively used for `audio_chunks` (see finding #2).

---

# New Suggestions

### Reliability issues (the "cuts out" stuff):

### ~~1. No audio device reconnection~~ ✅ RESOLVED

Audio stream health is now monitored every 10 seconds in the main loop. If the stream reports errors via the `status` callback parameter or becomes inactive (e.g., device disconnect), the app automatically attempts to restart the stream when not recording.

### ~~2. Hook refresh gap~~ ✅ RESOLVED

Individual hook tracking (finding #5 fix) minimizes the re-registration window. Hook refresh is skipped when recording is active, preventing stuck-state from missed events.

### ~~3. No health monitoring~~ ✅ RESOLVED

The main loop now checks `_audio_stream_healthy` and `_audio_stream.active` every 10 seconds (`HEALTH_CHECK_INTERVAL`). Unhealthy streams are automatically restarted with logging.

### System integration (the "no launcher" stuff):

### 4. No system tray — 📋 TBD

Needs design decisions: which toolkit (pystray, Qt, GTK), what state indicators to show, tray menu structure. Requires adding a new dependency and significant UI code.

### 5. No auto-start — 📋 TBD

Needs platform-specific implementation: `.desktop` file for Linux, Task Scheduler for Windows, LaunchAgent for macOS. Should be gated behind an `--install` / `--uninstall` flag.

### 6. No config file — 📋 TBD

Should persist settings to `~/.push_to_talk/config.json`. Needs decisions on: config schema, CLI args vs config file precedence, migration strategy for existing users.

---

## Summary

| # | Finding | Severity | Status |
|---|---------|----------|--------|
| 1 | Slow character-by-character typing | P0 | ✅ Resolved |
| 2 | Unbounded list growth in audio callback | P0 | ✅ Resolved |
| 3 | Unnecessary flatten in np.concatenate | P0 | ✅ Resolved |
| 4 | Full 2D array copy in audio callback | P1 | ✅ Resolved |
| 5 | Global hook teardown on refresh | P1 | ✅ Resolved |
| 6 | Hardcoded sleeps in paste path | P1 | ✅ Resolved |
| 7 | Unbounded transcription queue | P1 | ✅ Resolved |
| 8 | Synchronous model warm-up | P2 | ✅ Resolved |
| 9 | Conservative keyboard write delay | P2 | ✅ Resolved |
| 10 | Eager beep generation | P2 | ✅ Resolved |
| 11 | Unused deque import | P2 | ✅ Resolved |
| N1 | No audio device reconnection | P1 | ✅ Resolved |
| N2 | Hook refresh gap | P1 | ✅ Resolved |
| N3 | No health monitoring | P1 | ✅ Resolved |
| N4 | No system tray | P2 | 📋 TBD |
| N5 | No auto-start | P2 | 📋 TBD |
| N6 | No config file | P2 | 📋 TBD |
