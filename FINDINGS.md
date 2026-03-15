# Performance Review Findings

Review of `main.py` (~448 lines) for performance issues in the push-to-talk speech-to-text application.

---

## P0 - Critical Performance Issues

### 1. `keyboard.write()` character-by-character typing is extremely slow (line 312)

```python
keyboard.write(text, delay=0.01)
```

The default (non-paste) text output mode types each character with a 10ms delay. A 50-word transcription (~250 characters) takes **~2.5 seconds** just to type out. For a tool designed around fast voice-to-text, this is the dominant latency source after transcription itself.

**Recommendation:** Default to paste mode (`--paste`) or reduce the delay to 0. Consider making paste mode the default and keyboard typing the opt-in fallback.

---

### 2. Unbounded `audio_chunks` list growth during recording (lines 195, 203)

```python
self.audio_chunks = []
# ...
self.audio_chunks.append(indata.copy())
```

`audio_chunks` is a plain Python list that grows with `.append()` inside the audio callback (called from a real-time audio thread). Python lists use amortized O(1) append, but the periodic reallocation triggers GC pressure and can cause **audio glitches/dropouts** during longer recordings due to the GIL being held during list resizing.

**Recommendation:** Use a `collections.deque` (already imported but unused) or a pre-allocated `numpy` ring buffer. The `deque` import at line 30 suggests this was considered but never implemented.

---

### 3. `np.concatenate(..., axis=None)` forces a flatten + copy (line 262)

```python
audio = np.concatenate(chunks, axis=None)
```

Passing `axis=None` flattens every chunk before concatenating. Since each chunk is already shape `(1024, 1)`, this forces N intermediate flatten operations. For a 30-second recording (~470 chunks), this means ~470 unnecessary reshape operations plus the final concatenation copy.

**Recommendation:** Use `np.concatenate(chunks, axis=0).ravel()` or simply `np.concatenate(chunks).flatten()` which does a single concatenation followed by a single reshape. Even better, since `indata` shape is `(1024, 1)`, store `indata[:, 0].copy()` in the callback to avoid the flatten entirely:

```python
def _audio_callback(self, indata, frames, time_info, status):
    if self.recording:
        self.audio_chunks.append(indata[:, 0].copy())

# Then later:
audio = np.concatenate(chunks)
```

---

## P1 - Moderate Performance Issues

### 4. Audio callback copies full frame even when not recording (line 195)

```python
def _audio_callback(self, indata, frames, time_info, status):
    if self.recording:
        self.audio_chunks.append(indata.copy())
```

The callback itself is lightweight when not recording (just a boolean check), which is correct. However, when recording, `indata.copy()` copies the full 2D array `(1024, 1)`. Since only mono channel 0 is used, copying just the needed slice (`indata[:, 0].copy()`) would halve the per-callback allocation and eliminate the later flatten step (see finding #3).

### 5. Hook re-registration calls `keyboard.unhook_all()` globally (line 316)

```python
def _register_hooks(self):
    keyboard.unhook_all()
    keyboard.on_press_key(...)
    keyboard.on_release_key(...)
```

Every 300 seconds, `_register_hooks()` tears down **all** keyboard hooks and re-registers them. During the brief window between `unhook_all()` and the new `on_press_key()`, hotkey events are silently dropped. If the user presses the key during this gap, the press or release event is lost, which can leave the app in a stuck recording state.

**Recommendation:** Track hook handles and only remove/replace them individually, or use a flag to gate recording rather than relying on hook registration as a mutex.

### 6. Clipboard save/restore race condition adds latency (lines 298-309)

```python
old_clip = pyperclip.paste()
pyperclip.copy(text)
time.sleep(0.05)
keyboard.send("ctrl+v")
time.sleep(0.1)
pyperclip.copy(old_clip)
```

The paste path has **150ms of hardcoded sleep** (`0.05 + 0.1`). Additionally, `pyperclip.paste()` and `pyperclip.copy()` are synchronous subprocess calls on Linux (invoking `xclip` or `xsel`), adding further latency. The clipboard restore can also race with the actual paste operation if the target application is slow to read the clipboard.

**Recommendation:** Reduce or eliminate the sleeps, especially the 50ms pre-paste delay which is unnecessary on most systems. Consider a platform-specific clipboard API instead of pyperclip's subprocess-based approach. The restore could be deferred to a short timer to avoid the race.

### 7. Transcription queue has no backpressure (line 134, 236)

```python
self._transcription_queue = queue.Queue()
# ...
self._transcription_queue.put(chunks)
```

The queue is unbounded. If the user records multiple clips faster than they can be transcribed (e.g., with large models), audio data accumulates in memory. Each 30-second recording at 16kHz float32 is ~1.9MB, but slower models can take longer than 30s to transcribe, so clips will queue up.

**Recommendation:** Use `queue.Queue(maxsize=2)` or `queue.Queue(maxsize=3)` and handle the full case (e.g., drop oldest, warn user, or block recording until the queue drains).

---

## P2 - Minor / Optimization Opportunities

### 8. Model warm-up blocks startup unnecessarily (lines 168-171)

```python
silence = np.zeros(self.sample_rate, dtype=np.float32)
segments, _ = self.model.transcribe(silence, language=self.language)
list(segments)
```

The warm-up transcription runs synchronously during `__init__`, adding 1-3 seconds to startup time. While warm-up is valuable for reducing first-transcription latency, it could run in a background thread so the app is responsive sooner.

**Recommendation:** Move warm-up to a background thread and gate the first real transcription on its completion (e.g., with a `threading.Event`).

### 9. `keyboard.write()` delay parameter is per-character, not per-word (line 312)

```python
keyboard.write(text, delay=0.01)
```

The `delay=0.01` (10ms) is applied between every character. This is unnecessarily conservative for modern applications. Most text input fields can handle much faster injection.

**Recommendation:** If paste mode is not used, reduce delay to `0.002` or `0` and let the keyboard library handle its own timing. Test on target platforms to find the minimum reliable delay.

### 10. `generate_beep()` is called at module import time (lines 51-53)

```python
BEEP_START = generate_beep(freq=600, duration=0.06)
BEEP_STOP = generate_beep(freq=800, duration=0.06)
BEEP_ERROR = generate_beep(freq=300, duration=0.15)
```

Three numpy array allocations happen at import time even if `--no-beep` is passed. This is a trivial cost (~negligible memory) but violates the principle of lazy initialization.

**Recommendation:** Low priority. Could wrap in a lazy initializer or move into `PushToTalk.__init__` gated on the `beep` flag, but the practical impact is minimal.

### 11. Unused `deque` import (line 30)

```python
from collections import deque
```

`deque` is imported but never used. This was likely intended for `audio_chunks` (see finding #2) but was never implemented.

**Recommendation:** Either use `deque` for `audio_chunks` (recommended, see #2) or remove the import.

---

# New Suggestions 

Reliability issues (the "cuts out" stuff):
No audio device reconnection — if your mic disconnects/reconnects (Bluetooth, USB), the InputStream dies silently. No recovery path.
Hook refresh gap — the 300s re-registration still has a brief window where events can be missed. On Windows especially, keyboard hooks can get deregistered by the OS if your thread stalls (e.g., during a heavy transcription).
No health monitoring — if the audio stream errors out, nothing restarts it.
System integration (the "no launcher" stuff):
4. No system tray — no visual indicator of state (idle/recording/transcribing), no quick access to settings.
5. No auto-start — no .desktop file, no Windows Task Scheduler entry, no --install flag.
6. No config file — every setting requires CLI args. Should persist to ~/.push_to_talk/config.json

---

## Summary

| # | Finding | Severity | Estimated Impact |
|---|---------|----------|-----------------|
| 1 | Slow character-by-character typing | P0 | ~2.5s added latency per transcription |
| 2 | Unbounded list growth in audio callback | P0 | Audio glitches during long recordings |
| 3 | Unnecessary flatten in np.concatenate | P0 | ~470 extra reshape ops for 30s recording |
| 4 | Full 2D array copy in audio callback | P1 | 2x per-callback allocation overhead |
| 5 | Global hook teardown on refresh | P1 | Missed hotkey events, stuck state risk |
| 6 | Hardcoded sleeps in paste path | P1 | 150ms unnecessary latency |
| 7 | Unbounded transcription queue | P1 | Memory growth under load |
| 8 | Synchronous model warm-up | P2 | 1-3s slower startup |
| 9 | Conservative keyboard write delay | P2 | Higher latency in non-paste mode |
| 10 | Eager beep generation | P2 | Trivial import-time cost |
| 11 | Unused deque import | P2 | Dead code |
