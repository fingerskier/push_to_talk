# Remaining Findings

Items that need further design and planning before implementation.

---

## System Integration

### 1. No system tray — 📋 TBD

Needs design decisions: which toolkit (pystray, Qt, GTK), what state indicators to show, tray menu structure. Requires adding a new dependency and significant UI code.

### 2. No auto-start — 📋 TBD

Needs platform-specific implementation: `.desktop` file for Linux, Task Scheduler for Windows, LaunchAgent for macOS. Should be gated behind an `--install` / `--uninstall` flag.

### 3. No config file — 📋 TBD

Should persist settings to `~/.push_to_talk/config.json`. Needs decisions on: config schema, CLI args vs config file precedence, migration strategy for existing users.
