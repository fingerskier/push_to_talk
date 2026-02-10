# push_to_talk

Run as an administrator.

Select where you want to _"type"_, Hold "Ins" key, speak, release key and it will input the text.


## Issues

If it crashes intially, then run with the `--cpu` flag (which I have to do on my Windows PC w/ RTX 3060.)

`tiny` model actually works okay...and fst;  for day-to-day use I prefer the `small` model- there's a second of lag but surprisingly good accuracy.


## Options

```
push_to_talk.exe [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `base` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `--key` | `insert` | Push-to-talk key |
| `--language` | `en` | Language code |
| `--paste` | off | Use clipboard paste instead of keyboard typing |
| `--no-beep` | off | Disable audio feedback beeps |
| `--device` | auto | Audio input device index (use `--list-devices` to see options) |
| `--list-devices` | — | List audio input devices and exit |
| `--cpu` | off | Force CPU mode (skip CUDA/GPU) |

**Example:**
```
push_to_talk.exe --model small --key "scroll lock" --paste
```

### Keys

Common key names for `--key`:

`insert` (default), `scroll lock`, `pause`, `caps lock`, `num lock`,
`f1`–`f12`,
`right ctrl`, `right alt`, `right shift`,
`home`, `end`, `page up`, `page down`


## Administrative

To trigger the build:

```
git tag v1.2.3
git push origin --tags
```

...the GitHub action will trigger, build and release.