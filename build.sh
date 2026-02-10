#!/usr/bin/env bash
set -e

echo "Building push-to-talk binary..."

pyinstaller --onefile --console \
    --name push-to-talk \
    --hidden-import=faster_whisper \
    --hidden-import=onnxruntime \
    --collect-all=ctranslate2 \
    --collect-data=faster_whisper \
    main.py

echo ""
if [ -f dist/push-to-talk ]; then
    echo "Build succeeded: dist/push-to-talk"
    echo "Note: Run with sudo or add user to 'input' group for keyboard hooks."
    echo "Note: Ensure libportaudio2 is installed (e.g. sudo apt install libportaudio2)."
else
    echo "Build FAILED. Check output above for errors."
    exit 1
fi
