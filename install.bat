@echo off
echo Installing push-to-talk dependencies...
pip install faster-whisper sounddevice numpy keyboard pyperclip
echo.
echo Done! Run with:  python push_to_talk.py
echo.
echo Recommended: Run your terminal as Administrator for global hotkey capture.
pause
