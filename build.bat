@echo off
echo Building push-to-talk.exe ...

pyinstaller --onefile --console ^
    --name push-to-talk ^
    --hidden-import=faster_whisper ^
    --hidden-import=onnxruntime ^
    --collect-all=ctranslate2 ^
    --collect-data=faster_whisper ^
    --uac-admin ^
    main.py

echo.
if exist dist\push-to-talk.exe (
    echo Build succeeded: dist\push-to-talk.exe
) else (
    echo Build FAILED. Check output above for errors.
)
pause
