@echo off
title Spider-Man AR Studio
echo Launching Spider-Man AR Studio...

if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe main.py %*
) else (
    python main.py %*
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with an error code: %ERRORLEVEL%
    pause
)
