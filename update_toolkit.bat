@echo off
chcp 65001 >nul
title Orakul Studio - AI-Toolkit Updater

echo =======================================================
echo          Orakul Viking Engine - Auto Updater
echo =======================================================
echo.

if exist "venv_312\Scripts\activate.bat" (
    call venv_312\Scripts\activate.bat
) else (
    echo [!] Warning: venv_312 folder not found! Make sure the toolkit is properly installed.
)

python updater.py

echo.
pause