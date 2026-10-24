@echo off
chcp 65001 >nul
title Orakul Studio - AI-Toolkit Updater

echo =======================================================
echo          Orakul Viking Engine - Auto Updater
echo =======================================================
echo.

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo [!] Warning: venv folder not found! Make sure the toolkit is properly installed.
)

python updater.py

echo.
pause