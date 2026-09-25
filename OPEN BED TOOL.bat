@echo off
rem Opens the bed colour tool already pointed at the Mayfair Sleep folders.
rem Paths are relative to this file, so the drive letter can change freely.
cd /d "%~dp0"
python bedtool.py --beds "..\Beds Stock Images" --swatches "..\Colours" --masks "..\_AUTO MASKS\regions"
if errorlevel 1 (
  echo.
  echo Something went wrong. If it says 'python' is not recognised,
  echo install Python from python.org and tick "Add python.exe to PATH".
  pause
)
