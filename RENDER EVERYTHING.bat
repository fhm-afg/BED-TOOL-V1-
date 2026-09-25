@echo off
rem Renders every masked bed in every colour, no window. Results land in
rem _BED TOOL\output. Beds without a usable mask are listed and skipped.
cd /d "%~dp0"
python bedtool.py --batch --beds "..\Beds Stock Images" --swatches "..\Colours" --masks "..\_AUTO MASKS\regions" --out "output"
echo.
pause
