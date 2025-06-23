@echo off
REM Copyright (c) 2025 ama-prof-divi Project. All Rights Reserved.
REM
REM Author: carl <carlwang1212@gmail.com>
REM


REM Get the directory containing the script file
SET "SCRIPT_DIRECTORY=%~dp0"

python -m unittest discover -s "%SCRIPT_DIRECTORY%scripts" -p "test_*.py" %*
