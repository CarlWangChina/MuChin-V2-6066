@echo off
REM Copyright (c) 2025 ama-prof-divi Project. All Rights Reserved.
REM
REM Author: carl <carlwang1212@gmail.com>
REM


REM Get the directory containing the script file
SET "SCRIPT_DIRECTORY=%~dp0"

REM Check if a parameter is passed
IF "%~1"=="" (
    echo Usage: %0 ^<test_python_file^> [unittest_args]
    exit /b 1
)

python -m unittest discover -s "%SCRIPT_DIRECTORY%scripts" -p "%~1" %2 %3 %4 %5 %6 %7 %8 %9
