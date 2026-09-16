@echo off
chcp 65001 > nul
setlocal

echo ==========================================================
echo  Running MedGemma 27B Feature Extraction (Tier: FULL)
echo ==========================================================

REM Default to Ollama with 20 notes, repeat 2
if "%~1"=="" (
    python scripts\experiments\medgemma_extraction.py --tier full --backend ollama --notes 20 --repeat 2 --out results\full.json
) else (
    python scripts\experiments\medgemma_extraction.py --tier full %*
)

if %ERRORLEVEL% equ 0 (
    echo.
    echo [SUCCESS] MedGemma 27B extraction completed. Results in results\full.json
) else (
    echo.
    echo [ERROR] Script failed with error level %ERRORLEVEL%
)

endlocal
