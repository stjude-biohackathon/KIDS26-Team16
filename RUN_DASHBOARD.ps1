$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "SCOGS-Scribe Qwen Dashboard" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

if (-not $env:PCAI_API_KEY) {
    Write-Host ""
    Write-Host "PCAI_API_KEY is not set in this PowerShell session." -ForegroundColor Yellow
    Write-Host 'Set it first with: $env:PCAI_API_KEY="YOUR_VIRTUAL_KEY"' -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

python -m pip install -r requirements_pcai_dashboard.txt
python -m streamlit run pcai_dashboard.py
