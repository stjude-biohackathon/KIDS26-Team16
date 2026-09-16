# PowerShell script to run MedGemma 27B feature extraction on NVIDIA Windows
# Usage:
#   .\scripts\run_windows_27b.ps1 -Backend ollama -Notes 20
#   .\scripts\run_windows_27b.ps1 -Backend hf -Quant 4bit -Notes 20

param (
    [string]$Backend = "ollama",
    [string]$Quant = "none",
    [int]$Notes = 20,
    [int]$Repeat = 2,
    [string]$Out = "results\full.json",
    [string]$HostUrl = "http://localhost:11434"
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Running MedGemma 27B Feature Extraction (Tier: FULL) " -ForegroundColor Cyan
Write-Host " Backend: $Backend | Quant: $Quant | Notes: $Notes | Repeat: $Repeat" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$scriptPath = Join-Path $PSScriptRoot "experiments\medgemma_extraction.py"

if ($Backend -eq "hf") {
    python $scriptPath --tier full --backend hf --quant $Quant --notes $Notes --repeat $Repeat --out $Out
} elseif ($Backend -eq "ollama") {
    python $scriptPath --tier full --backend ollama --host $HostUrl --notes $Notes --repeat $Repeat --out $Out
} else {
    python $scriptPath --tier full --backend $Backend --host $HostUrl --notes $Notes --repeat $Repeat --out $Out
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[SUCCESS] MedGemma 27B extraction completed. Results saved to $Out" -ForegroundColor Green
} else {
    Write-Host "`n[ERROR] Extraction script exited with error code $LASTEXITCODE" -ForegroundColor Red
}
