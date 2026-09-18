$env:PCAI_MAX_TOKENS="4096"
$env:PCAI_ATTEMPTS="1"
Remove-Item Env:PCAI_CONCURRENCY -ErrorAction SilentlyContinue
python -m shiny run dashboard/interactive_dashboard.py
