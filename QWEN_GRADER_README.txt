PCAI_2 QWEN 4K SEQUENTIAL

Default grading configuration:
  Model: Qwen/Qwen3.8-27B-FP8
  PCAI_MAX_TOKENS: 4096
  Outcome concurrency: 1 (hard-coded; exactly one outcome at a time)
  Qwen thinking: disabled with /no_think and enable_thinking=False when supported
  Attempts per outcome: 1 by default

Run in PowerShell:
  $env:PCAI_API_KEY="YOUR_KEY"
  $env:PCAI_MAX_TOKENS="4096"
  python -m shiny run dashboard/interactive_dashboard.py

To test 8K output budget:
  $env:PCAI_MAX_TOKENS="8192"
  python -m shiny run dashboard/interactive_dashboard.py

Note: PCAI/Bifrost may enforce its own provider-side request timeout. Raising max_tokens gives the model more output room; it does not increase that server-side timeout.
