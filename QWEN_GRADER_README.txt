PCAI_2 QWEN GRADER BUILD

Default live grading model:
  Qwen/Qwen3.8-27B-FP8

The GPT-OSS model remains defined as an optional comparison model, but the live
Shiny dashboard and standalone grader now default to Qwen.

Default completion cap:
  PCAI_MAX_TOKENS=4096

To raise it in PowerShell before launching:
  $env:PCAI_MAX_TOKENS="8192"

Launch:
  cd <path-to-PCAI>
  python -m shiny run dashboard/interactive_dashboard.py

The app still grades one outcome at a time for gateway stability.
