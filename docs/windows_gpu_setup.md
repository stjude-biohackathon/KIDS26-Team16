# Running MedGemma 27B on NVIDIA Windows

This document details how to set up and run the `full` reporting tier (**MedGemma 27B**) on a Windows machine equipped with an NVIDIA GPU (CUDA).

---

## Hardware Requirements & Matrix

| Hardware Setup | Quantization / Mode | VRAM & System RAM Footprint | Notes |
| :--- | :--- | :--- | :--- |
| **RTX 4080 (16 GB VRAM) + 32 GB DDR5 RAM** | **4-bit Quantized** (Ollama GGUF Q4 or `bitsandbytes` 4-bit) | Model weights consume ~14.2 GB. Ollama/HF loads ~95–100% of layers directly in the 16 GB GDDR6X VRAM; the **32 GB DDR5 RAM** provides safety buffer for KV cache & Windows DWM overhead without OOM. | **Recommended target setup.** Fast tokens/sec (~15–25 t/s). |
| **RTX 3090 / RTX 4090 (24 GB VRAM)** | **4-bit or 8-bit Quantized** | Fits 100% inside 24 GB VRAM with substantial headroom. | Runs with zero CPU offload needed. |
| **Multi-GPU / Server ($\ge 56\text{ GB}$ VRAM: 2x 3090/4090, A6000, A100)** | **Full precision / bfloat16** | ~54 GB VRAM required. | Full unquantized inference. |

---

## Option 1: Running via Ollama on Windows (Recommended)

1. **Install Ollama for Windows:**
   - Download the Windows installer from [ollama.com/download/windows](https://ollama.com/download/windows).
   - Ensure NVIDIA GPU drivers and CUDA are detected automatically by Ollama.

2. **Pull / Import MedGemma 27B:**
   ```powershell
   ollama pull medgemma-27b-text-it
   # Or if importing from Hugging Face / GGUF:
   # ollama run google/medgemma-27b-text-it
   ```

3. **Run the Extraction Experiment:**
   ```powershell
   # 20 notes, repeated twice for temperature-0 consistency:
   python scripts\experiments\medgemma_extraction.py --tier full --backend ollama --notes 20 --repeat 2 --out results\full.json
   ```
   Or using the helper script:
   ```powershell
   .\scripts\run_windows_27b.ps1 -Backend ollama -Notes 20 -Repeat 2
   ```

---

## Option 2: Running via Hugging Face / PyTorch with CUDA

If you prefer running directly in Python without Ollama:

1. **Install PyTorch with CUDA 12.1+ support:**
   ```powershell
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

2. **Install Transformers & Quantization packages:**
   ```powershell
   pip install transformers accelerate bitsandbytes sentencepiece
   ```

3. **Run MedGemma 27B with 4-bit quantization (Fits on 24GB RTX 3090/4090):**
   ```powershell
   python scripts\experiments\medgemma_extraction.py --tier full --backend hf --quant 4bit --notes 20 --repeat 2 --out results\full.json
   ```
   Or for unquantized / multi-GPU:
   ```powershell
   python scripts\experiments\medgemma_extraction.py --tier full --backend hf --quant none --notes 20 --repeat 2 --out results\full.json
   ```

---

## Output & Verification

The script will:
1. Verify verbatim quotes from the clinical note.
2. Evaluate features against the SCOGS decision tables.
3. Measure temperature-0 consistency across repeated runs.
4. Output wall-clock profiling and tokens-per-second metrics.
5. Save full test results and clinical text extracts to `results/full.json`.
