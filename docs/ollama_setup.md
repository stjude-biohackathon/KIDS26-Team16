# Full MedGemma 27B with Ollama

This project supports **`google/medgemma-27b-text-it` in unquantized F16 or BF16**
through Ollama. A model's parameter count and its weight precision are different
properties: both are checked before inference. There is no automatic fallback
to a smaller or quantized model.

## 1. Plan memory and disk space

27 billion parameters at two bytes per parameter require approximately **54 GB
of weights**, before runtime buffers and the KV cache. An **80 GB GPU** is a
practical starting point for sequential 16k-context inference; actual fit depends
on Ollama, model conversion, context, and workload.

A 16/24 GB GPU cannot hold these weights by itself. Ollama may offload to system
RAM, but that does not make the model smaller and can be much slower. Allow at
least roughly 64 GB of system RAM for CPU/offloaded use, preferably more; this is
not a fit guarantee. Multiple GPUs also need enough aggregate memory and runtime
headroom. Check actual placement with `ollama ps` rather than assuming GPU-only
execution.

Budget over **110 GB of free disk** if keeping both a GGUF source and Ollama's
imported copy. Conversion from original weights and temporary files may need
substantially more. Model files belong outside the Git repository.

## 2. Install and start Ollama

### Windows

Install [Ollama for Windows](https://ollama.com/download/windows) and a current
driver supported by Ollama for your GPU. Launch the Ollama application, then
open a fresh PowerShell window:

```powershell
ollama --version
ollama list
```

The application normally starts the local service. If no service is running,
use `ollama serve` in a separate terminal. Do not start a second server on the
same port.

### Linux

Follow the [official Linux installation instructions](https://docs.ollama.com/linux).
Check the service on systemd-based installations:

```bash
systemctl status ollama --no-pager
ollama --version
ollama list
```

If your installation has no service manager, run `ollama serve` in a separate
terminal. Do not launch a second copy when the service already runs.

Python never loads CUDA or the weights; it talks to the Ollama HTTP API.
The same notebook code therefore works on either operating system.

## 3. Obtain and import verified full-precision weights

Use a trusted **F16/BF16 GGUF conversion of the exact MedGemma text model**,
or convert the original [Google weights](https://huggingface.co/google/medgemma-27b-text-it)
using a compatible [llama.cpp converter](https://github.com/ggml-org/llama.cpp).
Access to Google's weights may require accepting the Health AI Developer
Foundations terms and authenticating with Hugging Face. Never put tokens in
notebooks or commit them.

Record the original model revision, conversion tool/version, and file checksum.
A filename or a `27b` tag does not prove model identity. Metadata confirms
architecture, size, and precision, **not** that a file contains the intended
medical fine-tuning or was never quantized and re-expanded. Start from original
full-precision weights or a verifiable conversion.

Create a plain-text file named `Modelfile` beside the GGUF, using its actual
filename:

```dockerfile
FROM ./medgemma-27b-text-it-F16.gguf
PARAMETER num_ctx 16384
```

From that directory, run these commands on Windows or Linux:

```bash
ollama create medgemma-27b-f16 -f Modelfile
ollama show medgemma-27b-f16
```

**Do not add `--quantize`.** Ollama imports a GGUF at its existing precision.
For BF16, point `FROM` at your BF16 file and create the local alias
`medgemma-27b-bf16` instead. Use the relative filename to avoid Windows drive/path
quoting problems; otherwise follow Ollama's path syntax for your installation.

For split GGUF weights, current Ollama versions support a `FROM` wildcard that
matches all shards, with their original split filenames intact. Follow
[Ollama's import guide](https://docs.ollama.com/import); upgrade if your version
does not support that format. Do not concatenate binary shards yourself.

These aliases are **created locally**, not guaranteed public registry tags.
There is intentionally no unverified `ollama pull medgemma-27b-text-it` shortcut.
Use the model's correct Gemma chat template; check `ollama show --modelfile`
if a custom conversion fails to follow instructions.

## 4. Verify from the repository

Use the Python environment installed in the [README](../README.md). On Windows,
replace `python` below with `.\.venv\Scripts\python.exe` if not activated.

```bash
python scripts/experiments/medgemma_extraction.py --check-model --timeout 600
```

For the BF16 alias:

```bash
python scripts/experiments/medgemma_extraction.py --model medgemma-27b-bf16 --check-model --timeout 600
```

The gate requires Gemma 3 architecture, an actual parameter count in the rounded
27B range (26-29 billion), consistent F16/BF16 precision metadata, a model digest,
and a successful synthetic JSON generation without tokenizer artifacts.
Unsupported or missing metadata stops the run before patient notes are sent.

Then run two cases before a larger experiment:

```bash
python scripts/experiments/medgemma_extraction.py --cohort scd_primary --prompt-stage 3 --notes 2 --timeout 600 --out results/smoke.json
```

Every real extraction repeats the preflight. The gate detects configuration and
basic generation failures; it is not a test of clinical accuracy.

## 5. Runtime settings and troubleshooting

| Symptom | Action |
| --- | --- |
| Cannot reach Ollama | Start the app/service; check `ollama list` and `--host http://localhost:11434` |
| Model not found | Import it first; pass the exact alias shown in `ollama list` |
| Precision/size rejected | Reimport verified original F16/BF16 27B weights; renaming a tag will not help |
| Missing metadata | Upgrade Ollama/check the conversion; do not disable the gate |
| Out of memory | Reduce concurrent requests/context where appropriate, or use more memory; do not change precision |
| Timeout while loading/offloading | Increase `--timeout`; check Ollama logs and available RAM/VRAM |
| Incomplete output or unusable JSON | Inspect `raw_reply`, model template, and completion budget; raise `--num-predict` if needed |
| Long note loses evidence | Check that prompt plus completion fits `--num-ctx`; increasing context also increases memory use |
| Output already exists | Use a new result path; existing runs and human reviews are intentionally protected |

The client explicitly sends `num_ctx` (default 16384) and keeps the model loaded
for ten minutes after requests, rather than pinning it indefinitely. To free
memory when finished:

```bash
ollama stop medgemma-27b-f16
```

For reproducible sequential runs, keep client concurrency at one and configure
the Ollama service's `OLLAMA_NUM_PARALLEL=1` before starting/restarting it.
On Windows, quit the tray application before changing its environment and
relaunch it. On Linux systemd installations, use a service environment override;
setting a variable only in the notebook does not reconfigure an existing server.
See the [Ollama FAQ](https://docs.ollama.com/faq).

`--host` takes precedence over `OLLAMA_HOST`, which must include `http://` or
`https://`. Default to localhost. If collaborators use a shared GPU machine,
prefer an authenticated SSH tunnel to its local Ollama port; do not expose an
unauthenticated Ollama service to the internet. Notes are sent to the configured
server, so use only an environment authorized to process that data.
