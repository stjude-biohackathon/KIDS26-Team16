"""Ollama transport and fail-closed checks for full 16-bit MedGemma 27B."""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

# LOCAL: the 27B F16 weights are not installed on this machine; gemma4:12b is.
# The strict preflight below still refuses it, so callers that want it must pass
# strict=False (the CLI's --force-model, and the dashboard's availability check).
DEFAULT_MODEL = "medgemma-1.5-4b-it"
DEFAULT_HOST = "http://localhost:11434"
WEIGHTS = "google/medgemma-27b-text-it"
GGUF_FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 4: "Q4_1_SOME_F16", 7: "Q8_0", 8: "Q5_0",
    9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
    15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS",
    21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S",
    27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
    36: "TQ1_0", 37: "TQ2_0",
}


def request_json(host: str, endpoint: str, body: dict | None = None,
                 timeout: int = 300) -> dict:
    host = host.rstrip("/")
    parts = urlsplit(host)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError("--host must be a complete http:// or https:// Ollama URL")
    request = urllib.request.Request(
        host + endpoint,
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama {endpoint}: HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {host}: {exc}. Start Ollama and check --host."
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"Ollama {endpoint} returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"Ollama {endpoint} did not return a JSON object")
    if result.get("error"):
        raise RuntimeError(f"Ollama {endpoint}: {result['error']}")
    return result


def validate_model(info: dict, strict: bool = True, model: str | None = None) -> str:
    """Validate model metadata against supported architectures.

    Supported model families:
      - MedGemma 27B (gemma3 architecture, ~26-29B parameters, F16/BF16)
      - MedGemma 1.5 4B-IT (gemma3 architecture, ~3.9B parameters)
      - Gemma 4 21B (gemma4 architecture, ~21B parameters)

    When strict=False, bypasses parameter/architecture checks for experimental models.
    """
    details = info.get("details", {})
    if not strict:
        return str(details.get("quantization_level") or "custom")
    metadata = info.get("model_info", {})
    count = metadata.get("general.parameter_count")
    arch = metadata.get("general.architecture")
    raw_type = metadata.get("general.file_type")
    quant = details.get("quantization_level")
    precision = GGUF_FILE_TYPES.get(raw_type) if raw_type is not None else quant

    target = (model or "").lower()
    if "4b" in target or "medgemma-1.5" in target:
        if not isinstance(count, (int, float)) or not 3e9 <= count < 5e9:
            raise ValueError(f"Expected MedGemma 1.5 4B parameters (~3.9B); got parameter count {count!r}")
        if arch != "gemma3":
            raise ValueError(f"Expected MedGemma 1.5 4B with gemma3 architecture; got {arch!r}")
        return str(precision or quant or "custom")
    elif "12b" in target or "21b" in target or ("gemma4" in target and "27b" not in target):
        if arch != "gemma4" and (not isinstance(count, (int, float)) or not 10e9 <= count < 25e9):
            raise ValueError(f"Expected Gemma 4 12B/21B architecture or parameter count; got arch={arch!r}, count={count!r}")
        return str(precision or quant or "custom")
    else:
        # Default / MedGemma 27B strict validation
        if not isinstance(count, (int, float)) or not 26e9 <= count < 29e9:
            raise ValueError(f"Only full 27B models are supported; parameter count is {count!r}")
        if arch != "gemma3":
            raise ValueError("Expected MedGemma 27B text weights with gemma3 architecture")
        if raw_type is not None:
            if precision not in {"F16", "BF16"}:
                raise ValueError(f"Only F16/BF16 weights are allowed; GGUF file type is {raw_type!r}")
            if quant is not None and quant != precision:
                raise ValueError(f"Conflicting model precision metadata: {precision} versus {quant}")
        if precision not in {"F16", "BF16"}:
            raise ValueError(f"Only unquantized F16/BF16 weights are allowed; got {precision!r}")
        return precision


def call_ollama(prompt: str, model: str, host: str, stats: dict | None = None,
                timeout: int = 300, fmt=None, seed: int = 0, temperature: float = 0.0,
                num_predict: int = 1024, repeat_penalty: float = 1.1,
                num_ctx: int = 16384, **kwargs) -> str:
    start = time.monotonic()
    response = request_json(host, "/api/generate", {
        "model": model, "prompt": prompt, "stream": False,
        "format": fmt if fmt is not None else "json", "keep_alive": "10m",
        "options": {"temperature": temperature, "seed": seed, "num_ctx": num_ctx,
                    "num_predict": num_predict, "repeat_penalty": repeat_penalty},
    }, timeout=timeout)
    reply = response.get("response")
    if not isinstance(reply, str) or not reply.strip() or response.get("done") is not True:
        raise RuntimeError("Ollama returned an empty or incomplete generation")
    if stats is not None:
        for key in ("prompt_eval_count", "eval_count"):
            stats[key] = stats.get(key, 0) + response.get(key, 0)
        stats["eval_duration_sec"] = (stats.get("eval_duration_sec", 0.0)
                                      + response.get("eval_duration", 0) / 1e9)
        stats["total_duration_sec"] = (stats.get("total_duration_sec", 0.0)
                                       + time.monotonic() - start)
    return reply


def preflight(model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST,
              timeout: int = 300, num_ctx: int = 16384, strict: bool = True) -> dict:
    info = request_json(host, "/api/show", {"model": model}, timeout=timeout)
    precision = validate_model(info, strict=strict, model=model)
    tags = request_json(host, "/api/tags", timeout=timeout)
    canonical = model if ":" in model.rsplit("/", 1)[-1] else f"{model}:latest"
    entry = next((item for item in tags.get("models", [])
                  if item.get("name") == canonical or item.get("model") == canonical
                  or item.get("name") == model or item.get("model") == model
                  or str(item.get("name", "")).rstrip(":latest") == model.rstrip(":latest")), None)
    if entry is None or not entry.get("digest"):
        raise ValueError(f"No model digest found for {canonical}; check `ollama list`")
    reply = call_ollama(
        'Return exactly this JSON object: {"present": false, "findings": []}.',
        model, host, timeout=timeout, num_ctx=num_ctx, num_predict=128,
    )
    if re.search(r"\[UNK_BYTE_|\u2581", reply):
        raise ValueError("Tokenizer artifacts in preflight reply; verify the GGUF conversion")
    try:
        parsed = json.loads(reply)
    except json.JSONDecodeError as exc:
        raise ValueError("Preflight generation was not valid JSON; check the model template") from exc
    if parsed != {"present": False, "findings": []}:
        raise ValueError("Preflight model did not follow the JSON instruction; check weights/template")
    return {"quant": precision, "model_digest": entry["digest"], "model_info": info}
