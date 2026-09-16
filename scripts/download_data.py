"""Download and prepare clinical datasets for SCOGS.

Downloads:
- PMC-Patients-V2.json & PMC-Patients.csv from Hugging Face (zhengyun21/PMC-Patients)
- Generates or updates the 978 Sickle Cell Disease (SCD) cohort cache (PMC-Patients/scd_cache.json)

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --files v2 --force
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
PMC_DIR = ROOT / "PMC-Patients"
DATA_DIR = ROOT / "data"

URLS = {
    "v2": {
        "url": "https://huggingface.co/datasets/zhengyun21/PMC-Patients/resolve/main/PMC-Patients-V2.json",
        "file": PMC_DIR / "PMC-Patients-V2.json",
        "description": "Full PMC-Patients V2 dataset (JSON, ~800MB, 250k patients)",
    },
    "csv": {
        "url": "https://huggingface.co/datasets/zhengyun21/PMC-Patients/resolve/main/PMC-Patients.csv",
        "file": PMC_DIR / "PMC-Patients.csv",
        "description": "PMC-Patients V1 dataset (CSV, ~520MB, 167k patients)",
    },
}


def download_file(url: str, dest: pathlib.Path, description: str, force: bool = False):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"[EXISTS] {dest.name} ({size_mb:.1f} MB) already present. Use --force to re-download.")
        return

    print(f"\n[DOWNLOADING] {description}...")
    print(f"URL: {url}")
    print(f"Destination: {dest}")

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SCOGS-Downloader"}
    req = urllib.request.Request(url, headers=headers)

    t0 = time.time()
    try:
        with urllib.request.urlopen(req) as response, open(dest, "wb") as out_file:
            total_length = response.headers.get("Content-Length")
            total_bytes = int(total_length) if total_length else None
            downloaded = 0
            block_size = 1024 * 1024  # 1MB chunks

            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_bytes:
                    pct = 100 * downloaded / total_bytes
                    mb_done = downloaded / (1024 * 1024)
                    mb_total = total_bytes / (1024 * 1024)
                    speed = mb_done / max(time.time() - t0, 0.001)
                    sys.stdout.write(f"\r  Progress: {pct:.1f}% ({mb_done:.1f}/{mb_total:.1f} MB) at {speed:.2f} MB/s")
                else:
                    mb_done = downloaded / (1024 * 1024)
                    sys.stdout.write(f"\r  Downloaded: {mb_done:.1f} MB")
                sys.stdout.flush()
        print(f"\n[SUCCESS] Saved to {dest} in {time.time() - t0:.1f}s")
    except Exception as e:
        if dest.exists():
            dest.unlink()
        print(f"\n[ERROR] Download failed: {e}")
        raise


def build_scd_cache():
    v2_path = PMC_DIR / "PMC-Patients-V2.json"
    cache_path = PMC_DIR / "scd_cache.json"

    if not v2_path.exists():
        print(f"[SKIP] Cannot build scd_cache.json because {v2_path.name} is missing.")
        return

    print(f"\n[FILTERING] Building Sickle Cell Disease cohort cache from {v2_path.name}...")
    pat = re.compile(r"sickle cell|\bSCD\b|HbSS|HbSC", re.I)
    raw = json.loads(v2_path.read_text(encoding="utf-8"))
    scd = [r for r in raw if pat.search(r.get("patient", ""))]
    scd.sort(key=lambda r: r["patient_uid"])
    cache_path.write_text(json.dumps(scd, indent=2), encoding="utf-8")
    print(f"[SUCCESS] Extracted {len(scd)} SCD patient cases into {cache_path} ({cache_path.stat().st_size / (1024*1024):.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--files", choices=["all", "v2", "csv"], default="all",
                        help="which datasets to download (default: all)")
    parser.add_argument("--force", action="store_true", help="force re-download existing files")
    args = parser.parse_args()

    print("=" * 60)
    print("SCOGS Dataset Downloader")
    print("=" * 60)

    if args.files in {"all", "v2"}:
        download_file(URLS["v2"]["url"], URLS["v2"]["file"], URLS["v2"]["description"], args.force)
    if args.files in {"all", "csv"}:
        download_file(URLS["csv"]["url"], URLS["csv"]["file"], URLS["csv"]["description"], args.force)

    build_scd_cache()

    print("\n" + "=" * 60)
    print("Dataset Status:")
    for key, info in URLS.items():
        exists = info["file"].exists()
        size = f"{info['file'].stat().st_size / (1024*1024):.1f} MB" if exists else "NOT FOUND"
        print(f"  - {info['file'].name:25s} [{size}]")
    cache_file = PMC_DIR / "scd_cache.json"
    cache_status = f"{cache_file.stat().st_size / (1024*1024):.2f} MB" if cache_file.exists() else "NOT FOUND"
    print(f"  - {cache_file.name:25s} [{cache_status}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
