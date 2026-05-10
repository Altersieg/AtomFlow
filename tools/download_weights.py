#!/usr/bin/env python3
# ============================================================================
# tools/download_weights.py
# ----------------------------------------------------------------------------
# Downloads the AtomFlow pre-exported weight .bin and the Llama-3.2 tokenizer
# directory from a public HuggingFace Hub repository, using only the Python
# standard library (urllib.request).  No 'huggingface_hub', no 'requests',
# no 'torch', no 'transformers'.  Supports resumable downloads via HTTP
# Range and SHA-256 verification.
#
# Default layout after a successful run:
#   models/
#   ├── llama3_2_atomflow.bin
#   └── Llama-3.2-3B-Instruct/
#       ├── tokenizer.json
#       ├── tokenizer_config.json
#       └── special_tokens_map.json
#
# Configure repo via ATOMFLOW_HF_REPO environment variable, e.g.:
#     export ATOMFLOW_HF_REPO="yourname/atomflow-llama3.2-3b"
#
# Usage:
#     python3 tools/download_weights.py                # download all missing
#     python3 tools/download_weights.py --force        # re-download all
#     python3 tools/download_weights.py --verify       # only check SHA256
# ============================================================================

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parent.parent
MODELS_DIR  = REPO_ROOT / "models"

DEFAULT_HF_REPO = "Altersieg/test4AtomFlow"
HF_REPO = os.environ.get("ATOMFLOW_HF_REPO", DEFAULT_HF_REPO)

HF_URL_TEMPLATE = "https://huggingface.co/{repo}/resolve/main/{path}"


@dataclass
class Asset:
    remote_path: str          # path inside the HF repo
    local_path:  Path         # where to save locally
    sha256:      str | None   # expected hex digest, None = skip check
    size_bytes:  int | None   # advisory, for sanity check only

    @property
    def url(self) -> str:
        return HF_URL_TEMPLATE.format(repo=HF_REPO, path=self.remote_path)


# Fill in real SHA-256 once you upload.  Leaving as None means "skip check".
ASSETS: list[Asset] = [
    Asset(
        remote_path = "llama3_2_atomflow.bin",
        local_path  = MODELS_DIR / "llama3_2_atomflow.bin",
        sha256      = "de9815833f54defb05ec3d3315ceba86b283bd8d706a59e0b26f28ae3036d251",
        size_bytes  = 5_227_327_744,
    ),
    Asset(
        remote_path = "Llama-3.2-3B-Instruct/tokenizer.json",
        local_path  = MODELS_DIR / "Llama-3.2-3B-Instruct" / "tokenizer.json",
        sha256      = "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4",
        size_bytes  = 9_085_657,
    ),
    Asset(
        remote_path = "Llama-3.2-3B-Instruct/tokenizer_config.json",
        local_path  = MODELS_DIR / "Llama-3.2-3B-Instruct" / "tokenizer_config.json",
        sha256      = "9823dcfdc1121869029da45192238e85cf44f0b232a6d9dc20e4fe6f4242a14e",
        size_bytes  = 54_528,
    ),
    Asset(
        remote_path = "Llama-3.2-3B-Instruct/special_tokens_map.json",
        local_path  = MODELS_DIR / "Llama-3.2-3B-Instruct" / "special_tokens_map.json",
        sha256      = "6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec",
        size_bytes  = 296,
    ),
]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
CHUNK = 1 << 20   # 1 MiB


def _human(n: float) -> str:
    for u in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:6.1f} {u}"
        n /= 1024
    return f"{n:.1f} TiB"


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _download_with_resume(url: str, dst: Path) -> None:
    """Stream URL to dst.  Resumes if dst.part exists."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")
    start = part.stat().st_size if part.exists() else 0

    req = urllib.request.Request(url)
    if start > 0:
        req.add_header("Range", f"bytes={start}-")

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            # Total size: honour Content-Range when present
            total = None
            cr = resp.headers.get("Content-Range")
            cl = resp.headers.get("Content-Length")
            if cr:
                total = int(cr.split("/")[-1])
            elif cl:
                total = int(cl) + start

            mode = "ab" if start > 0 else "wb"
            downloaded = start
            t0 = time.time()
            t_last = t0
            print(f"    → {url}")
            if start > 0:
                print(f"      resuming from {_human(start)}")

            with part.open(mode) as out:
                while True:
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - t_last >= 0.5 or downloaded == total:
                        pct = 100.0 * downloaded / total if total else 0.0
                        speed = (downloaded - start) / max(now - t0, 1e-6)
                        eta = (total - downloaded) / speed if total and speed > 0 else 0
                        bar_w = 30
                        fill = int(bar_w * pct / 100) if total else 0
                        bar = "█" * fill + "·" * (bar_w - fill)
                        sys.stdout.write(
                            f"\r      [{bar}] {pct:5.1f}%  "
                            f"{_human(downloaded)} / {_human(total or 0)}  "
                            f"{_human(speed)}/s  eta {eta:4.0f}s "
                        )
                        sys.stdout.flush()
                        t_last = now
            print()
    except urllib.error.HTTPError as e:
        sys.exit(f"[FATAL] HTTP {e.code} {e.reason} for {url}")
    except urllib.error.URLError as e:
        sys.exit(f"[FATAL] network error: {e.reason} for {url}")

    part.rename(dst)


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
def _check_sha(asset: Asset) -> bool:
    if asset.sha256 is None:
        return True
    got = _sha256_of(asset.local_path)
    if got.lower() != asset.sha256.lower():
        print(f"    ✗ SHA-256 mismatch for {asset.local_path.name}")
        print(f"        expected : {asset.sha256}")
        print(f"        got      : {got}")
        return False
    print(f"    ✓ SHA-256 ok for {asset.local_path.name}")
    return True


def _need_download(asset: Asset, force: bool) -> bool:
    if force:
        return True
    if not asset.local_path.exists():
        return True
    # Size sanity check only (SHA verified separately)
    if asset.size_bytes and asset.local_path.stat().st_size != asset.size_bytes:
        print(f"    ⚠ size mismatch for {asset.local_path.name}; re-downloading")
        return True
    return False


def fetch_all(force: bool = False, verify_only: bool = False) -> int:
    print(f"[INFO] HF repo       : {HF_REPO}"
          + ("  (default)" if HF_REPO == DEFAULT_HF_REPO else "  (env override)"))
    print(f"[INFO] Target dir    : {MODELS_DIR}")
    print(f"[INFO] Assets        : {len(ASSETS)}")
    print()

    failures = 0
    for i, a in enumerate(ASSETS, 1):
        print(f"[{i}/{len(ASSETS)}] {a.remote_path}")
        if verify_only:
            if not a.local_path.exists():
                print(f"    ✗ missing: {a.local_path}")
                failures += 1
            else:
                if not _check_sha(a):
                    failures += 1
            continue

        if _need_download(a, force):
            _download_with_resume(a.url, a.local_path)
        else:
            print(f"    ✓ exists  {a.local_path}  "
                  f"({_human(a.local_path.stat().st_size)})")

        if not _check_sha(a):
            failures += 1

    print()
    if failures:
        print(f"[FAIL] {failures} asset(s) failed verification.")
        return 1
    print("[DONE] All assets present and verified.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download AtomFlow weights and tokenizer from HuggingFace Hub."
    )
    parser.add_argument("--force",  action="store_true",
                        help="re-download even if files exist")
    parser.add_argument("--verify", action="store_true",
                        help="only verify SHA-256, do not download")
    args = parser.parse_args()
    return fetch_all(force=args.force, verify_only=args.verify)


if __name__ == "__main__":
    sys.exit(main())
