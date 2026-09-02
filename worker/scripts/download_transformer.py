#!/usr/bin/env python3
"""
Download the GGUF transformer into the Docker image at build time.

Run from worker/Dockerfile (RUN python …) so `from_single_file` reads the 10 GB
transformer off the container's local NVMe instead of the RunPod network volume.
The URL is sourced from worker/manifest.json (single source of truth), so a model
swap only touches the manifest + the --dest/env in the Dockerfile.

Stdlib only — no pip deps — so this layer runs before `pip install` and never
invalidates on dependency/code changes.

Usage:
  python worker/scripts/download_transformer.py \
      --manifest worker/manifest.json \
      --dest /app/models/unet/qwen-image-edit-2511-Q3_K_L.gguf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("download_transformer")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR.parent / "manifest.json"
DEFAULT_DEST = Path("/app/models/unet/qwen-image-edit-2511-Q3_K_L.gguf")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_transformer_entry(manifest: dict[str, Any], dest: Path) -> dict[str, Any]:
    """Return the url-kind manifest entry whose dest basename matches `dest`."""
    target = dest.name
    for entry in manifest.get("files", []):
        entry_dest = entry.get("dest", "")
        if Path(entry_dest).name != target:
            continue
        source = entry.get("source") or {}
        if source.get("kind") != "url" or not source.get("url"):
            raise ValueError(
                f"Manifest entry for {target!r} is not a downloadable url source: {source!r}",
            )
        return entry
    raise ValueError(
        f"No manifest entry found with dest basename {target!r} "
        f"(looked in {len(manifest.get('files', []))} entries)",
    )


def _download_url(url: str, dest: Path) -> None:
    """Stream `url` → `dest` via a .part temp, failing loudly on a short read."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info("Downloading %s → %s", url, dest)
    with urllib.request.urlopen(url) as resp:  # raises HTTPError on non-2xx
        expected = resp.headers.get("Content-Length")
        expected = int(expected) if expected is not None else None
        with tmp.open("wb") as out:
            shutil.copyfileobj(resp, out)
    written = tmp.stat().st_size
    if expected is not None and written != expected:
        tmp.unlink(missing_ok=True)
        raise IOError(
            f"Short read: got {written} bytes, expected {expected} from {url}",
        )
    tmp.replace(dest)
    logger.info("Downloaded %.1f MB → %s", written / 1024 / 1024, dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()

    if not args.manifest.is_file():
        logger.error("Manifest not found: %s", args.manifest)
        return 1

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entry = _find_transformer_entry(manifest, args.dest)
    url = entry["source"]["url"]
    expected_sha = entry.get("sha256")

    if args.dest.is_file():
        if expected_sha:
            if _sha256_file(args.dest) == expected_sha:
                logger.info("[skip] exists and sha256 matches: %s", args.dest)
                return 0
            logger.warning("[redo] sha256 mismatch, re-downloading: %s", args.dest)
        else:
            size_mb = args.dest.stat().st_size / 1024 / 1024
            logger.info("[skip] exists (%.1f MB): %s", size_mb, args.dest)
            return 0

    _download_url(url, args.dest)

    if expected_sha:
        digest = _sha256_file(args.dest)
        if digest != expected_sha:
            args.dest.unlink(missing_ok=True)
            logger.error("sha256 mismatch after download: got %s…", digest[:12])
            return 1
        logger.info("sha256 verified: %s", args.dest)

    return 0


if __name__ == "__main__":
    sys.exit(main())
