#!/usr/bin/env python3
"""
Populate a RunPod network volume (or local path) with model files per worker/manifest.json.

Usage:
  PYTHONPATH=  # N/A
  python worker/scripts/provision_volume.py
  MANIFEST_PATH=/path/to/manifest.json python worker/scripts/provision_volume.py

Requires: huggingface_hub, httpx

Env:
  HF_TOKEN — optional, for gated repos
  VOLUME_ROOT — overrides manifest volume_root
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("provision_volume")


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _default_manifest_path() -> Path:
    return _script_dir().parent / "manifest.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_url(url: str, dest: Path) -> None:
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    with httpx.Client(follow_redirects=True, timeout=None) as client:
        with client.stream("GET", url, headers={"Accept": "*/*"}) as r:
            r.raise_for_status()
            with open(partial, "wb") as f:
                for chunk in r.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
    partial.replace(dest)
    logger.info("Downloaded URL -> %s", dest)


def _hf_file(repo: str, remote_path: str, dest: Path) -> None:
    import shutil

    from huggingface_hub import hf_hub_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(
        repo_id=repo,
        filename=remote_path,
        token=os.environ.get("HF_TOKEN"),
    )
    shutil.copy2(cached, dest)
    logger.info("HF file %s/%s -> %s", repo, remote_path, dest)


def _hf_snapshot(repo: str, dest_dir: Path, exclude: list[str]) -> None:
    from huggingface_hub import snapshot_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo,
        local_dir=str(dest_dir),
        ignore_patterns=exclude,
        token=os.environ.get("HF_TOKEN"),
    )
    logger.info("HF snapshot %s -> %s", repo, dest_dir)


def _process_entry(vol: Path, entry: dict[str, Any]) -> None:
    dest_rel = entry["dest"]
    dest = vol / dest_rel
    source = entry["source"]
    kind = source["kind"]
    sha256_expect = entry.get("sha256")

    if kind in ("hf_file", "url"):
        if dest.exists() and dest.is_file():
            if sha256_expect:
                got = _sha256_file(dest)
                if got.lower() == sha256_expect.lower():
                    logger.info("[skip] OK sha256: %s", dest)
                    return
                logger.warning("Checksum mismatch, re-downloading: %s", dest)
            else:
                logger.info("[skip] exists: %s", dest)
                return

    if kind == "hf_file":
        repo = source["repo"]
        path = source["path"]
        _hf_file(repo, path, dest)
    elif kind == "url":
        _download_url(source["url"], dest)
    elif kind == "hf_snapshot":
        repo = source["repo"]
        exclude = source.get("exclude", [])
        # If directory exists and non-empty, skip (idempotent for POC)
        if dest.is_dir() and any(dest.iterdir()):
            logger.info("[skip] snapshot dir non-empty: %s", dest)
            return
        _hf_snapshot(repo, dest, exclude)
    else:
        raise ValueError(f"Unknown source kind: {kind}")

    if kind != "hf_snapshot" and sha256_expect and dest.is_file():
        got = _sha256_file(dest)
        if got.lower() != sha256_expect.lower():
            raise RuntimeError(
                f"sha256 mismatch for {dest}: expected {sha256_expect}, got {got}"
            )


def main() -> int:
    manifest_path = Path(os.environ.get("MANIFEST_PATH", _default_manifest_path()))
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        return 1

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    volume_root = Path(os.environ.get("VOLUME_ROOT", manifest.get("volume_root", "/runpod-volume")))
    logger.info("Volume root: %s", volume_root)
    volume_root.mkdir(parents=True, exist_ok=True)

    files = manifest.get("files", [])
    for i, entry in enumerate(files):
        logger.info("--- [%d/%d] %s ---", i + 1, len(files), entry.get("dest"))
        try:
            _process_entry(volume_root, entry)
        except Exception as e:
            logger.exception("Failed on %s: %s", entry.get("dest"), e)
            return 1

    logger.info("Provision complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
