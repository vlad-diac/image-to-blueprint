#!/usr/bin/env python3
"""
Populate a RunPod network volume from worker/manifest.json.

Paths in the manifest must match worker/handler.py (_build_pipeline).

Usage:
  python worker/scripts/provision_volume.py

Env:
  HF_TOKEN       — optional, for gated repos
  RUNPOD_VOLUME  — volume mount path (default: /runpod-volume)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("provision_volume")

SCRIPT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = SCRIPT_DIR.parent / "manifest.json"

VOL = Path(os.environ.get("RUNPOD_VOLUME", "/runpod-volume"))
HF_HOME = VOL / "huggingface-cache"
HF_HUB_CACHE = HF_HOME / "hub"

# Marker files the handler needs inside the config snapshot (not repo root).
SNAPSHOT_MARKERS = (
    "transformer/config.json",
    "vae/config.json",
    "tokenizer/tokenizer_config.json",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_ready(path: Path, expected_sha256: str | None) -> bool:
    if not path.is_file():
        return False
    size_mb = path.stat().st_size / 1024 / 1024
    if expected_sha256:
        digest = _sha256_file(path)
        if digest != expected_sha256:
            logger.warning(
                "[redo] sha256 mismatch for %s (got %s…)", path, digest[:12],
            )
            return False
    logger.info("[skip] exists (%.1f MB): %s", size_mb, path)
    return True


def _snapshot_ready(path: Path) -> bool:
    if not path.is_dir():
        return False
    missing = [m for m in SNAPSHOT_MARKERS if not (path / m).is_file()]
    if missing:
        logger.warning(
            "[redo] snapshot incomplete at %s — missing: %s",
            path,
            ", ".join(missing),
        )
        return False
    logger.info("[skip] snapshot complete: %s", path)
    return True


def _download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info("Downloading URL → %s", dest)
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.replace(dest)


def _hf_local_dir(dest: Path, remote_path: str) -> Path:
    """Directory D where hf_hub_download(..., local_dir=D) writes D/remote_path == dest."""
    base = dest.parent
    for _ in range(len(Path(remote_path).parts) - 1):
        base = base.parent
    return base


def _provision_hf_file(
    dest: Path,
    repo: str,
    remote_path: str,
    token: str | None,
) -> None:
    from huggingface_hub import hf_hub_download

    local_dir = _hf_local_dir(dest, remote_path)
    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info("HF file %s/%s → %s", repo, remote_path, dest)
    hf_hub_download(
        repo_id=repo,
        filename=remote_path,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        token=token,
    )
    if not dest.is_file():
        raise FileNotFoundError(
            f"Expected file at {dest} after hf_hub_download "
            f"({repo}/{remote_path}, local_dir={local_dir})",
        )


def _provision_hf_snapshot(
    dest: Path,
    repo: str,
    exclude: list[str],
    token: str | None,
) -> None:
    from huggingface_hub import snapshot_download

    if dest.is_dir() and not _snapshot_ready(dest):
        logger.info("Removing incomplete snapshot: %s", dest)
        shutil.rmtree(dest)

    dest.mkdir(parents=True, exist_ok=True)
    logger.info("HF snapshot %s → %s (exclude %s)", repo, dest, exclude)
    snapshot_download(
        repo_id=repo,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        ignore_patterns=exclude or None,
        token=token,
    )
    if not _snapshot_ready(dest):
        raise RuntimeError(
            f"Snapshot at {dest} is still missing required config files "
            f"({', '.join(SNAPSHOT_MARKERS)})",
        )


def _provision_entry(entry: dict[str, Any], token: str | None) -> None:
    dest = VOL / entry["dest"]
    source = entry["source"]
    kind = source["kind"]
    expected_sha = entry.get("sha256")

    if kind == "hf_snapshot":
        if _snapshot_ready(dest):
            return
        _provision_hf_snapshot(
            dest,
            repo=source["repo"],
            exclude=list(source.get("exclude") or []),
            token=token,
        )
        return

    if kind in ("hf_file", "url"):
        if _file_ready(dest, expected_sha):
            return

    if kind == "hf_file":
        _provision_hf_file(
            dest,
            repo=source["repo"],
            remote_path=source["path"],
            token=token,
        )
        if expected_sha and _sha256_file(dest) != expected_sha:
            raise RuntimeError(f"sha256 mismatch after download: {dest}")
        return

    if kind == "url":
        _download_url(source["url"], dest)
        if expected_sha and _sha256_file(dest) != expected_sha:
            raise RuntimeError(f"sha256 mismatch after download: {dest}")
        return

    raise ValueError(f"Unknown source kind: {kind!r}")


def main() -> int:
    if not MANIFEST_PATH.is_file():
        logger.error("Manifest not found: %s", MANIFEST_PATH)
        return 1

    for d in (VOL / "models", HF_HOME, HF_HUB_CACHE):
        d.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    with MANIFEST_PATH.open(encoding="utf-8") as fh:
        manifest = json.load(fh)

    token = os.environ.get("HF_TOKEN")
    if token:
        logger.info("HF_TOKEN detected — gated repos enabled")

    volume_root = Path(manifest.get("volume_root", "/runpod-volume"))
    if volume_root.resolve() != VOL.resolve():
        logger.warning(
            "manifest volume_root=%s but RUNPOD_VOLUME=%s — using %s",
            volume_root,
            VOL,
            VOL,
        )

    entries = manifest.get("files") or []
    logger.info("Provisioning %d entries from %s", len(entries), MANIFEST_PATH)

    for entry in entries:
        logger.info("=== %s ===", entry.get("dest"))
        _provision_entry(entry, token)

    logger.info("Provisioning complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
