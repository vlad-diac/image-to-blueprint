#!/usr/bin/env python3
"""
Populate a RunPod network volume with model files.

Usage:
  python worker/scripts/provision_volume.py

Env:
  HF_TOKEN       — optional, for gated repos
  RUNPOD_VOLUME  — volume mount path (default: /runpod-volume)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("provision_volume")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

VOL               = Path(os.environ.get("RUNPOD_VOLUME", "/runpod-volume"))
BASE_DIR          = VOL / "models"
UNET_DIR          = BASE_DIR / "unet"
VAE_DIR           = BASE_DIR / "vae"
TEXT_ENCODER_DIR  = BASE_DIR / "text_encoders"
LORA_DIR          = BASE_DIR / "loras"
HF_HOME           = VOL / "huggingface-cache"
HF_HUB_CACHE      = HF_HOME / "hub"

# ---------------------------------------------------------------------------
# Create directories
# ---------------------------------------------------------------------------

for _d in [BASE_DIR, UNET_DIR, VAE_DIR, TEXT_ENCODER_DIR, LORA_DIR, HF_HOME, HF_HUB_CACHE]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# HuggingFace cache configuration
# Must be set before importing huggingface_hub so the library picks them up.
# ---------------------------------------------------------------------------

os.environ["HF_HOME"]                  = str(HF_HOME)
os.environ["HF_HUB_CACHE"]             = str(HF_HUB_CACHE)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"   # requires: pip install hf_transfer

from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402


def _exists_file(path: Path) -> bool:
    if path.is_file():
        logger.info("[skip] exists (%.1f MB): %s", path.stat().st_size / 1024 / 1024, path)
        return True
    return False


def _exists_dir(path: Path) -> bool:
    if path.is_dir() and any(path.iterdir()):
        logger.info("[skip] dir non-empty: %s", path)
        return True
    return False


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if token:
        logger.info("HF_TOKEN detected — gated repos enabled")

    # -----------------------------------------------------------------------
    # 1. Base Qwen Image Edit model snapshot
    #    Stored flat (local_dir_use_symlinks=False) so the path is fixed and
    #    the handler never needs to resolve a hash-based snapshot directory.
    # -----------------------------------------------------------------------
    snapshot_dest = BASE_DIR / "Qwen--Qwen-Image-Edit-2511"
    if not _exists_dir(snapshot_dest):
        logger.info("=== Downloading base Qwen model snapshot ===")
        snapshot_download(
            repo_id="Qwen/Qwen-Image-Edit-2511",
            local_dir=str(snapshot_dest),
            local_dir_use_symlinks=False,
            # Skip large binary weights — they are downloaded individually below.
            ignore_patterns=["*.safetensors", "*.bin", "*.gguf"],
            token=token,
        )

    # -----------------------------------------------------------------------
    # 2. GGUF transformer
    # -----------------------------------------------------------------------
    gguf_dest = UNET_DIR / "qwen-image-edit-2511-Q3_K_L.gguf"
    if not _exists_file(gguf_dest):
        logger.info("=== Downloading GGUF transformer ===")
        hf_hub_download(
            repo_id="unsloth/Qwen-Image-Edit-2511-GGUF",
            filename="qwen-image-edit-2511-Q3_K_L.gguf",
            local_dir=str(UNET_DIR),
            token=token,
        )

    # -----------------------------------------------------------------------
    # 3. VAE
    #    The repo path is split_files/vae/<name>, so hf_hub_download places
    #    the file at VAE_DIR/split_files/vae/<name>.
    # -----------------------------------------------------------------------
    vae_dest = VAE_DIR / "split_files" / "vae" / "qwen_image_vae.safetensors"
    if not _exists_file(vae_dest):
        logger.info("=== Downloading VAE ===")
        hf_hub_download(
            repo_id="Comfy-Org/Qwen-Image_ComfyUI",
            filename="split_files/vae/qwen_image_vae.safetensors",
            local_dir=str(VAE_DIR),
            token=token,
        )

    # -----------------------------------------------------------------------
    # 4. Text encoder
    #    Flat filename — file lands directly at TEXT_ENCODER_DIR/<name>.
    # -----------------------------------------------------------------------
    te_dest = TEXT_ENCODER_DIR / "qwen_2.5_vl_7b_fp8_scaled.safetensors"
    if not _exists_file(te_dest):
        logger.info("=== Downloading text encoder ===")
        hf_hub_download(
            repo_id="f5aiteam/CLIP",
            filename="qwen_2.5_vl_7b_fp8_scaled.safetensors",
            local_dir=str(TEXT_ENCODER_DIR),
            token=token,
        )

    # -----------------------------------------------------------------------
    # 5. Lightning LoRA
    # -----------------------------------------------------------------------
    lightning_dest = LORA_DIR / "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
    if not _exists_file(lightning_dest):
        logger.info("=== Downloading Lightning LoRA ===")
        hf_hub_download(
            repo_id="lightx2v/Qwen-Image-Edit-2511-Lightning",
            filename="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
            local_dir=str(LORA_DIR),
            token=token,
        )

    # -----------------------------------------------------------------------
    # 6. Multiple Angles LoRA
    # -----------------------------------------------------------------------
    angles_dest = LORA_DIR / "qwen-image-edit-2511-multiple-angles-lora.safetensors"
    if not _exists_file(angles_dest):
        logger.info("=== Downloading Multiple Angles LoRA ===")
        hf_hub_download(
            repo_id="fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
            filename="qwen-image-edit-2511-multiple-angles-lora.safetensors",
            local_dir=str(LORA_DIR),
            token=token,
        )

    logger.info("✅ Provisioning complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
