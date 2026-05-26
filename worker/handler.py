"""RunPod serverless handler: warm-loaded Qwen-Image-Edit pipeline."""

from __future__ import annotations

import base64
import io
import logging
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# HuggingFace cache — must be set before any HF / transformers import so the
# libraries pick up the volume-backed cache and never attempt network calls.
# ---------------------------------------------------------------------------
_VOL_EARLY = Path(os.environ.get("RUNPOD_VOLUME", "/runpod-volume"))
os.environ.setdefault("HF_HOME",      str(_VOL_EARLY / "huggingface-cache"))
os.environ.setdefault("HF_HUB_CACHE", str(_VOL_EARLY / "huggingface-cache" / "hub"))
os.environ["HF_HUB_OFFLINE"]      = "1"   # never download at runtime
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import runpod
import torch
from PIL import Image

from pipeline_utils import QwenEditPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

VOL = Path(os.environ.get("RUNPOD_VOLUME", "/runpod-volume"))
MODELS = VOL / "models"


def _check(p: Path) -> Path:
    """Log whether a model path exists, then return it."""
    if p.exists():
        size = p.stat().st_size / 1024 / 1024 if p.is_file() else None
        logger.info("✓ found  %s%s", p, f"  ({size:.1f} MB)" if size else "")
    else:
        logger.error("✗ MISSING  %s", p)
    return p


def _build_pipeline() -> QwenEditPipeline:
    attn = os.environ.get("ATTN_BACKEND", "_native_flash")
    offload = os.environ.get("ENABLE_OFFLOAD", "").lower() in ("1", "true", "yes")
    compile_te = os.environ.get("COMPILE_TEXT_ENCODER", "").lower() in ("1", "true", "yes")

    # hf_hub_download preserves the repo's subdirectory structure under local_dir,
    # so the VAE (split_files/vae/…) is nested one level deeper than its parent dir.
    transformer_path = _check(MODELS / "unet"          / "qwen-image-edit-2511-Q3_K_L.gguf")
    vae_path         = _check(MODELS / "vae"            / "split_files" / "vae"           / "qwen_image_vae.safetensors")
    te_path          = _check(MODELS / "text_encoders"  / "qwen_2.5_vl_7b_fp8_scaled.safetensors")
    snapshot_path    = _check(MODELS / "Qwen--Qwen-Image-Edit-2511")
    lora_angles      = _check(MODELS / "loras"          / "qwen-image-edit-2511-multiple-angles-lora.safetensors")
    lora_lightning   = _check(MODELS / "loras"          / "Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors")

    pipe = (
        QwenEditPipeline()
        .load(
            components={
                "default_repo": "Qwen/Qwen-Image-Edit-2511",
                "default_local": str(snapshot_path),
                "transformer": {"path": str(transformer_path)},
                "vae": {"path": str(vae_path)},
                "text_encoder": {
                    "path": str(te_path),
                    "format": "fp8_scaled",
                },
            },
            dtype=torch.bfloat16,
            device="cuda",
            enable_offload=offload,
            compile_text_encoder=compile_te,
            attention_backend=attn if attn else None,
        )
        .add_lora(lora_angles, name="angles")
        .add_lora(lora_lightning, name="lightning")
    )
    pipe.flush_loras()
    return pipe


logger.info("Loading pipeline (cold start)…")
PIPE = _build_pipeline()
logger.info("Pipeline ready.")


def handler(event: dict) -> dict:
    inp = event.get("input") or {}
    job_id = event.get("id")
    if not job_id:
        import uuid
        job_id = str(uuid.uuid4())

    raw_b64 = inp.get("image_b64")
    if not raw_b64:
        raise ValueError("input.image_b64 is required")

    img = Image.open(io.BytesIO(base64.b64decode(raw_b64))).convert("RGB")

    positive = inp.get("positive_prompt", "")
    if not str(positive).strip():
        raise ValueError("input.positive_prompt is required")

    negative = inp.get("negative_prompt") or ""
    steps    = int(inp.get("steps", 4))
    cfg      = float(inp.get("cfg", 1.0))
    seed     = inp.get("seed")
    if seed is not None:
        seed = int(seed)

    out = PIPE.run(
        image=img,
        positive_prompt=positive,
        negative_prompt=negative,
        steps=steps,
        cfg=cfg,
        seed=seed,
    )

    job_dir  = VOL / "jobs" / str(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    out_path = job_dir / "output.png"
    out.save(out_path)

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    rel = job_dir.relative_to(VOL)
    return {
        "image_b64": image_b64,
        "job_dir": str(rel).replace("\\", "/"),
        "width": out.width,
        "height": out.height,
    }


runpod.serverless.start({"handler": handler})
