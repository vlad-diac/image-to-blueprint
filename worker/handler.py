"""RunPod serverless handler: warm-loaded Qwen-Image-Edit pipeline."""

from __future__ import annotations

import base64
import io
import logging
import os
import random
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

import subprocess
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
    # so the VAE (repo path: split_files/vae/…) is nested under its parent dir.
    transformer_path = _check(MODELS / "unet"         / "qwen-image-edit-2511-Q3_K_L.gguf")
    vae_path         = _check(MODELS / "vae"          / "split_files" / "vae" / "qwen_image_vae.safetensors")
    te_path          = _check(MODELS / "text_encoders" / "qwen_2.5_vl_7b_fp8_scaled.safetensors")
    snapshot_path    = MODELS / "Qwen--Qwen-Image-Edit-2511"
    _check(snapshot_path)
    tx_cfg = snapshot_path / "transformer" / "config.json"
    if not tx_cfg.is_file():
        logger.error(
            "✗ MISSING  %s — config snapshot incomplete; "
            "re-run worker/scripts/provision_volume.py on the volume",
            tx_cfg,
        )
    lora_angles      = _check(MODELS / "loras"          / "qwen-image-edit-2511-multiple-angles-lora.safetensors")
    lora_lightning   = _check(MODELS / "loras"          / "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")

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
    steps = int(inp.get("steps", 4))
    cfg = float(inp.get("cfg", 1.0))
    num_images = max(1, min(4, int(inp.get("num_images", 3))))

    seed_raw = inp.get("seed")
    if seed_raw is not None:
        base = int(seed_raw)
        seeds_list = [base + i for i in range(num_images)]
    else:
        seeds_list = [random.randint(0, 2**32 - 1) for _ in range(num_images)]

    imgs = PIPE.run(
        image=img,
        positive_prompt=positive,
        negative_prompt=negative,
        steps=steps,
        cfg=cfg,
        num_images=num_images,
        seeds=seeds_list,
    )

    job_dir = VOL / "jobs" / str(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).parent / "scripts" / "png_to_svg.py"

    payload_images = []
    for i, out in enumerate(imgs):
        png_path = job_dir / f"output_{i}.png"
        out.save(png_path)

        # Generate SVG using the script
        try:
            logger.info(f"Running SVG conversion for {png_path}")
            result = subprocess.run(
                [
                    "python3",
                    str(script_path),
                    png_path.name,
                    "--output-dir",
                    str(job_dir),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info(f"SVG generation stdout: {result.stdout}")
            logger.info(f"SVG generation stderr: {result.stderr}")

            # Log what directories were created
            svg_base = job_dir / "svg"
            if svg_base.exists():
                svg_subdirs = list(svg_base.glob(f"{png_path.stem}_*_centerline"))
                logger.info(f"Found {len(svg_subdirs)} SVG subdirectories: {svg_subdirs}")
            else:
                logger.warning(f"SVG base directory does not exist: {svg_base}")

            # Find the generated SVG file (script creates timestamped directories)
            svg_dirs = sorted((job_dir / "svg").glob(f"{png_path.stem}_*_centerline"))
            logger.info(f"Looking for SVG in: {job_dir / 'svg' / f'{png_path.stem}_*_centerline'}")

            if svg_dirs:
                svg_path = svg_dirs[-1] / "output.svg"
                logger.info(f"Found SVG directory: {svg_dirs[-1]}, checking for: {svg_path}")
                if svg_path.exists():
                    svg_content = svg_path.read_text(encoding="utf-8")
                    svg_b64 = base64.b64encode(svg_content.encode("utf-8")).decode("ascii")
                    logger.info(f"Successfully encoded SVG for output_{i}, size: {len(svg_b64)} bytes")
                else:
                    logger.warning(f"SVG file not found at {svg_path}")
                    svg_b64 = None
            else:
                logger.warning(f"No SVG directory found for {png_path.stem} in {job_dir / 'svg'}")
                # List what's actually there
                svg_base = job_dir / "svg"
                if svg_base.exists():
                    contents = list(svg_base.iterdir())
                    logger.warning(f"Contents of {svg_base}: {contents}")
                svg_b64 = None
        except subprocess.CalledProcessError as e:
            logger.error(f"SVG generation subprocess failed for output_{i}")
            logger.error(f"Return code: {e.returncode}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            svg_b64 = None
        except Exception as ex:
            logger.exception(f"Unexpected error generating SVG for output_{i}: {ex}")
            svg_b64 = None

        payload_images.append({
            "index": i,
            "seed": int(seeds_list[i]),
            "svg_b64": svg_b64,
        })

    first = imgs[0]
    rel = job_dir.relative_to(VOL)
    return {
        "job_dir": str(rel).replace("\\", "/"),
        "width": first.width,
        "height": first.height,
        "num_images": num_images,
        "images": payload_images,
    }


runpod.serverless.start({"handler": handler})
