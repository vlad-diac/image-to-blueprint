"""
main.py — YAML-driven CLI runner for standalone Qwen-Image-Edit-2511 inference.

The pipeline is always assembled from explicit components. Each component
(transformer, VAE, text encoder) may either point at a local file or be omitted
to fall back to a HuggingFace download (or a local HF snapshot).

Usage:
  python standalone/main.py tests/ship_blueprint_top.yaml
  python standalone/main.py tests/ship_blueprint_top.yaml --seed 99 --steps 4
  python standalone/main.py tests/ship_blueprint_top.yaml --output out.png --offload

CLI overrides: --seed, --steps, --cfg, --output, --device, --dtype, --offload, --compile
All other settings come from YAML.

YAML schema (see standalone/tests/ship_blueprint_top.yaml for a full example):

  input:  path/to/input.png
  output: path/to/output.png

  models:
    # HF repo used for small config / tokenizer / processor files, and as the
    # default for any component below that has no `path` set.
    default_repo:  Qwen/Qwen-Image-Edit-2511
    # default_local: path/to/local/Qwen--Qwen-Image-Edit-2511   # optional snapshot

    transformer:
      path: path/to/qwen-image-edit-2511.gguf            # .gguf | .safetensors | (omit → HF download)

    vae:
      path: path/to/qwen_image_vae.safetensors           # .safetensors  | (omit → HF download)

    text_encoder:
      path: path/to/qwen_2.5_vl_7b_fp8_scaled.safetensors  # ComfyUI FP8-scaled | (omit → HF download)
      format: fp8_scaled                                   # only single-file format supported today

    loras:
      - name: my-lora
        path: path/to/lora.safetensors
        weight: 1.0
        bypass: false

  inference:
    positive_prompt: |
      <sks> front view ...
    negative_prompt: ""
    sampler:   euler         # (currently informational — scheduler is fixed for Lightning)
    scheduler: simple
    steps:     4
    cfg:       1.0
    denoise:   1.0
    seed:      randomize     # integer or "randomize"

  hardware:
    device:   cuda
    dtype:    bfloat16
    offload:  false
    compile:  false   # torch.compile text encoder (aot_eager backend, no Triton required)
                      # reduces CPU dispatch gaps between FP8 layers; works on Windows
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline_utils import QwenEditPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


# ---------------------------------------------------------------------------
# YAML loading + path resolution
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        sys.exit("PyYAML is required.\nInstall it with:  pip install PyYAML")
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve(val: str | None, base: Path) -> str | None:
    """Resolve a path string relative to `base`. Returns None if val is None."""
    if val is None:
        return None
    p = Path(val)
    return str(p if p.is_absolute() else (base / p).resolve())


def _parse_seed(raw: Any) -> int | None:
    if raw is None or str(raw).strip().lower() == "randomize":
        return None
    return int(raw)


def _build_components_dict(models_cfg: dict, base_dir: Path) -> dict[str, Any]:
    """Resolve YAML 'models' block into the components dict for build_pipeline()."""
    components: dict[str, Any] = {
        "default_repo": models_cfg.get("default_repo", "Qwen/Qwen-Image-Edit-2511"),
        "default_local": _resolve(models_cfg.get("default_local"), base_dir),
    }
    for key in ("transformer", "vae", "text_encoder"):
        sub = models_cfg.get(key) or {}
        entry: dict[str, Any] = {"path": _resolve(sub.get("path"), base_dir)}
        if "format" in sub:
            entry["format"] = sub["format"]
        components[key] = entry
    return components


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone Qwen-Image-Edit-2511 inference (YAML-driven).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "config",
        metavar="YAML",
        help="Path to the YAML config file.",
    )
    parser.add_argument("--seed",    type=int,   default=None, metavar="N",
                        help="RNG seed (overrides YAML; default: randomize).")
    parser.add_argument("--steps",   type=int,   default=None, metavar="N",
                        help="Denoising steps (overrides YAML).")
    parser.add_argument("--cfg",     type=float, default=None, metavar="FLOAT",
                        help="True-CFG scale (overrides YAML).")
    parser.add_argument("--output",  default=None, metavar="PATH",
                        help="Output file path (overrides YAML).")
    parser.add_argument("--device",  default=None, choices=["cuda", "cpu", "mps"],
                        help="Target device (overrides YAML).")
    parser.add_argument("--dtype",   default=None, choices=list(_DTYPE_MAP),
                        help="Compute dtype (overrides YAML).")
    parser.add_argument("--offload", action="store_true", default=False,
                        help="Enable CPU offloading to reduce peak VRAM.")
    parser.add_argument("--compile", action="store_true", default=False,
                        help="torch.compile the text encoder for better GPU utilization. "
                             "First forward triggers a one-time ~30–90 s JIT compilation.")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Config file not found: {config_path}")
    config_dir = config_path.resolve().parent

    data = _load_yaml(config_path)
    logger.info("Loaded config: %s", config_path.resolve())

    # ---- I/O ----
    input_str = _resolve(data.get("input"), config_dir)
    if not input_str:
        sys.exit("YAML must contain an 'input' key.")
    input_path = Path(input_str)
    if not input_path.exists():
        sys.exit(f"Input image not found: {input_path}")

    output_str = args.output or _resolve(data.get("output"), config_dir) or "output_edit.png"
    output_path = Path(output_str)

    # ---- models ----
    models_cfg = data.get("models", {})
    components = _build_components_dict(models_cfg, config_dir)
    loras_cfg: list[dict[str, Any]] = models_cfg.get("loras", [])

    # ---- inference ----
    inf_cfg = data.get("inference", {})
    positive_prompt = inf_cfg.get("positive_prompt", "")
    negative_prompt = inf_cfg.get("negative_prompt", "")
    if not positive_prompt:
        sys.exit("YAML inference.positive_prompt is required.")

    steps   = args.steps  if args.steps  is not None else int(inf_cfg.get("steps",   4))
    cfg_val = args.cfg    if args.cfg    is not None else float(inf_cfg.get("cfg",    1.0))
    denoise = float(inf_cfg.get("denoise", 1.0))
    seed    = args.seed   if args.seed   is not None else _parse_seed(inf_cfg.get("seed"))
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    # ---- hardware ----
    hw_cfg = data.get("hardware", {})
    device  = args.device  or hw_cfg.get("device",  "cuda")
    dtype_s = args.dtype   or hw_cfg.get("dtype",   "bfloat16")
    offload  = args.offload or bool(hw_cfg.get("offload", False))
    compile_ = args.compile or bool(hw_cfg.get("compile", False))
    dtype    = _DTYPE_MAP.get(dtype_s, torch.bfloat16)

    # ---- load image ----
    image = Image.open(input_path).convert("RGB")
    logger.info("Input: %s  (%dx%d)", input_path, *image.size)

    # ---- build pipeline ----
    pipeline = QwenEditPipeline().load(
        components=components,
        dtype=dtype,
        device=device,
        enable_offload=offload,
        compile_text_encoder=compile_,
    )

    for lora in loras_cfg:
        lora_path = _resolve(lora.get("path"), config_dir)
        if not lora_path:
            logger.warning("Lora entry missing 'path', skipping: %s", lora)
            continue
        pipeline.add_lora(
            path=lora_path,
            weight=float(lora.get("weight", 1.0)),
            name=lora.get("name"),
            bypass=bool(lora.get("bypass", False)),
        )

    # ---- run ----
    output_image = pipeline.run(
        image=image,
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        cfg=cfg_val,
        seed=seed,
        denoise=denoise,
    )

    # ---- save ----
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_image.save(str(output_path))
    logger.info("Saved: %s", output_path.resolve())


if __name__ == "__main__":
    main()
