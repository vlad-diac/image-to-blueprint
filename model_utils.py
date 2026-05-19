"""
model_utils.py — explicit-component loaders for Qwen-Image-Edit-2511 inference.

Each pipeline component (transformer, VAE, text encoder, tokenizer, processor) is
loaded by its own builder. A local path overrides HF download on a per-component
basis. `build_pipeline()` assembles them into a `QwenImageEditPlusPipeline`.

Per-component sources currently supported:

    transformer   : .gguf            (diffusers GGUFQuantizationConfig)
                    .safetensors     (diffusers from_single_file)
                    HF subfolder     (from_pretrained, when path is omitted)

    vae           : .safetensors     (AutoencoderKLQwenImage.from_single_file)
                    HF subfolder     (from_pretrained, when path is omitted)

    text_encoder  : .safetensors     (ComfyUI FP8-scaled, via fp8_loader.py)
                    HF subfolder     (from_pretrained, when path is omitted)

    tokenizer/    : HF subfolder only (tiny — no single-file alternative).
    processor       Falls back to default_repo / default_local.

For each component the "config source" (default_local or default_repo) is used
to fetch the small JSON config / tokenizer / processor files. No weight is ever
downloaded for a component that has a local `path` set.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Optional

import torch
from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    GGUFQuantizationConfig,
    QwenImageEditPlusPipeline,
    QwenImageTransformer2DModel,
)
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

from fp8_loader import load_qwen25vl_from_fp8_scaled

logger = logging.getLogger(__name__)

DEFAULT_REPO = "Qwen/Qwen-Image-Edit-2511"

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
DEFAULT_LORA_DIR = _PROJECT_ROOT / "models" / "loras"


# ---------------------------------------------------------------------------
# Scheduler (Lightning 4-step config)
# ---------------------------------------------------------------------------

def build_lightning_scheduler() -> FlowMatchEulerDiscreteScheduler:
    """
    FlowMatchEulerDiscreteScheduler tuned for the Qwen-Image-Edit-2511 Lightning
    4-step LoRA. The distillation used AuraFlow shift=3, which maps to
    base_shift = max_shift = log(3) in diffusers' exponential dynamic shifting.
    """
    config = {
        "base_image_seq_len": 256,
        "max_image_seq_len": 8192,
        "base_shift": math.log(3),
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_dynamic_shifting": True,
        "use_beta_sigmas": False,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    return FlowMatchEulerDiscreteScheduler.from_config(config)


# ---------------------------------------------------------------------------
# Config-source resolution
# ---------------------------------------------------------------------------

def _resolve_config_source(default_repo: str, default_local: Optional[str | Path]) -> str:
    """
    Pick the source used for small config / tokenizer / processor files.

    Local snapshot is used when present; otherwise the HF repo ID.
    Returns a string suitable to pass as the first argument of
    `from_pretrained(...)`.
    """
    if default_local and Path(default_local).is_dir():
        resolved = str(Path(default_local).resolve())
        logger.info("Config source: local snapshot %s", resolved)
        return resolved
    if default_local:
        logger.warning(
            "default_local '%s' not found — falling back to HF repo '%s'",
            default_local, default_repo,
        )
    logger.info("Config source: HF repo %s", default_repo)
    return default_repo


def _load_subfolder_config(config_source: str, subfolder: str) -> dict[str, Any]:
    """
    Fetch a diffusers component's `<subfolder>/config.json` either from a local
    snapshot directory or from a HuggingFace repo, and return it as a dict
    with diffusers-internal bookkeeping keys removed (so it can be fed straight
    into `<Model>.from_config(...)`).
    """
    candidate = Path(config_source) / subfolder / "config.json"
    if candidate.exists():
        path = candidate
        logger.info("Loading %s/config.json from local: %s", subfolder, path)
    else:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(repo_id=config_source, filename=f"{subfolder}/config.json"))
        logger.info("Loading %s/config.json from HF: %s", subfolder, path)

    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)
    cfg.pop("_name_or_path", None)
    return cfg


# ---------------------------------------------------------------------------
# Per-component builders
# ---------------------------------------------------------------------------

def build_transformer(
    path: Optional[str | Path],
    config_source: str,
    dtype: torch.dtype = torch.bfloat16,
    attention_backend: Optional[str] = None,
) -> QwenImageTransformer2DModel:
    if path is None:
        logger.info("Transformer source: HF subfolder of %s", config_source)
        transformer = QwenImageTransformer2DModel.from_pretrained(
            config_source, subfolder="transformer", torch_dtype=dtype,
        )
    else:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Transformer file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".gguf":
            logger.info("Transformer source: GGUF %s", path)
            transformer = QwenImageTransformer2DModel.from_single_file(
                str(path),
                quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                torch_dtype=dtype,
                config=config_source,
                subfolder="transformer",
            )
        elif suffix == ".safetensors":
            logger.info("Transformer source: single safetensors %s", path)
            transformer = QwenImageTransformer2DModel.from_single_file(
                str(path),
                torch_dtype=dtype,
                config=config_source,
                subfolder="transformer",
            )
        else:
            raise ValueError(
                f"Unsupported transformer file extension: {suffix!r} "
                "(must be .gguf or .safetensors)"
            )

    if attention_backend is not None:
        transformer.set_attention_backend(attention_backend)
        logger.info("Transformer attention backend: %s", attention_backend)

    return transformer


def build_vae(
    path: Optional[str | Path],
    config_source: str,
    dtype: torch.dtype = torch.bfloat16,
) -> AutoencoderKLQwenImage:
    """
    Build an AutoencoderKLQwenImage.

    `AutoencoderKLQwenImage` is not in diffusers' `from_single_file` registry
    (despite inheriting the mixin), so the single-file path is implemented
    manually:

      1. Fetch `vae/config.json` (small).
      2. Run the file's checkpoint through `convert_wan_vae_to_diffusers`
         from `diffusers.loaders.single_file_utils`. The Qwen image VAE shares
         the Wan VAE architecture, so the Wan converter remaps the ComfyUI
         `encoder.downsamples.N.residual.M.*` / `decoder.upsamples.N.*` /
         `encoder.middle.*` / `decoder.middle.*` keys into the diffusers
         `encoder.down_blocks.N.resnets.M.*` / `up_blocks.N.*` /
         `mid_block.resnets.*` naming.
      3. Build the model from config (meta-init) and `load_state_dict`
         the remapped checkpoint via `assign=True`.
    """
    if path is None:
        logger.info("VAE source: HF subfolder of %s", config_source)
        return AutoencoderKLQwenImage.from_pretrained(
            config_source, subfolder="vae", torch_dtype=dtype,
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"VAE file not found: {path}")
    if path.suffix.lower() != ".safetensors":
        raise ValueError(f"VAE must be a .safetensors file, got: {path.suffix}")

    import safetensors.torch as st
    from accelerate import init_empty_weights
    from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers

    logger.info("VAE source: single safetensors %s", path)
    config = _load_subfolder_config(config_source, "vae")
    with init_empty_weights():
        vae = AutoencoderKLQwenImage.from_config(config)

    raw_sd = st.load_file(str(path))
    logger.info("VAE raw state dict: %d tensors loaded.", len(raw_sd))

    converted_sd = convert_wan_vae_to_diffusers(raw_sd)
    logger.info(
        "VAE converted state dict: %d tensors (Wan → diffusers key remap).",
        len(converted_sd),
    )

    missing, unexpected = vae.load_state_dict(converted_sd, strict=False, assign=True)
    if unexpected:
        logger.warning("VAE unexpected keys (%d, first 5): %s", len(unexpected), unexpected[:5])
    if missing:
        logger.warning("VAE missing keys (%d, first 5): %s", len(missing), missing[:5])

    return vae.to(dtype)


def build_text_encoder(
    path: Optional[str | Path],
    fmt: str,
    config_source: str,
    dtype: torch.dtype = torch.bfloat16,
) -> Qwen2_5_VLForConditionalGeneration:
    if path is None:
        logger.info("Text encoder source: HF subfolder of %s", config_source)
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config_source, subfolder="text_encoder", torch_dtype=dtype,
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Text encoder file not found: {path}")

    fmt = (fmt or "fp8_scaled").lower()
    if fmt == "fp8_scaled":
        return load_qwen25vl_from_fp8_scaled(
            path, config_source=config_source, compute_dtype=dtype,
            lazy=True,  # keep weights as FP8 on GPU, dequantize per-layer during forward
        )
    raise ValueError(
        f"Unsupported text_encoder format: {fmt!r} "
        "(supported: 'fp8_scaled' for single-file local weights)"
    )


def build_tokenizer(config_source: str):
    logger.info("Tokenizer source: %s", config_source)
    return AutoTokenizer.from_pretrained(config_source, subfolder="tokenizer")


def build_processor(config_source: str):
    logger.info("Processor source: %s", config_source)
    return AutoProcessor.from_pretrained(config_source, subfolder="processor")


# ---------------------------------------------------------------------------
# Top-level pipeline assembly
# ---------------------------------------------------------------------------

def build_pipeline(
    components: dict[str, Any],
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    enable_offload: bool = False,
    compile_text_encoder: bool = False,
    attention_backend: Optional[str] = None,
) -> QwenImageEditPlusPipeline:
    """
    Assemble a `QwenImageEditPlusPipeline` from an explicit-component config dict.

    `components` schema:

        {
            "default_repo":  "Qwen/Qwen-Image-Edit-2511",        # HF repo for configs / fallback
            "default_local": "/path/to/local/snapshot" | None,   # optional local snapshot
            "transformer":   {"path": str | None},
            "vae":           {"path": str | None},
            "text_encoder":  {"path": str | None, "format": "fp8_scaled" | None},
        }

    For each of {transformer, vae, text_encoder}: when `path` is None, the
    component is downloaded from `default_local` if present else `default_repo`.
    Tokenizer and processor always come from the config source.

    `compile_text_encoder`: when True, applies torch.compile to the text encoder
    after loading. This fuses the FP8-cast + scale + matmul per-layer into a
    single CUDA kernel, eliminating intermediate BF16 buffers and CPU dispatch gaps
    that cause GPU utilization to drop between layers.

    `attention_backend`: diffusers attention backend for the transformer
    (e.g. `_native_flash`, `flash`, `_flash_3_hub`). None keeps the default.
    """
    if device == "cuda":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

    default_repo = components.get("default_repo") or DEFAULT_REPO
    default_local = components.get("default_local")
    config_source = _resolve_config_source(default_repo, default_local)

    tx_cfg = components.get("transformer") or {}
    vae_cfg = components.get("vae") or {}
    te_cfg = components.get("text_encoder") or {}

    transformer = build_transformer(
        path=tx_cfg.get("path"),
        config_source=config_source,
        dtype=dtype,
        attention_backend=attention_backend,
    )
    vae = build_vae(
        path=vae_cfg.get("path"),
        config_source=config_source,
        dtype=dtype,
    )
    text_encoder = build_text_encoder(
        path=te_cfg.get("path"),
        fmt=te_cfg.get("format", "fp8_scaled"),
        config_source=config_source,
        dtype=dtype,
    )
    tokenizer = build_tokenizer(config_source)
    processor = build_processor(config_source)
    scheduler = build_lightning_scheduler()

    logger.info("Assembling QwenImageEditPlusPipeline from explicit components.")
    pipe = QwenImageEditPlusPipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        processor=processor,
        transformer=transformer,
    )

    if enable_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    if compile_text_encoder and device == "cuda":
        # Use 'aot_eager' backend: eliminates Python per-kernel dispatch overhead
        # without requiring Triton (which is Linux-only and absent on Windows).
        # On Linux with a full Triton install, switch backend to 'inductor' for
        # full kernel fusion and even better GPU utilization.
        backend = "aot_eager"
        logger.info(
            "torch.compile(text_encoder, backend=%r) — "
            "first forward will trigger a one-time JIT trace.",
            backend,
        )
        try:
            pipe.text_encoder = torch.compile(pipe.text_encoder, backend=backend)
        except Exception as exc:
            logger.warning(
                "torch.compile failed (%s: %s) — continuing without compilation.",
                type(exc).__name__, exc,
            )

    return pipe


# ---------------------------------------------------------------------------
# LoRA key remapping (ComfyUI → diffusers)
# ---------------------------------------------------------------------------

_COMFYUI_LORA_PREFIX = "diffusion_model."
_DIFFUSERS_LORA_PREFIX = "transformer."


def remap_lora_keys_if_needed(lora_path: Path) -> Optional[Path]:
    """
    Inspect a LoRA's first key. If it uses the ComfyUI prefix `diffusion_model.*`,
    rewrite all keys to `transformer.*` and save alongside the original with the
    suffix `_diffusers_keys.safetensors`. Returns the new path, or None if no
    remap was needed.
    """
    try:
        import safetensors.torch as st
    except ImportError:
        logger.warning("safetensors not installed — skipping LoRA key inspection.")
        return None

    sd = st.load_file(str(lora_path))
    first_key = next(iter(sd), "")

    if not first_key.startswith(_COMFYUI_LORA_PREFIX):
        return None

    logger.info(
        "LoRA '%s' uses ComfyUI key prefix — remapping to diffusers format.",
        lora_path.name,
    )
    remapped = {
        _DIFFUSERS_LORA_PREFIX + k[len(_COMFYUI_LORA_PREFIX):]: v
        for k, v in sd.items()
    }

    tmp_path = lora_path.parent / (lora_path.stem + "_diffusers_keys.safetensors")
    st.save_file(remapped, str(tmp_path))
    logger.info("Saved remapped LoRA to: %s", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# HF cache cleanup
# ---------------------------------------------------------------------------

def clear_models(
    model_ids: list[str] | None = None,
    remove_lora_remaps: bool = True,
) -> None:
    """
    Delete cached HuggingFace model weights and optional LoRA remap temp files.

    Args:
        model_ids:          List of HF repo IDs to purge from the local cache.
                            Defaults to [DEFAULT_REPO].
        remove_lora_remaps: When True, also deletes the *_diffusers_keys.safetensors
                            temp files written by remap_lora_keys_if_needed().
    """
    if model_ids is None:
        model_ids = [DEFAULT_REPO]

    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        logger.error(
            "huggingface_hub is required for clear_models(). "
            "Install it with: pip install huggingface_hub"
        )
        return

    cache_info = scan_cache_dir()
    total_freed = 0

    for model_id in model_ids:
        matched = [r for r in cache_info.repos if r.repo_id == model_id]
        if not matched:
            logger.info("Model not found in HF cache: %s", model_id)
            continue

        for repo in matched:
            commit_hashes = [rev.commit_hash for rev in repo.revisions]
            if not commit_hashes:
                continue
            strategy = cache_info.delete_revisions(*commit_hashes)
            freed = strategy.expected_freed_size
            strategy.execute()
            total_freed += freed
            logger.info("Cleared %s — freed %.1f MB", model_id, freed / 1024 / 1024)

    if remove_lora_remaps:
        remapped = list(DEFAULT_LORA_DIR.glob("*_diffusers_keys.safetensors"))
        for f in remapped:
            f.unlink()
            logger.info("Deleted remapped LoRA: %s", f.name)
        if remapped:
            logger.info("Removed %d remapped LoRA temp file(s)", len(remapped))

    logger.info("Total freed: %.1f MB", total_freed / 1024 / 1024)
