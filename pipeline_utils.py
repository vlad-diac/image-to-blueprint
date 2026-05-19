"""
pipeline_utils.py — fluent QwenEditPipeline wrapper.

Wraps `model_utils.build_pipeline` (which constructs a `QwenImageEditPlusPipeline`
from explicit per-component configs) and adds LoRA management + inference.

Typical usage:

    pipe = (
        QwenEditPipeline()
        .load(
            components={
                "default_repo":  "Qwen/Qwen-Image-Edit-2511",
                "default_local": None,
                "transformer":   {"path": "models/unet/qwen-image-edit-2511-Q3_K_M.gguf"},
                "vae":           {"path": "models/vae/qwen_image_vae.safetensors"},
                "text_encoder":  {"path": "models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
                                  "format": "fp8_scaled"},
            },
            dtype=torch.bfloat16,
            device="cuda",
        )
        .add_lora("models/loras/qwen-image-edit-2511-multiple-angles-lora.safetensors",
                  weight=1.0, name="angles")
        .add_lora("models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
                  weight=1.0, name="lightning")
    )
    out = pipe.run(image, positive_prompt=prompt, steps=4, cfg=1.0)
"""

from __future__ import annotations

import gc
import logging
import random
import sys
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import build_pipeline, remap_lora_keys_if_needed

logger = logging.getLogger(__name__)


class QwenEditPipeline:
    """Fluent builder around `QwenImageEditPlusPipeline`."""

    def __init__(self) -> None:
        self._pipe = None
        self._device: str = "cuda"
        self._pending_loras: list[tuple[Path, str, float]] = []
        self._loaded_adapters: list[tuple[str, float]] = []

    def unload(self) -> None:
        """Release GPU/CPU memory held by the underlying diffusers pipeline."""
        pipe = self._pipe
        self._pipe = None
        self._pending_loras.clear()
        self._loaded_adapters.clear()
        if pipe is None:
            return

        for attr in ("transformer", "text_encoder", "vae", "tokenizer", "processor"):
            try:
                setattr(pipe, attr, None)
            except Exception:
                pass

        try:
            if hasattr(pipe, "to"):
                pipe.to("cpu")
        except Exception:
            pass

        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Pipeline unloaded and CUDA cache cleared.")

    # ------------------------------------------------------------------
    # Pipeline assembly
    # ------------------------------------------------------------------

    def load(
        self,
        components: dict[str, Any],
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        enable_offload: bool = False,
        compile_text_encoder: bool = False,
        attention_backend: Optional[str] = None,
    ) -> "QwenEditPipeline":
        """
        Build the underlying `QwenImageEditPlusPipeline` from an explicit-component
        config dict. See `model_utils.build_pipeline` for the `components` schema.

        `compile_text_encoder`: fuse FP8-cast + scale + matmul per-layer into a
        single CUDA kernel (torch.compile). Eliminates intermediate BF16 allocations
        and CPU dispatch gaps — significantly improves GPU utilization at the cost
        of a one-time ~30–90 s compilation on the first forward pass.
        """
        self._device = device
        self._pipe = build_pipeline(
            components=components,
            dtype=dtype,
            device=device,
            enable_offload=enable_offload,
            compile_text_encoder=compile_text_encoder,
            attention_backend=attention_backend,
        )
        return self

    # ------------------------------------------------------------------
    # LoRA adapters
    # ------------------------------------------------------------------

    def add_lora(
        self,
        path: str | Path,
        weight: float = 1.0,
        name: Optional[str] = None,
        bypass: bool = False,
    ) -> "QwenEditPipeline":
        """
        Register a LoRA adapter to be loaded on the first `run()` call.

        Args:
            path:   Path to a .safetensors LoRA file. ComfyUI-format keys
                    (diffusion_model.*) are remapped automatically.
            weight: Adapter scale applied via `set_adapters()` (default 1.0).
            name:   Logical adapter name. Defaults to the file stem.
            bypass: When True, the LoRA is silently skipped (no-op).
        """
        if bypass:
            return self

        lora_path = Path(path)
        adapter_name = name if name is not None else lora_path.stem
        self._pending_loras.append((lora_path, adapter_name, weight))
        return self

    def flush_loras(self) -> None:
        """Load all pending LoRA adapters into the pipeline."""
        self._flush_loras()

    def update_lora_weights(self, specs: list[tuple[str, float]]) -> None:
        """Update weights for already-loaded adapters (bypassed adapters are absent)."""
        if self._pipe is None or not self._loaded_adapters:
            return

        loaded_names = {n for n, _ in self._loaded_adapters}
        names = [n for n, w in specs if n and n in loaded_names]
        weights = [w for n, w in specs if n and n in loaded_names]
        if not names:
            return

        self._pipe.set_adapters(names, adapter_weights=weights)
        self._loaded_adapters = list(zip(names, weights))
        logger.info("Updated adapter weights: %s -> %s", names, weights)

    def _flush_loras(self) -> None:
        if not self._pending_loras:
            return

        for lora_path, adapter_name, weight in self._pending_loras:
            if not lora_path.exists():
                logger.warning("LoRA file not found, skipping: %s", lora_path)
                continue

            effective_path = remap_lora_keys_if_needed(lora_path) or lora_path

            try:
                self._pipe.load_lora_weights(
                    str(effective_path.parent),
                    weight_name=effective_path.name,
                    adapter_name=adapter_name,
                )
                self._loaded_adapters.append((adapter_name, weight))
                logger.info("Loaded LoRA '%s' from %s", adapter_name, effective_path.name)
            except Exception as exc:
                logger.warning(
                    "Failed to load LoRA '%s' (%s): %s — skipping.",
                    adapter_name, lora_path.name, exc,
                )

        self._pending_loras.clear()

        if self._loaded_adapters:
            names = [n for n, _ in self._loaded_adapters]
            weights = [w for _, w in self._loaded_adapters]
            self._pipe.set_adapters(names, adapter_weights=weights)
            logger.info("Active adapters: %s (weights: %s)", names, weights)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run(
        self,
        image: Image.Image,
        positive_prompt: str,
        negative_prompt: str = " ",
        steps: int = 4,
        cfg: float = 1.0,
        seed: Optional[int] = None,
        denoise: float = 1.0,
    ) -> Image.Image:
        """
        Run a single edit pass and return the output PIL image.

        Args:
            image:           Input PIL image (RGB).
            positive_prompt: Edit instruction.
            negative_prompt: Negative prompt — only active when cfg > 1.0.
            steps:           Number of denoising steps.
            cfg:             True-CFG scale. Values ≤ 1.0 disable CFG guidance.
            seed:            RNG seed. None → random.
            denoise:         Denoising strength (1.0 = full denoise). Currently
                             only used for logging; diffusers' QwenImageEditPlus
                             pipeline starts from full noise.
        """
        if self._pipe is None:
            raise RuntimeError("Pipeline not loaded. Call .load() before .run().")

        self._flush_loras()

        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        generator_device = "cpu" if self._device == "mps" else self._device
        generator = torch.Generator(device=generator_device).manual_seed(seed)

        effective_neg = negative_prompt if cfg > 1.0 else " "

        logger.info(
            "Inference: steps=%d  cfg=%.2f  denoise=%.2f  seed=%d  device=%s",
            steps, cfg, denoise, seed, self._device,
        )

        with torch.inference_mode():
            result = self._pipe(
                image=image,
                prompt=positive_prompt,
                negative_prompt=effective_neg,
                true_cfg_scale=cfg,
                num_inference_steps=steps,
                guidance_scale=None,
                generator=generator,
            )

        return result.images[0]
