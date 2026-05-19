"""
fp8_loader.py — load ComfyUI's FP8-scaled safetensors into a transformers Qwen2.5-VL model.

ComfyUI stores the Qwen2.5-VL text encoder weights as FP8 (float8_e4m3fn) with
per-tensor scale factors. The state-dict layout (per comfy/utils.py::convert_old_quants):

    <prefix>scaled_fp8                  → marker tensor
    <prefix>foo.bar.weight              → torch.float8_e4m3fn
    <prefix>foo.bar.scale_weight        → per-tensor scalar multiplier
    <prefix>foo.bar.scale_input         → optional input scale (always 1.0 here, dropped)

Two loading modes are supported (controlled by the `lazy` parameter):

EAGER (lazy=False) — dequantize at load time
    All FP8 weights are multiplied by their scale and cast to BF16 immediately.
    Simple, but stores the full 7B parameters as BF16 → ~14 GB on GPU.
    Numerically equivalent to the lazy path for inference.

LAZY (lazy=True, default) — dequantize at forward time  ← matches ComfyUI behaviour
    FP8 weights stay as float8_e4m3fn in GPU memory (~7 GB).
    nn.Linear layers that have FP8 weights are replaced with Fp8Linear, which
    dequantizes one layer at a time during the forward pass and immediately
    releases the temporary BF16 copy. Peak VRAM overhead is one layer (~100 MB)
    rather than the full 14 GB BF16 copy.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_FP8_MARKER_SUFFIX = "scaled_fp8"
_SCALE_WEIGHT_SUFFIX = ".scale_weight"
_SCALE_INPUT_SUFFIX = ".scale_input"


# ---------------------------------------------------------------------------
# Fp8Linear — on-the-fly dequantization
# ---------------------------------------------------------------------------

class Fp8Linear(nn.Module):
    """
    Drop-in replacement for nn.Linear that stores its weight as float8_e4m3fn
    and a per-tensor scale_weight buffer. On each forward pass the weight is
    dequantized to `compute_dtype`, the matmul runs, and the temporary BF16
    copy is released.

    This exactly mirrors ComfyUI's fp8_ops.Linear behaviour, cutting the text
    encoder's VRAM footprint from ~14 GB (BF16) to ~7 GB (FP8).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype

        # weight stays FP8; requires_grad=False (no training).
        # These are placeholder shapes — overwritten by load_state_dict(assign=True).
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.register_buffer("scale_weight", torch.ones((), dtype=torch.float32))
        if has_bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=compute_dtype),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast FP8 → compute dtype, then scale in-place to avoid a second allocation.
        # torch.compile fuses these two ops + F.linear into a single kernel,
        # eliminating the intermediate BF16 buffer entirely.
        w = self.weight.to(self.compute_dtype)
        w *= self.scale_weight.to(self.compute_dtype)
        return F.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, compute_dtype={self.compute_dtype}"
        )


# ---------------------------------------------------------------------------
# Qwen2.5-VL layout detection + old → new key remap
# ---------------------------------------------------------------------------
#
#   Old layout (≤ v4.51.3, what the ComfyUI FP8 file targets):
#       Qwen2_5_VLForConditionalGeneration
#         ├─ visual.*
#         ├─ model.embed_tokens.*
#         ├─ model.layers.*
#         └─ lm_head.*
#
#   New layout (≥ ~v4.52, current):
#       Qwen2_5_VLForConditionalGeneration
#         ├─ model.visual.*
#         ├─ model.language_model.embed_tokens.*
#         ├─ model.language_model.layers.*
#         └─ lm_head.*

def _detect_qwen25vl_layout(keys) -> str:
    keys = set(keys)
    if any(k.startswith("model.visual.") or k.startswith("model.language_model.") for k in keys):
        return "new"
    if any(k.startswith("visual.") or k.startswith("model.embed_tokens") or k.startswith("model.layers.") for k in keys):
        return "old"
    return "unknown"


def _remap_qwen25vl_old_to_new(key: str) -> str:
    if key.startswith("visual."):
        return "model." + key
    if key.startswith("model."):
        rest = key[len("model."):]
        head = rest.split(".", 1)[0]
        if head in ("visual", "language_model"):
            return key  # already new
        return "model.language_model." + rest
    return key


# ---------------------------------------------------------------------------
# FP8 prefix detection
# ---------------------------------------------------------------------------

def _detect_fp8_prefix(keys: list[str]) -> Optional[str]:
    candidates = [k for k in keys if k.endswith(_FP8_MARKER_SUFFIX)]
    if not candidates:
        return None
    if len(candidates) > 1:
        logger.warning("Multiple scaled_fp8 markers found: %s — using first.", candidates)
    return candidates[0][: -len(_FP8_MARKER_SUFFIX)]


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------

def inspect_fp8_file(fp8_path: str | Path, head: int = 8) -> dict:
    """
    Open a (possibly FP8-scaled) safetensors file and report its prefix,
    detected Qwen2.5-VL layout, dtype sample, and first N prefix-stripped keys.
    """
    import safetensors.torch as _st  # noqa: F401
    from safetensors import safe_open

    fp8_path = Path(fp8_path)
    if not fp8_path.exists():
        raise FileNotFoundError(fp8_path)

    with safe_open(str(fp8_path), framework="pt") as f:
        keys = list(f.keys())
        prefix = _detect_fp8_prefix(keys)
        dtypes: dict[str, int] = {}
        for k in keys[:64]:
            t = f.get_tensor(k)
            dt = str(t.dtype)
            dtypes[dt] = dtypes.get(dt, 0) + 1

    pref = prefix or ""
    stripped = [k[len(pref):] for k in keys if not k.endswith(_FP8_MARKER_SUFFIX)]
    layout = _detect_qwen25vl_layout(stripped)
    return {
        "path": str(fp8_path),
        "prefix": prefix,
        "is_fp8_scaled": prefix is not None,
        "n_tensors": len(keys),
        "detected_qwen25vl_layout": layout,
        "first_keys (prefix-stripped)": stripped[:head],
        "dtype_sample (first 64)": dtypes,
    }


# ---------------------------------------------------------------------------
# State-dict streaming — two modes
# ---------------------------------------------------------------------------

def _stream_dequantized_state_dict(
    fp8_path: Path,
    compute_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """
    EAGER: walk the file, dequantize every FP8 weight immediately
    (weight.to(compute_dtype) * scale_weight), strip prefix, return plain BF16 dict.
    """
    from safetensors import safe_open

    with safe_open(str(fp8_path), framework="pt") as f:
        all_keys = set(f.keys())
        prefix = _detect_fp8_prefix(list(all_keys))

        if prefix is None:
            logger.info("No `scaled_fp8` marker — treating file as plain safetensors.")
            return {k: f.get_tensor(k) for k in all_keys}

        marker_dtype = f.get_tensor(prefix + _FP8_MARKER_SUFFIX).dtype
        logger.info(
            "FP8 scaled prefix=%r  marker dtype=%s  target compute dtype=%s",
            prefix, marker_dtype, compute_dtype,
        )

        out: dict[str, torch.Tensor] = {}
        n_dequant = 0
        for k in all_keys:
            if k == prefix + _FP8_MARKER_SUFFIX:
                continue
            if k.endswith(_SCALE_WEIGHT_SUFFIX) or k.endswith(_SCALE_INPUT_SUFFIX):
                continue
            if not k.startswith(prefix):
                out[k] = f.get_tensor(k)
                continue

            rk = k[len(prefix):]
            t = f.get_tensor(k)
            scale_key = f"{prefix}{rk[:-len('.weight')]}{_SCALE_WEIGHT_SUFFIX}" if rk.endswith(".weight") else None

            if scale_key and scale_key in all_keys:
                scale = f.get_tensor(scale_key)
                t = t.to(compute_dtype) * scale.to(compute_dtype)
                n_dequant += 1
            elif t.dtype.is_floating_point and t.dtype != compute_dtype:
                t = t.to(compute_dtype)

            out[rk] = t

        logger.info("Dequantized %d FP8 weight tensors to %s.", n_dequant, compute_dtype)
        return out


def _stream_raw_state_dict(
    fp8_path: Path,
    compute_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """
    LAZY: walk the file without dequantizing FP8 weights. Returns:
      - FP8 weight tensors kept as float8_e4m3fn
      - scale_weight tensors as float32  (consumed by Fp8Linear.scale_weight)
      - all other tensors cast to compute_dtype
    input scales (.scale_input) are dropped — they are always 1.0 in this file.
    """
    from safetensors import safe_open

    with safe_open(str(fp8_path), framework="pt") as f:
        all_keys = set(f.keys())
        prefix = _detect_fp8_prefix(list(all_keys))

        if prefix is None:
            logger.info("No `scaled_fp8` marker — treating file as plain safetensors.")
            sd = {}
            for k in all_keys:
                t = f.get_tensor(k)
                sd[k] = t.to(compute_dtype) if t.dtype.is_floating_point else t
            return sd

        marker_dtype = f.get_tensor(prefix + _FP8_MARKER_SUFFIX).dtype
        logger.info(
            "FP8 scaled prefix=%r  marker dtype=%s  (lazy — keeping FP8 in memory)",
            prefix, marker_dtype,
        )

        # Which keys are FP8 weights (have a .scale_weight sibling)?
        fp8_weight_keys = {
            base + ".weight"
            for k in all_keys
            if k.endswith(_SCALE_WEIGHT_SUFFIX)
            for base in (k[:-len(_SCALE_WEIGHT_SUFFIX)],)
            if base + ".weight" in all_keys
        }

        out: dict[str, torch.Tensor] = {}
        for k in all_keys:
            if k == prefix + _FP8_MARKER_SUFFIX:
                continue
            if k.endswith(_SCALE_INPUT_SUFFIX):
                continue
            if not k.startswith(prefix):
                t = f.get_tensor(k)
                out[k] = t.to(compute_dtype) if t.dtype.is_floating_point else t
                continue

            rk = k[len(prefix):]
            t = f.get_tensor(k)

            if k in fp8_weight_keys:
                out[rk] = t                       # keep as float8_e4m3fn
            elif k.endswith(_SCALE_WEIGHT_SUFFIX):
                out[rk] = t.to(torch.float32)     # scale_weight stays f32
            elif t.dtype.is_floating_point:
                out[rk] = t.to(compute_dtype)     # biases, norms, embeddings → BF16
            else:
                out[rk] = t

        n_fp8 = sum(1 for v in out.values() if v.dtype == torch.float8_e4m3fn)
        logger.info("Kept %d tensors as FP8, rest cast to %s.", n_fp8, compute_dtype)
        return out


# ---------------------------------------------------------------------------
# Module replacement
# ---------------------------------------------------------------------------

def _replace_fp8_linears(
    model: nn.Module,
    fp8_layer_names: set[str],
    compute_dtype: torch.dtype,
) -> int:
    """
    Walk `model` and replace each nn.Linear whose full dotted name is in
    `fp8_layer_names` with an Fp8Linear of the same shape.
    Returns the number of modules replaced.
    """
    replaced = 0
    for full_name in fp8_layer_names:
        parts = full_name.rsplit(".", 1)
        if len(parts) == 1:
            parent, attr = model, full_name
        else:
            try:
                parent = model.get_submodule(parts[0])
            except AttributeError:
                logger.debug("Skipping %s: parent module not found.", full_name)
                continue
            attr = parts[1]

        orig = getattr(parent, attr, None)
        if not isinstance(orig, nn.Linear):
            logger.debug("Skipping %s: not an nn.Linear (%s).", full_name, type(orig))
            continue

        setattr(
            parent,
            attr,
            Fp8Linear(
                in_features=orig.in_features,
                out_features=orig.out_features,
                has_bias=orig.bias is not None,
                compute_dtype=compute_dtype,
            ),
        )
        replaced += 1

    return replaced


# ---------------------------------------------------------------------------
# Utility: materialize any remaining meta tensors after load_state_dict
# ---------------------------------------------------------------------------

def _materialize_meta_tensors(model: nn.Module, dtype: torch.dtype) -> int:
    n_fixed = 0
    for name, mod in model.named_modules():
        for pname, p in list(mod._parameters.items()):
            if p is not None and p.device.type == "meta":
                mod._parameters[pname] = nn.Parameter(torch.zeros(p.shape, dtype=dtype))
                logger.warning("Materialized missing parameter as zeros: %s.%s", name, pname)
                n_fixed += 1
        for bname, b in list(mod._buffers.items()):
            if b is not None and b.device.type == "meta":
                mod._buffers[bname] = torch.zeros(b.shape, dtype=dtype)
                logger.warning("Materialized missing buffer as zeros: %s.%s", name, bname)
                n_fixed += 1
    return n_fixed


# ---------------------------------------------------------------------------
# Main public entry point
# ---------------------------------------------------------------------------

def load_qwen25vl_from_fp8_scaled(
    fp8_path: str | Path,
    config_source: str,
    compute_dtype: torch.dtype = torch.bfloat16,
    lazy: bool = True,
):
    """
    Load a ComfyUI FP8-scaled safetensors file into a
    `transformers.Qwen2_5_VLForConditionalGeneration` model.

    Args:
        fp8_path:       Path to the ComfyUI scaled-FP8 .safetensors file.
        config_source:  HF repo ID or local snapshot dir — only config.json
                        is fetched (~5 KB, no weight download).
        compute_dtype:  Compute dtype for matmuls and non-FP8 tensors (BF16).
        lazy:           True (default) → keep weights as FP8 in memory and
                        dequantize per-layer during forward (matches ComfyUI,
                        ~7 GB VRAM for the text encoder).
                        False → dequantize everything at load time (~14 GB).

    Returns:
        A `Qwen2_5_VLForConditionalGeneration` on CPU, ready to be moved to
        the target device. When lazy=True, the Linear layers inside the model
        are Fp8Linear instances.
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration

    fp8_path = Path(fp8_path)
    if not fp8_path.exists():
        raise FileNotFoundError(f"FP8 scaled text encoder not found: {fp8_path}")

    logger.info("Loading text encoder config from: %s", config_source)
    config = AutoConfig.from_pretrained(config_source, subfolder="text_encoder")

    # ---- 1. Stream the state dict ----------------------------------------
    stream_fn = _stream_raw_state_dict if lazy else _stream_dequantized_state_dict
    mode = "lazy (FP8 in memory)" if lazy else "eager (pre-dequantized BF16)"
    logger.info("Streaming FP8 file [%s]: %s", mode, fp8_path)
    sd = stream_fn(fp8_path, compute_dtype)
    logger.info("Loaded %d tensors into state dict.", len(sd))

    # ---- 2. Detect and apply layout remap ---------------------------------
    file_layout = _detect_qwen25vl_layout(sd.keys())
    # Peek at the model layout without allocating real weights.
    with init_empty_weights():
        _probe = Qwen2_5_VLForConditionalGeneration(config)
    model_layout = _detect_qwen25vl_layout(_probe.state_dict().keys())
    del _probe
    logger.info("Qwen2.5-VL layouts → file: %s, model: %s", file_layout, model_layout)

    if file_layout == "old" and model_layout == "new":
        logger.info("Remapping state-dict keys: old → new Qwen2.5-VL layout.")
        sd = {_remap_qwen25vl_old_to_new(k): v for k, v in sd.items()}
    elif file_layout == "new" and model_layout == "old":
        logger.warning(
            "FP8 file uses new layout but transformers expects old. "
            "No automatic remap for this direction — load may fail."
        )

    # ---- 3. Build the model skeleton -------------------------------------
    logger.info("Instantiating Qwen2.5-VL on meta device.")
    with init_empty_weights():
        model = Qwen2_5_VLForConditionalGeneration(config)

    # ---- 4. (Lazy only) Replace nn.Linear → Fp8Linear for FP8 layers ----
    if lazy:
        # Collect the module names of FP8 layers from the (remapped) state dict.
        # A layer is FP8 if it has both a .weight key and a .scale_weight key.
        fp8_layer_names = {
            k[: -len(".weight")]
            for k in sd
            if k.endswith(".weight") and (k[: -len(".weight")] + ".scale_weight") in sd
        }
        n_replaced = _replace_fp8_linears(model, fp8_layer_names, compute_dtype)
        logger.info(
            "Replaced %d nn.Linear → Fp8Linear (FP8 weights will stay in GPU memory).",
            n_replaced,
        )

    # ---- 5. Load state dict ----------------------------------------------
    if sd:
        logger.info(
            "Example post-remap key: %r  (model expects keys like %r)",
            next(iter(sd)),
            next(iter(model.state_dict())),
        )

    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if unexpected:
        logger.warning("Unexpected keys (%d, first 5): %s", len(unexpected), unexpected[:5])
    if missing:
        logger.warning("Missing keys after load (%d, first 5): %s", len(missing), missing[:5])

    n_fixed = _materialize_meta_tensors(model, compute_dtype)
    if n_fixed:
        logger.warning(
            "%d meta tensors zero-filled — FP8 file may be incomplete.",
            n_fixed,
        )

    if lazy:
        logger.info(
            "Text encoder ready (lazy FP8). "
            "FP8 weights occupy ~%.1f GB; BF16 copy materialises only during forward.",
            sum(p.numel() for p in model.parameters() if p.dtype == torch.float8_e4m3fn) / 1e9,
        )

    return model
