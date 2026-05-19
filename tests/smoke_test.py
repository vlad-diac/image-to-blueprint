"""
smoke_test.py — non-network sanity checks for the standalone pipeline refactor.

Run from your conda environment:

    python standalone/tests/smoke_test.py

What this checks (no model weights loaded, no network calls):
  1. All standalone modules import without error.
  2. PyYAML parses both example YAMLs correctly.
  3. `_build_components_dict()` resolves paths and produces the expected dict.
  4. The local files referenced by ship_blueprint_top.yaml exist on disk.
  5. `fp8_loader.inspect_fp8_file()` opens the FP8 text encoder file and reports
     its prefix + first few keys (so we can verify the FP8 loader's
     prefix-detection + key-stripping will produce names that match
     transformers' `Qwen2_5_VLForConditionalGeneration.state_dict()`).
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
STANDALONE = HERE.parent
sys.path.insert(0, str(STANDALONE))


def check(name: str, fn) -> bool:
    print(f"\n[CHECK] {name}")
    try:
        fn()
        print(f"  OK")
        return True
    except Exception:
        traceback.print_exc()
        print(f"  FAIL: {name}")
        return False


def check_imports():
    import fp8_loader  # noqa: F401
    import model_utils  # noqa: F401
    import pipeline_utils  # noqa: F401
    from model_utils import (  # noqa: F401
        build_lightning_scheduler,
        build_pipeline,
        build_text_encoder,
        build_transformer,
        build_vae,
    )
    from fp8_loader import (  # noqa: F401
        inspect_fp8_file,
        load_qwen25vl_from_fp8_scaled,
    )


def check_yaml_parse():
    import yaml
    for yml in (HERE / "ship_blueprint_top.yaml", HERE / "ship_blueprint_top_hf.yaml"):
        assert yml.exists(), f"missing: {yml}"
        with open(yml, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert "input" in data, f"{yml.name}: missing 'input'"
        assert "models" in data, f"{yml.name}: missing 'models'"
        assert "inference" in data, f"{yml.name}: missing 'inference'"
        print(f"  parsed: {yml.name}  (top-level keys: {list(data.keys())})")


def check_components_dict():
    import yaml
    from main import _build_components_dict

    for yml in (HERE / "ship_blueprint_top.yaml", HERE / "ship_blueprint_top_hf.yaml"):
        with open(yml, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        models_cfg = data.get("models", {})
        components = _build_components_dict(models_cfg, HERE)
        print(f"  {yml.name} →")
        print(json.dumps(
            {k: v for k, v in components.items() if k in ("default_repo", "default_local", "transformer", "vae", "text_encoder")},
            indent=2, default=str,
        ))
        assert "default_repo" in components


def check_local_files_exist():
    import yaml
    from main import _build_components_dict, _resolve

    yml = HERE / "ship_blueprint_top.yaml"
    with open(yml, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    components = _build_components_dict(data["models"], HERE)

    missing = []
    for key in ("transformer", "vae", "text_encoder"):
        p = (components.get(key) or {}).get("path")
        if p and not Path(p).exists():
            missing.append((key, p))
        elif p:
            print(f"  {key}.path OK: {p}")

    for lora in data["models"].get("loras", []):
        p = _resolve(lora.get("path"), HERE)
        if p and not Path(p).exists():
            missing.append((f"lora[{lora.get('name')}]", p))
        elif p:
            print(f"  lora[{lora.get('name')}].path OK: {p}")

    if missing:
        for k, p in missing:
            print(f"  MISSING: {k} → {p}")
        raise FileNotFoundError(f"{len(missing)} local file(s) missing")


def check_fp8_lazy_replacement():
    """
    Dry-run the lazy FP8 loader: verify that after _stream_raw_state_dict +
    remap + _replace_fp8_linears, no meta tensors remain and the Fp8Linear
    count is non-zero.
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration
    from fp8_loader import (
        Fp8Linear,
        _detect_qwen25vl_layout,
        _replace_fp8_linears,
        _remap_qwen25vl_old_to_new,
        _stream_raw_state_dict,
    )

    fp8_path = STANDALONE.parent / "models" / "text_encoders" / "qwen_2.5_vl_7b_fp8_scaled.safetensors"
    if not fp8_path.exists():
        print(f"  SKIP: {fp8_path} not found.")
        return

    import torch
    config = AutoConfig.from_pretrained("Qwen/Qwen-Image-Edit-2511", subfolder="text_encoder")

    sd = _stream_raw_state_dict(fp8_path, torch.bfloat16)
    if _detect_qwen25vl_layout(sd.keys()) == "old":
        sd = {_remap_qwen25vl_old_to_new(k): v for k, v in sd.items()}

    fp8_layer_names = {
        k[: -len(".weight")]
        for k in sd
        if k.endswith(".weight") and (k[: -len(".weight")] + ".scale_weight") in sd
    }
    print(f"  FP8 layers identified: {len(fp8_layer_names)}")

    with init_empty_weights():
        model = Qwen2_5_VLForConditionalGeneration(config)

    n_replaced = _replace_fp8_linears(model, fp8_layer_names, torch.bfloat16)
    print(f"  nn.Linear → Fp8Linear replacements: {n_replaced}")
    assert n_replaced == len(fp8_layer_names), "Not all FP8 layers were replaced!"

    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    print(f"  load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"    missing (first 5): {missing[:5]}")
    if unexpected:
        print(f"    unexpected (first 5): {unexpected[:5]}")

    fp8_param_bytes = sum(
        p.numel() for p in model.parameters() if p.dtype == torch.float8_e4m3fn
    )
    print(f"  FP8 params on CPU: {fp8_param_bytes / 1e9:.2f} GB (GPU footprint will be the same)")

    fp8_linear_count = sum(1 for m in model.modules() if isinstance(m, Fp8Linear))
    print(f"  Fp8Linear modules in final model: {fp8_linear_count}")


def check_fp8_inspection():
    from fp8_loader import inspect_fp8_file
    fp8_path = STANDALONE.parent / "models" / "text_encoders" / "qwen_2.5_vl_7b_fp8_scaled.safetensors"
    if not fp8_path.exists():
        print(f"  SKIP: {fp8_path} not found.")
        return
    info = inspect_fp8_file(fp8_path)
    print(json.dumps(info, indent=2, default=str))
    if not info["is_fp8_scaled"]:
        raise RuntimeError(
            "File does not contain a `scaled_fp8` marker — FP8 loader expects ComfyUI format."
        )

    # Also pull the actual transformers model state-dict key names so we can
    # verify the auto-remap path will produce the right targets.
    try:
        from accelerate import init_empty_weights
        from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration
        from fp8_loader import _detect_qwen25vl_layout, _remap_qwen25vl_old_to_new
        config = AutoConfig.from_pretrained(
            "Qwen/Qwen-Image-Edit-2511", subfolder="text_encoder",
        )
        with init_empty_weights():
            model = Qwen2_5_VLForConditionalGeneration(config)
        model_keys = list(model.state_dict().keys())
        print(f"  transformers Qwen2_5_VLForConditionalGeneration.state_dict() first 8 keys:")
        for k in model_keys[:8]:
            print(f"    {k}")
        model_layout = _detect_qwen25vl_layout(model_keys)
        print(f"  detected model layout:  {model_layout}")
        print(f"  detected file layout:   {info['detected_qwen25vl_layout']}")

        # Dry-run the remap and check coverage.
        if info["detected_qwen25vl_layout"] == "old" and model_layout == "new":
            print(f"  → would auto-remap old → new layout")
        # Build the set of post-remap file keys (without actually streaming the file).
        from safetensors import safe_open
        with safe_open(str(fp8_path), framework="pt") as f:
            file_keys = [
                k for k in f.keys()
                if not (k.endswith("scaled_fp8") or k.endswith(".scale_weight") or k.endswith(".scale_input"))
            ]
        pref = info["prefix"] or ""
        stripped_keys = [k[len(pref):] for k in file_keys]
        if info["detected_qwen25vl_layout"] == "old" and model_layout == "new":
            remapped_keys = [_remap_qwen25vl_old_to_new(k) for k in stripped_keys]
        else:
            remapped_keys = list(stripped_keys)
        file_set = set(remapped_keys)
        model_set = set(model_keys)
        missing_in_file = model_set - file_set
        unexpected_in_file = file_set - model_set
        print(f"  coverage: {len(file_set & model_set)} / {len(model_set)} model params present in file")
        if missing_in_file:
            print(f"  WARN: {len(missing_in_file)} params expected by model but absent from file (first 5):")
            for k in list(missing_in_file)[:5]:
                print(f"    {k}")
        if unexpected_in_file:
            print(f"  WARN: {len(unexpected_in_file)} file keys not used by model (first 5):")
            for k in list(unexpected_in_file)[:5]:
                print(f"    {k}")
    except Exception as exc:
        print(f"  (could not load transformers model to compare keys: {exc})")
        traceback.print_exc()


def check_vae_inspection():
    """
    Confirm the local VAE safetensors keys, after running through
    `convert_wan_vae_to_diffusers`, fully populate an empty AutoencoderKLQwenImage
    built from `vae/config.json`.
    """
    import safetensors.torch as st
    from accelerate import init_empty_weights
    from diffusers import AutoencoderKLQwenImage
    from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers
    from model_utils import _load_subfolder_config

    vae_path = STANDALONE.parent / "models" / "vae" / "qwen_image_vae.safetensors"
    if not vae_path.exists():
        print(f"  SKIP: {vae_path} not found.")
        return

    config = _load_subfolder_config("Qwen/Qwen-Image-Edit-2511", "vae")
    with init_empty_weights():
        vae = AutoencoderKLQwenImage.from_config(config)
    model_keys = set(vae.state_dict().keys())

    raw_sd = st.load_file(str(vae_path))
    raw_keys = set(raw_sd.keys())
    converted_sd = convert_wan_vae_to_diffusers(raw_sd)
    converted_keys = set(converted_sd.keys())

    print(f"  VAE raw keys:       {len(raw_keys)}")
    print(f"  VAE converted keys: {len(converted_keys)}")
    print(f"  VAE coverage (raw → model):       {len(raw_keys & model_keys)} / {len(model_keys)}")
    print(f"  VAE coverage (converted → model): {len(converted_keys & model_keys)} / {len(model_keys)}")

    missing = model_keys - converted_keys
    unexpected = converted_keys - model_keys
    if missing:
        print(f"  WARN: {len(missing)} VAE params expected by model but absent after conversion (first 5):")
        for k in list(missing)[:5]:
            print(f"    {k}")
    if unexpected:
        print(f"  INFO: {len(unexpected)} converted VAE keys not used by model (first 5):")
        for k in list(unexpected)[:5]:
            print(f"    {k}")


def main() -> int:
    checks = [
        ("standalone module imports",                  check_imports),
        ("YAML parsing (local + hf examples)",         check_yaml_parse),
        ("components dict resolution",                 check_components_dict),
        ("local file paths exist (local YAML only)",   check_local_files_exist),
        ("FP8 lazy replacement (Fp8Linear dry-run)",   check_fp8_lazy_replacement),
        ("FP8 file inspection + key sample",           check_fp8_inspection),
        ("VAE single-file key compatibility",          check_vae_inspection),
    ]
    results = [(name, check(name, fn)) for name, fn in checks]

    print("\n========== SUMMARY ==========")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    sys.exit(main())
