"""
app.py — Gradio UI for YAML-driven Qwen-Image-Edit-2511 pipeline runs.

Launch:
  python app.py
  python app.py --listen          # bind 0.0.0.0 for remote access
"""

from __future__ import annotations

import argparse
import io
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from main import _DTYPE_MAP, _build_components_dict, _load_yaml, _resolve
from pipeline_utils import QwenEditPipeline

TESTS_DIR = ROOT / "tests"
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
MAX_LORAS = 5
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# ---------------------------------------------------------------------------
# Logging capture for the UI log textbox
# ---------------------------------------------------------------------------

class _LogBufferHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self._buffer = io.StringIO()

    def emit(self, record: logging.LogRecord) -> None:
        self._buffer.write(self.format(record) + "\n")

    def getvalue(self) -> str:
        return self._buffer.getvalue()

    def clear(self) -> None:
        self._buffer = io.StringIO()


_log_handler = _LogBufferHandler()
_log_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
)
logging.getLogger().addHandler(_log_handler)
logging.getLogger().setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_yaml_configs() -> list[str]:
    if not TESTS_DIR.is_dir():
        return []
    return sorted(p.name for p in TESTS_DIR.glob("*.yaml"))


def _list_input_images() -> list[str]:
    if not INPUT_DIR.is_dir():
        return []
    files = [
        p.name
        for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files)


def _component_label(sub: dict | None, default_repo: str, base_dir: Path) -> str:
    raw_path = (sub or {}).get("path")
    if raw_path:
        resolved = _resolve(raw_path, base_dir)
        fmt = (sub or {}).get("format")
        return f"{resolved}" + (f"  (format: {fmt})" if fmt else "")
    return f"HuggingFace: {default_repo}"


def _seed_ui_from_yaml(raw: Any) -> tuple[str, int | None]:
    if raw is None or str(raw).strip().lower() == "randomize":
        return "Randomized", None
    return "Fixed", int(raw)


def _empty_lora_updates() -> list[Any]:
    """Default hidden LoRA slot updates."""
    out: list[Any] = []
    for _ in range(MAX_LORAS):
        out.extend(
            [
                gr.update(visible=False),
                gr.update(value=""),
                gr.update(value=1.0),
                gr.update(value=False),
                gr.update(value=""),
            ]
        )
    return out


def _lora_slot_updates(
    loras: list[dict[str, Any]], config_dir: Path
) -> list[Any]:
    out: list[Any] = []
    for i in range(MAX_LORAS):
        if i < len(loras):
            entry = loras[i]
            name = entry.get("name") or Path(str(entry.get("path", "lora"))).stem
            path = _resolve(entry.get("path"), config_dir) or ""
            out.extend(
                [
                    gr.update(visible=True),
                    gr.update(value=name),
                    gr.update(value=float(entry.get("weight", 1.0))),
                    gr.update(value=bool(entry.get("bypass", False))),
                    gr.update(value=path),
                ]
            )
        else:
            out.extend(
                [
                    gr.update(visible=False),
                    gr.update(value=""),
                    gr.update(value=1.0),
                    gr.update(value=False),
                    gr.update(value=""),
                ]
            )
    return out


def load_pipeline_config(yaml_name: str | None) -> tuple[Any, ...]:
    """Populate all UI blocks from a tests/*.yaml file."""
    if not yaml_name:
        empty_images = _list_input_images()
        return (
            gr.update(choices=empty_images, value=empty_images[0] if empty_images else None),
            None,
            "",
            "",
            "",
            *_empty_lora_updates(),
            "",
            "",
            "Randomized",
            gr.update(value=None, interactive=False),
            4,
            1.0,
            1.0,
            None,
            gr.update(interactive=False, value="Initializing…"),
            None,
        )

    config_path = TESTS_DIR / yaml_name
    if not config_path.exists():
        raise gr.Error(f"Config not found: {config_path}")

    data = _load_yaml(config_path)
    config_dir = config_path.resolve().parent
    models_cfg = data.get("models", {})
    default_repo = models_cfg.get("default_repo", "Qwen/Qwen-Image-Edit-2511")
    inf_cfg = data.get("inference", {})

    transformer_lbl = _component_label(models_cfg.get("transformer"), default_repo, config_dir)
    vae_lbl = _component_label(models_cfg.get("vae"), default_repo, config_dir)
    te_lbl = _component_label(models_cfg.get("text_encoder"), default_repo, config_dir)

    loras_cfg: list[dict[str, Any]] = models_cfg.get("loras", [])
    lora_updates = _lora_slot_updates(loras_cfg, config_dir)

    seed_type, seed_val = _seed_ui_from_yaml(inf_cfg.get("seed"))
    seed_number_update = (
        gr.update(value=seed_val, interactive=True)
        if seed_type == "Fixed"
        else gr.update(value=None, interactive=False)
    )

    images = _list_input_images()
    yaml_input = _resolve(data.get("input"), config_dir)
    default_image: str | None = None
    if yaml_input:
        stem = Path(yaml_input).name
        if stem in images:
            default_image = stem
    if default_image is None and images:
        default_image = images[0]

    preview_path = str(INPUT_DIR / default_image) if default_image else None

    return (
        gr.update(choices=images, value=default_image),
        preview_path,
        transformer_lbl,
        vae_lbl,
        te_lbl,
        *lora_updates,
        inf_cfg.get("positive_prompt", ""),
        inf_cfg.get("negative_prompt", ""),
        seed_type,
        seed_number_update,
        int(inf_cfg.get("steps", 4)),
        float(inf_cfg.get("cfg", 1.0)),
        float(inf_cfg.get("denoise", 1.0)),
        str(config_path),
        gr.update(interactive=False, value="Initializing…"),
        None,
    )


def preview_input_image(image_name: str | None) -> str | None:
    if not image_name:
        return None
    path = INPUT_DIR / image_name
    return str(path) if path.exists() else None


def on_seed_type_change(seed_type: str) -> Any:
    if seed_type == "Fixed":
        return gr.update(interactive=True)
    return gr.update(value=None, interactive=False)


def on_bypass_pending() -> tuple[Any, Any, str]:
    return (
        gr.update(interactive=False, value="Initializing…"),
        None,
        "Reinitializing…",
    )


def initialize_models(
    config_path_str: str | None,
    *lora_bypasses: Any,
) -> tuple[Any, Any, str]:
    _log_handler.clear()
    logger = logging.getLogger(__name__)

    if not config_path_str:
        return (
            None,
            gr.update(interactive=False, value="Run Pipeline"),
            "No pipeline configuration loaded.",
        )

    config_path = Path(config_path_str)
    if not config_path.exists():
        return (
            None,
            gr.update(interactive=False, value="Run Pipeline"),
            f"Config file not found: {config_path}",
        )

    try:
        data = _load_yaml(config_path)
        config_dir = config_path.resolve().parent
        models_cfg = data.get("models", {})
        hw_cfg = data.get("hardware", {})

        device = hw_cfg.get("device", "cuda")
        dtype_s = hw_cfg.get("dtype", "bfloat16")
        offload = bool(hw_cfg.get("offload", False))
        compile_ = bool(hw_cfg.get("compile", False))
        attention_backend = hw_cfg.get("attention_backend")
        dtype = _DTYPE_MAP.get(dtype_s, torch.bfloat16)

        components = _build_components_dict(models_cfg, config_dir)
        loras_cfg: list[dict[str, Any]] = models_cfg.get("loras", [])

        logger.info("Initializing models from %s", config_path.name)
        pipeline = QwenEditPipeline().load(
            components=components,
            dtype=dtype,
            device=device,
            enable_offload=offload,
            compile_text_encoder=compile_,
            attention_backend=attention_backend,
        )

        for i, lora in enumerate(loras_cfg):
            lora_path = _resolve(lora.get("path"), config_dir)
            if not lora_path:
                continue
            bypass = bool(lora_bypasses[i]) if i < len(lora_bypasses) else False
            pipeline.add_lora(
                path=lora_path,
                weight=float(lora.get("weight", 1.0)),
                name=lora.get("name"),
                bypass=bypass,
            )

        pipeline.flush_loras()

        log_text = _log_handler.getvalue()
        status = "Models ready."
        if log_text:
            status = f"{status}\n\n{log_text}"
        return (
            pipeline,
            gr.update(interactive=True, value="Run Pipeline"),
            status,
        )

    except Exception as exc:
        logger.exception("Model initialization failed")
        log_text = _log_handler.getvalue() + f"\nERROR: {exc}"
        return (
            None,
            gr.update(interactive=False, value="Run Pipeline"),
            f"Init failed: {exc}\n\n{log_text}",
        )


def run_pipeline(
    pipeline: QwenEditPipeline | None,
    config_path_str: str | None,
    image_name: str | None,
    positive_prompt: str,
    negative_prompt: str,
    seed_type: str,
    seed_value: float | None,
    steps: float,
    cfg: float,
    denoise: float,
    # LoRA slots (name, weight, bypass, path) x MAX_LORAS
    *lora_fields: Any,
) -> tuple[Any, str]:
    _log_handler.clear()
    logger = logging.getLogger(__name__)

    if pipeline is None:
        return None, "Models not initialized. Load a configuration and wait for initialization."

    if not config_path_str:
        return None, "Load a pipeline configuration first."

    config_path = Path(config_path_str)
    if not config_path.exists():
        return None, f"Config file not found: {config_path}"

    if not image_name:
        return None, "Select an input image."

    input_path = INPUT_DIR / image_name
    if not input_path.exists():
        return None, f"Input image not found: {input_path}"

    if not positive_prompt.strip():
        return None, "Positive prompt is required."

    if seed_type == "Randomized":
        seed: int | None = None
    else:
        if seed_value is None:
            return None, "Enter a seed value when using Fixed seed type."
        seed = int(seed_value)
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    weight_specs: list[tuple[str, float]] = []
    for i in range(MAX_LORAS):
        base = i * 4
        name, weight, bypass, path = (
            lora_fields[base],
            float(lora_fields[base + 1] or 1.0),
            bool(lora_fields[base + 2]),
            lora_fields[base + 3],
        )
        if not path or bypass:
            continue
        adapter_name = name or Path(str(path)).stem
        weight_specs.append((adapter_name, weight))

    try:
        image = Image.open(input_path).convert("RGB")
        logger.info("Input: %s  (%dx%d)", input_path, *image.size)

        if weight_specs:
            pipeline.update_lora_weights(weight_specs)

        output_image = pipeline.run(
            image=image,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt or "",
            steps=int(steps),
            cfg=float(cfg),
            seed=seed,
            denoise=float(denoise),
        )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"{config_path.stem}-{Path(image_name).stem}-{stamp}.png"
        output_path = OUTPUT_DIR / out_name
        output_image.save(str(output_path))
        logger.info("Saved: %s", output_path.resolve())

        log_text = _log_handler.getvalue()
        return output_image, log_text

    except Exception as exc:
        logger.exception("Pipeline run failed")
        log_text = _log_handler.getvalue() + f"\nERROR: {exc}"
        return None, log_text


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    yaml_choices = _list_yaml_configs()
    image_choices = _list_input_images()

    lora_rows: list[dict[str, Any]] = []
    lora_names: list[gr.Textbox] = []
    lora_weights: list[gr.Number] = []
    lora_bypasses: list[gr.Checkbox] = []
    lora_paths: list[gr.Textbox] = []

    with gr.Blocks(title="Image to Blueprint", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Image to Blueprint — Pipeline UI")

        config_state = gr.State(value=None)
        pipeline_state = gr.State(value=None)

        with gr.Row():
            pipeline_dd = gr.Dropdown(
                label="Load Pipeline",
                choices=yaml_choices,
                value=yaml_choices[0] if yaml_choices else None,
            )
            load_btn = gr.Button("Load", variant="secondary")

        with gr.Group():
            gr.Markdown("### Input Image")
            with gr.Row():
                image_dd = gr.Dropdown(
                    label="Image",
                    choices=image_choices,
                    value=image_choices[0] if image_choices else None,
                )
                image_preview = gr.Image(
                    label="Preview",
                    type="filepath",
                    interactive=False,
                    height=320,
                )

        with gr.Group():
            gr.Markdown("### Base Model")
            transformer_txt = gr.Textbox(label="Transformer", interactive=False, lines=1)
            vae_txt = gr.Textbox(label="VAE", interactive=False, lines=1)
            text_encoder_txt = gr.Textbox(label="Text Encoder", interactive=False, lines=1)

        with gr.Group():
            gr.Markdown("### LoRAs")
            for i in range(MAX_LORAS):
                with gr.Row(visible=False) as row:
                    name_tb = gr.Textbox(
                        label="Name",
                        interactive=False,
                        scale=2,
                        container=True,
                    )
                    weight_nb = gr.Number(label="Weight", value=1.0, minimum=0.0, maximum=2.0, step=0.05)
                    bypass_cb = gr.Checkbox(label="Bypass", value=False)
                    path_tb = gr.Textbox(visible=False)
                lora_rows.append({"row": row, "name": name_tb, "weight": weight_nb, "bypass": bypass_cb, "path": path_tb})
                lora_names.append(name_tb)
                lora_weights.append(weight_nb)
                lora_bypasses.append(bypass_cb)
                lora_paths.append(path_tb)

        with gr.Group():
            gr.Markdown("### Configuration")
            positive_tb = gr.Textbox(label="Positive Prompt", lines=8)
            negative_tb = gr.Textbox(label="Negative Prompt", lines=2)
            with gr.Row():
                seed_type_rb = gr.Radio(
                    label="Seed Type",
                    choices=["Fixed", "Randomized"],
                    value="Randomized",
                )
                seed_nb = gr.Number(label="Seed", precision=0, interactive=False)
            with gr.Row():
                cfg_nb = gr.Number(label="CFG", value=1.0, minimum=0.0, maximum=20.0, step=0.1)
                steps_nb = gr.Number(label="Steps", value=4, precision=0, minimum=1, maximum=100, step=1)
                denoise_nb = gr.Number(label="Denoise", value=1.0, minimum=0.0, maximum=1.0, step=0.05)

        model_status_md = gr.Markdown("No pipeline loaded.")
        run_btn = gr.Button("Run Pipeline", variant="primary", interactive=False)

        with gr.Group():
            gr.Markdown("### Output")
            with gr.Row():
                output_image = gr.Image(label="Result", type="pil", interactive=False, height=400)
                log_tb = gr.Textbox(label="Log", lines=20, max_lines=40, interactive=False)

        load_outputs = [
            image_dd,
            image_preview,
            transformer_txt,
            vae_txt,
            text_encoder_txt,
            *[c for slot in lora_rows for c in (slot["row"], slot["name"], slot["weight"], slot["bypass"], slot["path"])],
            positive_tb,
            negative_tb,
            seed_type_rb,
            seed_nb,
            steps_nb,
            cfg_nb,
            denoise_nb,
            config_state,
            run_btn,
            pipeline_state,
        ]

        _init_inputs = [config_state, *lora_bypasses]
        _init_outputs = [pipeline_state, run_btn, model_status_md]

        load_fn = load_pipeline_config
        pipeline_dd.change(load_fn, inputs=[pipeline_dd], outputs=load_outputs).then(
            initialize_models, inputs=_init_inputs, outputs=_init_outputs
        )
        load_btn.click(load_fn, inputs=[pipeline_dd], outputs=load_outputs).then(
            initialize_models, inputs=_init_inputs, outputs=_init_outputs
        )

        image_dd.change(preview_input_image, inputs=[image_dd], outputs=[image_preview])

        seed_type_rb.change(on_seed_type_change, inputs=[seed_type_rb], outputs=[seed_nb])

        for slot in lora_rows:
            slot["bypass"].change(
                on_bypass_pending,
                outputs=[run_btn, pipeline_state, model_status_md],
            ).then(initialize_models, inputs=_init_inputs, outputs=_init_outputs)

        run_inputs = [
            pipeline_state,
            config_state,
            image_dd,
            positive_tb,
            negative_tb,
            seed_type_rb,
            seed_nb,
            steps_nb,
            cfg_nb,
            denoise_nb,
        ]
        for slot in lora_rows:
            run_inputs.extend([slot["name"], slot["weight"], slot["bypass"], slot["path"]])

        run_btn.click(
            run_pipeline,
            inputs=run_inputs,
            outputs=[output_image, log_tb],
        )

        if yaml_choices:
            demo.load(load_fn, inputs=[pipeline_dd], outputs=load_outputs).then(
                initialize_models, inputs=_init_inputs, outputs=_init_outputs
            )

    return demo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image to Blueprint — Gradio UI")
    parser.add_argument(
        "--listen",
        action="store_true",
        help="Bind to 0.0.0.0 (accessible on the network). Default: 127.0.0.1 only.",
    )
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    server_name = "0.0.0.0" if args.listen else "127.0.0.1"
    demo = build_ui()
    demo.launch(server_name=server_name, server_port=args.port)


if __name__ == "__main__":
    main()
