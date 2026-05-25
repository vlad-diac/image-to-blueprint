# Standalone Pipeline vs ComfyUI Workflow — Model Loading & Memory

This document maps the ComfyUI graph in [`workflow.json`](../workflow.json) onto the
standalone `diffusers`-based pipeline in `standalone/`, with the focus squarely on
**how each component is loaded** and **where memory goes during a run**.

> Reference checkpoint set (everything local):
> ```
> models/unet/qwen-image-edit-2511-Q3_K_M.gguf                            9.92 GB
> models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors              9.38 GB
> models/vae/qwen_image_vae.safetensors                                   0.25 GB
> models/loras/qwen-image-edit-2511-multiple-angles-lora.safetensors      0.29 GB
> models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors 0.85 GB
> ```

---

## 1. High-level shape

### ComfyUI graph (`workflow.json`)

```
UnetLoaderGGUF (113)            qwen-image-edit-2511-Q3_K_M.gguf
   └─ LoraLoaderModelOnly (109) angles            (weight 1.0)
        └─ LoraLoaderModelOnly (102) lightning    (weight 1.0)
             └─ ModelSamplingAuraFlow (94)        (shift 3.1)
                  └─ CFGNorm (98)                 (strength 1.0)
                       └─ KSampler (106)          (euler / simple / 4 steps / cfg 1.0)

CLIPLoader (93)   qwen_2.5_vl_7b_fp8_scaled.safetensors   type="qwen_image"
VAELoader  (95)   qwen_image_vae.safetensors
LoadImage  (41) → FluxKontextImageScale (107)
        ├─ VAEEncode (105) ───────────────────────────────► KSampler.latent_image
        ├─ TextEncodeQwenImageEditPlus (112, +) → FluxKontextMultiRef (97, "index_timestep_zero") → KSampler.positive
        └─ TextEncodeQwenImageEditPlus (100, -) → FluxKontextMultiRef (96, "index_timestep_zero") → KSampler.negative

KSampler → VAEDecode (103) → SaveImage (9)
```

### Standalone pipeline

```python
QwenEditPipeline()
    .load(components={transformer, vae, text_encoder, ...}, ...)   # build_pipeline()
    .add_lora("multiple-angles", w=1.0)
    .add_lora("lightning",       w=1.0)
    .run(image, prompt, steps=4, cfg=1.0)                          # diffusers __call__
```

Under the hood `QwenImageEditPlusPipeline` is the diffusers equivalent of the
ComfyUI graph's CLIP / VAE / transformer trio plus a scheduler. The conditioning
nodes (`TextEncodeQwenImageEditPlus`, `FluxKontextMultiReferenceLatentMethod`)
collapse into the pipeline's internal `encode_prompt()` + reference-latent code
path.

---

## 2. Node-to-code mapping

| ComfyUI node                            | Standalone counterpart                                                                  | Source                                                                                          |
|-----------------------------------------|-----------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| `UnetLoaderGGUF` (113)                  | `build_transformer(path=".gguf", ...)` via `GGUFQuantizationConfig` + `from_single_file`| `model_utils.py:142`                                                                            |
| `CLIPLoader` (93) — FP8-scaled          | `build_text_encoder(..., fmt="fp8_scaled")` → `load_qwen25vl_from_fp8_scaled(lazy=True)`| `model_utils.py:242`, `fp8_loader.py:386`                                                       |
| `VAELoader` (95)                        | `build_vae(path=".safetensors", ...)` — manual `convert_wan_vae_to_diffusers` remap     | `model_utils.py:180`                                                                            |
| `LoraLoaderModelOnly` (102, 109)        | `QwenEditPipeline.add_lora(...)` → `pipe.load_lora_weights(...)` + ComfyUI-key remap     | `pipeline_utils.py:93`, `model_utils.py:386`                                                    |
| `ModelSamplingAuraFlow` (94, shift=3.1) | `build_lightning_scheduler()` — `base_shift = max_shift = log(3)` (≈1.0986)             | `model_utils.py:65`                                                                             |
| `CFGNorm` (98, strength=1)              | **No-op** at `cfg=1.0`; the standalone never builds it.                                 | —                                                                                               |
| `TextEncodeQwenImageEditPlus` (100,112) | `pipe.encode_prompt(...)` inside `QwenImageEditPlusPipeline.__call__`                   | diffusers                                                                                       |
| `FluxKontextMultiRef` (96, 97)          | Diffusers' default reference-latent layout (≈ "index_timestep_zero" for single ref)     | diffusers                                                                                       |
| `FluxKontextImageScale` (107)           | Preprocessor inside `QwenImageEditPlusPipeline.__call__`                                | diffusers                                                                                       |
| `VAEEncode` / `VAEDecode` (105 / 103)   | Pipeline's `vae.encode` / `vae.decode`                                                  | diffusers                                                                                       |
| `KSampler` (106)                        | `pipe(... num_inference_steps=4, true_cfg_scale=1.0, generator=...)`                    | `pipeline_utils.py:155`                                                                         |
| `SaveImage` (9)                         | `output_image.save(path)`                                                               | `main.py:260`                                                                                   |

---

## 3. Component-by-component memory & loading

### 3.1 Transformer — GGUF (Q3_K_M)

Same approach in both:

| Property              | ComfyUI (`UnetLoaderGGUF`)                       | Standalone (`build_transformer`)                     |
|-----------------------|--------------------------------------------------|------------------------------------------------------|
| File format           | `.gguf` (Q3_K_M)                                 | `.gguf` (Q3_K_M)                                     |
| Loader                | `custom_nodes/ComfyUI-GGUF`                      | `QwenImageTransformer2DModel.from_single_file(...)` with `GGUFQuantizationConfig(compute_dtype=bf16)` |
| Resident GPU weight   | Quantized blocks (~3.5 GB)                       | Quantized blocks (~3.5 GB)                           |
| Dequant policy        | Per-tile during matmul                           | Per-tile during matmul (diffusers GGUF kernel)       |
| Config / arch         | Read from GGUF header                            | Pulled from `default_repo` / `default_local`'s `transformer/config.json` only — **no transformer weight is downloaded**  |

If `transformer.path` is omitted, the standalone falls back to
`QwenImageTransformer2DModel.from_pretrained(default_repo, subfolder="transformer", torch_dtype=bf16)`,
which downloads the **BF16** checkpoint (~40 GB on disk, ~14 GB in VRAM). Don't
do this unless you intend to.

#### `UnetLoaderGGUF` library stack (ComfyUI side)

`UnetLoaderGGUF` is implemented across four library layers in
`custom_nodes/ComfyUI-GGUF/`. Understanding them makes it clear why the
standalone can replace the node with a single diffusers call.

**Layer 1 — `gguf` (pip package from llama.cpp)**

`gguf.GGUFReader(path)` memory-maps the file and exposes:

- `.tensors` — each tensor's raw bytes, name, and quantization type enum
  (`gguf.GGMLQuantizationType`: Q4_K, Q3_K, Q8_0, …)
- `.fields` — metadata: `general.architecture` (e.g. `"qwen_image"`),
  `general.type`, custom `comfy.gguf.orig_shape.*` entries written by the
  ComfyUI GGUF exporter to preserve the logical weight shape
- `gguf.GGML_QUANT_SIZES` — block size and byte size per quant type, used to
  parse raw blocks in `dequant.py`

The tensors arrive as numpy arrays via mmap — the file is **never fully loaded
into RAM** at read time.

**Layer 2 — `dequant.py` + `GGMLTensor` (local module, pure PyTorch)**

`loader.py:gguf_sd_loader()` wraps every tensor in a `GGMLTensor`:

```python
state_dict[sd_key] = GGMLTensor(
    torch_tensor,           # still quantized raw bytes
    tensor_type=tensor.tensor_type,
    tensor_shape=shape,     # logical shape from GGUF metadata
)
```

`GGMLTensor` is a `torch.Tensor` subclass that carries `tensor_type` and
`tensor_shape` as extra attributes. **The quantized bytes are never expanded at
load time.** `dequant.py` implements a pure-PyTorch dequantization kernel for
every GGML quant format (BF16, Q8_0, Q5_1/0, Q4_1/0, Q6_K … Q2_K, IQ4_NL,
IQ4_XS). These run on GPU so the expand step happens where the compute is.

**Layer 3 — `GGMLOps` / `GGMLLayer` (custom operations layer)**

`GGMLOps` is a subclass of `comfy.ops.manual_cast`. It replaces every standard
`nn.Linear`, `Conv2d`, `Embedding`, `LayerNorm`, and `GroupNorm` in the model
with a GGUF-aware version whose `forward` does:

```python
def forward_ggml_cast_weights(self, input):
    weight, bias = self.cast_bias_weight(input)   # dequantize NOW
    return F.linear(input, weight, bias)           # matmul; expanded weight freed
```

`cast_bias_weight` calls `dequantize_tensor(tensor, dtype, self.dequant_dtype)`,
which expands one layer's quantized bytes to BF16/FP32 for the duration of the
matmul, then lets the allocation be freed. The resident GPU footprint is the
quantized blocks (~3.5 GB for Q3_K_M) plus at most one layer's full-precision
expansion at a time.

The `dequant_dtype` and `patch_dtype` knobs exposed on `UnetLoaderGGUFAdvanced`
control what dtype the temporary expansion targets (`None` → match input,
`"target"` → match compute dtype, or an explicit `torch.float32`/`float16`/
`bfloat16`).

**Layer 4 — `comfy.sd.load_diffusion_model_state_dict` (architecture detection)**

This is the ComfyUI-specific entry point that makes `UnetLoaderGGUF` aware of
model architecture:

```python
model = comfy.sd.load_diffusion_model_state_dict(
    sd, model_options={"custom_operations": ops}
)
```

It inspects the state dict's key names and tensor shapes to detect the
architecture (flux, sdxl, sd3, qwen_image, …), instantiates the correct PyTorch
`nn.Module` subclass, and calls `load_state_dict(sd)`. The
`model_options["custom_operations"]` hook is what substitutes `GGMLOps` for
every layer constructor — without it the model would allocate full-precision
parameters and `load_state_dict` would try to copy quantized bytes into them.

**Layer 5 — `GGUFModelPatcher` (LoRA + device management)**

`GGUFModelPatcher` extends `comfy.model_patcher.ModelPatcher`, ComfyUI's
object responsible for moving components between CPU and VRAM and applying LoRA
patches. The key override is `patch_weight_to_device`: when the target weight is
a `GGMLTensor`, LoRA deltas are stored directly on the tensor object
(`out_weight.patches = [(patches, key)]`) rather than being applied eagerly.
They are folded in lazily inside `GGMLLayer.get_weight()` on each forward pass,
after dequantization and before the matmul.

There is also an mmap-release pass in `GGUFModelPatcher.load()`: because
`gguf.GGUFReader` memory-maps the file, Windows never releases the file handle
while tensors hold a reference to the mapped memory. On the first model load the
patcher does a round-trip (`module.to(load_device).to(offload_device)`) on every
module that is still linked to the mmap, breaking the reference and allowing the
OS to reclaim the file handle.

#### Equivalence with the standalone

| `UnetLoaderGGUF` layer            | Standalone equivalent                                                           |
|-----------------------------------|---------------------------------------------------------------------------------|
| `gguf.GGUFReader` (file parse)    | Same — diffusers calls it internally via `GGUFQuantizationConfig`               |
| `dequant.py` (per-quant kernels)  | Same math — diffusers' GGUF path applies the same block dequantization          |
| `comfy.sd.load_diffusion_model_state_dict` | `QwenImageTransformer2DModel.from_single_file(path, quantization_config=...)` |
| `GGMLOps` custom ops              | `GGUFQuantizationConfig(compute_dtype=dtype)` passed to `from_single_file`      |
| `GGUFModelPatcher` (LoRA / offload) | diffusers PEFT `load_lora_weights` + `enable_model_cpu_offload`               |
| `folder_paths` (file discovery)   | Explicit `Path(path)` in `build_transformer()`                                  |

The only thing that does **not** translate is `comfy.sd` itself — it is
ComfyUI-internal and not pip-installable. The standalone replaces it entirely
with `diffusers`' `from_single_file` + `GGUFQuantizationConfig`.

### 3.2 Text encoder — FP8-scaled Qwen2.5-VL 7B

This is the only spot where the two stacks needed real reverse-engineering and
where memory diverges most sharply.

**File layout** (per `comfy/utils.py::convert_old_quants`):

```
<prefix>scaled_fp8                     marker tensor
<prefix>foo.bar.weight                 torch.float8_e4m3fn
<prefix>foo.bar.scale_weight           per-tensor scalar (broadcast)
<prefix>foo.bar.scale_input            optional input scale (always 1.0, dropped)
```

**Standalone supports two paths** in `fp8_loader.py:386`:

| Mode               | When                              | Storage on GPU                                   | Peak VRAM (text encoder) |
|--------------------|-----------------------------------|--------------------------------------------------|--------------------------|
| `lazy=True` (default) | matches ComfyUI's `fp8_ops.Linear` | `torch.float8_e4m3fn` (1 byte / param)           | ~7 GB                    |
| `lazy=False`       | reference / debug                 | BF16 (2 bytes / param) — full eager dequant      | ~14 GB                   |

Lazy works by:

1. `accelerate.init_empty_weights()` skeletons a `Qwen2_5_VLForConditionalGeneration`
   on the meta device (no allocations).
2. Every `nn.Linear` whose name appears in `{key for key in sd if key+".scale_weight" in sd}` is replaced with `Fp8Linear`
   (`fp8_loader.py:47`). 358 layers in total.
3. The state dict is streamed tensor-by-tensor from disk (`safetensors.safe_open`)
   so peak CPU RAM stays around ~17 GB instead of the naive ~32 GB of
   `safetensors.load_file`.
4. FP8 weights and FP32 `scale_weight` buffers go straight into the model via
   `load_state_dict(sd, assign=True)`.

`Fp8Linear.forward`:

```python
w = self.weight.to(self.compute_dtype)   # cast FP8 → BF16
w *= self.scale_weight.to(...)            # in-place multiply
return F.linear(x, w, self.bias)          # then matmul; temp w is freed
```

Net effect: weights stay FP8; one layer's BF16 copy (~50–200 MB) lives only for
the duration of that layer's forward pass. **Numerically identical** to
ComfyUI's per-forward FP8 dequant.

#### Old → new Qwen2.5-VL layout remap

The ComfyUI FP8 file targets transformers ≤4.51.3 (`visual.*`,
`model.embed_tokens.*`, `model.layers.*`). transformers ≥4.52 nests everything
under `model.visual.*` and `model.language_model.*`. `fp8_loader.py:118-136`
auto-detects both layouts and rewrites keys on the fly. The smoke test
(`tests/smoke_test.py:check_fp8_inspection`) prints a `coverage: N / M`
diagnostic that flags any mismatch before model construction.

### 3.3 VAE — Wan-architecture, single-file

| Step                          | What happens                                                                                                            | Where                       |
|-------------------------------|-------------------------------------------------------------------------------------------------------------------------|-----------------------------|
| Config                        | `vae/config.json` fetched from `default_repo`                                                                            | `model_utils.py:114`        |
| Empty model                   | `init_empty_weights()` + `AutoencoderKLQwenImage.from_config(config)` (meta device)                                      | `model_utils.py:221`        |
| Checkpoint                    | `safetensors.torch.load_file(qwen_image_vae.safetensors)` — ~250 MB, ok to fully load                                    | `model_utils.py:224`        |
| Key remap                     | `convert_wan_vae_to_diffusers(raw_sd)` — Qwen image VAE shares the Wan VAE architecture                                  | `model_utils.py:227`        |
| Load                          | `vae.load_state_dict(converted_sd, strict=False, assign=True).to(bf16)`                                                  | `model_utils.py:233`        |

Why this is necessary: `AutoencoderKLQwenImage` inherits from `FromOriginalModelMixin`
but is **not** registered in diffusers' `from_single_file` map, so the Wan key
remap (which the Qwen VAE happens to be compatible with — same `encoder.downsamples.N.residual.M.*` → `encoder.down_blocks.N.resnets.M.*` convention) has to be invoked manually.

VRAM footprint: ~0.3 GB resident.

### 3.4 LoRAs (multiple-angles + lightning)

| Step                  | ComfyUI (`LoraLoaderModelOnly`)                 | Standalone (`QwenEditPipeline.add_lora`)                                          |
|-----------------------|-------------------------------------------------|-----------------------------------------------------------------------------------|
| Loaded                | Eagerly into the model wrapper                  | **Deferred** — registered in `_pending_loras`, flushed on first `run()` call      |
| Key prefix            | `diffusion_model.*` (ComfyUI convention)        | Remapped to `transformer.*` and written to `<name>_diffusers_keys.safetensors` on first run (`model_utils.py:386`) |
| Composition           | Sequential merge into the model graph           | `pipe.load_lora_weights(... adapter_name=...)` + `pipe.set_adapters(names, weights)` (PEFT) |
| Weight footprint      | Folded into the transformer at the matmul site  | Held as separate LoRA adapter tensors; applied as `W + ΔW @ scale` per step       |
| `bypass=true`         | n/a                                             | Silent no-op on `add_lora()`                                                      |

LoRA tensors are small (~0.3 GB + ~0.85 GB) and live on the same device as the
transformer.

### 3.5 Scheduler

`ModelSamplingAuraFlow(shift=3.1)` in ComfyUI uses a sigmoid-of-shifted-logits
time schedule. The standalone uses diffusers'
`FlowMatchEulerDiscreteScheduler` with the **exponential dynamic shifting**
config `base_shift = max_shift = log(3) ≈ 1.0986` (`model_utils.py:65`). That
matches the AuraFlow `shift=3` distillation used to train the Lightning
4-step LoRA. The slight (3.1 vs 3) difference is intentional and matches the
LoRA's training config.

---

## 4. The big picture: where the memory goes

### 4.1 ComfyUI — sequential / phased

ComfyUI's model manager treats each loader as a separately swappable unit. The
runtime moves components between CPU RAM and VRAM as the graph executes:

```
phase 1: text encoder ON GPU  → encode prompts → text encoder OFF
phase 2: VAE         ON GPU  → encode reference → VAE OFF
phase 3: transformer ON GPU  → sample 4 steps  → transformer OFF
phase 4: VAE         ON GPU  → decode latent  → VAE OFF
```

Peak VRAM at any one moment ≈ `max(component_size) + activations` rather than
`sum(component_sizes) + activations`. With FP8 ops on the text encoder, that
peak is roughly:

```
~7 GB (text encoder)    │ activations 1–2 GB │   ≈   9 GB peak
~3.5 GB (GGUF Q3 transformer) │ activations 1–2 GB │   ≈   5 GB peak
```

This is why a 12 GB GPU runs the workflow comfortably.

### 4.2 Standalone — concurrent / resident

`diffusers` instantiates the full `QwenImageEditPlusPipeline` once and then
keeps every component on the target device for the lifetime of the pipeline
object (`build_pipeline()` calls `pipe.to(device)` after construction —
`model_utils.py:354`).

With the default `enable_offload=False`:

```
text_encoder   (FP8 lazy)   ≈  7.0 GB
transformer    (GGUF Q3)    ≈  3.5 GB
vae            (BF16)       ≈  0.3 GB
LoRA adapters  (BF16)       ≈  1.2 GB
activations    (cfg=1.0)    ≈  1.0–2.0 GB
─────────────────────────────────────
total peak                  ≈ 13–14 GB
```

So a 16 GB GPU is comfortable; 12 GB requires offloading (see §5).

#### Activation memory tip — `cfg=1.0`

When `true_cfg_scale ≤ 1.0`, `QwenImageEditPlusPipeline` skips the negative
forward pass entirely. The standalone enforces this in
`pipeline_utils.py:190` by setting the negative prompt to a single space when
`cfg ≤ 1.0`, which keeps the prompt cache empty and roughly halves transformer
activation memory.

### 4.3 Where the standalone wins / loses vs ComfyUI

| Dimension                          | ComfyUI                                | Standalone                                                |
|------------------------------------|----------------------------------------|-----------------------------------------------------------|
| FP8 text encoder dequant           | per-forward via `fp8_ops.Linear`       | per-forward via `Fp8Linear` (`lazy=True` — equivalent)    |
| GGUF transformer                   | per-tile dequant inside matmul         | per-tile dequant inside matmul (diffusers GGUF kernel — equivalent) |
| LoRA composition                   | eager merge into model wrapper         | PEFT-managed adapters, scaled per step                    |
| Multi-component phasing            | **Yes** — automatic CPU↔GPU swapping   | **No** by default — opt-in via `enable_offload=True`      |
| First-run network traffic          | Zero (loaders read local files only)   | ~5 MB of small JSON configs from `default_repo` (unless `default_local` is set) |
| Disk re-use                        | Native local files only                | Native local files **or** `from_pretrained` HF download   |

---

## 5. Tuning knobs in the standalone

All exposed via the YAML `hardware:` block, with CLI overrides (`--offload`,
`--compile`, `--dtype`, `--device`).

| Knob                  | What it does                                                                                                                  | When to flip it                                                                       |
|-----------------------|-------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| `offload: true`       | `pipe.enable_model_cpu_offload()` — diffusers' equivalent of ComfyUI phasing. Moves modules to GPU just before their forward. | Anything under ~16 GB VRAM, or when running multiple pipelines in the same process.   |
| `compile: true`       | `torch.compile(pipe.text_encoder, backend="aot_eager")` — fuses `cast → scale → matmul` per Fp8Linear into one kernel.        | When the 358-layer Python dispatch overhead is visible (GPU util drops between layers). One-time ~30–90 s JIT trace on first forward. |
| `dtype: bfloat16`     | Compute dtype for matmuls and non-FP8 tensors                                                                                  | Keep BF16 on Ampere+; FP16 only as a fallback for older cards.                        |
| `text_encoder.format` | `fp8_scaled` (only supported format)                                                                                          | If you have the ComfyUI FP8 file. Otherwise omit `path:` and let HF supply BF16.      |
| LoRA `bypass: true`   | Skip a LoRA entry without removing it from the YAML                                                                            | Quick A/B without editing the file structure.                                         |

### Why `aot_eager` not `inductor`

`inductor` requires Triton, which is Linux-only. `aot_eager` removes the
Python-side per-kernel dispatch (which is the dominant CPU bottleneck for
small FP8 layers) without needing Triton, so it works on Windows. On Linux,
switching backend to `inductor` will additionally fuse the cast+scale+matmul
into a single CUDA kernel for ~2× better GPU utilization on the text encoder.

---

## 6. Quick correctness checklist

When verifying the standalone matches the workflow:

1. **Coverage** — `python standalone/tests/smoke_test.py` should report
   `coverage: N / N model params present in file` for both the FP8 text
   encoder and the VAE. Anything less means a key mismatch.
2. **Scheduler shift** — `build_lightning_scheduler()` must use
   `base_shift = max_shift = log(3)` to match the Lightning 4-step LoRA's
   training-time AuraFlow shift=3.
3. **CFG** — `cfg=1.0` (matches workflow's `KSampler` widget); `CFGNorm` is a
   no-op at this value.
4. **Reference latent layout** — `FluxKontextMultiReferenceLatentMethod`'s
   `"index_timestep_zero"` is the diffusers default for a single reference
   image; no override is needed.
5. **LoRA order / weight** — both adapters at weight 1.0 (Lightning second so
   it's the "outer" LoRA in ComfyUI's chain; PEFT doesn't care about order
   when weights are linear).

---

## 7. Flash attention backends — standalone vs ComfyUI, and how to improve

### 7.1 Current state (default run)

ComfyUI explicitly selects its attention backend at startup (see §3.1 library stack). The standalone uses
diffusers' default, which is `"native"` — PyTorch's `F.scaled_dot_product_attention` (SDPA) via
`AttnProcessor2_0`. On CUDA the SDPA dispatcher tries its registered backends in order (Flash → Efficient
→ Math), but because no SDP backend flags are set explicitly in the standalone, the choice is left to
PyTorch's own heuristics and what happens to be compiled into the installed torch wheel.

The text encoder (`Qwen2_5_VLForConditionalGeneration`, loaded via `fp8_loader.py`) has attention layers
built by transformers, which also resolve to SDPA internally. Same story: PyTorch's heuristics, no
explicit control.

Net effect: on a RunPod A100 or H100, you are probably already getting PyTorch's flash SDP kernel for
the transformer and text encoder attention layers — but accidentally, not intentionally, and with no
fallback strategy if a particular shape or mask pattern triggers the less-efficient math path.

### 7.2 Diffusers `set_attention_backend()` — the new unified API

Since diffusers ≥0.35.0 (August 2025), every `nn.Module` in a diffusers model tree that uses the
attention dispatcher exposes `set_attention_backend(name)` / `reset_attention_backend()` and an
`attention_backend(name)` context manager. `QwenImageTransformer2DModel` is fully supported.

The full backend menu for RunPod Linux:

| Backend name | Package required | GPU target | Notes |
|---|---|---|---|
| `"native"` | none (default) | any | PyTorch SDPA; picks Flash/Efficient/Math per shape |
| `"_native_flash"` | none | CUDA, sm≥80 | Force PyTorch's flash SDP sub-kernel only |
| `"_native_cudnn"` | none | CUDA | cuDNN attention — useful on Windows; on Linux use flash instead |
| `"flash"` | `flash-attn ≥2.6.3` | CUDA, sm≥80 | Dao-AI-Lab FA2; supports masks via variable-length path (PR #13479) |
| `"flash_varlen"` | `flash-attn ≥2.6.3` | CUDA, sm≥80 | Same, variable-length variant (handles ragged batch) |
| `"_flash_3_hub"` | `kernels` (no compile) | H100 (sm90) only | FA3 from HF kernels hub — best option on H100 pods |
| `"flash_4_hub"` | `kernels` (no compile) | H100/Blackwell | FA4 (experimental) |
| `"sage"` | `sageattention ≥2.1.1` | CUDA, sm≥80 | INT8 QK, FP16 PV — fastest but numerically approximate |
| `"xformers"` | `xformers ≥0.0.29` | CUDA | Memory-efficient attention; no mask length restriction |

**Important attention-mask note.** The QwenImage transformer passes `encoder_hidden_states_mask` for
padding positions in the joint image+text attention. The plain `"flash"` backend raised a runtime error
on masked inputs until PR #13479 (merged April 2026) which added a `flash_varlen_forward` code path that
handles masks by packing sequences before and unpacking after the kernel call. Always use a diffusers
version that includes this fix when using `"flash"` with this model.

### 7.3 Where the gain comes from

The transformer has **60 dual-stream DiT blocks** with `attention_head_dim=128` and
`num_attention_heads=24`. At 512×512 latent resolution the image sequence length is ~4096 tokens; at
1024×1024 it is ~16 384. Standard SDPA materialises the full `(B, H, Nq, Nk)` attention score matrix in
HBM at each layer. FlashAttention tiles it in SRAM and never writes the full matrix, cutting HBM traffic
from O(N²) to O(N). For the DiT at 1024-res, that is a ~16× reduction in attention memory bandwidth per
layer — which typically translates to **15–30% wall-clock speedup** on the transformer forward pass,
with the exact gain depending on GPU memory bandwidth.

The text encoder (7B Qwen2.5-VL, 28 layers, sequence length ~3 000 tokens for a typical prompt + image
tokens) also benefits — each transformer layer there runs the same SDPA bottleneck. However, because the
text encoder uses a transformers-native attention module (not diffusers `set_attention_backend`), it
picks up flash attention only through the PyTorch SDPA flags — see §7.5.

### 7.4 What to implement

**Step 1 — add `attention_backend` to `build_transformer()`** in `model_utils.py`:

```python
def build_transformer(
    path,
    config_source,
    dtype=torch.bfloat16,
    attention_backend=None,          # new
):
    ...
    transformer = QwenImageTransformer2DModel.from_single_file(...)
    if attention_backend is not None:
        transformer.set_attention_backend(attention_backend)
        logger.info("Transformer attention backend: %s", attention_backend)
    return transformer
```

**Step 2 — surface it through `build_pipeline()`:**

```python
def build_pipeline(components, dtype, device, enable_offload=False,
                   compile_text_encoder=False,
                   attention_backend=None):   # new
    ...
    transformer = build_transformer(
        path=tx_cfg.get("path"),
        config_source=config_source,
        dtype=dtype,
        attention_backend=attention_backend,
    )
```

**Step 3 — surface it through `QwenEditPipeline.load()`:**

```python
def load(self, components, dtype=torch.bfloat16, device="cuda",
         enable_offload=False, compile_text_encoder=False,
         attention_backend=None):   # new
    self._pipe = build_pipeline(
        ...,
        attention_backend=attention_backend,
    )
```

**Step 4 — add explicit SDPA flags for the text encoder**, at the top of `build_pipeline()`, so the
transformers-native attention also benefits without needing a separate API call:

```python
# Enable all PyTorch SDPA sub-backends; flash-attn being installed
# makes FLASH_ATTENTION the best sub-kernel choice automatically.
if device == "cuda":
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
```

**Step 5 — add to the YAML hardware block:**

```yaml
hardware:
  device: cuda
  dtype: bfloat16
  offload: false
  compile: false
  attention_backend: flash        # or _flash_3_hub on H100, sage, null for default
```

### 7.5 RunPod install commands

**A100 pod (sm80) — recommended:**

```bash
pip install flash-attn --no-build-isolation   # FA2; ~2 min build on A100
```

Then in the YAML: `attention_backend: flash`

**H100 pod (sm90) — best option:**

```bash
pip install kernels    # tiny package; downloads pre-built FA3 kernel from HF Hub
```

Then in the YAML: `attention_backend: _flash_3_hub`
FA3 is roughly 1.5–2× faster than FA2 on H100 due to the Hopper async pipeline.

**Any CUDA pod — zero-install baseline improvement** (no extra packages):

Add this to `build_pipeline()` before constructing the pipeline:

```python
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
```

And set `attention_backend: _native_flash` in the YAML. This forces PyTorch's own flash sub-kernel for
the transformer and benefits the text encoder attention too, with no compile step and no new packages.

### 7.6 What does NOT change

- The GGUF transformer weights are still quantized blocks. The attention backend change only affects
  the attention computation (QKV projection → softmax → weighted sum), not how the linear layers
  dequantize their weights. FlashAttention takes BF16 Q/K/V as input — the GGUF dequant path already
  produces BF16, so the two are fully compatible.
- The FP8 text encoder lazy path (`fp8_loader.py`) is unaffected. Fp8Linear dequantises each linear
  layer including the QKV projections; the attention kernel sits downstream of that and will use
  whatever PyTorch SDPA backend is active.
- Numerical output is **identical** between `"native"` and `"flash"` / `"_flash_3_hub"` for BF16.
  `"sage"` is intentionally approximate (INT8 quantisation of QK); use only after confirming output
  quality.
- The correctness checklist in §6 is unchanged — flash attention does not affect conditioning,
  scheduler, LoRA scale, or reference latent layout.
