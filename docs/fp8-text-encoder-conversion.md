# How the text encoder FP8 conversion works

## 0. Where the truth lives

`worker/handler.py` does not convert anything. It only sets a flag. The real definition of the conversion is `fp8_loader.py` at the repo root. The data shape it consumes is a ComfyUI "scaled FP8" safetensors file, and that shape is documented and parsed in `fp8_loader.py`.

The switch in the handler:

```python
# worker/handler.py:81-84
"text_encoder": {
    "path": str(te_path),        # qwen_2.5_vl_7b_fp8_scaled.safetensors
    "format": "fp8_scaled",      # this string is the only trigger
},
```

The call chain from that flag to the loader (each hop is a real function):

| Step | Code |
|---|---|
| Handler builds the pipeline | `worker/handler.py:73-91` (`.load(components=...)`) |
| Fluent wrapper forwards to builder | `pipeline_utils.py:109-116` (`build_pipeline(...)`) |
| Builder loads the text encoder component | `model_utils.py:374-380` (`build_text_encoder(...)`) |
| Builder routes `fp8_scaled` to the loader | `model_utils.py:276-282` |
| The actual conversion | `fp8_loader.py:399-509` (`load_qwen25vl_from_fp8_scaled`) |

Note on the two forwarding hops. `handler.py` lives in `worker/` but imports `pipeline_utils` and (transitively) `fp8_loader` from the repo root. That works because `pipeline_utils.py:43` inserts its own directory (the root) onto `sys.path` before importing `model_utils`, and `model_utils.py:50` imports the loader:

```python
# model_utils.py:50
from fp8_loader import load_qwen25vl_from_fp8_scaled
```

The `fp8_scaled` branch, with `lazy=True` hard-wired:

```python
# model_utils.py:276-282
if fmt == "fp8_scaled":
    return load_qwen25vl_from_fp8_scaled(
        path, config_source=config_source, compute_dtype=dtype,
        lazy=True,   # keep weights FP8 on GPU, dequantize per layer at forward
        device=device,
    )
```

So the handler always takes the lazy path. Everything below is that path.

---

## 1. Vocabulary you need first

**scaled FP8 / `float8_e4m3fn`.** An 8-bit float: 1 sign, 4 exponent, 3 mantissa bits, "fn" = finite (no infinities, one NaN). Max magnitude 448, only 8 steps per binade. Too coarse to hold raw weights, so each weight tensor carries a separate scale. The dtype is named at `fp8_loader.py:76`.

**The `scaled_fp8` marker.** A sentinel tensor whose key ends in `scaled_fp8`. Its presence means "this file is scaled FP8." Suffix constant at `fp8_loader.py:38`, detection at `fp8_loader.py:146-152`.

**`scale_weight`.** One float32 scalar per weight tensor. The true weight is `weight_fp8 * scale_weight`. Suffix constant at `fp8_loader.py:39`.

**`scale_input`.** An optional activation scale. Always 1.0 in this file, so it is dropped. Suffix constant at `fp8_loader.py:40`, dropped at `fp8_loader.py:298`.

**prefix.** ComfyUI may nest the encoder under a namespace. The text before `scaled_fp8` is that prefix, stripped from every key. Computed at `fp8_loader.py:152`.

**the join key.** A weight and its scale share the same base name: `foo.bar.weight` pairs with `foo.bar.scale_weight`. That shared base is how the code decides a tensor is FP8. Used at `fp8_loader.py:286-292` and again at `fp8_loader.py:470-474`.

**lazy vs eager.** Two modes. Eager multiplies weight by scale at load and stores BF16 (`fp8_loader.py:198-249`). Lazy keeps FP8 in memory and multiplies during the forward pass (`fp8_loader.py:252-319`). Selected at `fp8_loader.py:437`.

**`Fp8Linear`.** A drop-in `nn.Linear` that stores an FP8 weight plus a `scale_weight` buffer and dequantizes on each forward. Defined at `fp8_loader.py:47-100`.

**meta device / `init_empty_weights`.** A way to build a model whose tensors have shape and dtype but no storage, so a 7B skeleton costs no memory. Used at `fp8_loader.py:446` and `fp8_loader.py:463`.

**`assign=True`.** A `load_state_dict` mode that adopts the loaded tensor object wholesale instead of copying into the existing one. It is what lets FP8 tensors survive into the model. Used at `fp8_loader.py:489`.

**layout (old vs new).** Two Qwen2.5-VL key naming schemes. The file uses the old one, current transformers uses the new one. Detected at `fp8_loader.py:121-127`, remapped at `fp8_loader.py:130-139`.

**`compute_dtype`.** The dtype matmuls run in, BF16 here. Threaded through as the second argument everywhere.

---

## 2. The data shape, read off the parser

The file has levels. Whole file, then prefix namespace, then a per-layer triplet, then individual tensors. The layout is documented at `fp8_loader.py:6-11`:

```
<prefix>scaled_fp8                  → marker tensor
<prefix>foo.bar.weight              → torch.float8_e4m3fn
<prefix>foo.bar.scale_weight        → per-tensor float32 multiplier
<prefix>foo.bar.scale_input         → optional, always 1.0, dropped
```

The join that binds a layer together is the shared base `foo.bar`. The loader never trusts a dtype flag to decide "is this FP8." It asks "does a `.scale_weight` sibling exist for this `.weight`." That test is the whole classifier:

```python
# fp8_loader.py:286-292  (which weights are FP8)
fp8_weight_keys = {
    base + ".weight"                          # rebuild the weight key
    for k in all_keys
    if k.endswith(_SCALE_WEIGHT_SUFFIX)       # start from every scale_weight
    for base in (k[:-len(_SCALE_WEIGHT_SUFFIX)],)  # base = key minus ".scale_weight"
    if base + ".weight" in all_keys           # keep only if the .weight really exists
}
```

Why a scale is mandatory. Real weights cluster near zero, under 0.1. Cast straight to FP8 they fall into the cramped subnormal range and mostly flush toward zero. Dividing by a per-tensor scale (about `max|w| / 448`) spreads the tensor across FP8's full range, and multiplying back at compute time recovers most of the precision. Activations are never quantized here, because `scale_input` is 1.0, so this is weight-only FP8.

---

## 3. Orchestration: `load_qwen25vl_from_fp8_scaled`

This function runs the parts below in order. Signature and defaults at `fp8_loader.py:399-405`. The parts run top to bottom, so I present them in that order.

### 3a. Fetch config only, no weights

```python
# fp8_loader.py:433-434
config = AutoConfig.from_pretrained(config_source, subfolder="text_encoder")
```

`config_source` resolves to the local snapshot's `text_encoder/` subfolder. Only the small `config.json` is read. No weight download. This gives the model shape the FP8 weights will pour into.

### 3b. Pick the streaming mode

```python
# fp8_loader.py:437-440
stream_fn = _stream_raw_state_dict if lazy else _stream_dequantized_state_dict
sd = stream_fn(fp8_path, compute_dtype, device)
```

Since the handler passes `lazy=True`, `stream_fn` is `_stream_raw_state_dict`. That is the next part.

---

## 4. Streaming the state dict, lazy mode

`_stream_raw_state_dict` at `fp8_loader.py:252-319` walks the file once and sorts every tensor into three buckets, reading each straight onto the target device. It first builds `fp8_weight_keys` (shown above), then routes:

```python
# fp8_loader.py:296-315  (per-key routing, trimmed)
if k == prefix + _FP8_MARKER_SUFFIX:
    continue                                   # drop the marker
if k.endswith(_SCALE_INPUT_SUFFIX):
    continue                                   # drop scale_input (always 1.0)
...
if k in fp8_weight_keys:
    out[rk] = t                                # keep FP8 weight untouched
elif k.endswith(_SCALE_WEIGHT_SUFFIX):
    out[rk] = t.to(torch.float32)              # scale stays float32
elif t.dtype.is_floating_point:
    out[rk] = t.to(compute_dtype)              # bias, norm, embedding → BF16
else:
    out[rk] = t                                # rare integer buffers pass through
```

`rk` is the prefix-stripped name, so keys line up with the transformers module tree. The result `sd` holds FP8 weights as FP8, scales as float32, everything else as BF16. That is the memory win: FP8 weights are about 7 GB, versus about 14 GB if they were BF16.

The eager alternative at `fp8_loader.py:198-249` does the multiply at this stage instead:

```python
# fp8_loader.py:241  (eager only)
t = t.to(compute_dtype) * scale.to(compute_dtype)   # dequantize now, store BF16
```

Same math, different timing. Lazy defers this exact line to the forward pass.

---

## 5. Layout detection and remap

The file was exported against the old Qwen2.5-VL key scheme. Current transformers uses the new one. The mapping, documented at `fp8_loader.py:107-118`:

```
visual.*             → model.visual.*
model.embed_tokens.* → model.language_model.embed_tokens.*
model.layers.*       → model.language_model.layers.*
lm_head.*            → lm_head.*   (unchanged)
```

The loader detects the scheme on both sides. It reads the file's scheme from `sd`, and the model's scheme from a throwaway probe built on the meta device just to inspect its keys:

```python
# fp8_loader.py:444-450
file_layout = _detect_qwen25vl_layout(sd.keys())
with init_empty_weights():                     # probe costs no memory
    _probe = Qwen2_5_VLForConditionalGeneration(config)
model_layout = _detect_qwen25vl_layout(_probe.state_dict().keys())
del _probe
```

If file is old and model is new, every key is rewritten:

```python
# fp8_loader.py:452-454
if file_layout == "old" and model_layout == "new":
    sd = {_remap_qwen25vl_old_to_new(k): v for k, v in sd.items()}
```

The rewriter at `fp8_loader.py:130-139` is idempotent. Keys already under `model.visual` or `model.language_model` are returned unchanged, so running it twice is safe. Without this step every file key would be "unexpected" and every model key "missing."

---

## 6. The empty skeleton

```python
# fp8_loader.py:463-464
with init_empty_weights():
    model = Qwen2_5_VLForConditionalGeneration(config)
```

Every parameter is on the meta device: shape and dtype, no storage. A 7B model built this way costs almost nothing. Real tensors arrive at load time.

---

## 7. Swapping in `Fp8Linear`

Only in lazy mode. First the loader finds which module names are FP8, using the same join rule as before, now applied to the remapped `sd`:

```python
# fp8_loader.py:470-474
fp8_layer_names = {
    k[: -len(".weight")]                                  # module path
    for k in sd
    if k.endswith(".weight")
    and (k[: -len(".weight")] + ".scale_weight") in sd    # has a scale sibling
}
n_replaced = _replace_fp8_linears(model, fp8_layer_names, compute_dtype, device)  # :475
```

`_replace_fp8_linears` at `fp8_loader.py:326-368` walks to each parent module and swaps the `nn.Linear` for an `Fp8Linear` of matching shape:

```python
# fp8_loader.py:355-365
setattr(parent, attr, Fp8Linear(
    in_features=orig.in_features,
    out_features=orig.out_features,
    has_bias=orig.bias is not None,
    compute_dtype=compute_dtype,
    device=device,          # real (empty) FP8 tensors on the GPU, not meta
))
```

Unlike the rest of the skeleton, these `Fp8Linear` modules allocate real empty FP8 tensors right away, because `__init__` gets a `device` (`fp8_loader.py:75-79`). They are placeholders that load will overwrite.

---

## 8. Loading with `assign=True`

```python
# fp8_loader.py:489
missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
```

`assign=True` is the crux. The default path does `param.data.copy_(loaded)`, which needs real storage in the destination (meta tensors have none) and forces the loaded tensor into the destination's dtype (which would corrupt FP8). With `assign=True`, load instead does `module._parameters[name] = loaded_tensor`, adopting the loaded object as is. So:

- meta placeholders for normal params get replaced by real BF16 tensors,
- `Fp8Linear.weight` placeholders get replaced by the real FP8 tensors, FP8 preserved,
- `Fp8Linear.scale_weight` buffers receive the float32 scales.

`strict=False` tolerates the dropped marker and `scale_input` keys, and any genuinely absent weights. Both are only logged (`fp8_loader.py:490-493`).

---

## 9. The backstop

```python
# fp8_loader.py:495-500
n_fixed = _materialize_meta_tensors(model, compute_dtype, device)
if n_fixed:
    logger.warning("%d meta tensors zero-filled — FP8 file may be incomplete.", n_fixed)
```

Anything still on the meta device after load (a param with no key in `sd`) is zero-filled by `_materialize_meta_tensors` (`fp8_loader.py:375-392`). This never leaves a meta tensor in the model, which would crash at forward. It is a safety net, not a normal path.

---

## 10. The runtime dequantization

This is where lazy mode pays off, once per layer per forward:

```python
# fp8_loader.py:88-94
def forward(self, x):
    w = self.weight.to(self.compute_dtype)          # FP8 → BF16, a fresh buffer
    w *= self.scale_weight.to(self.compute_dtype)   # apply scale in place on w
    return F.linear(x, w, self.bias)                # matmul, then w is freed
```

Line by line:

- `self.weight` is FP8 and stays FP8 in VRAM. `.to(bf16)` allocates a new tensor `w`, so the stored weight is never mutated.
- `w *= ...` mutates that fresh `w`, avoiding a third allocation. This is the same `fp8 * scale` product the eager path computed at load (`fp8_loader.py:241`), so lazy and eager are bit identical, only the timing differs.
- After `F.linear` returns, `w` has no references and is reclaimed. Peak overhead is one layer, roughly 100 MB, not the full 14 GB.

---

## 11. Optional fusion

Back in the builder, if `compile_text_encoder` is on:

```python
# model_utils.py:404-412
if compile_text_encoder and target_device.type == "cuda":
    backend = "aot_eager"
    pipe.text_encoder = torch.compile(pipe.text_encoder, backend=backend)
```

The handler wires this from an env var (`worker/handler.py:54`, passed at `worker/handler.py:89`). Compilation fuses the cast, the scale, and the matmul from `Fp8Linear.forward` into fewer kernels, so the BF16 buffer `w` need not be written to global memory at all. With the `inductor` backend on Linux this becomes a single fused kernel. Default here is `aot_eager`, which mainly removes Python dispatch overhead.

---

## Map: file contents to code that handles them

| On-disk thing | How it is handled | Code |
|---|---|---|
| `scaled_fp8` marker | detected to find prefix, then dropped | `fp8_loader.py:146-152`, `:296` |
| `foo.bar.weight` (FP8) | kept as `float8_e4m3fn` | `fp8_loader.py:308-309` |
| `foo.bar.scale_weight` | cast to float32, feeds the buffer | `fp8_loader.py:310-311` |
| `foo.bar.scale_input` | dropped (always 1.0) | `fp8_loader.py:298` |
| bias / norm / embedding | cast to BF16 | `fp8_loader.py:312-313` |
| old-scheme key names | remapped to new scheme | `fp8_loader.py:452-454` |
| the FP8 Linear layers | replaced by `Fp8Linear` | `fp8_loader.py:475` |
| every tensor into the model | adopted via `assign=True` | `fp8_loader.py:489` |
| any missing tensor | zero-filled backstop | `fp8_loader.py:495-500` |
| the `fp8 * scale` math | at forward (lazy) or load (eager) | `fp8_loader.py:92-93` / `:241` |

One layer with no runtime match: activation scaling. `scale_input` exists in the format but is unused here because it is 1.0, so there is no activation-quantization code path.

---

## Plain recap

1. The handler flips one flag, `format: "fp8_scaled"`, and that flag routes the text encoder through `fp8_loader.py` with `lazy=True`.
2. The file stores each 7B weight as 8-bit FP8 plus one float32 scale, joined by a shared base name. The true weight is `fp8 * scale`.
3. Lazy loading keeps the FP8 weights as FP8 (about 7 GB), casts scales to float32, and casts everything else to BF16.
4. Old key names are rewritten to the current transformers scheme.
5. An empty 7B skeleton is built on the meta device, its FP8 Linear layers are swapped for `Fp8Linear`, and `assign=True` pours the tensors in while keeping FP8 intact.
6. At inference, each `Fp8Linear` multiplies its FP8 weight by the scale into a short-lived BF16 buffer, does the matmul, and frees it. Optionally `torch.compile` fuses those steps.

Net result: a standard Qwen2.5-VL text encoder that lives in VRAM at half the size, and computes results identical to the full BF16 model.
