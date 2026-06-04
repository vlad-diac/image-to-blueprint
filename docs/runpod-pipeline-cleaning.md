# RunPod Pipeline Cleaning Notes

A breakdown of `worker/handler.py` separating the bare-minimum work needed to
produce an edited image from the safety nets, logging, and convenience layers
that surround it. Use this list when stripping the handler down to its
essentials (e.g. for a leaner cold path or for porting the pipeline elsewhere).

## Absolutely necessary for the pipeline run

These steps cannot be removed without breaking inference.

1. **Set HuggingFace cache + offline env vars before any HF import**
   (`worker/handler.py` lines 16–20). `HF_HOME`, `HF_HUB_CACHE`,
   `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1` must be in place before
   `transformers` / `diffusers` are imported, otherwise the libraries try to
   resolve weights over the network and the container fails on a cold start
   (RunPod workers have no public-internet egress to the HF hub by default).

2. **Import the core runtime libs**: `runpod` (the serverless entry-point),
   `torch` (CUDA + bf16), `PIL.Image` (decoding the input image), and
   `pipeline_utils.QwenEditPipeline` (the actual model wrapper).

3. **Resolve model file paths on the network volume**
   (lines 56–61). The pipeline needs concrete on-disk paths for:
   - `unet/qwen-image-edit-2511-Q3_K_L.gguf` — quantised transformer
   - `vae/split_files/vae/qwen_image_vae.safetensors` — VAE
   - `text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors` — text encoder
   - `Qwen--Qwen-Image-Edit-2511` — HF snapshot directory (tokenizer,
     processor, scheduler configs)
   - Both LoRAs (`multiple-angles` and `Lightning-4steps`)

4. **Build the pipeline at module import time** (`_build_pipeline()` →
   `PIPE = _build_pipeline()`, lines 89–91). Calling `QwenEditPipeline().load(...)`
   instantiates `QwenImageEditPlusPipeline` with the explicit component
   configs, moves it to CUDA in bf16, and primes the attention backend. This
   is what makes the handler warm — subsequent invocations reuse `PIPE`.

5. **Attach the LoRAs and flush them into the transformer**
   (`.add_lora(...).add_lora(...)` + `pipe.flush_loras()`, lines 82–85). The
   Lightning LoRA is what enables the 4-step inference path; the angles LoRA
   shapes the camera/view distribution. Without flushing, the adapters are
   queued but not active.

6. **Decode the input image from base64 to RGB PIL**
   (lines 103–107). `Image.open(io.BytesIO(base64.b64decode(...))).convert("RGB")`
   produces the `PIL.Image` the pipeline expects.

7. **Require a positive prompt** (lines 109–111). The model is conditioned on
   text; an empty prompt is not a valid generation request.

8. **Resolve generation parameters**: `negative_prompt`, `steps`, `cfg`,
   `num_images`, and `seeds` (lines 113–123). The seed list is required so
   each of the N outputs is deterministic / distinguishable; the pipeline’s
   `run()` consumes one seed per image.

9. **Call `PIPE.run(...)`** (lines 125–133). This is the actual diffusion
   inference. It returns a list of `PIL.Image` outputs.

10. **Return the generated images** (lines 138–159). At minimum the caller
    needs the PNG bytes — the handler base64-encodes each output and ships
    them inside the JSON response under `images[*].image_b64`.

11. **Register the handler with RunPod**: `runpod.serverless.start({"handler": handler})`
    (line 162). Without this the worker never receives jobs.

## Additional checks, boilerplate, and conveniences

These can be removed, inlined, or simplified without changing the model
output — they exist for observability, defensiveness, or ergonomics.

1. **`_check(p)` existence logger** (lines 39–46). It only prints `✓ found`
   or `✗ MISSING` and returns the path unchanged. It never raises, so
   `_build_pipeline` would still try to load the file even if it’s missing.
   Useful during cold-start debugging; dead weight in production.

2. **`logging.basicConfig` + `logger.info("Loading pipeline…")` / `"Pipeline ready."`**
   (lines 28–33, 89, 91). Pure observability. The pipeline runs identically
   without any of it.

3. **Env-flag plumbing** in `_build_pipeline` (lines 50–52):
   `ATTN_BACKEND`, `ENABLE_OFFLOAD`, `COMPILE_TEXT_ENCODER`. These are
   knobs to experiment with attention kernels, CPU offload, and
   `torch.compile` of the text encoder. For a fixed production config they
   can be hard-coded and the env reads dropped.

4. **`job_id` fallback to `uuid4()`** (lines 97–101). RunPod always
   populates `event["id"]`, so the `uuid` fallback is defensive and never
   fires in practice.

5. **`num_images` clamp `max(1, min(4, ...))`** (line 116). Guards against
   absurd inputs but is not part of the model contract — the pipeline would
   happily run with any positive integer (modulo VRAM).

6. **Defaulting `negative_prompt` to `""`** (line 113). The pipeline accepts
   `None` or empty; the explicit default is for clarity.

7. **Validation of `image_b64` presence** (lines 103–105). Same idea — a
   well-formed caller always supplies it; the explicit `ValueError` is a
   nicer error than a `TypeError` from `b64decode(None)`.

8. **Saving each output PNG to `job_dir`** (lines 135–141). Files are
   written under `/runpod-volume/jobs/<job_id>/output_<i>.png` for later
   retrieval / debugging, but the response already carries the base64
   bytes, so the disk write is purely a side-effect. Removing it would
   speed up the handler and reduce volume IO.

9. **Returning `job_dir`, `width`, `height`, `num_images`** in the response
   (lines 151–158). Metadata for the caller’s convenience — the `images`
   array is the only field strictly required to reconstruct the outputs.

10. **The fluent `.load(...).add_lora(...).add_lora(...)` chain** in
    `_build_pipeline` (lines 63–84). The `QwenEditPipeline` wrapper is a
    convenience layer over `model_utils.build_pipeline`; a stripped-down
    handler could call `build_pipeline` directly and load LoRAs manually.

11. **Re-reading `RUNPOD_VOLUME` twice** (lines 16 and 35, `_VOL_EARLY` vs
    `VOL`). The early copy is required (it has to run before HF imports);
    the second module-level `VOL` is a stylistic duplicate that could just
    reuse `_VOL_EARLY`.

12. **Module-level `MODELS = VOL / "models"` constant** (line 36). Trivial,
    but only used inside `_build_pipeline`; it could live as a local.
