# Worker — RunPod Serverless Deployment

Single RunPod serverless endpoint that wraps the standalone Qwen-Image-Edit pipeline ([pipeline_utils.py](pipeline_utils.py), [model_utils.py](model_utils.py), [fp8_loader.py](fp8_loader.py) — see [docs/PIPELINE.md](docs/PIPELINE.md) for how the FP8 text encoder, GGUF transformer, VAE and LoRAs load). Weights live on a RunPod network volume (~21 GB: GGUF Q3 transformer + FP8 text encoder + VAE + 2 LoRAs + config snapshot); the container image is lightweight. The pipeline is built **once at module import** in [worker/handler.py](worker/handler.py) — FlashBoot keeps the warm worker hot so subsequent `/run` calls skip the 13 GB load entirely. The handler decodes the base64 input, runs inference, writes `output.png` to `/runpod-volume/jobs/<id>/`, and returns the result inline.

## Prerequisites

- A RunPod account + API key
- `runpodctl` CLI installed and authenticated
- A Docker registry (Docker Hub, GHCR, etc.) you can push to
- A GPU pod target (A6000 / A100 / H100)

## Deploy

1. **Create the network volume** (one-time):
   ```bash
   runpodctl network-volume create \
     --name blueprint-vol --size 60 --data-center-id US-KS-2
   ```
   Note the returned volume ID.

2. **Provision models onto the volume** — spin up a one-shot GPU pod that mounts the volume and runs the provisioner. It reads [worker/manifest.json](worker/manifest.json) (HF file/snapshot + direct URL entries, idempotent, optional sha256) and writes everything to `/runpod-volume/models/...`:
   ```bash
   runpodctl pod create \
     --network-volume-id <volume-id> \
     --image runpod/base:0.7.0-cuda \
     --command "pip install huggingface_hub httpx && python /worker/scripts/provision_volume.py"
   ```
   Wait for it to exit cleanly, then terminate the pod.

3. **Build and push the worker image** (from repo root):
   ```bash
   docker build -t <your-registry>/blueprint-worker:0.1 -f worker/Dockerfile .
   docker push <your-registry>/blueprint-worker:0.1
   ```
   The image bundles [worker/handler.py](worker/handler.py) plus the pipeline modules ([model_utils.py](model_utils.py), [fp8_loader.py](fp8_loader.py), [pipeline_utils.py](pipeline_utils.py)) and installs [worker/requirements.txt](worker/requirements.txt) (project deps minus `gradio`, plus `runpod`).

4. **Create the serverless template**:
   ```bash
   runpodctl template create \
     --image <your-registry>/blueprint-worker:0.1 \
     --serverless --name blueprint-tpl
   ```

5. **Create the endpoint** — attach the volume, pick a GPU, enable FlashBoot:
   ```bash
   runpodctl serverless create \
     --template-id <template-id> \
     --network-volume-id <volume-id> \
     --gpu-id "A6000" \
     --workers-max 2 \
     --flash-boot true \
     --name blueprint-ep
   ```
   Note the returned **endpoint ID** — that's what the API needs in `RUNPOD_ENDPOINT_ID`.

6. **Smoke-test** with a base64-encoded PNG:
   ```bash
   curl -X POST https://api.runpod.ai/v2/<endpoint-id>/run \
     -H "Authorization: Bearer $RUNPOD_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"input":{"image_b64":"<base64>","positive_prompt":"...", "steps":4, "cfg":1.0}}'
   ```
   Then poll `GET /v2/<endpoint-id>/status/<id>`. First call cold-starts (loads weights from the volume); subsequent calls run warm.

## Handler input/output

**Input** (`event["input"]`): `image_b64` (required), `positive_prompt` (required), `negative_prompt`, `steps` (default 4), `cfg` (default 1.0), `seed`.

**Output**: `image_b64` (PNG), `job_dir` (`jobs/<id>` on the volume — audit only), `width`, `height`.

## Tuning

- `ATTN_BACKEND` env var on the endpoint: `_native_flash` (default, zero-install), `flash` (add `flash-attn` to requirements; A100), `_flash_3_hub` (add `kernels`; H100). See [docs/PIPELINE.md](docs/PIPELINE.md) §7 for the full backend menu.
- Updating models: edit [worker/manifest.json](worker/manifest.json) and re-run the provisioner pod — the script skips files already present with the right size/sha256.
