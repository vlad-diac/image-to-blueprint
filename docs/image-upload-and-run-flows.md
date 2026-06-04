# Image Upload & Run Trigger Flows

This document explains how images move through the system — from the client to the RunPod worker — and how runs are triggered. It covers three distinct flows:

1. **Volume provisioning** — how models are pre-loaded onto the pod's network volume (one-time setup)
2. **Run with inline image** — the primary runtime flow: client sends image + params, API forwards to RunPod, worker processes and returns outputs
3. **Run with a pre-stored volume image** — not yet implemented; described here as a future reference

---

## Architecture Overview

```
Client (browser)
    │
    │  multipart/form-data (image binary + run params)
    ▼
API (NestJS)  ──────────────────────────────────────────────────────────┐
    │                                                                    │
    │  stores Run row in Postgres (inputImage as Bytes)                  │
    │                                                                    │
    │  POST /run  { image_b64, positive_prompt, ... }                   │
    ▼                                                                    │
RunPod Serverless Endpoint                                              │
    │                                                                    │
    │  event.input = { image_b64, ... }                                 │
    ▼                                                                    │
Worker (Python)                                                          │
    │  1. Decode base64 → PIL Image                                      │
    │  2. Run inference pipeline                                         │
    │  3. Save output PNGs → /runpod-volume/jobs/{job_id}/output_N.png  │
    │  4. Return { images: [{ image_b64, seed, index }, ...] }          │
    ▼                                                                    │
API polls RunPod status ◄───────────────────────────────────────────────┘
    │
    │  on COMPLETED: decode base64 → store in RunImage table
    ▼
Client polls GET /runs/:id → receives status + output images
```

---

## Flow 1: Volume Provisioning (One-Time Setup)

This is **not a user-facing flow**. It runs once when a new pod network volume is created, before any worker is deployed.

### Script

[worker/scripts/provision_volume.py](../worker/scripts/provision_volume.py)

### What It Does

The script downloads and caches all models the inference pipeline needs onto the persistent network volume. Because workers set `HF_HUB_OFFLINE=1` at runtime, every model/LoRA **must** be on the volume before a worker starts.

### Volume Layout

```
/runpod-volume/
├── huggingface-cache/          # HF_HUB_CACHE — snapshot cache used by transformers
│   └── ...
└── models/
    ├── unet/
    │   └── qwen-image-edit-2511-Q3_K_L.gguf
    ├── vae/
    │   └── split_files/vae/
    │       └── qwen_image_vae.safetensors
    ├── text_encoders/
    │   └── qwen_2.5_vl_7b_fp8_scaled.safetensors
    ├── Qwen--Qwen-Image-Edit-2511/     # full HF snapshot for processor/config
    │   └── snapshots/...
    └── loras/
        ├── qwen-image-edit-2511-multiple-angles-lora.safetensors
        └── Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors
```

### Relevant Env Vars (Worker)

| Variable | Default | Purpose |
|---|---|---|
| `RUNPOD_VOLUME` | `/runpod-volume` | Mount point for the network volume |
| `HF_HOME` / `HF_HUB_CACHE` | set from volume path | Redirects all HF downloads to the volume |
| `HF_HUB_OFFLINE` | `1` | Blocks any HF network requests at runtime |
| `TRANSFORMERS_OFFLINE` | `1` | Blocks transformers from calling the HF hub |

---

## Flow 2: Run with Inline Payload Image (Primary Flow)

This is the **only implemented runtime flow**. The client uploads an image file with the run request; the API converts it to base64 and forwards it to the worker.

### Step 1 — Client Sends Multipart Request

**Endpoint:** `POST /runs`  
**Content-Type:** `multipart/form-data`

**Fields:**

| Field | Type | Required | Constraints | Default |
|---|---|---|---|---|
| `image` | File (binary) | Yes | max 40 MB | — |
| `positivePrompt` | string | Yes | non-empty after trim | — |
| `negativePrompt` | string | No | — | `""` |
| `steps` | number | No | 1–100 | `4` |
| `cfg` | number | No | — | `1.0` |
| `seed` | number | No | — | random |
| `numImages` | number | No | 1–4 | `3` |

**Controller:** [apps/api/src/runs/runs.controller.ts](../apps/api/src/runs/runs.controller.ts)

```
POST /runs
├── @UseInterceptors(FileInterceptor('image', { limits: { fileSize: 40MB } }))
├── Extracts file.buffer (raw bytes)
└── Calls runsService.createWithImage(buffer, dto)
```

### Step 2 — API Validates and Stores the Run

**Service:** [apps/api/src/runs/runs.service.ts](../apps/api/src/runs/runs.service.ts)

Validation rules:
- Image buffer must be present and non-empty
- `positivePrompt` must be non-empty after trim
- `steps` clamped to 1–100, default 4
- `numImages` clamped to 1–4, default 3
- `seed` converted to `BigInt` if provided

A `Run` row is created in Postgres with `status: QUEUED`:

```
Run {
  id:              UUID (auto-generated)
  status:          QUEUED
  inputImage:      Bytes   ← full image buffer stored in DB
  positivePrompt:  string
  negativePrompt:  string
  steps:           int
  cfg:             float
  seed:            BigInt?
  numImages:       int
  createdAt:       now()
}
```

**Schema:** [apps/api/prisma/schema.prisma](../apps/api/prisma/schema.prisma)

### Step 3 — API Submits Job to RunPod

The image buffer is base64-encoded and sent to the RunPod serverless endpoint:

```typescript
const image_b64 = buffer.toString('base64');

// Payload sent to RunPod POST /run
{
  image_b64:        string,   // base64-encoded source image
  positive_prompt:  string,
  negative_prompt:  string,
  steps:            number,
  cfg:              number,
  num_images:       number,
  seed?:            number    // if provided, worker increments per output image
}
```

**RunPod config:**
- Endpoint: `https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run`
- Auth: `Authorization: Bearer {RUNPOD_API_KEY}`

The `Run` row is updated with the returned `runpodJobId` and `status: IN_QUEUE`.

If submission fails, the run is immediately marked `FAILED` with the error message stored in `errorMessage`.

### Step 4 — Worker Processes the Job

**Handler:** [worker/handler.py](../worker/handler.py)

The worker receives `event.input` from RunPod and executes:

**4a. Decode input image**

```python
raw_b64 = event["input"]["image_b64"]
img = Image.open(io.BytesIO(base64.b64decode(raw_b64)))
```

**4b. Extract params**

```python
positive_prompt = event["input"]["positive_prompt"]   # required
negative_prompt = event["input"].get("negative_prompt", "")
steps           = event["input"].get("steps", 4)
cfg             = event["input"].get("cfg", 1.0)
num_images      = event["input"].get("num_images", 3)
seed            = event["input"].get("seed")          # optional
```

If `seed` is provided: `seeds = [seed + i for i in range(num_images)]`  
If not: `seeds = [random.randint(...) for _ in range(num_images)]`

**4c. Run inference pipeline**

```python
imgs = PIPE.run(
    image=img,
    positive_prompt=positive_prompt,
    negative_prompt=negative_prompt,
    steps=steps,
    cfg=cfg,
    num_images=num_images,
    seeds=seeds,
)
```

**4d. Save outputs to volume and encode as base64**

For each output image `i`:

```python
job_dir = VOL / "jobs" / str(job_id)    # /runpod-volume/jobs/{job_id}/
job_dir.mkdir(parents=True, exist_ok=True)

output_path = job_dir / f"output_{i}.png"
img.save(output_path, format="PNG")

buf = io.BytesIO()
img.save(buf, format="PNG")
image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
```

**4e. Return payload to RunPod**

```python
{
  "job_dir":    "jobs/{job_id}",   # relative path on the volume
  "width":      int,
  "height":     int,
  "num_images": int,
  "images": [
    { "index": 0, "seed": 12345, "image_b64": "..." },
    { "index": 1, "seed": 12346, "image_b64": "..." },
    { "index": 2, "seed": 12347, "image_b64": "..." },
  ]
}
```

### Step 5 — API Polls RunPod and Stores Outputs

The API polls `GET /status/{jobId}` on RunPod. When status becomes `COMPLETED`:

1. Extracts `output.images[]` from the worker response
2. Decodes each `image_b64` → `Buffer`
3. Wraps in a Postgres transaction:
   - Deletes any previous `RunImage` rows for this run
   - Inserts one `RunImage` row per output:

```
RunImage {
  id:      UUID
  runId:   string  (FK → Run.id)
  index:   int     (0, 1, 2, 3...)
  bytes:   Bytes   (decoded PNG buffer)
  seed:    BigInt?
  width:   int?
  height:  int?
}
```

4. Updates `Run`:

```
status:       SUCCEEDED
workerJobDir: "jobs/{job_id}"
durationMs:   delayMs + executionMs
numImages:    actual count stored
completedAt:  now()
```

### Image Storage Summary

| Data | Where stored |
|---|---|
| Input image | `Run.inputImage` (Postgres Bytes column) |
| Output images | `RunImage.bytes` (Postgres) + `/runpod-volume/jobs/{job_id}/output_N.png` |

---

## Flow 3: Run with Pre-Stored Volume Image

**Status: Not yet implemented.**

This flow would allow a client to reference an image that is already on the pod network volume (e.g., a previously uploaded or output image) without re-transmitting the raw bytes. It would require:

1. An upload endpoint (`POST /images`) that:
   - Accepts a multipart image
   - Saves it to the network volume (e.g., `/runpod-volume/uploads/{id}.png`)
   - Returns a reference ID or volume path

2. A modified `POST /runs` payload that accepts a `volumeImagePath` instead of (or in addition to) `image`

3. Worker-side support to read from the volume path instead of decoding base64:
   ```python
   if "volume_image_path" in event["input"]:
       img = Image.open(VOL / event["input"]["volume_image_path"])
   else:
       img = Image.open(io.BytesIO(base64.b64decode(event["input"]["image_b64"])))
   ```

Until this is implemented, every run requires the full image binary sent as multipart form data.

---

## Environment Variables Reference

### API (`apps/api/`)

| Variable | Required | Purpose |
|---|---|---|
| `DATABASE_URL` | Yes | Postgres connection string |
| `RUNPOD_API_KEY` | Yes | RunPod authentication |
| `RUNPOD_ENDPOINT_ID` | Yes | Serverless endpoint ID |
| `PORT` | No | API listen port (default 3001) |
| `WEB_ORIGIN` | No | CORS origin for the web client |

### Worker (`worker/`)

| Variable | Default | Purpose |
|---|---|---|
| `RUNPOD_VOLUME` | `/runpod-volume` | Network volume mount point |
| `HF_HOME` / `HF_HUB_CACHE` | derived from volume | Redirect HF cache to volume |
| `HF_HUB_OFFLINE` | `1` | Block HF network calls at runtime |
| `TRANSFORMERS_OFFLINE` | `1` | Block transformers hub calls |
| `ATTN_BACKEND` | `_native_flash` | Attention implementation |
| `ENABLE_OFFLOAD` | unset | GPU memory offloading |
| `COMPILE_TEXT_ENCODER` | unset | Compile text encoder for speed |

---

## Key Files Quick Reference

| File | Role |
|---|---|
| [apps/api/src/runs/runs.controller.ts](../apps/api/src/runs/runs.controller.ts) | HTTP endpoint, multipart file parsing |
| [apps/api/src/runs/runs.service.ts](../apps/api/src/runs/runs.service.ts) | Validation, DB writes, RunPod submission, output processing |
| [apps/api/src/runs/runs.helpers.ts](../apps/api/src/runs/runs.helpers.ts) | Shared helpers (status mapping, etc.) |
| [apps/api/src/runs/dto/create-run-multipart.dto.ts](../apps/api/src/runs/dto/create-run-multipart.dto.ts) | DTO with validation decorators |
| [apps/api/src/runpod/runpod.types.ts](../apps/api/src/runpod/runpod.types.ts) | RunPod input/output type definitions |
| [apps/api/prisma/schema.prisma](../apps/api/prisma/schema.prisma) | DB schema: Run, RunImage, enums |
| [worker/handler.py](../worker/handler.py) | RunPod worker entry point — decodes input, runs pipeline, saves outputs |
| [worker/scripts/provision_volume.py](../worker/scripts/provision_volume.py) | One-time volume setup: downloads all models |
