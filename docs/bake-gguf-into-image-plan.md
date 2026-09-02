# Plan — Bake ONLY the GGUF transformer into the Docker image (small image, big cold-start win)

## Context

Cold starts are dominated by reading the **10 GB GGUF transformer** off the RunPod
**network volume** (`/runpod-volume/models/unet/…Q3_K_L.gguf`). On 2026-09-01 that single
read took **4m43s–9m20s** (vs ~26 s on a healthy node), and workers cold-started **11×
in 3 hours** — each paying it. See `docs/run-backtrace.md`. The other artifacts are cheap
by comparison: FP8 text encoder usually a few seconds (occasionally ~70 s), VAE tiny,
LoRAs small.

Fix (minimal-image variant): **bake only the GGUF transformer into the image** so
`from_single_file` reads it from the container's **local NVMe** (~2–3 GB/s, consistent)
instead of the network volume. Leave the text encoder, VAE, LoRAs, and config snapshot on
the network volume. This removes the biggest, most-variable read while adding only ~10 GB
to the image.

Decisions (confirmed with user):
- **Only the GGUF** is baked in; keep the image as small as possible.
- **Download the GGUF at build time** via a small Python script run from the Dockerfile
  (`RUN python …`) — not downloaded locally + `COPY`ed. Keeps the repo and build context
  clean (no 10 GB file on disk, no `.gitignore`/`models/` carve-out). The download runs
  once during `docker build` and is baked into a layer; the RunPod worker never downloads
  at runtime, and a handler-only rebuild reuses the cached layer (the download layer sits
  above the code COPYs).
- Registry/image: **`dvladts/blueprint-worker`** (Docker Hub), built
  `docker build --platform linux/amd64 -f worker/Dockerfile -t dvladts/blueprint-worker:0.11 .`
  (bump tag from `0.10`).
- Scope of THIS work: **code changes only** (new download script + Dockerfile +
  `.dockerignore`). The user runs `docker build` (which performs the download) / `push` /
  redeploy.
- **Network volume stays mounted** (`qd816uyrme`) — it still serves the other weights and
  job output.

## Approach

Add a `TRANSFORMER_PATH` env override for just the transformer file, default it to the
baked-in `/app/models/unet/…gguf` in the Dockerfile, and have the Dockerfile `RUN` a tiny
stdlib-only Python script that downloads the GGUF (URL sourced from `worker/manifest.json`)
into that path during the build. All other loaders keep reading from
`MODELS = /runpod-volume/models` unchanged.

## Changes

### 1. `worker/handler.py` — override only the transformer path
- Line 58, currently:
  ```python
  transformer_path = _check(MODELS / "unet"         / "qwen-image-edit-2511-Q3_K_L.gguf")
  ```
  becomes:
  ```python
  _default_tx = MODELS / "unet" / "qwen-image-edit-2511-Q3_K_L.gguf"
  transformer_path = _check(Path(os.environ.get("TRANSFORMER_PATH", str(_default_tx))))
  ```
- `vae_path`, `te_path`, `snapshot_path`, and both LoRAs (lines 59–71) are **unchanged** —
  still `MODELS / …` on the volume. The transformer's `config.json` is read from the
  volume snapshot (`config=config_source, subfolder="transformer"`), so only the GGUF
  weights move into the image. Offline flags stay.

### 2. `worker/scripts/download_transformer.py` — NEW (build-time downloader)
- **Stdlib only** (`urllib.request`, `json`, `hashlib`, `argparse`) so it can run before
  `pip install` and never invalidates on dependency/code changes.
- Reads the transformer entry from `worker/manifest.json` (the `url`-kind entry whose
  `dest` is `models/unet/qwen-image-edit-2511-Q3_K_L.gguf`) — single source of truth, so a
  future model swap only touches the manifest.
- Streams the file to a target path (`--dest`, default
  `/app/models/unet/qwen-image-edit-2511-Q3_K_L.gguf`), writing to a `.part` temp then
  renaming on success (atomic; avoids a half-file being cached in a layer).
- **Verifies sha256 if present** in the manifest entry; skips re-download if the target
  already exists and matches. (Add a `sha256` to the manifest entry to harden this — none
  is set today.)
- Fails loudly (`sys.exit(1)`) on HTTP error / short read so a bad download can't be baked
  in silently.

### 3. `worker/Dockerfile` — download the GGUF during build + set env
- Put the **download layer early** (right after `FROM`/`WORKDIR`, before `pip install` and
  the code COPYs) so editing `requirements.txt` or app code never re-downloads 10 GB:
  ```dockerfile
  FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
  WORKDIR /app

  # Baked transformer: fetched at build time (early, stdlib-only, cache-stable layer)
  COPY worker/manifest.json                    /tmp/manifest.json
  COPY worker/scripts/download_transformer.py  /tmp/download_transformer.py
  RUN python /tmp/download_transformer.py \
        --manifest /tmp/manifest.json \
        --dest /app/models/unet/qwen-image-edit-2511-Q3_K_L.gguf
  ENV TRANSFORMER_PATH=/app/models/unet/qwen-image-edit-2511-Q3_K_L.gguf
  ```
- Everything below stays as-is (`pip install`, `COPY` code, `ENV PYTHONPATH=/app`,
  `ENV RUNPOD_VOLUME=/runpod-volume`, `CMD`). Because the download layer is above the
  `COPY worker/handler.py …` / `COPY worker/scripts/ …` lines, a handler-only change
  rebuilds only the lower layers and reuses the cached download.
- Optional hardening for rebuilds: a BuildKit cache mount
  (`RUN --mount=type=cache,target=/downloads …` download there then move into the image)
  so even a cache-busting rebuild reuses the previous download. Skip unless rebuilds get
  painful.

### 4. `.dockerignore` — NEW (repo root; currently missing)
Build context is the repo root (`docker build … .`). Without this the daemon sends
`venv/` (511M), `node_modules/` (357M), `.git`, etc. No local `models/` now (downloaded at
build), so just allowlist what the image needs:
```
*
!worker/
!model_utils.py
!fp8_loader.py
!pipeline_utils.py
```

### 5. `README-worker.md` — note the baked-GGUF option
Adjust the "container image is lightweight / weights on the volume" wording
(README-worker.md:3) to record that the GGUF transformer is fetched into the image at build
time (the rest stay on the volume), and reference the build command below.

## Build / push / deploy (user runs)

```bash
# First build downloads ~10 GB from HuggingFace into an image layer (needs network).
docker build --platform linux/amd64 -f worker/Dockerfile -t dvladts/blueprint-worker:0.11 .
docker push dvladts/blueprint-worker:0.11
```
Point the RunPod template/endpoint at `dvladts/blueprint-worker:0.11`, **keep the network
volume attached** (`qd816uyrme`), roll the workers. `TRANSFORMER_PATH` is baked into the
image — no endpoint env change needed.

## Verification
1. **Build**: completes without error; `docker history dvladts/blueprint-worker:0.11` shows
   one ~10 GB layer from the download `RUN`. Optionally
   `docker run --rm --platform linux/amd64 dvladts/blueprint-worker:0.11 ls -lh /app/models/unet/`
   shows the ~10 GB GGUF. (No local functional test — the Mac has no CUDA GPU.)
2. **On RunPod**, send one smoke-test job and read the logs. Success criteria:
   - Gap between `Transformer source: GGUF …` and `Some weights of the model checkpoint…`
     drops from **minutes to a few seconds** (now a local-disk read).
   - Remaining volume reads (text encoder / VAE / LoRAs) unchanged, typically tens of
     seconds total; `Pipeline ready.` arrives much sooner overall.
   - Output SVGs return normally; job output still in `/runpod-volume/jobs/<id>/`.

## Risks / notes
- **Build needs network** and re-downloads 10 GB on any cache miss for the download layer
  (editing `manifest.json`/`download_transformer.py`, `--no-cache`, or building on a fresh
  machine with no cache) — mitigated by placing it early + keeping the script stdlib-only
  so upper-layer edits don't bust it. A handler-only rebuild does **not** re-download.
- **Runtime never downloads** — the GGUF is baked into a layer; the RunPod worker only
  pulls the image (once per node) and reads the file from local disk.
- **Integrity**: rely on the script's sha256 check (add a `sha256` to the manifest entry to
  make it enforce, not just skip-if-present).
- **Image ~10 GB heavier** (base pytorch ≈ 7–8 GB + deps + 10 GB GGUF). First pull per node
  is a one-time cached cost; pairs well with keeping ≥1 warm worker.
- **Not fully cold-start-free** — the ~10 GB of other weights still read from the volume
  each cold start (usually fast, but the FP8 encoder can spike to ~70 s on a bad node). If
  that becomes the new bottleneck, the text encoder is the next candidate to bake.
- Disk: the 10 GB lives in Docker's build cache/image store on the build machine (not the
  repo) → keep ~30–40 GB free for the build.
