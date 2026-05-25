# Minimal Runpod Serverless Pipeline Deployment

**Executive Summary:** A single Runpod Serverless endpoint can host the entire pipeline. We attach a **network volume** (mounted at `/runpod-volume`【118†L210-L218】) containing large models so they persist across runs. We use a minimal Python handler (e.g. using HuggingFace’s `diffusers`) in a container based on `runpod/base` with a GPU. Key steps: create a network volume, build a Docker image, define a serverless template, and deploy an endpoint with the volume attached. Cold-start optimizations like FlashBoot and Runpod’s cached-model option reduce startup time【100†L296-L304】【118†L210-L218】. We set necessary environment variables (e.g. paths or API keys) via the Runpod console or CLI【96†L187-L194】【112†L311-L318】. Invocation is via Runpod’s HTTPS API (sync or async)【114†L170-L174】. The checklist at the end summarizes the commands and settings. 

```mermaid
graph TD
  Client[Client or Scheduler] -->|HTTPS POST| Endpoint[Runpod Serverless Endpoint]
  Endpoint -->|Loads model from volume| Volume[/runpod-volume (Network Volume)/]
  Endpoint -->|Returns output| Client
```

## Architecture & Components  
Use **one Serverless function** (endpoint) for the full pipeline. The handler code loads models from `/runpod-volume` (where the network volume is mounted【118†L210-L218】) and processes input. We do **not** split into multiple endpoints to keep it minimal. The container’s **ephemeral disk** holds temporary data, while the **network volume** (persistent NVMe SSD, 200–400 MB/s【101†L1-L4】) stores the model files. This avoids re-downloading models on each cold start【118†L210-L218】.

- **Serverless Endpoint:** Runs our handler. Configure GPU type (e.g. `A6000` or higher for a ~10–20GB model)【100†L203-L207】.  
- **Network Volume:** Create (via console or CLI) with size ≥ model size (e.g. 20–50GB). Mounts at `/runpod-volume` inside the function【118†L210-L218】. All model files go here (persisted beyond individual runs).  
- **Docker Image:** Based on `runpod/base` (CUDA/PyTorch)【96†L235-L242】. Install required libraries (e.g. `diffusers`, `torch`). Copy your handler. Use a lightweight ENTRYPOINT.  

## Runpod Features Used  
- **Serverless Functions:** Use queue-based endpoints for asynchronous tasks (default).  
- **Network Volumes:** Attach to the endpoint for persistent models【118†L210-L218】. (Attaching restricts to that datacenter【118†L210-L218】, but necessary for large models.)  
- **GPU Allocation:** In endpoint settings (or CLI) select a GPU tier with enough VRAM (48–80GB). For example, an A6000 (48GB) or H100 (80GB) is suitable for ~13GB Qwen model【100†L203-L207】.  
- **FlashBoot & Cached Models:** Enable **FlashBoot** on the endpoint to reuse warmed workers【100†L296-L304】. If a matching model is in Runpod’s cache, set it in “Model” to pre-load (reduces download time)【100†L308-L312】.  

## Handler & Docker Setup  
A minimal Python handler (`handler.py`) might lazy-load the model on first request:

```python
import os
import runpod
from diffusers import DiffusionPipeline

# Lazy-load pipeline for first invocation
pipe = None
def handler(event):
    global pipe
    if pipe is None:
        model_dir = "/runpod-volume/models"
        pipe = DiffusionPipeline.from_pretrained(model_dir, torch_dtype="auto")
        pipe = pipe.to("cuda")
    output = pipe(event["input"]["prompt"])
    return {"output": {"image": output.images[0].tolist()}}

runpod.serverless.start({"handler": handler})
```

**Dockerfile snippet:** (based on Runpod base image)  
```dockerfile
FROM runpod/base:0.4.0-cuda11.8.0   # Runpod CUDA base image【96†L235-L242】  
# Install Python libraries
RUN pip install torch diffusers  
# Copy handler code
COPY handler.py /handler.py  
CMD ["python", "-u", "/handler.py"]
```
(Alternatively use a newer tag of `runpod/base` with PyTorch already installed.) 

## Environment, Secrets & IAM  
- **Env Variables:** Set any config (e.g. paths, flags) in the Runpod console or via CLI when creating the endpoint【96†L187-L194】. Access in code with `os.environ`.  
- **Secrets:** If using an external bucket instead of volume, store access keys as **runtime env vars** only in the console (not in image)【94†L333-L342】【96†L270-L278】. For a pure volume-based pipeline, no external creds are needed.  
- **IAM/Keys:** Runpod uses its own API keys. No AWS IAM is involved unless you call external services. Ensure your Runpod API key is kept secret.  

## Volume Mount & IO  
Create and attach a network volume (e.g. via CLI):  
```bash
# Create a 50GB volume in US-KS-2 datacenter
runpodctl network-volume create --name my-vol --size 50 --data-center-id "US-KS-2"
```
Attach it when creating the endpoint: either in console or CLI:  
```bash
runpodctl serverless create \
  --name "pipeline-endpoint" \
  --template-id tpl_XXXXX \
  --gpu-id "A6000" \
  --network-volume-id vol_YYYYY \
  --workers-max 1 \
  --flash-boot true
```
This mounts the volume at `/runpod-volume`【118†L210-L218】 inside the function. The NVMe-backed storage (~200–400 MB/s) is fast for model loads【101†L1-L4】. One volume per endpoint, multiple if needed (then code must handle no shared data race)【118†L224-L232】.

## Compute Sizing  
Choose GPU with VRAM > model size. E.g. Qwen 7B image-edit (~10GB) + text encoder (~3GB) suggests a 48GB GPU like A6000【100†L203-L207】. You may set `--gpu-count` if using multiple GPUs, but 1 is enough. CPU and RAM are fixed by GPU selection. No additional CPU config is needed.

## Cold-Start Minimization  
- **FlashBoot:** Enable on the endpoint to “warm” workers between requests【100†L296-L304】.  
- **Cached Model:** If Runpod’s Hub has your model, select it so workers skip downloading【100†L306-L312】.  
- **Lazy Load:** As in handler above, load the model once per worker (store globally) so subsequent invocations reuse it.  
- **Container Image:** Keep it lean (no unnecessary libraries) so startup is quick. Only copy what’s needed.

## Networking & Security  
- **Egress:** By default, serverless has internet access. There’s no configurable VPC. Use data-center restrictions to control where it runs【100†L319-L327】. For external API calls, traffic goes over public internet.  
- **Least Privilege:** Only grant access needed. No extra network ports: serverless endpoints only expose the Runpod API.  
- **Secrets Management:** Store sensitive keys (if any) as encrypted runtime vars, not in code or container.

## Invocation Patterns  
- **HTTP API:** Use Runpod’s REST endpoints to call the function. For synchronous requests (wait for result), use `/runsync`; for asynchronous, use `/run` and poll `/status`【114†L170-L174】.  
```bash
curl -X POST https://api.runpod.ai/v2/<endpoint-id>/runsync \
  -H "authorization: Bearer $RUNPOD_API_KEY" \
  -H "content-type: application/json" \
  -d '{"input":{"prompt":"Example prompt"}}'
```
- **Cron/Queue:** Runpod has no built-in schedule. Use an external scheduler (cron, Lambda, GitHub Actions, etc.) to POST to this API when needed【120†L50-L58】. 
- **Payload:** Wrap your input in `{ "input": { ... } }` as shown【114†L204-L212】.

## Monitoring & Logging  
Use Runpod’s console or CLI (`runpodctl serverless list`) to check endpoint status. Handlers should `print` or `logging` to stdout; logs are viewable in the Runpod console. For errors, look at the “Jobs” tab. No custom monitoring setup is needed for minimal use, but you could integrate webhooks or log to external systems in production.

## Checklist

1. **Create network volume:** In Runpod console or CLI (name, size, datacenter).  
2. **Prepare Docker image:** `FROM runpod/base`, install libraries, add `handler.py`.  
3. **Build & push image:** E.g. `docker build -t myrepo/my-pipeline:latest .` and push to a registry.  
4. **Create serverless template:**  
   ```bash
   runpodctl template create --name pipeline-tpl \
     --image myrepo/my-pipeline:latest --serverless
   ```  
5. **Deploy endpoint:** Use Runpod UI or CLI: attach the template and network volume. Example CLI above. Enable **FlashBoot**.  
6. **Set env vars:** In console or via `--env KEY=VALUE`.  
7. **Test invocation:** Send a `/runsync` request (see above) or use `runpodctl send <endpoint-id>`.  
8. **Scale if needed:** Adjust `--workers-max` or use auto-scaling based on request count.

## Troubleshooting

- **Model not found:** Ensure models are in the volume at expected paths (e.g. `/runpod-volume/models`).  
- **Timeouts/Memory errors:** Increase GPU memory tier if OOM. Check `Execution timeout` setting (default 10 min).  
- **Volume attach issues:** Endpoint must be in same datacenter as volume. If jobs fail on volume I/O, verify mount at `/runpod-volume`.  
- **Cold starts slow:** Warm up by sending a dummy request after deployment, or rely on FlashBoot.

**Sources:** Official Runpod docs on [Network Volumes]【118†L210-L218】, [Storage Options]【91†L176-L184】, [Environment Variables]【96†L187-L194】, [Endpoint Settings (FlashBoot/GPU)]【100†L296-L304】【100†L203-L207】, and [API Requests]【114†L170-L174】.  

Here are the most relevant official docs and resources from the minimal RunPod serverless deployment research:

## Core RunPod Serverless

* [RunPod Serverless Overview](https://docs.runpod.io/serverless/overview?utm_source=chatgpt.com)
* [RunPod Serverless Endpoints](https://docs.runpod.io/serverless/endpoints/overview?utm_source=chatgpt.com)
* [RunPod Endpoint Configuration (FlashBoot, workers, scaling)](https://docs.runpod.io/serverless/endpoints/endpoint-configurations?utm_source=chatgpt.com)
* [RunPod Send API Requests (/run and /runsync)](https://docs.runpod.io/serverless/endpoints/send-requests?utm_source=chatgpt.com)
* [RunPod Serverless Workers Guide](https://docs.runpod.io/serverless/workers/overview?utm_source=chatgpt.com)

## Network Volumes / Storage

* [RunPod Network Volumes Documentation](https://docs.runpod.io/storage/network-volumes?utm_source=chatgpt.com)
* [RunPod Storage Overview](https://docs.runpod.io/serverless/storage/overview?utm_source=chatgpt.com)
* [RunPod Persistent Storage Docs](https://docs.runpod.io/storage/overview?utm_source=chatgpt.com)

## Deployment / CLI

* [runpodctl CLI Reference](https://docs.runpod.io/runpodctl/overview?utm_source=chatgpt.com)
* [runpodctl Serverless Commands](https://docs.runpod.io/runpodctl/reference/runpodctl-serverless?utm_source=chatgpt.com)
* [RunPod Templates Documentation](https://docs.runpod.io/pods/templates?utm_source=chatgpt.com)

## Environment Variables / Secrets

* [RunPod Environment Variables](https://docs.runpod.io/serverless/development/environment-variables?utm_source=chatgpt.com)

## Docker / Base Images

* [RunPod Base Docker Images](https://github.com/runpod/containers?utm_source=chatgpt.com)
* [RunPod Serverless Worker Examples](https://github.com/runpod-workers?utm_source=chatgpt.com)

## Cold Start Optimization

* [RunPod FlashBoot Documentation](https://docs.runpod.io/serverless/endpoints/endpoint-configurations?utm_source=chatgpt.com#reducing-worker-startup-times)
* [RunPod Cached Models Explanation](https://docs.runpod.io/serverless/endpoints/endpoint-configurations?utm_source=chatgpt.com#reducing-worker-startup-times)

## Diffusers / HF Loading

* [Diffusers from_single_file Documentation](https://huggingface.co/docs/diffusers/main/en/using-diffusers/other-formats?utm_source=chatgpt.com)
* [Diffusers GGUF Support](https://huggingface.co/docs/diffusers/main/en/quantization/gguf?utm_source=chatgpt.com)
* [PEFT LoRA Adapters Documentation](https://huggingface.co/docs/peft/index?utm_source=chatgpt.com)

## Flash Attention / Attention Backends

* [Diffusers Attention Backends Documentation](https://huggingface.co/docs/diffusers/main/en/optimization/attention_backends?utm_source=chatgpt.com)
* [FlashAttention Repository](https://github.com/Dao-AILab/flash-attention?utm_source=chatgpt.com)
* [PyTorch SDPA Documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html?utm_source=chatgpt.com)

## Recommended for Your Exact Setup

These are the most important docs for your architecture:

1. [RunPod Network Volumes Documentation](https://docs.runpod.io/storage/network-volumes?utm_source=chatgpt.com)
2. [RunPod Endpoint Configuration (FlashBoot)](https://docs.runpod.io/serverless/endpoints/endpoint-configurations?utm_source=chatgpt.com)
3. [Diffusers GGUF Support](https://huggingface.co/docs/diffusers/main/en/quantization/gguf?utm_source=chatgpt.com)
4. [Diffusers from_single_file Documentation](https://huggingface.co/docs/diffusers/main/en/using-diffusers/other-formats?utm_source=chatgpt.com)
5. [RunPod Send API Requests](https://docs.runpod.io/serverless/endpoints/send-requests?utm_source=chatgpt.com)
6. [RunPod Environment Variables](https://docs.runpod.io/serverless/development/environment-variables?utm_source=chatgpt.com)
