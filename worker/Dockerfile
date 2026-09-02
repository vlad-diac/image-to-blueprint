# PyTorch CUDA image (adjust tag if your RunPod GPUs need a specific CUDA toolkit).
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# Baked transformer: fetched at build time (early, stdlib-only, cache-stable layer)
# so editing requirements.txt or app code below never re-downloads 10 GB.
COPY worker/manifest.json                    /tmp/manifest.json
COPY worker/scripts/download_transformer.py  /tmp/download_transformer.py
RUN python /tmp/download_transformer.py \
      --manifest /tmp/manifest.json \
      --dest /app/models/unet/qwen-image-edit-2511-Q3_K_L.gguf
ENV TRANSFORMER_PATH=/app/models/unet/qwen-image-edit-2511-Q3_K_L.gguf

COPY worker/requirements.txt ./worker-requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r worker-requirements.txt

COPY model_utils.py fp8_loader.py pipeline_utils.py /app/
COPY worker/handler.py /app/handler.py
COPY worker/scripts/ /app/scripts/

ENV PYTHONPATH=/app
ENV RUNPOD_VOLUME=/runpod-volume

CMD ["python", "-u", "/app/handler.py"]
