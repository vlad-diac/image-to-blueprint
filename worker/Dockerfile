# PyTorch CUDA image (adjust tag if your RunPod GPUs need a specific CUDA toolkit).
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

WORKDIR /app

COPY worker/requirements.txt ./worker-requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r worker-requirements.txt

COPY model_utils.py fp8_loader.py pipeline_utils.py /app/
COPY worker/handler.py /app/handler.py

ENV PYTHONPATH=/app
ENV RUNPOD_VOLUME=/runpod-volume

CMD ["python", "-u", "/app/handler.py"]
