# Breeze TTS 2 server: a thin FastAPI wrapper (server.py) around BreezeBlue's inference code.
# The model is not baked in: it lives in the /models volume and loads on demand.
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TOKENIZERS_PARALLELISM=false \
    HF_HUB_ENABLE_HF_TRANSFER=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg sox libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

# Pinned inference code. flash-attn is not needed: the server loads with attn "eager".
ARG BREEZE_COMMIT=58ec70ce5fa4cc361bdebf77ec40d1365da00ab2
RUN git clone https://github.com/breezeblue-ai/breeze-tts /opt/breeze-infer \
    && git -C /opt/breeze-infer checkout ${BREEZE_COMMIT}
WORKDIR /opt/breeze-infer
RUN python -m pip install -r requirements.txt huggingface_hub hf_transfer

# Last so edits to the server don't rebuild the layers above.
COPY server.py fast_depth.py test_server.py ./
EXPOSE 8005
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8005"]
