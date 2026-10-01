"""Breeze TTS 2 server. Starts idle (no model, no GPU memory) and loads on the first batch or on
POST /api/load, so it can share one GPU with Chatterbox and an LLM that the app swaps in and out.

The model sits behind BreezeSynthesizer; create_app(synth) takes any object with the same
surface, which is how the tests run without torch."""
import base64
import gc
import io
import json
import logging
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("breeze")

SAMPLE_RATE = 24000
DEFAULT_SEED = 1234
DEFAULT_CFG = 1.0
INSTRUCTION_CFG = 4.0  # voice design and direction need guidance; plain cloning does not
MAX_TOKENS_CAP = 1500
CHARS_PER_SECOND = 15  # typical speech; used only to bound runaway generations
LENGTH_SLACK = 3.0  # allow 3x the expected length, plus 3 s


class OutOfMemory(Exception):
    """Out-of-memory from a synthesizer that isn't torch (the fakes in the tests raise this)."""


def is_oom(exc: BaseException) -> bool:
    if isinstance(exc, OutOfMemory):
        return True
    torch = sys.modules.get("torch")  # not imported yet means the error can't be torch's
    oom = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
    return oom is not None and isinstance(exc, oom)


def empty_cuda_cache() -> None:
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def template_name(request: dict) -> str:
    """Same rule as breeze_infer.templates.select_template_name, kept here so grouping needs no torch."""
    instruction = request.get("instruction")
    has_instruction = bool(isinstance(instruction, str) and instruction.strip())
    if request.get("ref_audio_path"):
        return "ref_edit_tata" if has_instruction else "ref_clone_tata"
    return "tts_instruction" if has_instruction else "tts_plain"


class BreezeSynthesizer:
    """The real model. torch and the Breeze modules are imported on load(), not at startup."""

    frame_rate: float | None = None  # codec frames per second; read from the tokenizer config on load

    def __init__(self, model_dir: str | None = None, repo_id: str | None = None):
        self.model_dir = Path(model_dir or os.environ.get("BREEZE_MODEL_DIR", "/models/breeze-tts-2"))
        self.repo_id = repo_id or os.environ.get("BREEZE_REPO_ID", "BreezeBlue/Breeze-TTS-2")
        self._runtime = None

    @property
    def loaded(self) -> bool:
        return self._runtime is not None

    def load(self) -> None:
        if self.loaded:
            return
        if not (self.model_dir / "config.json").exists():
            from huggingface_hub import snapshot_download
            log.info("downloading %s to %s", self.repo_id, self.model_dir)
            snapshot_download(self.repo_id, local_dir=str(self.model_dir))
        from breeze_infer.runtime import load_runtime, resolve_device, update_generation_config_for_breeze
        started = time.perf_counter()
        tokenizer, model, audio_tokenizer = load_runtime(
            self.model_dir, device=resolve_device(), attn_implementation="eager")  # a Path; a str fails
        update_generation_config_for_breeze(model)
        self._runtime = (tokenizer, model, audio_tokenizer)
        self.frame_rate = self._read_frame_rate()
        log.info("model loaded in %.1fs (codec %s frames/s)", time.perf_counter() - started, self.frame_rate)

    def _read_frame_rate(self) -> float | None:
        """Codec frames per second = sample rate / samples per frame, from the audio tokenizer's config."""
        try:
            cfg = json.loads((self.model_dir / "audio_tokenizer" / "config.json").read_text())
            return float(cfg["output_sample_rate"]) / float(cfg["decode_upsample_rate"])
        except (OSError, KeyError, ValueError, ZeroDivisionError):
            return None

    def unload(self) -> None:
        self._runtime = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def generate(self, chunk_requests: list[dict], cfg: float, seed: int, max_new_tokens: int) -> list[np.ndarray]:
        """One batched model call. All requests share a template and a cfg scale."""
        import torch
        from breeze_infer.runtime import set_all_seeds
        from breeze_infer.templates import get_template, prepare_inputs, select_template_name
        tokenizer, model, audio_tokenizer = self._runtime
        # No repetition_penalty: passing it to generate() trips a CUDA device-side assert.
        inputs = prepare_inputs(tokenizer, audio_tokenizer, model, chunk_requests,
                                get_template(select_template_name(chunk_requests[0])),
                                guidance_scale=cfg, guidance_scale_ref=None, guidance_scale_ins=None)
        set_all_seeds(seed)
        with torch.inference_mode():
            audios = model.generate(**inputs, output_audio=True, audio_tokenizer=audio_tokenizer,
                                    max_new_tokens=max_new_tokens)
        return [a.float().cpu().numpy().reshape(-1) for a in audios]


class BatchItem(BaseModel):
    id: str
    text: str
    voice: str | None = None
    ref_text: str | None = None
    instruction: str | None = None
    cfg_scale: float | None = None


class BatchRequest(BaseModel):
    items: list[BatchItem]
    seed: int | None = None


class SpeechRequest(BaseModel):
    input: str
    voice: str
    ref_text: str | None = None
    instruction: str | None = None
    response_format: str = "wav"


def encode_wav(audio: np.ndarray) -> str:
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def max_tokens_for(frame_rate: float | None, longest_text: int) -> int:
    """Cap a runaway generation: 3x the expected length of the longest text (~15 chars/s) plus 3 s."""
    if not frame_rate:
        return MAX_TOKENS_CAP
    seconds = LENGTH_SLACK * longest_text / CHARS_PER_SECOND + LENGTH_SLACK
    return min(MAX_TOKENS_CAP, math.ceil(seconds * frame_rate))


def create_app(synth, voices_dir: str | None = None, max_batch: int | None = None) -> FastAPI:
    voices = Path(voices_dir or os.environ.get("BREEZE_VOICES_DIR", "/voices"))
    max_batch = max(1, int(max_batch or os.environ.get("BREEZE_MAX_BATCH", 32)))
    lock = threading.Lock()  # one generation at a time; the GPU is shared and memory is tight
    app = FastAPI(title="Breeze TTS 2")

    def prepare(item: BatchItem) -> dict:
        """Validate one item and turn it into a Breeze request."""
        if not item.text.strip():
            raise HTTPException(400, f"{item.id}: text is empty")
        request = {"id": item.id, "text": item.text, "speaker": "S0"}
        instruction = (item.instruction or "").strip()
        if instruction:
            request["instruction"] = instruction
        if item.voice:
            if "/" in item.voice or "\\" in item.voice or ".." in item.voice:
                raise HTTPException(400, f"{item.id}: voice must be a bare file name")
            if not (item.ref_text or "").strip():
                raise HTTPException(400, f"{item.id}: a voice needs ref_text (what the clip says)")
            path = voices / item.voice
            if not path.is_file():
                raise HTTPException(404, f"{item.id}: voice '{item.voice}' not found")
            request["ref_audio_path"] = str(path)
            request["ref_text"] = item.ref_text.strip()
        elif not instruction:
            raise HTTPException(400, f"{item.id}: needs a voice or an instruction")
        return request

    def run_chunk(chunk: list[dict], cfg: float, seed: int, results: dict) -> None:
        """Generate one chunk into results[id] = (audio, error). On out-of-memory, split and retry."""
        longest = max(len(r["text"]) for r in chunk)
        tokens = max_tokens_for(getattr(synth, "frame_rate", None), longest)
        started = time.perf_counter()
        try:
            audios = synth.generate(chunk, cfg, seed, tokens)
            if len(audios) != len(chunk):
                raise RuntimeError(f"model returned {len(audios)} takes for {len(chunk)} items")
        except Exception as exc:
            if is_oom(exc):
                empty_cuda_cache()
                if len(chunk) > 1:
                    half = len(chunk) // 2
                    log.warning("out of memory at batch %d; retrying as %d + %d", len(chunk), half, len(chunk) - half)
                    run_chunk(chunk[:half], cfg, seed, results)
                    run_chunk(chunk[half:], cfg, seed, results)
                    return
                log.error("out of memory on a single item (%s)", chunk[0]["id"])
            else:
                log.exception("chunk of %d failed", len(chunk))
            for r in chunk:
                results[r["id"]] = (None, f"{type(exc).__name__}: {exc}")
            return
        elapsed = time.perf_counter() - started
        seconds = sum(len(a) for a in audios) / SAMPLE_RATE
        log.info("chunk of %d: %.1fs audio in %.1fs = %.2fx real time", len(chunk), seconds, elapsed,
                 seconds / elapsed if elapsed > 0 else 0.0)
        for r, audio in zip(chunk, audios):
            results[r["id"]] = (audio, None if len(audio) else "no audio generated")

    def run_batch(body: BatchRequest) -> dict:
        started = time.perf_counter()
        requests = [prepare(item) for item in body.items]
        ids = [r["id"] for r in requests]
        if len(set(ids)) != len(ids):
            raise HTTPException(400, "item ids must be unique")
        groups: dict[tuple[str, float], list[dict]] = {}
        for item, request in zip(body.items, requests):
            cfg = item.cfg_scale if item.cfg_scale is not None else (
                INSTRUCTION_CFG if "instruction" in request else DEFAULT_CFG)
            groups.setdefault((template_name(request), cfg), []).append(request)
        seed = body.seed if body.seed is not None else DEFAULT_SEED
        results: dict[str, tuple] = {}
        with lock:
            if not synth.loaded:
                synth.load()
            chunk_index = 0
            for (_, cfg), group in groups.items():
                group.sort(key=lambda r: len(r["text"]))  # a batch runs until its longest item ends
                for i in range(0, len(group), max_batch):
                    run_chunk(group[i:i + max_batch], cfg, seed + chunk_index, results)
                    chunk_index += 1
        out = []
        for rid in ids:
            audio, error = results[rid]
            out.append({"id": rid, "wav_b64": encode_wav(audio) if error is None else None,
                        "seconds": len(audio) / SAMPLE_RATE if error is None else 0.0, "error": error})
        return {"items": out, "elapsed_s": time.perf_counter() - started}

    @app.get("/health")
    def health():
        return {"status": "ok", "loaded": bool(synth.loaded), "max_batch": max_batch}

    @app.post("/api/load")
    def load():
        with lock:
            if not synth.loaded:
                synth.load()
        return {"loaded": True}

    @app.post("/api/unload")
    def unload():
        with lock:
            synth.unload()
        return {"loaded": False}

    @app.post("/v1/batch")
    def batch(body: BatchRequest):
        return run_batch(body)

    @app.post("/v1/audio/speech")
    def speech(body: SpeechRequest):
        if body.response_format != "wav":
            raise HTTPException(400, "only response_format 'wav' is supported")
        item = BatchItem(id="0", text=body.input, voice=body.voice, ref_text=body.ref_text,
                         instruction=body.instruction)
        result = run_batch(BatchRequest(items=[item]))["items"][0]
        if result["error"]:
            raise HTTPException(500, result["error"])
        return Response(base64.b64decode(result["wav_b64"]), media_type="audio/wav")

    return app


app = create_app(BreezeSynthesizer())
