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
import signal
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import BackgroundTasks, FastAPI, HTTPException, Response
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
SLOW_REAL_TIME = 1.5  # BREEZE_SLOW_RTF; 0 turns the slowdown guard off
SLOW_WINDOW = 4  # full batches
SLOW_RELOAD_COOLDOWN_SECONDS = 1800
# The app sends up to 64 short units or 32 longer ones per request (WORKLOG §56); BREEZE_MAX_BATCH.
DEFAULT_MAX_BATCH = 64
FULL_CHUNK = 24  # rows from which a chunk counts as full for the slowdown guard


class OutOfMemory(Exception):
    """Out-of-memory from a synthesizer that isn't torch (the fakes in the tests raise this)."""


def is_oom(exc: BaseException) -> bool:
    if isinstance(exc, OutOfMemory):
        return True
    torch = sys.modules.get("torch")  # not imported yet means the error can't be torch's
    oom = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
    return oom is not None and isinstance(exc, oom)


def is_allocator_fault(exc: BaseException) -> bool:
    """PyTorch's expandable-segments allocator failing to grow, which under WSL is a RuntimeError, not
    an OutOfMemoryError. On 2026-10-05 a full card first gave 'CUDA driver error: device not ready'
    (a growth that failed half-mapped), then every later growth over those pages '!handles_.at(i)
    INTERNAL ASSERT FAILED at .../CUDACachingAllocator.cpp' (pytorch#166234, #188008): a chunk of 32
    long units lost all 32 takes, three attempts running. Kernel errors read 'CUDA error: ...' and
    are not this."""
    message = str(exc)
    return isinstance(exc, RuntimeError) and ("CUDA driver error" in message or "CUDACachingAllocator" in message)


def restart_process() -> None:
    """Let uvicorn shut down; compose's restart policy (unless-stopped) starts a fresh process."""
    os.kill(os.getpid(), signal.SIGTERM)


def empty_cuda_cache() -> None:
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def reset_peak_memory() -> None:
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def chunk_memory_note() -> str:
    """'; GPU peak N MiB, reserved M MiB' for the chunk just generated, or '' without CUDA. Peak is what
    the chunk itself needed (since reset_peak_memory); reserved is what the allocator keeps afterwards.
    A reserve far above the peak is cached, fragmented memory, which under WSL can push the process
    past the card's dedicated memory into shared system memory (WORKLOG §51; compose sets
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True against it)."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return ""
    return (f"; GPU peak {torch.cuda.max_memory_allocated() >> 20} MiB, "
            f"reserved {torch.cuda.memory_reserved() >> 20} MiB")


def gpu_memory_report() -> str:
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return "no CUDA"
    free, total = torch.cuda.mem_get_info()
    return (f"GPU free {free >> 20} of {total >> 20} MiB; this process reserves "
            f"{torch.cuda.memory_reserved() >> 20} MiB, allocated {torch.cuda.memory_allocated() >> 20} MiB")


class SlowdownGuard:
    """Asks for a model reload when full batches have slowed far below their normal speed.

    On 2026-10-01 one book's full batches fell from ~3x real time to a median of 1.2x and stayed
    there for five hours; the next book, after an unload and load, ran at normal speed again. Over
    488 healthy full batches the median of 4 in a row never fell under 1.5x; the slow book got
    there 20 minutes in. If the cause is outside Breeze and a reload doesn't help, the next reload
    waits out a cooldown, so a slow GPU doesn't also lose a reload every few batches."""

    def __init__(self, threshold: float, full_batch: int, window: int = SLOW_WINDOW,
                 cooldown: float = SLOW_RELOAD_COOLDOWN_SECONDS, clock=time.monotonic):
        self.threshold = threshold
        self.full_batch = full_batch
        self.window = window
        self.cooldown = cooldown
        self.clock = clock
        self.recent: list[float] = []
        self.reload_due = False
        self.last_reload: float | None = None

    def record(self, size: int, real_time: float) -> None:
        """Note one finished chunk's speed (seconds of audio per second of generating)."""
        if self.threshold <= 0 or size < self.full_batch:
            return  # small chunks are slow by nature
        self.recent = (self.recent + [real_time])[-self.window:]
        if len(self.recent) < self.window:
            return
        median = statistics.median(self.recent)
        if median >= self.threshold:
            return
        self.recent = []
        if self.last_reload is not None and self.clock() - self.last_reload < self.cooldown:
            log.warning("still slow %.0f min after reloading: last %d full batches median %.2fx real time",
                        (self.clock() - self.last_reload) / 60, self.window, median)
            return
        log.warning("slowed down: last %d full batches median %.2fx real time (under %.2fx); "
                    "reloading the model before the next batch", self.window, median, self.threshold)
        self.reload_due = True

    def reset(self) -> None:
        """A fresh model starts a fresh window."""
        self.recent = []
        self.reload_due = False

    def reloaded(self) -> None:
        self.reset()
        self.last_reload = self.clock()


class ReferenceCache:
    """Encoded voice clips, so a clip is read and encoded once rather than for every item that uses
    it. The pinned runtime encodes the reference anew for each item of a batch (and again for a
    directed item's guidance prompt): 35-50 ms each on the RTX 4070, 1-3 s of every 32-item batch.
    An entry is keyed by the clip's path and checked against its size and modification time, so a
    replaced clip is encoded afresh; the synthesizer clears the cache when the model unloads. The
    codes are small CPU tensors (a 13 s clip is 165 x 16 int16) that the runtime only reads."""

    def __init__(self, encode):
        self.encode = encode
        self.codes: dict[str, tuple[tuple[int, int], object]] = {}

    def __call__(self, audio_tokenizer, audio_path):
        stat = os.stat(audio_path)
        signature = (stat.st_size, stat.st_mtime_ns)
        entry = self.codes.get(str(audio_path))
        if entry is None or entry[0] != signature:
            entry = (signature, self.encode(audio_tokenizer, audio_path))
            self.codes[str(audio_path)] = entry
        return entry[1]

    def clear(self) -> None:
        self.codes.clear()


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
        self.references: ReferenceCache | None = None

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
        self._cache_references()
        self._fast_depth(model)
        log.info("model loaded in %.1fs (codec %s frames/s)", time.perf_counter() - started, self.frame_rate)

    @staticmethod
    def _fast_depth(model) -> None:
        """Plain depth decoding as CUDA graph replays (fast_depth.py; 2.2-2.4x faster batches on
        2026-10-05, WORKLOG §53). BREEZE_FAST_DEPTH=0 keeps the stock decoder, as does a failed capture."""
        if os.environ.get("BREEZE_FAST_DEPTH", "1").strip().lower() in ("0", "off", "false", "no"):
            log.info("fast depth decoder off (BREEZE_FAST_DEPTH)")
            return
        stock = model.depth_decoder.generate
        started = time.perf_counter()
        try:
            import fast_depth
            fast = fast_depth.install(model, max_rows=max(1, int(os.environ.get("BREEZE_MAX_BATCH", DEFAULT_MAX_BATCH))))
        except Exception:
            model.depth_decoder.generate = stock
            log.exception("fast depth decoder off: capturing its CUDA graphs failed")
            return
        log.info("fast depth decoder: CUDA graphs for %s rows in %.1fs", "/".join(map(str, fast.buckets)),
                 time.perf_counter() - started)

    def _read_frame_rate(self) -> float | None:
        """Codec frames per second = sample rate / samples per frame, from the audio tokenizer's config."""
        try:
            cfg = json.loads((self.model_dir / "audio_tokenizer" / "config.json").read_text())
            return float(cfg["output_sample_rate"]) / float(cfg["decode_upsample_rate"])
        except (OSError, KeyError, ValueError, ZeroDivisionError):
            return None

    def _cache_references(self) -> None:
        """Route the runtime's reference encoding (templates._encode_prompt_audio, which every audio
        segment goes through) via a ReferenceCache, once per process."""
        if self.references is not None:
            return
        from breeze_infer import templates
        encode = getattr(templates, "_encode_prompt_audio", None)
        if not callable(encode):
            log.warning("reference cache off: breeze_infer.templates has no _encode_prompt_audio")
            return
        self.references = templates._encode_prompt_audio = ReferenceCache(encode)

    def unload(self) -> None:
        self._runtime = None
        if self.references is not None:
            self.references.clear()  # codes belong to the codec that made them
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


def create_app(synth, voices_dir: str | None = None, max_batch: int | None = None,
               guard: SlowdownGuard | None = None, restart=restart_process) -> FastAPI:
    voices = Path(voices_dir or os.environ.get("BREEZE_VOICES_DIR", "/voices"))
    max_batch = max(1, int(max_batch or os.environ.get("BREEZE_MAX_BATCH", DEFAULT_MAX_BATCH)))
    if guard is None:  # 3/4 of the batch size counts as full, at most FULL_CHUNK: 32 long units are a full chunk too
        guard = SlowdownGuard(float(os.environ.get("BREEZE_SLOW_RTF") or SLOW_REAL_TIME),
                              min(FULL_CHUNK, math.ceil(max_batch * 0.75)))
    lock = threading.Lock()  # one generation at a time; the GPU is shared and memory is tight
    # Set by an allocator fault: the half-mapped pages stay until the process ends (unloading the
    # model doesn't free them), so the next unload restarts the server instead.
    allocator_damaged = threading.Event()
    app = FastAPI(title="Breeze TTS 2")

    def ensure_loaded() -> None:
        """Load the model if it isn't, or reload it if the guard asked. Call with the lock held."""
        if guard.reload_due and synth.loaded:
            log.warning("reloading: %s", gpu_memory_report())
            synth.unload()
            synth.load()
            guard.reloaded()
            log.info("reloaded: %s", gpu_memory_report())
        elif not synth.loaded:
            synth.load()
            guard.reset()

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
        """Generate one chunk into results[id] = (audio, error). On out-of-memory (or an allocator
        fault, which is how WSL runs out), split and retry."""
        longest = max(len(r["text"]) for r in chunk)
        tokens = max_tokens_for(getattr(synth, "frame_rate", None), longest)
        started = time.perf_counter()
        reset_peak_memory()
        try:
            audios = synth.generate(chunk, cfg, seed, tokens)
            if len(audios) != len(chunk):
                raise RuntimeError(f"model returned {len(audios)} takes for {len(chunk)} items")
        except Exception as exc:
            fault = is_allocator_fault(exc)
            if not (fault or is_oom(exc)):
                log.exception("chunk of %d failed", len(chunk))
                for r in chunk:
                    results[r["id"]] = (None, f"{type(exc).__name__}: {exc}")
                return
            error = f"{type(exc).__name__}: {exc}"
        else:
            elapsed = time.perf_counter() - started
            seconds = sum(len(a) for a in audios) / SAMPLE_RATE
            real_time = seconds / elapsed if elapsed > 0 else 0.0
            log.info("chunk of %d: %.1fs audio in %.1fs = %.2fx real time%s", len(chunk), seconds, elapsed,
                     real_time, chunk_memory_note())
            guard.record(len(chunk), real_time)
            for r, audio in zip(chunk, audios):
                results[r["id"]] = (audio, None if len(audio) else "no audio generated")
            return

        # Leave the exception handler first: its traceback can hold GPU tensors from generate().
        empty_cuda_cache()
        if fault and not allocator_damaged.is_set():
            allocator_damaged.set()
            log.error("GPU allocator fault at a chunk of %d (%s); %s. Chunks that need more are split; "
                      "the next unload restarts the server to clear it", len(chunk), error.strip(),
                      gpu_memory_report())
        if len(chunk) > 1:
            half = len(chunk) // 2
            log.warning("out of memory at batch %d; retrying as %d + %d", len(chunk), half, len(chunk) - half)
            run_chunk(chunk[:half], cfg, seed, results)
            run_chunk(chunk[half:], cfg, seed, results)
            return
        log.error("out of memory on a single item (%s)", chunk[0]["id"])
        for r in chunk:
            results[r["id"]] = (None, error)

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
            ensure_loaded()
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
            ensure_loaded()
        return {"loaded": True}

    @app.post("/api/unload")
    def unload(background: BackgroundTasks):
        with lock:
            synth.unload()
            guard.reset()
        if allocator_damaged.is_set():
            # After the reply: the app unloads before the cast LLM runs, so nothing waits on Breeze.
            log.warning("restarting the server to clear the GPU allocator fault")
            background.add_task(restart)
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
