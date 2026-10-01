"""Client for the Breeze TTS 2 server (BREEZE_BASE_URL, e.g. http://breeze:8005).

Breeze is only fast when many sentences are generated together (measured 2026-10-01 on this
machine: 1 at a time 0.45x real time, 32 at a time 7.1x; Chatterbox 2.5x), so the app sends a
chapter's units in /v1/batch requests instead of one request per unit. The server loads its model
on demand and can be asked to give the GPU back (/api/unload), see core.engine_gpu.
"""
import base64
import io
import logging
import os
import time
from typing import Callable, List, Optional, Union

import requests
from pydub import AudioSegment

logger = logging.getLogger(__name__)

HEALTH_TIMEOUT_SECONDS = 10
LOAD_TIMEOUT_SECONDS = 300      # loading takes ~20-30 s; a cold cache downloads the model first
UNLOAD_TIMEOUT_SECONDS = 120
# A batch of 32 sentences takes ~20-60 s generating; this leaves room for a slow or busy GPU.
BATCH_READ_TIMEOUT_SECONDS = 1800
CONNECT_TIMEOUT_SECONDS = 10
# Same wait as OpenAITTSProvider._create_speech: a server that is still starting is waited for.
RETRYABLE_STATUS_CODES = (502, 503, 504)
SERVER_WAIT_TOTAL_SECONDS = 600
SERVER_WAIT_INITIAL_DELAY_SECONDS = 2.0
SERVER_WAIT_MAX_DELAY_SECONDS = 30.0

Take = Union[AudioSegment, str]  # the audio, or the server's reason it made none


def base_url() -> str:
    return os.environ.get("BREEZE_BASE_URL", "").strip().rstrip("/")


def configured() -> bool:
    return bool(base_url())


def health() -> Optional[dict]:
    """{"status", "loaded", "max_batch"}, or None when the server cannot be reached."""
    if not configured():
        return None
    try:
        response = requests.get(f"{base_url()}/health", timeout=HEALTH_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except Exception as error:
        logger.debug("Breeze health unavailable: %s", error)
        return None


def loaded() -> Optional[bool]:
    """Whether the model is in GPU memory; None when the server cannot be reached."""
    info = health()
    return None if info is None else bool(info.get("loaded"))


def load() -> bool:
    """Load the model (blocks ~20-30 s). False (logged) when it could not be done."""
    try:
        response = requests.post(f"{base_url()}/api/load", json={}, timeout=LOAD_TIMEOUT_SECONDS)
        response.raise_for_status()
        return bool(response.json().get("loaded"))
    except Exception as error:
        logger.warning("Breeze could not be loaded: %s", error)
        return False


def unload() -> bool:
    """Free Breeze's GPU memory. False (logged) when it could not be done."""
    try:
        response = requests.post(f"{base_url()}/api/unload", json={}, timeout=UNLOAD_TIMEOUT_SECONDS)
        response.raise_for_status()
        logger.info("Breeze model unloaded (GPU memory freed)")
        return True
    except Exception as error:
        logger.warning("Breeze could not be unloaded (%s)", error)
        return False


def _decode(item: dict) -> Take:
    if item.get("error"):
        return str(item["error"])
    if not item.get("wav_b64"):
        return "the server returned no audio"
    try:
        return AudioSegment.from_file(io.BytesIO(base64.b64decode(item["wav_b64"])), format="wav")
    except Exception as error:
        return f"unreadable audio from the server: {error}"


def synthesize_batch(items: List[dict], seed: Optional[int] = None, *,
                     sleep: Callable[[float], None] = time.sleep,
                     clock: Callable[[], float] = time.monotonic) -> List[Take]:
    """Generate every item ({"id", "text", "voice", "ref_text", "instruction", "cfg_scale"}) in one
    request; one result per item, in order: an AudioSegment, or the error string for that item.

    A connection error or 502/503/504 (the server still starting) is waited out for up to
    SERVER_WAIT_TOTAL_SECONDS with backoff, like OpenAITTSProvider._create_speech; anything else
    raises. `sleep`/`clock` are injectable so a test needs no real wait."""
    deadline = clock() + SERVER_WAIT_TOTAL_SECONDS
    delay = SERVER_WAIT_INITIAL_DELAY_SECONDS
    while True:
        try:
            response = requests.post(f"{base_url()}/v1/batch", json={"items": items, "seed": seed},
                                     timeout=(CONNECT_TIMEOUT_SECONDS, BATCH_READ_TIMEOUT_SECONDS))
            if response.status_code in RETRYABLE_STATUS_CODES:
                raise requests.ConnectionError(f"HTTP {response.status_code}")
            response.raise_for_status()
            break
        except (requests.ConnectionError, requests.Timeout) as error:
            remaining = deadline - clock()
            if remaining <= 0:
                logger.error("Breeze still unavailable after %ss, giving up: %s", SERVER_WAIT_TOTAL_SECONDS, error)
                raise
            wait_for = min(delay, remaining)
            logger.warning("Waiting for Breeze to become available, retrying in %.1fs: %s", wait_for, error)
            sleep(wait_for)
            delay = min(delay * 2, SERVER_WAIT_MAX_DELAY_SECONDS)
    results = response.json().get("items", [])
    if len(results) != len(items):
        raise RuntimeError(f"Breeze answered {len(results)} items for a batch of {len(items)}")
    return [_decode(item) for item in results]
