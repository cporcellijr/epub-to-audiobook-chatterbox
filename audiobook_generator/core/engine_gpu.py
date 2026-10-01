"""Which TTS model holds the one 12 GB GPU: Chatterbox (~4 GB), Breeze (~8 GB) or the cast LLM (~9 GB),
one at a time. A book needs its own engine loaded and the other unloaded; Kokoro runs elsewhere
and needs nothing.

The queue asks ready_for_book(engine) on its worker thread every tick, so it never blocks: a
handover (unload one model, load the other, ~20-30 s) runs in a background thread and the answer is
False until the wanted model reports loaded.

The LLM is only asked to leave: Ollama keeps a model loaded for its keep-alive (5 min here) after the
last request, so a book queued straight after its cast analysis would load Breeze beside it. Under
WSL an overfull GPU spills into Windows' RAM instead of failing; on 2026-10-01 that took Docker down.
"""
import logging
import threading
from typing import Callable, Optional

import requests

from audiobook_generator.core import breeze_client, cast_llm, chatterbox_control

logger = logging.getLogger(__name__)

LLM_STATUS_TIMEOUT_SECONDS = 5
LLM_UNLOAD_TIMEOUT_SECONDS = 30

_thread: Optional[threading.Thread] = None
_lock = threading.Lock()


def unload_llm() -> bool:
    """Ask the cast LLM's server to free the GPU now instead of at the end of its keep-alive.
    Ollama only (its /api/ps lists what is loaded); another server answers 404 and is left alone.
    True when nothing is left loaded; False (logged) when that could not be done, which never
    stops a handover: an unreachable LLM server holds no GPU memory worth waiting for."""
    base = cast_llm.llm_base_url()
    if not base:
        return True
    root = base[:-len("/v1")] if base.endswith("/v1") else base
    try:
        response = requests.get(f"{root}/api/ps", timeout=LLM_STATUS_TIMEOUT_SECONDS)
        if response.status_code == 404:
            return True
        response.raise_for_status()
        for model in response.json().get("models") or []:
            requests.post(f"{root}/api/generate", json={"model": model["name"], "keep_alive": 0},
                          timeout=LLM_UNLOAD_TIMEOUT_SECONDS).raise_for_status()
            logger.info("Cast LLM %s unloaded (GPU memory freed for Breeze)", model["name"])
        return True
    except Exception as error:
        logger.warning("Could not unload the cast LLM (%s); loading Breeze anyway", error)
        return False


def unload_breeze_if_loaded() -> bool:
    """Free Breeze's GPU memory before the LLM runs. True when it was unloaded; False when Breeze
    is not configured, not loaded or could not be unloaded."""
    if not breeze_client.configured() or not breeze_client.loaded():
        return False
    return breeze_client.unload()


def _wait_for_chatterbox_reload() -> None:
    reload_thread = chatterbox_control._reload_thread
    if reload_thread is not None and reload_thread.is_alive():
        reload_thread.join()


def _to_breeze() -> bool:
    _wait_for_chatterbox_reload()
    if chatterbox_control.model_loaded() and not chatterbox_control.unload():
        return False
    unload_llm()
    return breeze_client.load()


def prepare_breeze() -> None:
    """Wait for a handover, then load Breeze for a queue-reserved standalone request."""
    handover = _thread
    if handover is not None and handover.is_alive():
        handover.join()
    with _lock:
        if not _to_breeze():
            raise RuntimeError("Could not give Breeze the GPU. Try again once the speech services are ready.")


def _to_chatterbox() -> None:
    breeze_client.unload()


def _start(engine: str, target: Callable[[], object]) -> None:
    """Run one handover in the background unless one is already running (then wait for it)."""
    global _thread
    with _lock:
        if _thread is not None and _thread.is_alive():
            return
        logger.info("Switching the GPU to %s before the next book", engine)
        _thread = threading.Thread(target=target, name=f"gpu-to-{engine}", daemon=True)
        _thread.start()


def ready_for_book(engine: str) -> bool:
    """May a book for this engine start now? Never blocks for long; see the module docstring."""
    if engine == "kokoro":
        return True
    if engine == "breeze":
        if not breeze_client.configured():
            return True  # build_config refuses the book with a clear message
        if breeze_client.loaded() is True and not chatterbox_control.model_loaded():
            return True
        _start("breeze", _to_breeze)
        return False
    if breeze_client.configured() and breeze_client.loaded():
        _start("chatterbox", _to_chatterbox)
        return False
    if _thread is not None and _thread.is_alive():
        return False
    return chatterbox_control.ready_for_book()
