"""Which TTS model holds the one 12 GB GPU: Chatterbox (~4 GB), Breeze (~8 GB) or the cast LLM (~9 GB),
one at a time. A book needs its own engine loaded and the other unloaded; Kokoro runs elsewhere
and needs nothing.

The queue asks ready_for_book(engine) on its worker thread every tick, so it never blocks: a
handover (unload one model, load the other, ~20-30 s) runs in a background thread and the answer is
False until the wanted model reports loaded.
"""
import logging
import threading
from typing import Callable, Optional

from audiobook_generator.core import breeze_client, chatterbox_control

logger = logging.getLogger(__name__)

_thread: Optional[threading.Thread] = None
_lock = threading.Lock()


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


def _to_breeze() -> None:
    _wait_for_chatterbox_reload()
    if chatterbox_control.model_loaded():
        chatterbox_control.unload()
    breeze_client.load()


def _to_chatterbox() -> None:
    breeze_client.unload()


def _start(engine: str, target: Callable[[], None]) -> None:
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
