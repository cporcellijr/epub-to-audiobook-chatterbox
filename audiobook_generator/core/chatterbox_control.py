"""Unload and reload the Chatterbox model around a cast analysis.

The GPU cannot hold Chatterbox (about 4.4 GB idle) and a local LLM at once, so the analysis job
frees Chatterbox first (POST /api/unload) and brings it back afterwards (POST /restart_server,
which reloads the model and returns once it is loaded), then confirms with GET /api/model-info.
While Chatterbox is unloaded /v1/audio/speech answers 503, so the queue also asks here before
starting a book, and triggers a reload if an analysis died before it could.

Settings are read per call: LLM_UNLOAD_CHATTERBOX (default on) and the Chatterbox URL.
"""
import json
import logging
import os
import threading
import time
import urllib.request
from typing import Callable, Optional

logger = logging.getLogger(__name__)

INFO_TIMEOUT_SECONDS = 10
UNLOAD_TIMEOUT_SECONDS = 120
# A reload takes ~12 s from a warm cache, plus ~17 s to compile the token loop (F-45); a cold
# cache downloads the model first, hence the long allowance.
RELOAD_TIMEOUT_SECONDS = 900
POLL_SECONDS = 2.0
_OFF_VALUES = ("0", "off", "false", "no")


def chatterbox_url() -> str:
    """Chatterbox root URL (no /v1)."""
    explicit = os.environ.get("CHATTERBOX_URL", "").rstrip("/")
    if explicit:
        return explicit
    base = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def unload_enabled() -> bool:
    """LLM_UNLOAD_CHATTERBOX: unset or anything but 0/off/false/no means on."""
    return os.environ.get("LLM_UNLOAD_CHATTERBOX", "on").strip().lower() not in _OFF_VALUES


def _get_json(path: str, timeout: float) -> dict:
    with urllib.request.urlopen(f"{chatterbox_url()}{path}", timeout=timeout) as response:
        return json.load(response)


def _post(path: str, timeout: float) -> bytes:
    request = urllib.request.Request(f"{chatterbox_url()}{path}", data=b"{}",
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def model_loaded() -> Optional[bool]:
    """True/False from /api/model-info, or None when Chatterbox cannot be reached at all."""
    if not chatterbox_url():
        return None
    try:
        return bool(_get_json("/api/model-info", INFO_TIMEOUT_SECONDS).get("loaded"))
    except Exception as e:
        logger.debug(f"Chatterbox model-info unavailable: {e}")
        return None


def unload() -> bool:
    """Free Chatterbox's GPU memory. False (logged) when it could not be done; the caller then
    runs the LLM alongside Chatterbox rather than not at all."""
    try:
        _post("/api/unload", UNLOAD_TIMEOUT_SECONDS)
        logger.info("Chatterbox model unloaded (GPU memory freed for the LLM)")
        return True
    except Exception as e:
        logger.warning(f"Chatterbox could not be unloaded ({e}); continuing with it loaded")
        return False


def wait_until_loaded(timeout: float = RELOAD_TIMEOUT_SECONDS, sleep: Callable[[float], None] = time.sleep,
                      clock: Callable[[], float] = time.monotonic) -> bool:
    """Poll /api/model-info until it reports loaded, or the timeout passes."""
    deadline = clock() + timeout
    while True:
        if model_loaded():
            return True
        if clock() >= deadline:
            return False
        sleep(POLL_SECONDS)


def reload(timeout: float = RELOAD_TIMEOUT_SECONDS) -> bool:
    """Reload the model and wait until /api/model-info confirms it. /restart_server itself blocks
    until the load finishes; an error there still falls through to polling, since the model may
    well come up anyway (or the server may be restarting on its own)."""
    try:
        _post("/restart_server", timeout)
    except Exception as e:
        logger.warning(f"Chatterbox reload request failed ({e}); waiting for the model regardless")
    loaded = wait_until_loaded(timeout)
    if loaded:
        logger.info("Chatterbox model loaded again")
    else:
        logger.error(f"Chatterbox model still not loaded after {timeout:.0f}s; books will wait for it")
    return loaded


_reload_thread: Optional[threading.Thread] = None
_reload_lock = threading.Lock()


def ready_for_book() -> bool:
    """For the queue: may a book start now? False while Chatterbox reports its model unloaded, in
    which case a reload is kicked off in the background (once) so the next tick can proceed.
    An unreachable Chatterbox does not hold the queue: the book's own start-up wait (F-02)
    covers a server that is still coming up."""
    global _reload_thread
    if model_loaded() is not False:
        return True
    with _reload_lock:
        if _reload_thread is None or not _reload_thread.is_alive():
            logger.info("Chatterbox is unloaded; reloading it before the next book")
            _reload_thread = threading.Thread(target=reload, name="chatterbox-reload", daemon=True)
            _reload_thread.start()
    return False
