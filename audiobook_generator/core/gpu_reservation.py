import os
import uuid
import threading
from contextlib import contextmanager
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import json
import logging
import time

logger = logging.getLogger(__name__)
_held = threading.local()


def configured():
    return bool(os.environ.get("GPU_COORDINATOR_URL", "").strip())


def _post(path, body, timeout=600):
    base = os.environ["GPU_COORDINATOR_URL"].strip().rstrip("/")
    token_env = os.environ.get("GPU_COORDINATOR_TOKEN", "")
    url = base + path
    data = json.dumps(body).encode()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token_env}",
    }
    req = Request(url, data=data, headers=headers, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


@contextmanager
def session(engine):
    if not configured():
        yield
        return
    if engine not in ("breeze", "ollama", "strata", "chatterbox", "training"):
        yield
        return

    if getattr(_held, "engine", None):
        if _held.engine != engine:
            raise RuntimeError("Cannot switch engines inside an active GPU reservation")
        yield _held.token
        return

    token_env = os.environ.get("GPU_COORDINATOR_TOKEN", "")
    if not token_env:
        raise RuntimeError("GPU_COORDINATOR_TOKEN required nonempty")

    token = uuid.uuid4().hex
    acquired_token = None
    hb_thread = None
    hb_event = threading.Event()
    hb_error = [None]
    caller_failed = False
    body_done = threading.Event()

    def hb_worker():
        while not hb_event.wait(10):
            try:
                _post("/heartbeat", {"token": acquired_token}, timeout=15)
                hb_error[0] = None
            except Exception:
                if not body_done.is_set():
                    hb_error[0] = "GPU reservation heartbeat failed"
                    logger.warning("GPU reservation heartbeat failed; retrying")

    try:
        while True:
            try:
                result = _post("/acquire", {"engine": engine, "token": token}, timeout=600)
                break
            except HTTPError as e:
                if e.code == 409:
                    time.sleep(1)
                    continue
                raise RuntimeError(f"HTTP {e.code} on acquire") from None
            except (URLError, TimeoutError, OSError):
                raise RuntimeError("GPU coordinator is unavailable") from None

        acquired_token = result.get("token")
        if acquired_token != token:
            raise RuntimeError("Token mismatch in acquire response")

        hb_thread = threading.Thread(target=hb_worker, daemon=True)
        hb_thread.start()
        _held.engine, _held.token = engine, token
        yield acquired_token
    except BaseException:
        caller_failed = True
        raise
    finally:
        body_done.set()
        _held.engine, _held.token = None, None
        release_error = None
        try:
            _post("/release", {"token": token}, timeout=180)
        except Exception:
            release_error = "GPU release failed; the coordinator will recover the reservation"
            logger.warning(release_error)
        finally:
            hb_event.set()
            if hb_thread:
                hb_thread.join(timeout=20)

        if not caller_failed:
            if release_error:
                raise RuntimeError(release_error)
            if hb_error[0]:
                raise RuntimeError(hb_error[0])
