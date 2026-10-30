"""Resemble Enhance service for cleaning voice reference clips.

Cleans voice clips in the voices library, especially Breeze-designed voices.
The model needs about 4.5 GB of GPU memory at peak so it cannot run while
Breeze is loaded; the app unloads Breeze first.

Endpoints:
  GET /health -> {"status": "ok", "loaded": bool}
  POST /enhance -> raw WAV in, enhanced WAV out
  POST /api/unload -> {"loaded": false}
"""

import gc
import io
import os
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool

NFE = 128
SOLVER = "rk4"
LAMBD = 1.0
TAU = 0.5  # owner chose these by ear on 2026-10-30 (variant "8D": default tau, twice the steps, rk4)


class ResembleEnhancer:
    """The real model. The weights load on the first enhance (resemble_enhance caches them per run
    folder and device) and are dropped again by unload()."""

    def __init__(self):
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @staticmethod
    def _run_dir() -> Path:
        import resemble_enhance
        return Path(resemble_enhance.__file__).parent / "model_repo" / "enhancer_stage2"

    def load(self) -> None:
        if self._loaded:
            return
        if not (self._run_dir() / "ds" / "G" / "default" / "mp_rank_00_model_states.pt").exists():
            from resemble_enhance.enhancer.download import download
            download()  # first start: clone the weights into the volume
        self._loaded = True

    def unload(self) -> None:
        import torch
        from resemble_enhance.enhancer.inference import load_enhancer
        load_enhancer.cache_clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False

    def enhance(self, samples: np.ndarray, rate: int):
        import torch
        from resemble_enhance.enhancer.inference import enhance
        # A Path run_dir skips the package's git pull on every load (it joins paths with "/").
        out, out_rate = enhance(torch.from_numpy(samples), rate, "cuda", nfe=NFE, solver=SOLVER,
                                lambd=LAMBD, tau=TAU, run_dir=self._run_dir())
        return out.cpu().numpy().astype(np.float32), int(out_rate)


def create_app(enhancer):
    app = FastAPI(title="Resemble Enhance")
    lock = threading.Lock()
    idle_timer = None

    def cancel_idle():
        nonlocal idle_timer
        if idle_timer:
            idle_timer.cancel()
            idle_timer = None

    def schedule_idle():
        nonlocal idle_timer
        cancel_idle()
        seconds = float(os.environ.get("RESEMBLE_IDLE_UNLOAD_SECONDS", "120"))
        def _do():
            with lock:
                if enhancer.loaded:
                    enhancer.unload()
        idle_timer = threading.Timer(seconds, _do)
        idle_timer.daemon = True
        idle_timer.start()

    @app.get("/health")
    def health():
        return {"status": "ok", "loaded": enhancer.loaded}  # no lock: answers during a clean

    @app.post("/enhance")
    async def enhance_endpoint(request: Request):
        body = await request.body()
        if not body:
            raise HTTPException(400, "empty body")
        try:
            buf = io.BytesIO(body)
            data, sr = sf.read(buf, dtype="float32")
        except Exception:
            raise HTTPException(400, "undecodable audio")
        if data.size == 0:
            raise HTTPException(400, "empty audio")
        if data.ndim > 1:
            data = data.mean(axis=1)

        def work():
            with lock:
                if not enhancer.loaded:
                    enhancer.load()
                return enhancer.enhance(np.ascontiguousarray(data, dtype=np.float32), int(sr))

        out_samples, out_rate = await run_in_threadpool(work)  # seconds of GPU work: keep /health answering
        peak = np.max(np.abs(out_samples))
        if peak > 0:
            target = 10 ** (-1 / 20)
            scale = target / peak
            out_samples = out_samples * scale
        out_buf = io.BytesIO()
        sf.write(out_buf, out_samples, out_rate, format="WAV", subtype="PCM_16")
        schedule_idle()
        return Response(content=out_buf.getvalue(), media_type="audio/wav")

    @app.post("/api/unload")
    def unload():
        cancel_idle()
        with lock:
            if enhancer.loaded:
                enhancer.unload()
        return {"loaded": False}

    return app


if __name__ == "__main__":
    import uvicorn
    app = create_app(ResembleEnhancer())
    uvicorn.run(app, host="0.0.0.0", port=8006)
