"""Cleaning voice clips with the Resemble Enhance service (RESEMBLE_BASE_URL, e.g. http://resemble:8006).

A clip's room and colouring are copied by Breeze into every line spoken with it. Resemble Enhance
(resemble/) regenerates a clip clean: on 2026-10-30 the owner preferred the cleaned version of every
voice in the library, so a newly designed voice is cleaned too, once per design run (not per voice).

Resemble needs ~4.5 GB of GPU memory and Breeze up to 10.5 GB of the 12 GB card, so Breeze is unloaded
first; its next request loads it again. When Breeze can't be unloaded nothing is cleaned: an overfull
GPU spills into Windows' RAM under WSL (core.engine_gpu).
"""
import io
import logging
import os
from typing import List, Optional

import requests
from pydub import AudioSegment

from audiobook_generator.core import breeze_client, gpu_reservation, voice_measure, voice_transcripts

logger = logging.getLogger(__name__)

CLEAN_TIMEOUT_SECONDS = 300  # the first clip waits for the model to load (once ever: to download, ~1.6 GB)
UNLOAD_TIMEOUT_SECONDS = 60
MAX_LENGTH_CHANGE_MS = 500   # Resemble keeps a clip's length; anything else is not the same clip


def base_url() -> str:
    return os.environ.get("RESEMBLE_BASE_URL", "").strip().rstrip("/")


def configured() -> bool:
    return bool(base_url())


def _clean_one(name: str, folder: str) -> None:
    """Replace one clip with its cleaned version, keeping its recorded words and measurement (they
    are re-stamped with the new file's signature, so nothing is transcribed or measured again)."""
    path = os.path.join(folder, name)
    with open(path, "rb") as f:
        original = f.read()
    text = voice_transcripts.transcript(name, folder)  # still the saved one: the file is unchanged
    measurement = voice_measure.load_features().get(name)
    response = requests.post(f"{base_url()}/enhance", data=original, headers={"Content-Type": "audio/wav"},
                             timeout=CLEAN_TIMEOUT_SECONDS)
    response.raise_for_status()
    before = AudioSegment.from_file(io.BytesIO(original))
    after = AudioSegment.from_file(io.BytesIO(response.content), format="wav")
    if abs(len(after) - len(before)) > MAX_LENGTH_CHANGE_MS:
        raise ValueError(f"the cleaned clip is {len(after) / 1000:.1f} s long, the original {len(before) / 1000:.1f} s")
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(response.content)
    os.replace(tmp, path)
    if text:
        voice_transcripts.remember(name, text, folder)
    if measurement:
        kept = {key: measurement[key] for key in ("f0_median", "f0_range", "hnr")}
        voice_measure.save_features(name, kept, voice_measure.file_signature(path))


def clean_voices(file_names: List[str], voices_dir: Optional[str] = None) -> List[str]:
    """Clean these clips of the voices folder in place; returns the names cleaned. A clip that could
    not be cleaned stays as it was (logged), so this never fails the design that made it."""
    if not configured() or not file_names:
        return []
    folder = voices_dir or os.environ.get("TTS_VOICES_DIR") or ""
    cleaned = []
    with gpu_reservation.session("breeze"):  # held already inside a design or a book: re-entrant
        if breeze_client.configured() and breeze_client.loaded() and not breeze_client.unload():
            logger.warning("Breeze kept the GPU, so these voices stay as designed: %s", ", ".join(file_names))
            return []
        try:
            for name in file_names:
                try:
                    _clean_one(name, folder)
                except Exception as error:
                    logger.warning("Voice %s was not cleaned (kept as designed): %s", name, error)
                    continue
                cleaned.append(name)
                logger.info("Voice cleaned: %s", name)
        finally:
            try:
                requests.post(f"{base_url()}/api/unload", json={}, timeout=UNLOAD_TIMEOUT_SECONDS).raise_for_status()
            except requests.RequestException as error:
                logger.warning("Resemble could not be unloaded (it unloads itself when idle): %s", error)
    return cleaned
