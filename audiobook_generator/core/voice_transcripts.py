"""The exact words spoken in each voice clip, which Breeze needs next to the clip (its `ref_text`).

Transcribed once with the app's Whisper (core.speech_check) over the whole clip and saved in
VOICE_TRANSCRIPTS_FILE (app data, next to voice_features.json) with the voice file's size and
modification time, so a replaced clip is transcribed again.
"""
import json
import logging
import os
from typing import Dict, Optional

from pydub import AudioSegment

from audiobook_generator.core import speech_check
from audiobook_generator.core.voice_measure import file_signature

logger = logging.getLogger(__name__)

VOICE_TRANSCRIPTS_FILE = "voice_transcripts.json"


def _load(path: str) -> Dict[str, dict]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    voices = data.get("voices") if isinstance(data, dict) else None
    return {voice: entry for voice, entry in (voices or {}).items()
            if isinstance(entry, dict) and isinstance(entry.get("text"), str)}


def _write(voices: Dict[str, dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "voices": voices}, f, ensure_ascii=False, indent=1, sort_keys=True)
    os.replace(tmp, path)


def transcript(voice: str, voices_dir: Optional[str] = None, path: Optional[str] = None) -> Optional[str]:
    """The words in the voice's clip, or None when they are unknown: no voices folder, no clip, or
    no speech checker (SPEECH_CHECK_MODEL unset or Whisper unavailable) to transcribe a new one."""
    voices_dir = voices_dir or os.environ.get("TTS_VOICES_DIR")
    if not voices_dir:
        return None
    clip = os.path.join(voices_dir, voice)
    try:
        signature = file_signature(clip)
    except OSError:
        return None
    path = path or VOICE_TRANSCRIPTS_FILE
    saved = _load(path)
    entry = saved.get(voice)
    if entry and entry.get("signature") == signature:
        return entry["text"]
    checker = speech_check.get()
    if checker is None:
        return None
    try:
        text = checker.transcribe(AudioSegment.from_file(clip)).text.strip()
    except Exception as error:
        logger.warning("Could not transcribe the voice clip %s: %s", voice, error)
        return None
    if not text:
        return None
    # Logged the first time only: a clip Whisper mishears (Teen.mp3 comes out as "Hello. Hello. ...") shows here.
    logger.info("Voice transcript %s: %s", voice, text)
    saved[voice] = {"text": text, "signature": signature}
    try:
        _write(saved, path)
    except OSError as error:
        logger.warning("Could not save the voice transcripts: %s", error)
    return text
