"""Personal web UI: EPUB -> audiobook through a local Chatterbox server, plus a voice lab.

The OpenAI-compatible provider is exposed twice over: pointed at Chatterbox via OPENAI_BASE_URL
(the default engine), or at Kokoro via KOKORO_BASE_URL (opt-in second engine); Breeze (BREEZE_BASE_URL, opt-in third engine)
speaks through its own batch client. Upstream's
multi-provider UI stays in web_ui.py, whose process helpers are reused here.

Environment:
    OPENAI_BASE_URL      Chatterbox OpenAI endpoint, e.g. http://chatterbox:8004/v1
    CHATTERBOX_URL       Chatterbox root (default: OPENAI_BASE_URL without /v1)
    CHATTERBOX_CONFIG    Chatterbox config.yaml (read-only mount) for the saved delivery settings
    TTS_VOICES_DIR       Chatterbox voices folder (writable mount, for adding/deleting voices)
    OPENAI_DEFAULT_VOICE Voice selected by default
    EBOOK_LIBRARY_DIR    Ebook library (read-only mount) for the searchable book picker
    KOKORO_BASE_URL      Kokoro OpenAI endpoint, e.g. http://kokoro:8880/v1 (optional: leave unset
                          to hide the Engine choice entirely and behave exactly as without Kokoro)
    KOKORO_DEFAULT_VOICE Kokoro voice id selected by default (default: the server's own
                          default_voice, else "af_heart")
    BREEZE_BASE_URL      Breeze TTS 2 server, e.g. http://breeze:8005 (optional: unset hides the Breeze
                          choice; it speaks with the same voice files as Chatterbox)
    LLM_BASE_URL         Local OpenAI-compatible chat endpoint for cast analysis, e.g.
                          http://ollama:11434/v1 (optional: leave unset to hide the Cast voice mode)
    LLM_MODEL            Chat model name for cast analysis
    LLM_API_KEY          Key for that endpoint, if it wants one
    LLM_UNLOAD_CHATTERBOX  "off" keeps Chatterbox's model loaded during a cast analysis (default on)
"""
import glob
import io
import json
import logging
import multiprocessing
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

import gradio as gr
import yaml
from gradio_log import Log
from pydub import AudioSegment

from audiobook_generator.book_parsers.base_book_parser import get_book_parser
from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.core import cast as cast_store
from audiobook_generator.core import breeze_client, delivery, voice_measure, voice_transcripts
from audiobook_generator.core.cast_llm import llm_configured
from audiobook_generator.core.chapter_selection import preselect_chapters
from audiobook_generator.core.chatterbox_control import chatterbox_url
from audiobook_generator.core.dialogue import dialogue_lines
from audiobook_generator.tts_providers.openai_tts_provider import (
    PARAGRAPH_MARK, VOICE_MODE_CAST, VOICE_MODE_DIALOGUE, VOICE_MODE_SINGLE, VOICE_MODES,
)
from audiobook_generator.ui import library_index, web_ui
from audiobook_generator.ui.job_queue import BOOK, CAST, DONE, FAILED, QUEUED, RUNNING, UPLOAD_KEYS, JobQueue, job_kind
from audiobook_generator.ui.web_ui import (
    OUTPUT_ROOT,
    default_openai_voice,
    openai_voice_choices,
    safe_folder_name,
    suggest_output_dir,
    timestamped_output_dir,
)
from audiobook_generator.utils.log_handler import generate_unique_log_path

logger = logging.getLogger(__name__)

PREVIEW_PHRASE = (
    "The rain had stopped by the time they reached the old bridge. "
    "\"We should have turned back an hour ago,\" she said, pulling her coat tighter."
)
FALLBACK_SETTINGS = {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8}
SHORT_SAMPLE_SECONDS = 6.0
PAUSE_FILTER = ("silenceremove=start_periods=1:start_threshold=-40dB:stop_periods=-1:"
                "stop_duration=0.3:stop_threshold=-40dB:stop_silence=0.15")


def _post_json(path: str, payload: dict, timeout: float) -> bytes:
    request = urllib.request.Request(
        f"{chatterbox_url()}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _http_error_detail(error: urllib.error.HTTPError) -> str:
    try:
        return json.loads(error.read()).get("detail", str(error))
    except Exception:
        return str(error)


# ---- Kokoro (second engine) ----

# Engines that speak with the voices folder's files (Breeze clones the same clips as Chatterbox).
FILE_VOICE_ENGINES = ("chatterbox", "breeze")
BREEZE_ENGINE_CHOICE = ("Breeze TTS 2 (best quality, needs the GPU to itself)", "breeze")
KOKORO_ENGLISH_PREFIXES = ("af_", "am_", "bf_", "bm_")  # American/British female/male
KOKORO_TIMEOUT_SECONDS = 5
_KOKORO_PREFIX_LABELS = {
    "af": "American female", "am": "American male",
    "bf": "British female", "bm": "British male",
}
_KOKORO_GRADE_LETTERS = "ABCDF"


def kokoro_base_url() -> str:
    """Kokoro's OpenAI-compatible endpoint, or "" when it isn't configured."""
    return os.environ.get("KOKORO_BASE_URL", "").rstrip("/")


def _kokoro_grade_sort_key(grade: Optional[str]) -> tuple:
    """Sort key for a Kokoro voice grade: best (A+) first, ungraded last."""
    if not grade:
        return (len(_KOKORO_GRADE_LETTERS), 0)
    letter, modifier = grade[0].upper(), grade[1:]
    letter_rank = (_KOKORO_GRADE_LETTERS.index(letter) if letter in _KOKORO_GRADE_LETTERS
                  else len(_KOKORO_GRADE_LETTERS))
    modifier_rank = {"+": 0, "": 1, "-": 2}.get(modifier, 1)
    return (letter_rank, modifier_rank)


def _kokoro_voice_label(voice_id: str, grade: Optional[str]) -> str:
    """Readable label for a Kokoro voice id, e.g. "af_heart" + grade "A" -> "Heart · American female · A"."""
    prefix, _, rest = voice_id.partition("_")
    name = rest.replace("_", " ").title() or voice_id
    parts = [name, _KOKORO_PREFIX_LABELS.get(prefix, prefix)]
    if grade:
        parts.append(grade)
    return " · ".join(parts)


def kokoro_voices_and_default() -> Tuple[list, Optional[str]]:
    """(label, id) choices for Kokoro's English voices (af_/am_/bf_/bm_ prefixes only; the
    server's other prefixes are other languages), best grade first, plus the voice to preselect:
    KOKORO_DEFAULT_VOICE when it's offered, else the server's own default_voice, else "af_heart".

    One request to Kokoro. If it can't be reached, warns and falls back to offering just the
    configured (or built-in) default, so the dropdown is never left empty.
    """
    configured_default = os.environ.get("KOKORO_DEFAULT_VOICE", "").strip()
    fallback = configured_default or "af_heart"
    try:
        with urllib.request.urlopen(f"{kokoro_base_url()}/audio/voices", timeout=KOKORO_TIMEOUT_SECONDS) as resp:
            data = json.load(resp)
    except Exception as e:
        gr.Warning(f"Could not reach Kokoro for its voice list: {e}")
        return [(fallback, fallback)], fallback
    voices = data.get("voices", []) if isinstance(data, dict) else []
    english = [v for v in voices if isinstance(v, dict) and isinstance(v.get("id"), str)
              and v["id"].startswith(KOKORO_ENGLISH_PREFIXES)]
    english.sort(key=lambda v: (_kokoro_grade_sort_key(v.get("overall_grade")), v["id"]))
    choices = [(_kokoro_voice_label(v["id"], v.get("overall_grade")), v["id"]) for v in english]
    values = [value for _, value in choices]
    server_default = data.get("default_voice") if isinstance(data, dict) else None
    if configured_default in values:
        default = configured_default
    elif server_default in values:
        default = server_default
    else:
        default = values[0] if values else fallback
    return choices, default


# ---- Delivery settings (Chatterbox generation defaults) ----

# The Chatterbox server rewrites config.yaml non-atomically (copy, then fill), so a read can briefly
# see it missing, empty or mid-write; one retry after this short a delay rides out that window (F-52).
SETTINGS_READ_RETRY_DELAY_SECONDS = 0.2


def _read_generation_defaults(path: str) -> Optional[dict]:
    """One attempt to read Chatterbox's generation_defaults; None if `path` is missing, a directory,
    empty or unparsable, instead of raising."""
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    defaults = data.get("generation_defaults")
    return defaults if isinstance(defaults, dict) else None


def read_saved_settings() -> dict:
    """Chatterbox's saved delivery settings, read from its config file (never blocks on a busy
    server, and never raises). build_ui() calls this at start-up, so a bad read must fall back to
    defaults instead of taking the UI down with it."""
    path = os.environ.get("CHATTERBOX_CONFIG")
    settings = dict(FALLBACK_SETTINGS)
    if not path:
        return settings
    defaults = _read_generation_defaults(path)
    if defaults is None:
        time.sleep(SETTINGS_READ_RETRY_DELAY_SECONDS)
        defaults = _read_generation_defaults(path)
    if defaults is None:
        print(f"Could not read Chatterbox settings from '{path}' (missing, a directory, empty or "
              "unparsable after one retry); using defaults.")
        return settings
    for key in settings:
        if isinstance(defaults.get(key), (int, float)):
            settings[key] = float(defaults[key])
    return settings


def load_saved_settings() -> tuple:
    settings = read_saved_settings()
    return settings["exaggeration"], settings["cfg_weight"], settings["temperature"]


def delivery_baseline_text(exaggeration: float, cfg_weight: float, temperature: float) -> str:
    """The Make tab's one-line summary of the baseline adaptive delivery (or a set per-book
    baseline with it off) will use for this book, following the Voice lab sliders live."""
    return (f"Delivery for this book: exaggeration {float(exaggeration):g} · CFG {float(cfg_weight):g} · "
            f"temperature {float(temperature):g}, from the Voice lab.")


def save_settings(exaggeration: float, cfg_weight: float, temperature: float) -> str:
    """Store the sliders as Chatterbox's defaults, which every book request uses."""
    payload = {"generation_defaults": {
        "exaggeration": round(float(exaggeration), 2),
        "cfg_weight": round(float(cfg_weight), 2),
        "temperature": round(float(temperature), 2),
    }}
    try:
        _post_json("/save_settings", payload, timeout=300)
    except urllib.error.HTTPError as e:
        raise gr.Error(f"Chatterbox refused the settings: {_http_error_detail(e)}")
    except Exception as e:
        raise gr.Error(f"Could not reach Chatterbox: {e}")
    return (f"Saved: exaggeration {payload['generation_defaults']['exaggeration']}, "
            f"CFG {payload['generation_defaults']['cfg_weight']}, "
            f"temperature {payload['generation_defaults']['temperature']}. Books use these from now on.")


# ---- Voice preview ----

_current_preview_path: Optional[str] = None


def _delete_if_exists(path: Optional[str]) -> None:
    if path and os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass


def sweep_voice_previews() -> None:
    """Delete leftover preview files from earlier runs (F-30): otherwise the container's writable
    layer grows by one small MP3 per Play, for as long as the container lives."""
    for path in glob.glob(os.path.join(tempfile.gettempdir(), "voice_preview_*")):
        _delete_if_exists(path)


def preview_voice(voice: str, phrase: str, exaggeration: float, cfg_weight: float,
                  temperature: float, speed: float) -> str:
    """Speak the phrase with the slider values (not saved) and return the audio file path."""
    global _current_preview_path
    if not voice:
        raise gr.Error("Pick a voice first.")
    text = (phrase or "").strip() or PREVIEW_PHRASE
    payload = {
        "text": text,
        "voice_mode": "predefined",
        "predefined_voice_id": voice,
        "output_format": "mp3",
        "split_text": True,
        "chunk_size": 500,
        "exaggeration": float(exaggeration),
        "cfg_weight": float(cfg_weight),
        "temperature": float(temperature),
        "speed_factor": float(speed),
    }
    try:
        audio = _post_json("/tts", payload, timeout=600)
    except urllib.error.HTTPError as e:
        raise gr.Error(f"Chatterbox could not make the preview: {_http_error_detail(e)}")
    except Exception as e:
        raise gr.Error(f"Could not reach Chatterbox: {e}")
    handle, path = tempfile.mkstemp(prefix="voice_preview_", suffix=".mp3")
    with os.fdopen(handle, "wb") as f:
        f.write(audio)
    _delete_if_exists(_current_preview_path)
    _current_preview_path = path
    return path


def preview_delivery_range(voice: str, phrase: str, exaggeration: float, cfg_weight: float,
                           temperature: float, speed: float) -> str:
    """Voice lab: the phrase spoken soft, then normal, then excited, around the current sliders
    (each mood's gain and peak guard applied, same as a book would get), as one clip with ~1 s of
    silence between the takes. Reuses preview_voice's temp-file handling."""
    global _current_preview_path
    if not voice:
        raise gr.Error("Pick a voice first.")
    text = (phrase or "").strip() or PREVIEW_PHRASE
    baseline = delivery.Baseline(float(exaggeration), float(cfg_weight), float(temperature))
    combined: Optional[AudioSegment] = None
    for mood in (delivery.MOOD_SOFT, delivery.MOOD_NORMAL, delivery.MOOD_EXCITED):
        mood_exaggeration, mood_cfg_weight, mood_temperature, gain_db = delivery.preset(mood, baseline)
        payload = {
            "text": text,
            "voice_mode": "predefined",
            "predefined_voice_id": voice,
            "output_format": "mp3",
            "split_text": True,
            "chunk_size": 500,
            "exaggeration": mood_exaggeration,
            "cfg_weight": mood_cfg_weight,
            "temperature": mood_temperature,
            "speed_factor": float(speed),
        }
        try:
            audio_bytes = _post_json("/tts", payload, timeout=600)
        except urllib.error.HTTPError as e:
            raise gr.Error(f"Chatterbox could not make the {mood} sample: {_http_error_detail(e)}")
        except Exception as e:
            raise gr.Error(f"Could not reach Chatterbox: {e}")
        clip = delivery.guarded_gain(AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3"), gain_db)
        combined = clip if combined is None else combined + AudioSegment.silent(duration=1000) + clip
    handle, path = tempfile.mkstemp(prefix="voice_preview_", suffix=".mp3")
    with os.fdopen(handle, "wb") as f:
        combined.export(f, format="mp3")
    _delete_if_exists(_current_preview_path)
    _current_preview_path = path
    return path


def _breeze_sample(voice: str) -> str:
    """One-off Breeze sample of PREVIEW_PHRASE in a voice file, as one item of a batch."""
    global _current_preview_path
    if not breeze_client.configured():
        raise gr.Error("Breeze is not configured (BREEZE_BASE_URL).")
    ref_text = voice_transcripts.transcript(voice)
    if not ref_text:
        raise gr.Error(f"Breeze needs the words spoken in {voice}, and the speech check's Whisper model "
                       "could not transcribe it (SPEECH_CHECK_MODEL).")
    try:
        take = breeze_client.synthesize_batch([{"id": "sample", "text": PREVIEW_PHRASE, "voice": voice,
                                                "ref_text": ref_text, "instruction": None, "cfg_scale": None}])[0]
    except Exception as e:
        raise gr.Error(f"Could not reach Breeze: {e}")
    if isinstance(take, str):
        raise gr.Error(f"Breeze could not make the sample: {take}")
    handle, path = tempfile.mkstemp(prefix="voice_preview_", suffix=".mp3")
    with os.fdopen(handle, "wb") as f:
        take.export(f, format="mp3")
    _delete_if_exists(_current_preview_path)
    _current_preview_path = path
    return path


def _kokoro_sample(voice: str, speed: float) -> str:
    """One-off Kokoro sample of PREVIEW_PHRASE; mirrors preview_voice's temp-file handling (F-30)."""
    global _current_preview_path
    base_url = kokoro_base_url()
    if not base_url:
        raise gr.Error("Kokoro is not configured (KOKORO_BASE_URL).")
    payload = {"model": "kokoro", "voice": voice, "input": PREVIEW_PHRASE,
              "response_format": "mp3", "speed": float(speed), "instructions": None}
    request = urllib.request.Request(
        f"{base_url}/audio/speech", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            audio = response.read()
    except urllib.error.HTTPError as e:
        raise gr.Error(f"Kokoro could not make the sample: {_http_error_detail(e)}")
    except Exception as e:
        raise gr.Error(f"Could not reach Kokoro: {e}")
    handle, path = tempfile.mkstemp(prefix="voice_preview_", suffix=".mp3")
    with os.fdopen(handle, "wb") as f:
        f.write(audio)
    _delete_if_exists(_current_preview_path)
    _current_preview_path = path
    return path


def sample_voice(engine: str, voice: str, speed: float) -> str:
    """Speak PREVIEW_PHRASE with the Make tab's selected engine, voice and speed, so a voice can
    be auditioned before queuing a book. Chatterbox goes through preview_voice with its saved
    delivery settings (i.e. what a book will actually sound like); Kokoro has no delivery sliders
    to save, so it is asked directly, and so is Breeze (a Chatterbox preview would need Chatterbox
    loaded, which a Breeze book keeps off the GPU).
    """
    if not voice:
        raise gr.Error("Pick a voice first.")
    if engine == "kokoro":
        return _kokoro_sample(voice, speed)
    if engine == "breeze":
        return _breeze_sample(voice)
    settings = read_saved_settings()
    return preview_voice(voice, PREVIEW_PHRASE, settings["exaggeration"], settings["cfg_weight"],
                         settings["temperature"], speed)


# ---- Adding voices ----

def _audio_seconds(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip() or 0)


def voice_name_from_sample(sample: Optional[str]) -> str:
    """Suggest a voice name from the uploaded file's name."""
    if not sample:
        return ""
    return safe_folder_name(os.path.splitext(os.path.basename(sample))[0])


def add_voice(sample: Optional[str], name: str, remove_pauses: bool, replace: bool,
             engine: str = "chatterbox") -> tuple:
    """Save an uploaded sample into the Chatterbox voices folder as <name>.wav.

    engine gates the Make-tab dropdown update only: adding a voice always affects the Chatterbox
    voices folder and the Voice lab regardless of which engine the Make tab currently has
    selected, but a Chatterbox file name must never land in the Make-tab dropdown while Kokoro is
    selected there.
    """
    voices_dir = os.environ.get("TTS_VOICES_DIR")
    if not voices_dir or not os.path.isdir(voices_dir):
        raise gr.Error("The Chatterbox voices folder is not mounted (TTS_VOICES_DIR).")
    if not sample:
        raise gr.Error("Upload a voice sample first.")
    base = safe_folder_name(name or "") or voice_name_from_sample(sample)
    if not base:
        raise gr.Error("Give the voice a name.")
    file_name = f"{base}.wav"
    destination = os.path.join(voices_dir, file_name)
    if os.path.exists(destination) and not replace:
        raise gr.Error(f"A voice called '{file_name}' already exists. Tick 'Replace' or pick another name.")

    audio_filter = "aformat=channel_layouts=mono" + (f",{PAUSE_FILTER}" if remove_pauses else "")
    with tempfile.TemporaryDirectory() as tmp:
        converted = os.path.join(tmp, file_name)
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", sample,
             "-af", audio_filter, "-c:a", "pcm_s16le", converted],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not os.path.isfile(converted):
            raise gr.Error(f"Could not read that audio file: {result.stderr.strip()[:200]}")
        seconds = _audio_seconds(converted)
        if seconds < 1:
            raise gr.Error("No speech found in that sample.")
        shutil.copyfile(converted, destination)

    message = f"Added **{base}** ({seconds:.1f} s of speech{', pauses removed' if remove_pauses else ''})."
    if seconds < SHORT_SAMPLE_SECONDS:
        message += " It's short: Chatterbox sounds steadier with 10-15 s of speech."
    message += " " + _measure_after_add(file_name)
    choices = openai_voice_choices()
    make_tab_update = gr.update(choices=choices, value=file_name) if engine in FILE_VOICE_ENGINES else gr.update()
    return (message, gr.update(choices=choices, value=file_name), make_tab_update,
            gr.update(choices=own_voice_choices()))


# ---- Deleting voices ----

BUILT_IN_VOICES = frozenset({  # Chatterbox's 28 shipped voices (chatterbox/voices/); never deletable
    "Abigail.wav", "Adrian.wav", "Alexander.wav", "Alice.wav", "Austin.wav", "Axel.wav",
    "Connor.wav", "Cora.wav", "Elena.wav", "Eli.wav", "Emily.wav", "Everett.wav",
    "Gabriel.wav", "Gianna.wav", "Henry.wav", "Ian.wav", "Jade.wav", "Jeremiah.wav",
    "Jordan.wav", "Julian.wav", "Layla.wav", "Leonardo.wav", "Michael.wav", "Miles.wav",
    "Olivia.wav", "Ryan.wav", "Taylor.wav", "Thomas.wav",
})


def own_voice_names() -> List[str]:
    """Voice files sitting directly in TTS_VOICES_DIR that are not a Chatterbox built-in."""
    voices_dir = os.environ.get("TTS_VOICES_DIR")
    if not voices_dir or not os.path.isdir(voices_dir):
        return []
    return sorted(
        (name for name in os.listdir(voices_dir)
         if name.lower().endswith(web_ui.VOICE_FILE_EXTENSIONS) and name not in BUILT_IN_VOICES
         and os.path.isfile(os.path.join(voices_dir, name))),
        key=str.lower,
    )


def own_voice_choices() -> list:
    """(label, file name) pairs for voices this app added -- excludes Chatterbox's built-ins."""
    return [(os.path.splitext(name)[0], name) for name in own_voice_names()]


def _is_safe_voice_filename(name: str) -> bool:
    """True only for a plain file name: no path separators, no traversal."""
    return (bool(name) and name not in (".", "..") and "/" not in name and "\\" not in name
           and os.path.basename(name) == name)


def _voice_in_use(name: str, jobs: List[dict]) -> Optional[str]:
    """Title of the queued/running Chatterbox or Breeze job still using this voice file, if any."""
    for job in jobs:
        if job.get("status") not in (QUEUED, RUNNING):
            continue
        settings = job.get("settings") or {}
        if settings.get("engine", "chatterbox") in FILE_VOICE_ENGINES and settings.get("voice") == name:
            return job.get("title") or "a queued book"
    return None


def delete_own_voice(name: Optional[str], jobs: List[dict]) -> str:
    """Delete one of the owner's own voice files from TTS_VOICES_DIR.

    Refuses (gr.Error, file left untouched) a built-in voice, a name that isn't a plain existing
    file directly inside TTS_VOICES_DIR, or a voice a queued/running Chatterbox job still uses.
    """
    voices_dir = os.environ.get("TTS_VOICES_DIR")
    if not voices_dir or not os.path.isdir(voices_dir):
        raise gr.Error("The Chatterbox voices folder is not mounted (TTS_VOICES_DIR).")
    if not name:
        raise gr.Error("Pick a voice to delete.")
    if name in BUILT_IN_VOICES:
        raise gr.Error(f"'{name}' is a built-in Chatterbox voice and can't be deleted.")
    if not _is_safe_voice_filename(name):
        raise gr.Error("That is not a valid voice file name.")
    path = os.path.join(voices_dir, name)
    if not _within_root(path, voices_dir) or not os.path.isfile(path):
        raise gr.Error(f"'{name}' does not exist in the voices folder.")
    book = _voice_in_use(name, jobs)
    if book:
        raise gr.Error(f"'{name}' is used by '{book}' in the queue. Remove or finish that book first.")
    os.remove(path)
    voice_measure.forget_features(name)
    if name in cast_store.load_voice_genders():
        cast_store.save_voice_gender(name, None)
    return f"Deleted **{os.path.splitext(name)[0]}**."


def _voice_dropdown_after_delete(choices: list, current_value: Optional[str], deleted_name: str,
                                 default_value: Optional[str]) -> dict:
    """Refresh a voice dropdown's choices after a deletion; reselect the default only if the
    deleted voice was the one showing, otherwise leave the caller's current selection alone."""
    if current_value == deleted_name:
        return gr.update(choices=choices, value=default_value)
    return gr.update(choices=choices)


# ---- Measuring voices (for cast suggestions) ----

MEASURE_TIMEOUT_SECONDS = 180  # a request waits behind the chunk of a book being generated


def _chatterbox_voice_files() -> Dict[str, str]:
    """{voice file name: path} for every voice in the Chatterbox voices folder."""
    voices_dir = os.environ.get("TTS_VOICES_DIR")
    if not voices_dir or not os.path.isdir(voices_dir):
        return {}
    return {name: os.path.join(voices_dir, name) for name in os.listdir(voices_dir)
            if name.lower().endswith(web_ui.VOICE_FILE_EXTENSIONS) and os.path.isfile(os.path.join(voices_dir, name))}


def measure_voice(voice: str) -> dict:
    """Have Chatterbox speak voice_measure.MEASURE_TEXT with this voice (the saved delivery
    settings, as a book would), measure the speech and save it; returns the measurement."""
    path = _chatterbox_voice_files().get(voice)
    if not path:
        raise ValueError(f"'{voice}' is not in the voices folder")
    settings = read_saved_settings()
    payload = {"text": voice_measure.MEASURE_TEXT, "voice_mode": "predefined", "predefined_voice_id": voice,
               "output_format": "wav", "split_text": True, "chunk_size": 500, "speed_factor": 1.0,
               "exaggeration": settings["exaggeration"], "cfg_weight": settings["cfg_weight"],
               "temperature": settings["temperature"]}
    measurement = voice_measure.measure_audio(_post_json("/tts", payload, timeout=MEASURE_TIMEOUT_SECONDS))
    voice_measure.save_features(voice, measurement, voice_measure.file_signature(path))
    logger.info(f"Voice measured: {voice} {measurement}")
    return measurement


def voice_sound_words(voice: Optional[str]) -> str:
    """How a Chatterbox voice sounds, from its measurement ("low for a woman, husky"), ranked
    among the measured voices of its recorded gender."""
    voices = engine_voices_with_gender("chatterbox")
    traits = voice_measure.voice_traits(voices, voice_measure.load_features())
    return voice_measure.describe(traits.get(voice or ""), dict(voices).get(voice or "", "neutral"))


def _measure_after_add(voice: str) -> str:
    """Measure a voice just added; a failure (Chatterbox busy, unloaded or unreachable) never
    fails the add, it only leaves the voice for Measure voices."""
    try:
        measure_voice(voice)
    except Exception as e:
        logger.warning(f"Voice {voice} not measured after adding it: {e}")
        return "Not measured yet (Chatterbox couldn't be asked just now): press **Measure voices** later."
    return f"Measured: sounds {voice_sound_words(voice)}."


def voice_sound_text(voice: Optional[str]) -> str:
    """Voice lab: the selected voice's measurement, or how to get one."""
    if not voice:
        return ""
    measurement = voice_measure.load_features().get(voice)
    if not measurement:
        return "Not measured yet: press **Measure voices**."
    return (f"Sounds **{voice_sound_words(voice)}** ({measurement['f0_median']:.0f} Hz), "
            f"measured {measurement.get('measured', '')}.")


def measure_voices() -> str:
    """Voice lab: measure every Chatterbox voice not measured yet, or changed since (about 3 s
    each). Stops at the first sign that Chatterbox can't be reached rather than timing out on
    every voice."""
    files = _chatterbox_voice_files()
    if not files:
        raise gr.Error("The Chatterbox voices folder is not mounted (TTS_VOICES_DIR).")
    todo = voice_measure.voices_to_measure(files, voice_measure.load_features())
    if not todo:
        return f"All {len(files)} voices are measured."
    done, failed = 0, []
    for voice in todo:
        try:
            measure_voice(voice)
            done += 1
        except urllib.error.HTTPError as e:
            failed.append(f"{os.path.splitext(voice)[0]} ({_http_error_detail(e)})")
        except (urllib.error.URLError, OSError) as e:
            return (f"Measured {done} of {len(todo)}, then Chatterbox couldn't be reached ({e}). "
                    "Press **Measure voices** again when it's running.")
        except Exception as e:  # this voice's audio couldn't be measured; the others still can
            failed.append(f"{os.path.splitext(voice)[0]} ({e})")
    message = f"Measured {done} voice{'' if done == 1 else 's'}."
    if failed:
        message += f" Couldn't measure {', '.join(failed)}."
    return message


# ---- Audiobook generation ----

def build_config(input_file, output_dir: str, voice: str, speed: float, chapter_selection: list,
                 sentence_pause: float, paragraph_pause: float, output_m4b: bool, skip_existing: bool,
                 output_text: bool, title_mode: str, newline_mode: str, remove_endnotes: bool,
                 remove_reference_numbers: bool, search_and_replace_file, log_level: str,
                 paced_unit_mode: str = "sentence", engine: str = "chatterbox",
                 voice_mode: str = VOICE_MODE_SINGLE, dialogue_voice: Optional[str] = None,
                 cast_file: Optional[str] = None, adaptive_delivery: bool = False,
                 delivery_exaggeration: Optional[float] = None, delivery_cfg_weight: Optional[float] = None,
                 delivery_temperature: Optional[float] = None, tone_match: bool = True) -> GeneralConfig:
    """GeneralConfig for the OpenAI provider pointed at Chatterbox or Kokoro (pauses in seconds).

    paced_unit_mode, engine, the voice-mode arguments, the delivery arguments and tone_match all
    default so books queued before any of them existed still build (as Chatterbox, sentence units,
    one voice, adaptive delivery off, tone matching on). Kokoro always narrates in sentence units regardless of paced_unit_mode:
    paragraph mode's gap detector was tuned on Chatterbox audio and saves nothing on a server this
    fast (see the Narration units info text); delivery is ignored entirely for Kokoro.

    Breeze is the same: sentence units, no delivery sliders or adaptive delivery (moods come later as
    spoken instructions), tone matching as passed, and no OpenAI endpoint (it has its own client).

    Raises ValueError if engine is "kokoro" but KOKORO_BASE_URL isn't configured (or "breeze" without
    BREEZE_BASE_URL); queue_settings
    already refuses that earlier with a friendlier gr.Error at enqueue time, so this only guards a
    job that was queued while Kokoro was configured and lost that configuration before its turn.
    """
    config = GeneralConfig(None)
    config.input_file = input_file.name if hasattr(input_file, "name") else input_file
    config.output_folder = output_dir
    config.preview = False
    config.output_text = output_text
    config.skip_existing = skip_existing
    config.log = log_level
    config.worker_count = 1  # Chatterbox/Kokoro handle one request at a time from this app (Breeze: one batch)
    config.no_prompt = True
    config.title_mode = title_mode
    config.newline_mode = newline_mode
    config.chapter_start = 1
    config.chapter_end = -1
    config.chapter_selection = list(chapter_selection)
    config.remove_endnotes = remove_endnotes
    config.remove_reference_numbers = remove_reference_numbers
    config.search_and_replace_file = (search_and_replace_file.name if hasattr(search_and_replace_file, "name")
                                      else search_and_replace_file)
    config.tts = "openai"
    config.language = "en"
    config.output_format = "mp3"
    config.voice_name = voice
    config.instructions = None
    config.speed = float(speed)
    config.sentence_pause_ms = int(round(float(sentence_pause) * 1000))
    config.paragraph_pause_ms = int(round(float(paragraph_pause) * 1000))
    config.output_m4b = bool(output_m4b)
    config.voice_mode = voice_mode if voice_mode in VOICE_MODES else VOICE_MODE_SINGLE
    config.dialogue_voice = dialogue_voice or None
    config.cast_file = cast_file if config.voice_mode == VOICE_MODE_CAST else None

    engine = engine or "chatterbox"
    if engine == "kokoro":
        base_url = kokoro_base_url()
        if not base_url:
            raise ValueError("KOKORO_BASE_URL is not configured.")
        config.model_name = "kokoro"
        config.openai_base_url = base_url
        config.paced_unit_mode = "sentence"
        config.adaptive_delivery = False
        config.delivery_exaggeration = config.delivery_cfg_weight = config.delivery_temperature = None
        config.tone_match = False
    elif engine == "breeze":
        if not breeze_client.configured():
            raise ValueError("BREEZE_BASE_URL is not configured.")
        config.model_name = "breeze"
        config.openai_base_url = None
        config.paced_unit_mode = "sentence"
        config.adaptive_delivery = False
        config.delivery_exaggeration = config.delivery_cfg_weight = config.delivery_temperature = None
        config.tone_match = bool(tone_match)
    else:
        config.model_name = "chatterbox"
        config.openai_base_url = None
        config.paced_unit_mode = paced_unit_mode or "sentence"
        config.adaptive_delivery = bool(adaptive_delivery)
        config.delivery_exaggeration = delivery_exaggeration
        config.delivery_cfg_weight = delivery_cfg_weight
        config.delivery_temperature = delivery_temperature
        config.tone_match = bool(tone_match)
    return config


def _table_rows(table) -> list:
    """Rows of the chapter table, whether Gradio hands over a DataFrame, a dict or a list."""
    if table is None:
        return []
    if hasattr(table, "values") and hasattr(table, "columns"):
        return table.values.tolist()
    if isinstance(table, dict):
        return [list(row) for row in table.get("data", [])]
    return [list(row) for row in table]


def selected_chapter_numbers(table) -> list:
    return [int(row[0]) for row in _table_rows(table) if row and bool(row[1])]


QUEUE_FILE = "queue.json"
QUEUE_UPLOADS = "queue_uploads"


def _copy_for_queue(path, suffix: str) -> str:
    """Keep a private copy of an uploaded file: Gradio's temporary uploads can vanish before a
    queued book gets its turn."""
    os.makedirs(QUEUE_UPLOADS, exist_ok=True)
    handle, copy = tempfile.mkstemp(prefix="upload_", suffix=suffix, dir=QUEUE_UPLOADS)
    os.close(handle)
    shutil.copyfile(path, copy)
    return os.path.abspath(copy)


def sweep_orphaned_uploads(queue: JobQueue) -> None:
    """Delete files in queue_uploads/ that no job references (F-31): a crash or an error between
    copying an upload and queuing the job would otherwise leave it there forever."""
    if not os.path.isdir(QUEUE_UPLOADS):
        return
    referenced = set()
    for job in queue.jobs():
        for key in UPLOAD_KEYS:
            path = job["settings"].get(key)
            if path:
                referenced.add(os.path.abspath(path))
    for name in os.listdir(QUEUE_UPLOADS):
        path = os.path.abspath(os.path.join(QUEUE_UPLOADS, name))
        if os.path.isfile(path) and path not in referenced:
            try:
                os.remove(path)
            except OSError as e:
                print(f"Could not remove orphaned upload {path}: {e}")


def _within_root(path: str, root: str) -> bool:
    """True if `path` resolves (symlinks included) to `root` itself or somewhere inside it."""
    if not root or not path:
        return False
    real_root = os.path.realpath(root)
    real_path = os.path.realpath(path)
    return real_path == real_root or real_path.startswith(real_root + os.sep)


CHAPTER_WORK_FOLDER = ".chapters"  # mirrors core.audiobook_generator.CHAPTER_WORK_FOLDER


def _refuse_if_output_dir_unavailable(output_dir: str, skip_existing: bool, active_jobs: List[dict]) -> None:
    """Refuse an output folder another queued/running book already owns, and guard against silently
    overwriting a finished book unless the owner ticked 'Skip chapters already made' (F-12)."""
    target = os.path.realpath(output_dir)
    for job in active_jobs:
        if job.get("status") in (QUEUED, RUNNING):
            other = job.get("settings", {}).get("output_dir")
            if other and os.path.realpath(other) == target:
                raise gr.Error(f"'{output_dir}' is already queued as '{job.get('title', 'another book')}'. "
                               "Pick a different output folder.")
    if not skip_existing and os.path.isdir(target):
        names = os.listdir(target)
        has_book = any(name.lower().endswith(".m4b") for name in names) or CHAPTER_WORK_FOLDER in names
        if has_book:
            raise gr.Error(f"'{output_dir}' already has a book in it. Use a different output folder "
                           "(e.g. add the author's name), or tick 'Skip chapters already made' to resume an "
                           "unfinished book there (a finished book in that folder will be replaced).")


def queue_settings(library_book, input_file, chapter_table, output_dir: str, voice: str, speed: float,
                   sentence_pause: float, paragraph_pause: float, output_m4b: bool, skip_existing: bool,
                   output_text: bool, title_mode: str, newline_mode: str, remove_endnotes: bool,
                   remove_reference_numbers: bool, search_and_replace_file, log_level: str,
                   paced_unit_mode: str = "sentence", engine: str = "chatterbox",
                   voice_mode: str = VOICE_MODE_SINGLE, dialogue_voice: Optional[str] = None,
                   cast_key: Optional[str] = None, adaptive_delivery: bool = False,
                   delivery_exaggeration: Optional[float] = None, delivery_cfg_weight: Optional[float] = None,
                   delivery_temperature: Optional[float] = None, tone_match: bool = True,
                   active_jobs: Optional[List[dict]] = None) -> dict:
    """Validate the form and turn it into build_config keyword arguments for a queued book.

    In cast mode the book's saved cast (see cast_store) must be finished and its voices must
    belong to the engine; a private snapshot of it goes with the job, so later edits to the cast
    (or a re-analysis) never change a book that is already queued."""
    if library_book:
        if not os.path.isfile(library_book):
            raise gr.Error("Pick the book from the list as you type (or clear the box to use an upload).")
        if not _within_root(library_book, library_index.library_dir()):
            raise gr.Error("That book is outside the ebook library folder.")
    upload = input_file.name if hasattr(input_file, "name") else input_file
    if not library_book and not upload:
        raise gr.Error("Pick a book from the library or upload an EPUB first.")
    selection = selected_chapter_numbers(chapter_table)
    if not selection:
        raise gr.Error("Tick at least one chapter.")
    if not voice:
        raise gr.Error("Pick a voice.")
    engine = engine or "chatterbox"
    if engine == "kokoro" and not kokoro_base_url():
        raise gr.Error("Kokoro is not configured (set KOKORO_BASE_URL first), or switch the Engine "
                       "back to Chatterbox.")
    if engine == "breeze" and not breeze_client.configured():
        raise gr.Error("Breeze is not configured (set BREEZE_BASE_URL first), or switch the Engine "
                       "back to Chatterbox.")
    voice_mode = voice_mode or VOICE_MODE_SINGLE
    if voice_mode not in VOICE_MODES:
        raise gr.Error(f"Unknown voice mode '{voice_mode}'.")
    cast_source = None
    dialogue_voice = _dialogue_voice_setting(voice_mode, dialogue_voice)
    if voice_mode == VOICE_MODE_CAST:
        cast_source, cast = _finished_cast(cast_key)
        wrong = cast_store.voices_belong_to_engine(cast, engine)
        if wrong:
            raise gr.Error(f"The cast uses {', '.join(wrong)}, which is not a {engine} voice. Pick "
                           f"{engine} voices in the cast table, or switch the Engine.")
        if voice in (c.get("voice") for c in cast["characters"].values()):
            gr.Warning("The narrator's voice is also given to a character; they will sound the same.")
        gaps = cast_coverage_gaps(library_book or upload, cast, selection, title_mode, newline_mode,
                                  remove_endnotes, remove_reference_numbers,
                                  search_and_replace_file.name if hasattr(search_and_replace_file, "name")
                                  else search_and_replace_file)
        if gaps:
            raise gr.Error(f"The cast has no analysis for chapter{'s' if len(gaps) > 1 else ''} "
                           f"{', '.join(map(str, gaps))} as the book reads now (a chapter was ticked after the "
                           "analysis, or a text option changed). Press Analyse selected chapters again.")
    chosen = [voice] + ([dialogue_voice] if dialogue_voice else [])
    if voice_mode == VOICE_MODE_CAST:
        chosen += [c.get("voice") for c in cast["characters"].values()]
    gone = missing_voices(chosen, engine)
    if gone:
        raise gr.Error(f"{', '.join(gone)} {'is' if len(gone) == 1 else 'are'} not among the {engine} voices any "
                       "more (deleted or renamed?). Pick another voice.")
    output_dir = (output_dir or "").strip()
    if not output_dir:
        raise gr.Error("Set an output folder.")
    if not _within_root(output_dir, OUTPUT_ROOT):
        raise gr.Error(f"Output folder must be inside '{OUTPUT_ROOT}'.")
    _refuse_if_output_dir_unavailable(output_dir, bool(skip_existing), active_jobs or [])
    replace_file = (search_and_replace_file.name if hasattr(search_and_replace_file, "name")
                    else search_and_replace_file)
    if replace_file and not os.path.isfile(replace_file):
        raise gr.Error("The search & replace file could not be read.")
    if upload and not library_book and not os.path.isfile(upload):
        raise gr.Error("The uploaded EPUB could not be read.")
    # Every check above has passed, so copying now can never orphan one file because a later
    # validation failed (F-31); sweep_orphaned_uploads() is the backstop for anything else (a crash,
    # a disk error) that still leaves one behind.
    queued_input_file = library_book or _copy_for_queue(upload, ".epub")
    queued_replace_file = _copy_for_queue(replace_file, ".txt") if replace_file else None
    queued_cast_file = None
    if cast_source:
        cast["narrator_voice"], cast["engine"] = voice, engine
        cast_store.save_cast(cast_source, cast)
        queued_cast_file = _copy_for_queue(cast_source, ".json")
    return {
        "input_file": queued_input_file,
        "output_dir": output_dir, "voice": voice, "speed": float(speed),
        "chapter_selection": selection, "sentence_pause": float(sentence_pause),
        "paragraph_pause": float(paragraph_pause), "output_m4b": bool(output_m4b),
        "skip_existing": bool(skip_existing), "output_text": bool(output_text), "title_mode": title_mode,
        "newline_mode": newline_mode, "remove_endnotes": bool(remove_endnotes),
        "remove_reference_numbers": bool(remove_reference_numbers),
        "search_and_replace_file": queued_replace_file,
        "log_level": log_level, "paced_unit_mode": paced_unit_mode or "sentence",
        "engine": engine,
        "voice_mode": voice_mode, "dialogue_voice": dialogue_voice or None, "cast_file": queued_cast_file,
        "adaptive_delivery": bool(adaptive_delivery) and engine == "chatterbox",
        "delivery_exaggeration": delivery_exaggeration if engine == "chatterbox" else None,
        "delivery_cfg_weight": delivery_cfg_weight if engine == "chatterbox" else None,
        "delivery_temperature": delivery_temperature if engine == "chatterbox" else None,
        "tone_match": bool(tone_match) and engine != "kokoro",
    }


# ---- Cast (multi-voice) ----

CASTS_DIR = cast_store.CASTS_FOLDER  # inside the app data folder, next to queue.json
CAST_COLUMNS = ["Character", "Role", "Lines", "Gender", "Age", "Voice", "Sounds like", "Also called"]
VOICE_MODE_CHOICES = [("Single voice", VOICE_MODE_SINGLE), ("Narrator + dialogue voice", VOICE_MODE_DIALOGUE)]
CAST_MODE_CHOICE = ("Cast (LLM picks who speaks)", VOICE_MODE_CAST)
# Rough placeholder for the queue's time column: the LLM's real speed is unknown until
# experiments/multivoice/validate_multivoice.py reports time per 1,000 lines on the owner's machine.
ANALYSIS_SECONDS_PER_LINE = 0.5
_GENDER_CHOICES = [("female", "female"), ("male", "male"), ("unknown", "unknown")]
DELIVERY_CHOICES = [("from the profile", "auto"), ("a little more even", "even"), ("as the book", "book"),
                    ("a little more expressive", "expressive")]
# The cast editor's voice choice for a first-person book's narrator: their lines follow whatever
# narrator voice the book is queued with. Never stored as a voice; saving it clears the character's own.
NARRATOR_VOICE = "__narrator__"
NARRATOR_VOICE_CHOICE = ("(the narrator's voice)", NARRATOR_VOICE)
VOICE_GENDER_CHOICES = [("not set", ""), ("female", "female"), ("male", "male"), ("neutral (fits anyone)", "neutral")]


def voice_mode_choices() -> list:
    """The Voice mode options: cast mode is offered only when a chat endpoint is configured."""
    return VOICE_MODE_CHOICES + ([CAST_MODE_CHOICE] if llm_configured() else [])


def book_cast_key(library_book, input_file) -> Optional[str]:
    """The cast key of the book currently picked (library pick wins over an upload), or None."""
    book = library_book if library_book and os.path.isfile(library_book) else input_file
    book = book.name if hasattr(book, "name") else book
    if not book or not os.path.isfile(book):
        return None
    return cast_store.cast_key(book)


def cast_file_for(cast_key: str) -> str:
    return cast_store.cast_path(cast_key, CASTS_DIR)


def _finished_cast(cast_key: Optional[str]) -> Tuple[str, dict]:
    """(path, cast) of a finished analysis for this key; gr.Error otherwise."""
    if not cast_key:
        raise gr.Error("Pick a book first.")
    path = cast_file_for(cast_key)
    cast = cast_store.load_cast(path)
    if cast is None:
        raise gr.Error("This book has no cast yet: press Analyse selected chapters first.")
    if cast.get("status") != cast_store.STATUS_DONE:
        raise gr.Error("The cast analysis hasn't finished yet." if cast.get("status") == cast_store.STATUS_RUNNING
                       else f"The cast analysis failed ({cast.get('error') or 'see the log'}); run it again.")
    return path, cast


def engine_voices_with_gender(engine: str) -> List[Tuple[str, str]]:
    """(voice, gender) for every voice the engine offers: Kokoro's from its id prefixes,
    Chatterbox's from the owner's mapping (neutral when unset)."""
    if engine == "kokoro":
        choices, _ = kokoro_voices_and_default()
        return [(voice, cast_store.kokoro_voice_gender(voice)) for _, voice in choices]
    genders = cast_store.load_voice_genders()
    return [(voice, cast_store.voice_gender("chatterbox", voice, genders)) for _, voice in openai_voice_choices()]


def engine_voice_ids(engine: str) -> Optional[List[str]]:
    """Every voice the engine can speak with right now, or None when that can't be known (no
    Chatterbox voices folder mounted, Kokoro unreachable), in which case callers skip the check."""
    if engine == "kokoro":
        choices, _ = kokoro_voices_and_default()
        real = [value for label, value in choices if label != value]  # the unreachable fallback has label == id
        return real or None
    voices_dir = os.environ.get("TTS_VOICES_DIR")
    if not voices_dir or not os.path.isdir(voices_dir):
        return None
    return [name for name in os.listdir(voices_dir)
            if name.lower().endswith(web_ui.VOICE_FILE_EXTENSIONS) and os.path.isfile(os.path.join(voices_dir, name))]


def missing_voices(voices: List[Optional[str]], engine: str) -> List[str]:
    """The given voices the engine no longer has (deleted or renamed since they were chosen)."""
    known = engine_voice_ids(engine)
    if known is None:
        return []
    return sorted({v for v in voices if v and v not in known})


def cast_coverage_gaps(book: str, cast: dict, selection: List[int], title_mode: str, newline_mode: str,
                       remove_endnotes: bool, remove_reference_numbers: bool, search_and_replace_file) -> List[int]:
    """Ticked chapters the cast has no attributions for, as the book reads with these settings.

    Attributions are keyed by each chapter's text, so ticking a chapter that wasn't analysed, or
    changing an option that changes the text (paragraph detection, endnote removal, search &
    replace), leaves a chapter uncovered; at generation every quote in it would get the dialogue
    voice."""
    chapters = book_chapters(book, title_mode, newline_mode, remove_endnotes, remove_reference_numbers,
                             search_and_replace_file)
    analysed = cast.get("chapters", {})
    return [n for n in selection if 0 < n <= len(chapters) and cast_store.text_hash(chapters[n - 1][1]) not in analysed]


# The Dialogue voice choice that means "the narrator's voice". In cast mode it reads the lines whose
# speaker wasn't found (and characters with no voice yet), and it is the default there: a voice that
# belongs to no one, heard exactly where attribution failed, is the more jarring fallback (WORKLOG §29).
NARRATOR_FALLBACK = "(narrator)"


def dialogue_voice_choices(choices: list) -> list:
    return [("(the narrator's voice)", NARRATOR_FALLBACK)] + list(choices)


def _dialogue_voice_setting(voice_mode: str, dialogue_voice: Optional[str]) -> Optional[str]:
    """The dialogue voice to queue: None means the narrator's (cast mode only; "narrator + dialogue
    voice" mode needs a voice of its own, or it is single voice)."""
    voice = None if dialogue_voice in (None, "", NARRATOR_FALLBACK) else dialogue_voice
    if voice_mode == VOICE_MODE_DIALOGUE and not voice:
        raise gr.Error("Pick a dialogue voice: it speaks every quoted line in this mode.")
    return voice if voice_mode != VOICE_MODE_SINGLE else None


def engine_voice_choices(engine: str) -> list:
    return kokoro_voices_and_default()[0] if engine == "kokoro" else openai_voice_choices()


def cast_rows(cast: dict, engine: str) -> Tuple[list, list]:
    """(table rows, character keys in row order), most lines first. Role and "Sounds like" come
    from the character's profile and are blank for characters without one."""
    rows, keys = [], []
    labels = dict((value, label) for label, value in engine_voice_choices(engine))
    narrating = cast_store.narrating_character(cast)
    for key, character in cast_store.ranked_characters(cast):
        voice = character.get("voice") or ""
        profile = character.get("profile") or {}
        role = profile.get("role", "")
        shown = ("(narrator's voice)" if key == narrating
                 else labels.get(voice, voice) if voice else "(dialogue voice)")
        rows.append([character.get("name", key), "" if role == "unknown" else role, int(character.get("lines", 0)),
                     character.get("gender", "unknown"), character.get("age", "unknown"), shown,
                     profile.get("voice", ""), ", ".join(character.get("aliases", []))])
        keys.append(key)
    return rows, keys


def voice_match_text(character: dict, voice_label: str = "", voice_words: str = "") -> str:
    """What kind of voice the character's profile asks for, next to how their voice measures:
    "wants low pitch, clear · Olivia is low for a woman, clear, even"."""
    targets = cast_store.voice_targets(character)
    wants = ", ".join(w for w in (f"{targets['pitch']} pitch" if targets["pitch"] else "",
                                  targets["quality"] or "", targets["delivery"] or "") if w)
    if not wants and voice_words in ("", "not measured"):
        return ""
    parts = [f"wants {wants}" if wants else "no particular voice asked for"]
    if voice_label and voice_words:
        parts.append(f"{voice_label} is {voice_words}")
    return "**Voice match:** " + " · ".join(parts)


def character_delivery_text(character: dict, offset: Optional[float] = None) -> str:
    """How the character's lines are delivered against the book's sliders (adaptive delivery);
    offset is the character's within its cast (core.cast.exaggeration_offsets)."""
    offset = cast_store.exaggeration_offset(character) if offset is None else offset
    if offset > 0:
        how = f"a little more expressive than the book (exaggeration {offset:+.2f})"
    elif offset < 0:
        how = f"a little more even than the book (exaggeration {offset:+.2f})"
    else:
        how = "as the book"
    return f"**Delivery:** {how}" + (" (your setting)" if character.get("delivery") not in (None, "auto") else "")


def character_profile_text(character: dict, voice_match: str = "", delivery_offset: Optional[float] = None) -> str:
    """The clicked character's profile as Markdown: who they are, how they might sound, how their
    voice matches, and the first line they speak."""
    profile = character.get("profile") or {}
    facts = [f"**{character.get('name', '')}**"]
    if profile.get("role") and profile["role"] != "unknown":
        facts.append(profile["role"])
    facts.append(f"{character.get('gender', 'unknown')}, {character.get('age', 'unknown')}")
    lines = int(character.get("lines", 0))
    facts.append(f"{lines} line{'' if lines == 1 else 's'}")
    parts = [" · ".join(facts)]
    if profile.get("description"):
        parts.append(profile["description"])
    else:
        parts.append("*No profile: they are written for the characters with the most lines when the cast is "
                     "analysed (a cast analysed before profiles existed has none until it is analysed again).*")
    if profile.get("voice"):
        parts.append(f"**Sounds like:** {profile['voice']}")
    if voice_match:
        parts.append(voice_match)
    parts.append(character_delivery_text(character, delivery_offset))
    if profile.get("relationships"):
        parts.append(f"**Relationships:** {profile['relationships']}")
    first = profile.get("first_line")
    if isinstance(first, dict) and first.get("text"):
        parts.append(f"**First line** (chapter {first.get('chapter', '?')}): {first['text']}")
    return "\n\n".join(parts)


def engine_voice_traits(engine: str, voices: Optional[List[Tuple[str, str]]] = None) -> Dict[str, dict]:
    """Measured traits of the engine's voices (core.voice_measure), for matching; Kokoro voices
    aren't measured, so they match by gender only."""
    if engine == "kokoro":
        return {}
    return voice_measure.voice_traits(voices if voices is not None else engine_voices_with_gender(engine),
                                      voice_measure.load_features())


def _fill_missing_voices(cast: dict, path: str, engine: str, narrator_voice: Optional[str]) -> dict:
    """Give every character without a voice the automatic suggestion and save, so what the table
    shows is exactly what a queued book would use. A first-person book's narrating character gives
    up any suggested voice first: their lines are the narrator's."""
    if cast_store.release_narrating_voice(cast):
        cast_store.save_cast(path, cast)
    narrating = cast_store.narrating_character(cast)
    if any(not c.get("voice") for key, c in cast["characters"].items() if key != narrating):
        voices = engine_voices_with_gender(engine)
        suggestions = cast_store.suggest_voices(cast, voices, narrator_voice, engine_voice_traits(engine, voices))
        for key, voice in suggestions.items():
            cast["characters"][key]["voice"] = voice
        if suggestions:
            cast_store.save_cast(path, cast)
    return cast


def _fill_narrator_suggestion(cast: dict, path: str, engine: str,
                              narrator_voice: Optional[str] = None) -> Optional[dict]:
    """The narrator the book's tone asks for (cast["narrator_suggestion"]: voice and delivery
    sliders), worked out once and saved; None when there is no book tone, the engine's voices aren't
    measured (Kokoro), or nothing fits. Voices the owner picked for characters are never offered;
    the owner's narrator_voice is kept when the tone asks only for a gender it already has."""
    if cast.get("narrator_suggestion"):
        return cast["narrator_suggestion"]
    if engine == "kokoro" or not cast.get("book_tone"):
        return None
    voices = engine_voices_with_gender(engine)
    picked = tuple(c["voice"] for c in cast["characters"].values() if c.get("voice_picked") and c.get("voice"))
    voice = cast_store.suggest_narrator(cast, voices, engine_voice_traits(engine, voices), picked, narrator_voice)
    if not voice:
        return None
    saved = read_saved_settings()
    exaggeration, cfg_weight, temperature = cast_store.narrator_delivery(
        cast["book_tone"], (saved["exaggeration"], saved["cfg_weight"], saved["temperature"]))
    cast["narrator_suggestion"] = {"voice": voice, "exaggeration": exaggeration, "cfg_weight": cfg_weight,
                                   "temperature": temperature}
    cast_store.save_cast(path, cast)
    return cast["narrator_suggestion"]


def _narrator_updates(suggestion: Optional[dict]) -> tuple:
    """Updates for the Make tab's Voice and the Voice lab's three sliders (all four unchanged
    without a suggestion): Add to queue takes the narrator and the book's delivery from them."""
    if not suggestion:
        return gr.update(), gr.update(), gr.update(), gr.update()
    return (gr.update(value=suggestion["voice"]), gr.update(value=suggestion["exaggeration"]),
            gr.update(value=suggestion["cfg_weight"]), gr.update(value=suggestion["temperature"]))


def resuggest_cast_voices(cast_key: Optional[str], engine: str, narrator_voice: Optional[str],
                          confirmed: Optional[bool] = True) -> tuple:
    """Suggest again the narrator and every voice the owner didn't pick with Save (the browser
    confirms first; a falsy `confirmed` means cancelled). Returns the refreshed table, keys, a status
    line, and the narrator's Voice and slider updates."""
    if not confirmed:
        return (gr.update(),) * 7
    path, cast = _finished_cast(cast_key)
    cast.pop("narrator_suggestion", None)
    suggestion = _fill_narrator_suggestion(cast, path, engine, narrator_voice)
    cleared = cast_store.clear_suggested_voices(cast)
    cast = _fill_missing_voices(cast, path, engine, suggestion["voice"] if suggestion else narrator_voice)
    cast_store.save_cast(path, cast)
    kept = sum(1 for c in cast["characters"].values() if c.get("voice_picked"))
    rows, keys = cast_rows(cast, engine)
    message = f"Suggested voices again for {cleared} character{'' if cleared == 1 else 's'}"
    message += f"; kept the {kept} you picked." if kept else "."
    if suggestion:
        message += f" Narrator: {os.path.splitext(suggestion['voice'])[0]}."
    if not engine_voice_traits(engine):
        message += (" No voices are measured yet, so this matched by gender only: press **Measure voices** "
                    "in the Voice lab to match by sound.")
    return (gr.update(value=rows, visible=True), keys, message, *_narrator_updates(suggestion))


def _cast_summary(cast: dict, auto_pick: bool = False) -> str:
    stats = cast.get("stats", {})
    lines, unknown = int(stats.get("lines", 0)), int(stats.get("unknown_lines", 0))
    known = lines - unknown
    parts = [f"**Cast ready**: {len(cast['characters'])} characters, {known} of {lines} lines attributed"]
    if unknown:
        parts.append(f"{unknown} with no speaker found (read by the Dialogue voice setting: the narrator's "
                     "voice unless you pick another)")
    moods = cast_store.mood_counts(cast)
    mood_parts = [f"{moods[key]} {label}" for key, label in
                  (("soft", "soft"), ("emphatic", "emphasized"), ("excited", "excited")) if moods.get(key)]
    if mood_parts:
        parts.append(", ".join(mood_parts))
    if stats.get("invalid_after_retry"):
        parts.append(f"{stats['invalid_after_retry']} window(s) the LLM never answered usably")
    if stats.get("profiles"):
        parts.append(f"{stats['profiles']} character profile{'' if stats['profiles'] == 1 else 's'}")
    if cast.get("profile_error"):
        parts.append(f"profiles stopped early ({cast['profile_error']})")
    text = " · ".join(parts) + (". Click a character to see who they are." if auto_pick else
                                ". Click a character to see who they are, then **Add this book to queue**.")
    text = text + "\n\n" + book_tone_text(cast) if cast.get("book_tone") else text
    tellers = story_tellers_text(cast)
    return text + "  \n" + tellers if tellers else text


def story_tellers_text(cast: dict) -> str:
    """The first-person stories another character tells, each narrated in the teller's own voice
    (cast.chapter_narrator_voice); "" when there are none."""
    told: Dict[str, List[int]] = {}
    for entry in (cast.get("chapters") or {}).values():
        key = entry.get("narrator")
        if key and key != cast_store.pov_character(cast) and key in cast.get("characters", {}):
            told.setdefault(key, []).append(entry.get("number", 0))
    if not told:
        return ""
    parts = []
    for key, numbers in sorted(told.items(), key=lambda kv: min(kv[1])):
        character = cast["characters"][key]
        voice = character.get("voice")
        chapters = ", ".join(str(n) for n in sorted(numbers))
        parts.append(f"{character.get('name', key)} (chapter{'s' if len(numbers) > 1 else ''} {chapters}, "
                     f"{os.path.splitext(voice)[0] if voice else 'the narrator voice'})")
    return "**First-person stories, each narrated by its teller's voice:** " + "; ".join(parts)


def _unnamed_narrator(name: str) -> bool:
    """A first-person narrator the book never names: attribution calls them "I" (seen live)."""
    return cast_store.normalize_name(name) in ("i", "me", "myself", "narrator", "the narrator")


def book_tone_text(cast: dict) -> str:
    """The book's narration and the narrator picked for it, in one line."""
    tone = cast.get("book_tone") or {}
    named = tone.get("pov_character") and not _unnamed_narrator(tone["pov_character"])
    facts = [w for w in (f"{tone['point_of_view']} person" if tone.get("point_of_view") else "",
                         f"narrated by {tone['pov_character']}" if named else "",
                         tone.get("tone") or "", f"{tone['pace']} pace" if tone.get("pace") else "",
                         f"{tone['intensity']} narration" if tone.get("intensity") else "") if w]
    text = "**Book:** " + " · ".join(facts)
    suggestion = cast.get("narrator_suggestion")
    if suggestion:
        text += (f"  \n**Narrator:** {os.path.splitext(suggestion['voice'])[0]}, exaggeration "
                 f"{suggestion['exaggeration']:.2f} · CFG {suggestion['cfg_weight']:.2f} · temperature "
                 f"{suggestion['temperature']:.2f}, picked from the book's tone")
    narrating = cast_store.narrating_character(cast)
    if narrating:
        name = cast["characters"][narrating].get("name", narrating)
        text += ("  \nThe unnamed first-person narrator (\"I\") speaks their own lines in the narrator's voice."
                 if _unnamed_narrator(name) else
                 f"  \n{name} tells the story, so their lines are read in the narrator's voice.")
    return text


def cast_overview(cast_key: Optional[str], engine: str, narrator_voice: Optional[str], seen: Optional[list],
                  auto_pick: bool = False) -> tuple:
    """(table update, character keys, status text, seen) for the picked book's cast.

    `seen` is [path, mtime, status, engine, narrator] of what the table currently shows; when the
    cast file hasn't changed since, the table is left alone (gr.update()) so a 3-second refresh
    never disturbs a row the owner has selected. With auto_pick, the narrator the book's tone asks
    for is worked out first, and characters' voices are suggested around it.
    """
    if not cast_key:
        return gr.update(value=None, visible=False), [], "Pick a book, tick its chapters, then press **Analyse selected chapters**.", None
    path = cast_file_for(cast_key)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return gr.update(value=None, visible=False), [], "No cast yet for this book: press **Analyse selected chapters**.", None
    cast = cast_store.load_cast(path)
    if cast is None:
        return gr.update(value=None, visible=False), [], "The cast file could not be read; run **Analyse selected chapters** again.", None
    stamp = [path, mtime, cast.get("status"), engine, narrator_voice]
    if seen == stamp:
        return gr.update(), gr.update(), gr.update(), seen
    if cast.get("status") == cast_store.STATUS_RUNNING:
        done, total = cast_store.analysis_progress(cast)
        profiled, to_profile = cast_store.profile_progress(cast)
        if to_profile and done >= total:
            text = f"⏳ **Writing character profiles**: {profiled} of {to_profile} done."
        else:
            text = f"⏳ **Analysing cast**: {done} of {total} chapters done, {len(cast['characters'])} characters so far."
        return gr.update(value=None, visible=False), [], text, stamp
    if cast.get("status") == cast_store.STATUS_FAILED:
        return (gr.update(value=None, visible=False), [],
                f"✗ The cast analysis failed: {cast.get('error') or 'see the log'}. Press **Analyse selected chapters** to try again.",
                stamp)
    suggestion = _fill_narrator_suggestion(cast, path, engine, narrator_voice) if auto_pick else None
    cast = _fill_missing_voices(cast, path, engine, suggestion["voice"] if suggestion else narrator_voice)
    rows, keys = cast_rows(cast, engine)
    stamp[1] = os.path.getmtime(path)  # the fills may just have saved
    return gr.update(value=rows, visible=True), keys, _cast_summary(cast, auto_pick), stamp


def cast_panel_update(cast_key: Optional[str], engine: str, narrator_voice: Optional[str], seen: Optional[list],
                      auto_pick: bool) -> tuple:
    """cast_overview, plus the narrator: the first time a finished cast is shown (this book, this
    page) with Auto-pick on, the Make tab's Voice and the Voice lab sliders are set to the narrator
    the book's tone asks for; Add to queue takes both from there. Later refreshes leave them alone,
    so a change the owner makes afterwards wins."""
    table, keys, status, stamp = cast_overview(cast_key, engine, narrator_voice, seen, auto_pick)
    first_sight = not seen or not stamp or seen[0] != stamp[0] or seen[2] != cast_store.STATUS_DONE
    if not (auto_pick and first_sight and stamp and stamp[2] == cast_store.STATUS_DONE):
        return (table, keys, status, stamp, *_narrator_updates(None))
    cast = cast_store.load_cast(stamp[0]) or {}
    return (table, keys, status, stamp, *_narrator_updates(cast.get("narrator_suggestion")))


def select_cast_row(cast_key: Optional[str], keys: list, engine: str, evt: gr.SelectData) -> tuple:
    """Clicking a row loads that character into the editor (name, gender, voice) and shows their
    profile."""
    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if not cast_key or not (0 <= row < len(keys)):
        return None, "", gr.update(), gr.update(), "", gr.update()
    cast = cast_store.load_cast(cast_file_for(cast_key)) or {"characters": {}}
    character = cast["characters"].get(keys[row])
    if not character:
        return None, "", gr.update(), gr.update(), "", gr.update()
    choices = engine_voice_choices(engine)
    voice = character.get("voice") or ""
    voices = engine_voices_with_gender(engine)
    words = (voice_measure.describe(engine_voice_traits(engine, voices).get(voice), dict(voices).get(voice, "neutral"))
             if voice else "")
    label = dict((value, label) for label, value in choices).get(voice, voice)
    match = voice_match_text(character, label, words)
    offset = cast_store.exaggeration_offsets(cast).get(keys[row])
    if keys[row] == cast_store.pov_character(cast):  # a first-person narrator can follow the narrator's voice
        choices = [NARRATOR_VOICE_CHOICE, *choices]
        if keys[row] == cast_store.narrating_character(cast):
            voice, offset = NARRATOR_VOICE, 0.0
            match = "**Voice:** the narrator's: they tell the story, so one performer reads their lines too"
    return (keys[row], f"Editing **{character.get('name', keys[row])}**",
            gr.update(value=character.get("gender", "unknown")),
            gr.update(choices=choices, value=voice or None),
            character_profile_text(character, match, offset),
            gr.update(value=character.get("delivery") or "auto"))


def sample_character(cast_key: Optional[str], character_key: Optional[str], engine: str, voice: Optional[str],
                     delivery_choice: str, speed: float, exaggeration: float, cfg_weight: float,
                     temperature: float, narrator_voice: Optional[str] = None) -> str:
    """Cast editor: the selected character's first line in the editor's voice, with the editor's
    delivery around the book's current sliders (as a book would read it); any other text when the
    character has no first line. "(the narrator's voice)" samples the Make tab's narrator, which
    reads such lines with the book's own delivery. Kokoro plays its usual sample."""
    follows_narrator = voice == NARRATOR_VOICE
    voice = narrator_voice if follows_narrator else voice
    if not voice:
        raise gr.Error("Pick a voice first.")
    if engine == "kokoro":
        return _kokoro_sample(voice, speed)
    if engine == "breeze":
        return _breeze_sample(voice)
    cast = (cast_store.load_cast(cast_file_for(cast_key)) if cast_key else None) or {"characters": {}}
    character = dict(cast["characters"].get(character_key or "") or {})
    character["delivery"] = delivery_choice or "auto"
    first = (character.get("profile") or {}).get("first_line") or {}
    text = first.get("text") or PREVIEW_PHRASE
    if follows_narrator:
        offset = 0.0
    elif character_key in cast["characters"]:  # the editor's delivery, within this cast
        cast["characters"][character_key] = character
        offset = cast_store.exaggeration_offsets(cast)[character_key]
    else:
        offset = cast_store.exaggeration_offset(character)
    return preview_voice(voice, text, round(min(2.0, max(0.25, float(exaggeration) + offset)), 2), cfg_weight,
                         temperature, speed)


def merge_choices(cast_key: Optional[str], character_key: Optional[str]) -> dict:
    """The editor's "Same person as" list: every other character of the cast, most lines first."""
    cast = cast_store.load_cast(cast_file_for(cast_key)) if cast_key else None
    if not cast or character_key not in cast["characters"]:
        return gr.update(choices=[], value=None)
    return gr.update(choices=[(c.get("name", k), k) for k, c in cast_store.ranked_characters(cast) if k != character_key],
                     value=None)


def merge_cast_character(cast_key: Optional[str], character_key: Optional[str], target_key: Optional[str],
                         engine: str, confirmed: bool = True) -> tuple:
    """Merge the selected character into the one picked as the same person (a doctor and her first
    name, a name the book spells two ways): their lines, names and aliases move over, and the
    target keeps its voice. Returns (table, keys, status, selection, editor heading)."""
    if not confirmed:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    if not cast_key or not character_key:
        raise gr.Error("Click a character in the cast table first.")
    if not target_key:
        raise gr.Error("Pick who they are the same person as.")
    path = cast_file_for(cast_key)
    cast = cast_store.load_cast(path)
    if cast is None or character_key not in cast["characters"] or target_key not in cast["characters"]:
        raise gr.Error("That character is no longer in the cast (was it re-analysed?).")
    source = cast["characters"][character_key].get("name", character_key)
    target = cast["characters"][target_key].get("name", target_key)
    cast_store.merge_characters(cast, character_key, target_key)
    cast_store.save_cast(path, cast)
    rows, keys = cast_rows(cast, engine)
    return (gr.update(value=rows, visible=True), keys,
            f"Merged **{source}** into **{target}**: their lines now use {target}'s voice.", None,
            "Click a character in the table.")


def apply_cast_edit(cast_key: Optional[str], character_key: Optional[str], gender: str, voice: Optional[str],
                    engine: str, delivery_choice: str = "auto") -> tuple:
    """Save the editor's gender, voice and delivery for the selected character; returns the
    refreshed table."""
    if not cast_key or not character_key:
        raise gr.Error("Click a character in the cast table first.")
    path = cast_file_for(cast_key)
    cast = cast_store.load_cast(path)
    if cast is None or character_key not in cast["characters"]:
        raise gr.Error("That character is no longer in the cast (was it re-analysed?).")
    if voice == NARRATOR_VOICE:
        if character_key != cast_store.pov_character(cast):
            raise gr.Error("Only the character who narrates a first-person book can share the narrator's voice.")
        character = cast["characters"][character_key]
        character["gender"] = gender if gender in cast_store.GENDERS else "unknown"
        character["voice"], character["voice_picked"] = None, False  # follows the narrator again
        character["delivery"] = delivery_choice if delivery_choice in cast_store.DELIVERIES else "auto"
        cast_store.save_cast(path, cast)
        rows, keys = cast_rows(cast, engine)
        return (gr.update(value=rows, visible=True), keys,
                f"Saved **{character.get('name', character_key)}**: their lines use the narrator's voice.")
    if not voice:
        raise gr.Error("Pick a voice for the character.")
    if cast_store.voices_belong_to_engine({"characters": {"x": {"voice": voice}}}, engine):
        raise gr.Error(f"'{voice}' is not a {engine} voice.")
    if missing_voices([voice], engine):
        raise gr.Error(f"'{voice}' isn't one of the {engine} voices (was it deleted or renamed?).")
    character = cast["characters"][character_key]
    character["gender"] = gender if gender in cast_store.GENDERS else "unknown"
    character["voice"] = voice
    character["voice_picked"] = True  # Suggest voices again keeps it
    character["delivery"] = delivery_choice if delivery_choice in cast_store.DELIVERIES else "auto"
    cast_store.save_cast(path, cast)
    rows, keys = cast_rows(cast, engine)
    return gr.update(value=rows, visible=True), keys, f"Saved **{character.get('name', character_key)}**: {gender}, {voice}."


def analysis_settings(library_book, input_file, chapter_table, engine: str, voice: str, title_mode: str,
                      newline_mode: str, remove_endnotes: bool, remove_reference_numbers: bool,
                      search_and_replace_file, log_level: str = "INFO", auto_pick_voices: bool = True) -> dict:
    """Validate the form for a cast analysis and return the analysis job's settings.

    auto_pick_voices: a re-analysis keeps only the voices the owner saved in the cast editor, so
    every other character gets a fresh suggestion matched to its new profile; off, it keeps every
    voice the earlier analysis had."""
    if not llm_configured():
        raise gr.Error("No LLM is configured (set LLM_BASE_URL and LLM_MODEL first).")
    if library_book:
        if not os.path.isfile(library_book):
            raise gr.Error("Pick the book from the list as you type (or clear the box to use an upload).")
        if not _within_root(library_book, library_index.library_dir()):
            raise gr.Error("That book is outside the ebook library folder.")
    upload = input_file.name if hasattr(input_file, "name") else input_file
    if not library_book and not upload:
        raise gr.Error("Pick a book from the library or upload an EPUB first.")
    if upload and not library_book and not os.path.isfile(upload):
        raise gr.Error("The uploaded EPUB could not be read.")
    selection = selected_chapter_numbers(chapter_table)
    if not selection:
        raise gr.Error("Tick the chapters to analyse.")
    replace_file = (search_and_replace_file.name if hasattr(search_and_replace_file, "name")
                    else search_and_replace_file)
    if replace_file and not os.path.isfile(replace_file):
        raise gr.Error("The search & replace file could not be read.")
    book = library_book or upload
    key = cast_store.cast_key(book)
    return {
        "input_file": library_book or _copy_for_queue(upload, ".epub"),
        "chapter_selection": selection, "title_mode": title_mode, "newline_mode": newline_mode,
        "remove_endnotes": bool(remove_endnotes), "remove_reference_numbers": bool(remove_reference_numbers),
        "search_and_replace_file": _copy_for_queue(replace_file, ".txt") if replace_file else None,
        "engine": engine or "chatterbox", "voice": voice, "log_level": log_level,
        "cast_key": key, "cast_file": cast_file_for(key), "auto_pick_voices": bool(auto_pick_voices),
    }


def book_options_for_later(library_book, chapter_table, stats: list, output_dir: str, voice: str, speed: float,
                           sentence_pause: float, paragraph_pause: float, output_m4b: bool, skip_existing: bool,
                           output_text: bool, engine: str, paced_unit_mode: str, dialogue_voice: Optional[str],
                           adaptive_delivery: bool, exaggeration: float, cfg_weight: float, temperature: float,
                           tone_match: bool = True, active_jobs: Optional[List[dict]] = None) -> dict:
    """With Auto-pick on, the Make tab's book options go with the cast analysis so the book joins
    the queue once its cast is ready (queue_book_after_cast). What can already be checked is
    checked now, while the owner is at the page."""
    dialogue_voice = _dialogue_voice_setting(VOICE_MODE_CAST, dialogue_voice)
    output_dir = (output_dir or "").strip()
    if not output_dir:
        raise gr.Error("Set an output folder.")
    if not _within_root(output_dir, OUTPUT_ROOT):
        raise gr.Error(f"Output folder must be inside '{OUTPUT_ROOT}'.")
    _refuse_if_output_dir_unavailable(output_dir, bool(skip_existing), active_jobs or [])
    return {
        "from_library": bool(library_book), "output_dir": output_dir, "voice": voice, "speed": float(speed),
        "sentence_pause": float(sentence_pause), "paragraph_pause": float(paragraph_pause),
        "output_m4b": bool(output_m4b), "skip_existing": bool(skip_existing), "output_text": bool(output_text),
        "paced_unit_mode": paced_unit_mode or "sentence", "dialogue_voice": dialogue_voice,
        "adaptive_delivery": bool(adaptive_delivery), "exaggeration": exaggeration, "cfg_weight": cfg_weight,
        "temperature": temperature, "tone_match": bool(tone_match),
        "estimate_seconds": generation_estimate(chapter_table, stats, engine or "chatterbox"),
    }


def queue_book_after_cast(queue: JobQueue, job: dict) -> Optional[str]:
    """JobQueue.on_done: a cast analysis started with Auto-pick on puts its book in the queue as
    soon as the cast is ready, with the narrator and delivery the book's tone asks for (as the page
    would set them) and every other option as the Make tab stood at Analyse. The book waits for
    Start queued books like any other. Returns the analysis job's note."""
    later = job["settings"].get("then_queue")
    if job_kind(job) != CAST or not later:
        return None
    s = job["settings"]
    engine = s.get("engine") or "chatterbox"
    try:
        path, cast = _finished_cast(s.get("cast_key"))
        suggestion = _fill_narrator_suggestion(cast, path, engine, later["voice"])
        voice = suggestion["voice"] if suggestion else later["voice"]
        _fill_missing_voices(cast, path, engine, voice)
        source = suggestion or later
        book_settings = queue_settings(
            s["input_file"] if later["from_library"] else None, s["input_file"],
            [[n, True] for n in s["chapter_selection"]], later["output_dir"], voice, later["speed"],
            later["sentence_pause"], later["paragraph_pause"], later["output_m4b"], later["skip_existing"],
            later["output_text"], s["title_mode"], s["newline_mode"], s["remove_endnotes"],
            s["remove_reference_numbers"], s.get("search_and_replace_file"), s.get("log_level") or "INFO",
            later["paced_unit_mode"], engine, VOICE_MODE_CAST, later["dialogue_voice"], s["cast_key"],
            later["adaptive_delivery"], source["exaggeration"], source["cfg_weight"], source["temperature"],
            later.get("tone_match", True), active_jobs=queue.jobs())
    except gr.Error as e:
        message = getattr(e, "message", None) or str(e)
        logger.warning(f"Queue: '{job['title']}' is ready but its book was not queued: {message}")
        return f"book not queued: {message}"
    title = os.path.basename(book_settings["output_dir"].rstrip("/\\")) or "Book"
    queue.add(title, book_settings, len(book_settings["chapter_selection"]), later["estimate_seconds"], voice)
    logger.info(f"Queue: cast ready, added '{title}' (waits for Start queued books)")
    return "book added to the queue"


def _book_on_its_way(job: dict) -> bool:
    """A waiting book, or an analysis that will add its book to the queue when it finishes."""
    if job_kind(job) == BOOK:
        return job["status"] == QUEUED
    return job["status"] in (QUEUED, RUNNING) and bool(job["settings"].get("then_queue"))


def start_available(queue: JobQueue) -> bool:
    """Start queued books shows while books are held, including ones whose analysis is still
    running: pressing it early lets each book start once every analysis is done."""
    return queue.preparing and any(_book_on_its_way(job) for job in queue.jobs())


def enqueue_button_update(voice_mode: str, auto_pick: bool) -> dict:
    """Add to queue is hidden in Cast mode with Auto-pick on: the book joins the queue by itself."""
    return gr.update(visible=not (voice_mode == VOICE_MODE_CAST and auto_pick))


def analysis_estimate(table, stats: list) -> float:
    """Seconds the LLM pass is guessed to take for the ticked chapters (ANALYSIS_SECONDS_PER_LINE)."""
    stats = stats or []
    chosen = [stats[n - 1] for n in selected_chapter_numbers(table) if 0 < n <= len(stats)]
    return sum((s[3] if len(s) > 3 else 0) for s in chosen) * ANALYSIS_SECONDS_PER_LINE


def voice_mode_changed(voice_mode: str) -> tuple:
    """Show the dialogue voice for the two multi-voice modes and the cast panel for cast mode."""
    multi = voice_mode in (VOICE_MODE_DIALOGUE, VOICE_MODE_CAST)
    return gr.update(visible=multi), gr.update(visible=voice_mode == VOICE_MODE_CAST)


def voice_gender_of(voice: Optional[str]) -> dict:
    """Voice lab: the recorded gender of a Chatterbox voice ("" when none)."""
    return gr.update(value=cast_store.load_voice_genders().get(voice or "", ""))


def save_voice_gender(voice: Optional[str], gender: str) -> str:
    """Voice lab: record a Chatterbox voice's gender for cast suggestions."""
    if not voice:
        raise gr.Error("Pick a voice first.")
    cast_store.save_voice_gender(voice, gender or None)
    label = dict((value, label) for label, value in VOICE_GENDER_CHOICES).get(gender or "", "not set")
    return f"**{os.path.splitext(voice)[0]}**: {label}. Cast suggestions use this."


def stats_for_estimate(table, stats: list, settings: dict) -> list:
    """The chapter stats a queued job's time estimate needs. They live in the page's session, which
    an app restart empties while the open page still shows its chapter table (every estimate then
    came out 0, 2026-09-30), so they are read from the book again when they don't cover the ticked
    chapters. `settings` are the job's (book and parsing options)."""
    stats = stats or []
    if all(0 < n <= len(stats) for n in selected_chapter_numbers(table)):
        return stats
    try:
        chapters = book_chapters(settings["input_file"], settings["title_mode"], settings["newline_mode"],
                                 settings["remove_endnotes"], settings["remove_reference_numbers"],
                                 settings.get("search_and_replace_file"))
    except Exception as error:
        logger.warning(f"Could not read the chapters again for the time estimate: {error}")
        return stats
    return [chapter_stats(text) for _, text in chapters]


def generation_estimate(table, stats: list, engine: str = "chatterbox") -> float:
    """Seconds the engine needs for the ticked chapters."""
    stats = stats or []
    chosen = [stats[n - 1] for n in selected_chapter_numbers(table) if 0 < n <= len(stats)]
    chars_per_second, generation_speed, paced_overhead = _engine_estimate_constants(engine)
    return sum(s[0] for s in chosen) / chars_per_second / generation_speed * paced_overhead


def _status_label(job: dict) -> str:
    note = f" · {job['note']}" if job.get("note") else ""
    if job["status"] == RUNNING:
        verb = "analysing cast" if job_kind(job) == CAST else "generating"
        return f"▶ {verb} · {JobQueue.chapters_done(job)} of {job['chapters']} chapters done"
    if job["status"] == QUEUED:
        return f"waiting{note}"
    if job["status"] == DONE:  # a cast analysis says whether its book joined the queue
        return f"✓ done {job['finished']}" + (note if job_kind(job) == CAST else "")
    if job["status"] == FAILED:
        return f"✗ failed{note}"
    return f"■ stopped{note}"


QUEUE_COLUMNS = ["#", "Book", "Voice", "Chapters", "Status", "Generating time"]


def _voice_column(job: dict) -> str:
    """Voice column text: the bare (extension-stripped) file name for Chatterbox, prefixed with
    the engine name for Kokoro (its ids carry no file extension to strip) and Breeze; a cast analysis says
    so, and a multi-voice book adds its mode."""
    if job_kind(job) == CAST:
        return "cast analysis (LLM)"
    settings = job.get("settings", {})
    engine = settings.get("engine", "chatterbox")
    voice = (f"Kokoro · {job['voice']}" if engine == "kokoro"
             else f"Breeze · {os.path.splitext(job['voice'])[0]}" if engine == "breeze"
             else os.path.splitext(job["voice"])[0])
    mode = settings.get("voice_mode", VOICE_MODE_SINGLE)
    if mode == VOICE_MODE_CAST:
        return f"{voice} + cast"
    if mode == VOICE_MODE_DIALOGUE:
        return f"{voice} + dialogue voice"
    return voice


def queue_view(queue: JobQueue) -> tuple:
    """(table rows, job ids in row order, one-line queue status)."""
    jobs = queue.jobs()
    rows = [[n, job["title"], _voice_column(job), job["chapters"], _status_label(job),
             _duration(job["estimate_seconds"])] for n, job in enumerate(jobs, start=1)]
    ids = [job["id"] for job in jobs]
    left = 0.0
    for job in jobs:
        if job["status"] == QUEUED:
            left += job["estimate_seconds"]
        elif job["status"] == RUNNING:
            left += job["estimate_seconds"] * (1 - JobQueue.chapters_done(job) / max(1, job["chapters"]))
    to_go = sum(1 for job in jobs if job["status"] in (QUEUED, RUNNING))
    if not jobs and not queue.preparing:
        status = ("The queue is empty. Pick a book, then **Analyse selected chapters** (Cast mode) or "
                  "**Add this book to queue**.")
    elif queue.paused:
        status = f"⏸ **Queue paused** · {to_go} job(s) waiting. Press **Resume queue** to continue."
    elif queue.preparing:
        waiting_books = sum(job_kind(job) == BOOK and job["status"] == QUEUED for job in jobs)
        if waiting_books:
            status = (f"🎭 **Preparing casts** · {waiting_books} book(s) waiting. "
                      "Analyse and add any other books, then press **Start queued books**. "
                      "A book already generating will finish first.")
        elif any(job_kind(job) == CAST and job["status"] in (QUEUED, RUNNING) and job["settings"].get("then_queue")
                 for job in jobs):
            status = ("🎭 **Preparing casts** · each book joins the queue when its analysis finishes. Pick more "
                      "books, or press **Start queued books** once you're done picking.")
        else:
            status = ("🎭 **Preparing cast** · choose voices when analysis finishes, "
                      "then press **Add this book to queue**.")
    elif to_go:
        status = f"▶ **Working** · {to_go} book(s) to go · {_about(left)} of generating left."
    else:
        status = "✓ **All done.** Finished books are in the audiobook library."
    return rows, ids, status


# ---- Chapter list ----

# Measured on a finished book (Elena, speed 1.0): 20.2 characters of text per second of speech.
# Generation assumes Chatterbox's compiled token loop (F-45, TTS_COMPILE=on): 3.87x real time for one
# long request, and sentence-sized requests take 26% longer than that, since the fixed per-request cost
# weighs more once tokens are fast (live, 2026-09-28, a 1,159-character chapter). Without it: 1.8x
# and 14% (A/B, 2026-09-27). Other voices read at somewhat different paces.
CHARS_PER_AUDIO_SECOND = 20.2
GENERATION_SPEED = 3.87
PACED_GENERATION_OVERHEAD = 1.26

# Measured live 2026-09-28 against Kokoro's af_heart voice through the real paced path
# (paced_units + OpenAITTSProvider, sentence pause 0.35s / paragraph pause 0.9s, speed 1.0): two
# invented paragraphs (6 sentences, 355 characters of speech) sent as 6 real paced requests took
# 1.22s wall time for 20.95s of audio (ffprobe); the same text as one undivided request took
# 0.41s for 21.72s of audio -- 52.9x real time, close to the ~0.09s-per-5s figure from the
# earlier live check. 355 chars / 20.95s = 16.9 chars/s. The paced run took 3.07x the wall time
# the bulk rate implies for that much audio: Kokoro generates so fast that many small
# sentence-sized requests' fixed per-request overhead dominates far more than it does for
# Chatterbox. Single small sample, one voice; not re-validated across books or voices.
KOKORO_CHARS_PER_AUDIO_SECOND = 16.9
KOKORO_GENERATION_SPEED = 52.9
KOKORO_PACED_GENERATION_OVERHEAD = 3.07
# Breeze speaks about as fast as Chatterbox (same chars per audio second). Its server generates at 7.1x
# real time with 32 sentences per request (measured 2026-10-01); with the speech check on every take and
# the retried ones that is a GUESS of 4x until a real book has been timed.
BREEZE_GENERATION_SPEED = 4.0
CHAPTER_COLUMNS = ["#", "Include", "Chapter", "Starts with", "Listening"]
_SENTENCE_END = re.compile(r"[.!?\u2026]+[\"'\u201d\u2019)\]]*(?=\s|$)")


def _duration(seconds: float) -> str:
    minutes = round(seconds / 60)
    if seconds < 60:
        return "under 1 min"
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60} min"


def _about(seconds: float) -> str:
    return "less than a minute" if seconds < 60 else f"about {_duration(seconds)}"


def chapter_stats(text: str) -> list:
    """[characters, sentences, paragraphs, dialogue lines] of a chapter parsed with paragraph marks."""
    paragraphs = [p for p in (" ".join(part.split()) for part in text.split(PARAGRAPH_MARK)) if p]
    plain = " ".join(paragraphs)
    return [len(plain), max(1, len(_SENTENCE_END.findall(plain))), max(1, len(paragraphs)), len(dialogue_lines(text))]


def _engine_estimate_constants(engine: str) -> Tuple[float, float, float]:
    """(chars_per_audio_second, generation_speed, paced_overhead) for the given engine."""
    if engine == "kokoro":
        return KOKORO_CHARS_PER_AUDIO_SECOND, KOKORO_GENERATION_SPEED, KOKORO_PACED_GENERATION_OVERHEAD
    if engine == "breeze":
        return CHARS_PER_AUDIO_SECOND, BREEZE_GENERATION_SPEED, 1.0
    return CHARS_PER_AUDIO_SECOND, GENERATION_SPEED, PACED_GENERATION_OVERHEAD


def listening_seconds(stats: list, speed, sentence_pause, paragraph_pause, engine: str = "chatterbox") -> float:
    """Speech plus the inserted pauses (both shrink with speed)."""
    characters, sentences, paragraphs = stats[:3]
    chars_per_second, _, _ = _engine_estimate_constants(engine)
    pauses = (max(0, sentences - paragraphs) * float(sentence_pause or 0)
              + max(0, paragraphs - 1) * float(paragraph_pause or 0))
    return (characters / chars_per_second + pauses) / float(speed or 1.0)


def book_chapters(book: str, title_mode: str, newline_mode: str, remove_endnotes: bool,
                  remove_reference_numbers: bool, search_and_replace_file) -> list:
    """(title, text with paragraph marks) per chapter, numbered exactly as the generator numbers them."""
    config = GeneralConfig(None)
    config.input_file = book
    config.title_mode = title_mode
    config.newline_mode = newline_mode
    config.remove_endnotes = remove_endnotes
    config.remove_reference_numbers = remove_reference_numbers
    config.search_and_replace_file = (search_and_replace_file.name if hasattr(search_and_replace_file, "name")
                                      else search_and_replace_file)
    chapters = get_book_parser(config).get_chapters(f" {PARAGRAPH_MARK}")
    return [(title, text) for title, text in chapters if text.strip()]  # same filter as the generator


def chapter_summary(table, stats: list, speed, sentence_pause, paragraph_pause,
                    engine: str = "chatterbox") -> str:
    """One-line summary of the ticked chapters."""
    rows = _table_rows(table)
    if not rows:
        return ""
    picked = selected_chapter_numbers(rows)
    if not picked:
        return "⚠️ No chapters ticked."
    stats = stats or []
    chosen = [stats[n - 1] for n in picked if 0 < n <= len(stats)]
    audio_seconds = sum(listening_seconds(s, speed, sentence_pause, paragraph_pause, engine) for s in chosen)
    chars_per_second, generation_speed, paced_overhead = _engine_estimate_constants(engine)
    speech_seconds = sum(s[0] for s in chosen) / chars_per_second
    skipped = len(rows) - len(picked)
    unticked = f" ({skipped} unticked)" if skipped else ""
    return (f"**{len(picked)} of {len(rows)} chapters** ticked{unticked} · "
            f"**{_about(audio_seconds)}** of audio at {float(speed or 1.0):g}× · "
            f"{_about(speech_seconds / generation_speed * paced_overhead)} to generate. "
            f"They'll be numbered 1–{len(picked)} in the finished book.")


def chapter_overview(library_book, input_file, speed, sentence_pause, paragraph_pause, engine, title_mode: str,
                     newline_mode: str, remove_endnotes: bool, remove_reference_numbers: bool,
                     search_and_replace_file) -> tuple:
    """Chapter table with story chapters pre-ticked and front/back matter unticked."""
    valid_library_pick = bool(library_book) and os.path.isfile(library_book)
    book = library_book if valid_library_pick else input_file
    if not book:
        if library_book and not input_file:
            # Typed text that doesn't match a library book, with nothing uploaded either: say so
            # instead of silently hiding the table (F-42c).
            return (gr.update(value=None, visible=False), [],
                    "Pick the book from the list as you type (or clear the box to use an upload).")
        return gr.update(value=None, visible=False), [], ""
    book = book.name if hasattr(book, "name") else book
    try:
        chapters = book_chapters(book, title_mode, newline_mode, remove_endnotes, remove_reference_numbers,
                                 search_and_replace_file)
    except Exception as e:
        return gr.update(value=None, visible=False), [], f"Could not read chapters from this book: {e}"
    if not chapters:
        return gr.update(value=None, visible=False), [], "No chapters with text found in this book."
    plain = [(title, " ".join(text.replace(PARAGRAPH_MARK, " ").split())) for title, text in chapters]
    include = preselect_chapters(plain)
    stats = [chapter_stats(text) for _, text in chapters]
    rows = [[number, ticked, title.replace("_", " "), text[:70],
             _duration(listening_seconds(stat, speed, sentence_pause, paragraph_pause, engine))]
            for number, ((title, text), ticked, stat) in enumerate(zip(plain, include, stats), start=1)]
    return (gr.update(value=rows, visible=True), stats,
            chapter_summary(rows, stats, speed, sentence_pause, paragraph_pause, engine))


def retime_chapters(table, stats: list, speed, sentence_pause, paragraph_pause,
                    engine: str = "chatterbox") -> dict:
    """Speed, pauses or engine changed: update the Listening column, keeping the ticks."""
    rows = _table_rows(table)
    stats = stats or []
    for row in rows:
        number = int(row[0])
        if 0 < number <= len(stats):
            row[4] = _duration(listening_seconds(stats[number - 1], speed, sentence_pause, paragraph_pause, engine))
    return gr.update(value=rows) if rows else gr.update()


def tick_all_chapters(table) -> dict:
    rows = _table_rows(table)
    for row in rows:
        row[1] = True
    return gr.update(value=rows) if rows else gr.update()


def library_output_dir(library_book: Optional[str]) -> dict:
    """Picking a library book points the output folder at audiobook_output/<its title>."""
    if not library_book:
        return gr.update()
    folder = safe_folder_name(library_index.book_title(library_book, library_index.load_index()))
    return gr.update(value=os.path.join(OUTPUT_ROOT, folder) if folder else timestamped_output_dir())


def uploaded_book_selected(input_file) -> tuple:
    """An upload takes over from the library pick and names the output folder."""
    if not input_file:
        return gr.update(), gr.update()
    return suggest_output_dir(input_file), gr.update(value=None)


def refresh_library() -> dict:
    return gr.update(choices=library_index.book_choices(library_index.refresh_index()))


def refresh_voices() -> tuple:
    """Page-load handler: re-list Chatterbox voices for the Make tab, the Voice lab and the
    delete dropdown so newly added or removed voice files appear without a restart. Chatterbox
    is always the starting engine on a fresh page load, so this only needs its own voice list."""
    choices = openai_voice_choices()
    default = default_openai_voice(choices)
    return (gr.update(choices=choices, value=default), gr.update(choices=choices, value=default),
            gr.update(choices=own_voice_choices()), _dialogue_voice_update(choices, default))


def _dialogue_voice_update(choices: list, default: Optional[str]) -> dict:
    """The Dialogue voice dropdown for an engine's voices: the narrator's voice first, and selected
    when the page starts in cast mode (an LLM is configured)."""
    return gr.update(choices=dialogue_voice_choices(choices),
                     value=NARRATOR_FALLBACK if llm_configured() else default)


def engine_changed(engine: str) -> tuple:
    """Switching the Make tab's engine swaps the Voice, Dialogue voice and cast-editor voice
    dropdowns to that engine's own choices and default (Chatterbox's file list, or Kokoro's
    English voices from its live API; Breeze clones the same voice files as Chatterbox), and shows
    the adaptive delivery checkbox and its baseline line only while Chatterbox is selected (delivery is
    ignored entirely for Kokoro and Breeze)."""
    if engine == "kokoro":
        choices, default = kokoro_voices_and_default()
    else:
        choices = openai_voice_choices()
        default = default_openai_voice(choices)
    is_chatterbox = engine not in ("kokoro", "breeze")
    return (gr.update(choices=choices, value=default), _dialogue_voice_update(choices, default),
            gr.update(choices=choices, value=None), gr.update(visible=is_chatterbox), gr.update(visible=is_chatterbox))


def _sync_if_chatterbox(value: str, engine: str) -> dict:
    """Pass a voice value through to the paired dropdown only while the Make tab's engine is
    Chatterbox: a Kokoro id must never land in the (Chatterbox-only) Voice lab, and a Chatterbox
    file name must never land in the Make-tab dropdown while Kokoro is selected there."""
    return gr.update(value=value) if engine in FILE_VOICE_ENGINES else gr.update()


# ---- Layout ----

def build_ui(queue: Optional[JobQueue] = None) -> gr.Blocks:
    web_ui.webui_log_file = generate_unique_log_path("EtA_WebUI")
    web_ui.webui_log_file.touch()
    if queue is None:  # standalone build (tests): a throwaway queue that never starts on its own
        queue = JobQueue(os.path.join(tempfile.mkdtemp(), QUEUE_FILE), build_config,
                         lambda: str(web_ui.webui_log_file.absolute()))
    choices = openai_voice_choices()
    default_voice = default_openai_voice(choices)
    kokoro_configured = bool(kokoro_base_url())
    engine_choices = [("Chatterbox", "chatterbox")]
    if kokoro_configured:
        engine_choices.append(("Kokoro", "kokoro"))
    if breeze_client.configured():
        engine_choices.append(BREEZE_ENGINE_CHOICE)
    # Breeze is the default once it is set up: the owner retired Chatterbox for it (2026-10-01).
    initial_engine = "breeze" if breeze_client.configured() else "chatterbox"
    saved = read_saved_settings()  # also seeds the Make tab's initial delivery baseline line

    def refresh_queue() -> tuple:
        rows, ids, status = queue_view(queue)
        return gr.update(value=rows), ids, status, gr.update(visible=start_available(queue))

    def enqueue(library_book, input_file, chapter_table, stats, *settings) -> tuple:
        job_settings = queue_settings(library_book, input_file, chapter_table, *settings, active_jobs=queue.jobs())
        stats = stats_for_estimate(chapter_table, stats, job_settings)
        title = os.path.basename(job_settings["output_dir"].rstrip("/\\")) or "Book"
        position = queue.add(title, job_settings, len(job_settings["chapter_selection"]),
                             generation_estimate(chapter_table, stats, job_settings["engine"]),
                             job_settings["voice"])
        queue.tick()
        gr.Info((f"Added '{title}' to the queue; it will wait for Start queued books."
                 if queue.preparing else f"Added '{title}' to the queue" +
                 ("." if position <= 1 else f" (#{position} in line).")))
        return refresh_queue()

    def queue_analysis(library_book, input_file, chapter_table, stats, output_dir, voice, speed, sentence_pause,
                       paragraph_pause, output_m4b, skip_existing, output_text, title_mode, newline_mode,
                       remove_endnotes, remove_reference_numbers, search_and_replace_file, log_level,
                       paced_unit_mode, engine, voice_mode, dialogue_voice, cast_key, adaptive_delivery,
                       exaggeration, cfg_weight, temperature, tone_match, auto_pick_voices) -> tuple:
        """Analyse this cast before generating queued books; a running book finishes first. With
        Auto-pick on, the book follows its analysis into the queue (queue_book_after_cast)."""
        job_settings = analysis_settings(library_book, input_file, chapter_table, engine, voice, title_mode,
                                         newline_mode, remove_endnotes, remove_reference_numbers,
                                         search_and_replace_file, log_level, auto_pick_voices)
        for job in queue.jobs():
            if (job_kind(job) == CAST and job["status"] in (QUEUED, RUNNING)
                    and job["settings"].get("cast_key") == job_settings["cast_key"]):
                raise gr.Error("This book's cast analysis is already in the queue.")
        stats = stats_for_estimate(chapter_table, stats, job_settings)
        if auto_pick_voices:
            job_settings["then_queue"] = book_options_for_later(
                library_book, chapter_table, stats, output_dir, voice, speed, sentence_pause, paragraph_pause,
                output_m4b, skip_existing, output_text, engine, paced_unit_mode, dialogue_voice,
                adaptive_delivery, exaggeration, cfg_weight, temperature, tone_match, active_jobs=queue.jobs())
        book = job_settings["input_file"]
        title = f"Cast: {library_index.book_title(book, library_index.load_index()) or os.path.basename(book)}"
        # Books already started keep going (this analysis runs after the current one, and its book
        # joins them); otherwise a new batch begins and its books wait for Start queued books.
        started = not queue.preparing and any(job_kind(job) == BOOK and job["status"] in (QUEUED, RUNNING)
                                              for job in queue.jobs())
        if not started:
            queue.set_preparing(True)
        queue.set_paused(False)
        queue.add(title, job_settings, len(job_settings["chapter_selection"]),
                  analysis_estimate(chapter_table, stats), voice, kind=CAST)
        queue.tick()
        if started:
            gr.Info("Cast analysis queued after the current book; " +
                    ("this book then joins the books already started." if auto_pick_voices else
                     "add the book to the queue when it's ready."))
        else:
            gr.Info("Cast analysis queued; the book joins the queue when its cast is ready. Audiobook jobs wait "
                    "until you start them." if auto_pick_voices else
                    "Cast analysis queued. Audiobook jobs will wait until you start them.")
        return (*refresh_queue(), job_settings["cast_key"],
                "⏳ Cast analysis queued. This panel updates as it runs.")

    def refresh_cast(voice_mode, cast_key, engine, voice, seen, auto_pick) -> tuple:
        if voice_mode != VOICE_MODE_CAST:
            return gr.update(), gr.update(), gr.update(), seen, *_narrator_updates(None)
        return cast_panel_update(cast_key, engine, voice, seen, auto_pick)

    def delete_voice(name: Optional[str], current_lab_voice: str, current_voice: str, engine: str) -> tuple:
        """Delete an own voice after the browser confirms (see the button's js=); None means the
        owner cancelled the confirm, so nothing changes."""
        if name is None:
            return gr.update(), gr.update(), gr.update(), gr.update()
        message = delete_own_voice(name, queue.jobs())
        choices = openai_voice_choices()
        default = default_openai_voice(choices)
        lab_update = _voice_dropdown_after_delete(choices, current_lab_voice, name, default)
        voice_update = (_voice_dropdown_after_delete(choices, current_voice, name, default)
                        if engine in FILE_VOICE_ENGINES else gr.update())
        return message, lab_update, voice_update, gr.update(choices=own_voice_choices(), value=None)

    def select_job(ids: list, evt: gr.SelectData) -> tuple:
        row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        job = next((j for j in queue.jobs() if 0 <= row < len(ids) and j["id"] == ids[row]), None)
        if not job:
            return None, ""
        return job["id"], f"Selected: **{job['title']}**"

    def remove_selected(job_id) -> tuple:
        if not job_id:
            raise gr.Error("Click a book in the queue first.")
        if not queue.remove(job_id):
            raise gr.Error("That book is generating: press Stop current book first.")
        return (*refresh_queue(), None, "")

    def retry_selected(job_id) -> tuple:
        if not job_id or not queue.retry(job_id):
            raise gr.Error("Pick a failed or stopped book to retry.")
        gr.Info("Re-queued; chapters already made are kept.")
        return (*refresh_queue(), None, "")

    def clear_finished() -> tuple:
        queue.clear_finished()
        return refresh_queue()

    def pause_queue() -> tuple:
        queue.set_paused(True)
        return refresh_queue()

    def resume_queue() -> tuple:
        queue.set_paused(False)
        queue.tick()
        return refresh_queue()

    def start_books() -> tuple:
        queue.set_preparing(False)
        queue.set_paused(False)
        queue.tick()
        return refresh_queue()

    def stop_current() -> tuple:
        if queue.stop_current():
            gr.Info("Stopped. The queue is paused: press Resume queue to go on to the next book.")
        return refresh_queue()

    with gr.Blocks(analytics_enabled=False, title="Audiobook Maker") as ui:
        with gr.Tab("Make audiobook"):
            with gr.Row(equal_height=True):
                with gr.Column():
                    library_book = gr.Dropdown(library_index.book_choices(library_index.load_index()), value=None,
                                               label="Book", filterable=True, allow_custom_value=True,
                                               info="Type part of a title or author to search the library.")
                    with gr.Accordion("Or upload an EPUB", open=False):
                        input_file = gr.File(label="EPUB file", file_types=[".epub"], file_count="single")
                with gr.Column():
                    output_dir = gr.Textbox(label="Output folder", value=timestamped_output_dir,
                                            info="Filled in from the book title; lands in the audiobook library.")
                    with gr.Row(equal_height=True):
                        engine = gr.Dropdown(engine_choices, value=initial_engine, label="Engine",
                                             visible=len(engine_choices) > 1, scale=1)
                        voice = gr.Dropdown(choices, value=default_voice, label="Voice",
                                            allow_custom_value=True, scale=2)
                        with gr.Column(scale=0, min_width=100):
                            sample_button = gr.Button("▶ Sample", size="sm")
                    sample_audio = gr.Audio(label="Sample", show_label=False, autoplay=True,
                                            interactive=False, visible=False)
                    speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed",
                                      info="1.0 recommended (other speeds are stretched after generation).")
            # Cast is the default whenever an LLM is configured (without one it isn't offered).
            initial_mode = VOICE_MODE_CAST if llm_configured() else VOICE_MODE_SINGLE
            with gr.Row(equal_height=True):
                voice_mode = gr.Radio(voice_mode_choices(), value=initial_mode, label="Voice mode", scale=2,
                                      info="Single voice reads everything as before. The other modes give quoted "
                                           "lines their own voice; Cast asks the local LLM who speaks each line.")
                dialogue_voice = gr.Dropdown(dialogue_voice_choices(choices), label="Dialogue voice", scale=1,
                                             value=NARRATOR_FALLBACK if initial_mode == VOICE_MODE_CAST else default_voice,
                                             allow_custom_value=True, visible=initial_mode != VOICE_MODE_SINGLE,
                                             info="Quoted lines. In cast mode, only lines whose speaker wasn't "
                                                  "found: the narrator's voice unless you pick another.")
            with gr.Row(equal_height=True):
                adaptive_delivery = gr.Checkbox(
                    True, label="Adaptive delivery", visible=initial_engine == "chatterbox",
                    info="Dialogue tagged whispered/shouted (and, in Cast mode, the LLM's own read) is spoken "
                         "softer or more excited around this book's baseline, instead of one flat delivery.")
                delivery_baseline_info = gr.Markdown(
                    delivery_baseline_text(saved["exaggeration"], saved["cfg_weight"], saved["temperature"]),
                    visible=initial_engine == "chatterbox")
            with gr.Row(equal_height=True):
                sentence_pause = gr.Slider(0.0, 1.5, value=0.35, step=0.05, label="Pause after sentences (s)")
                paragraph_pause = gr.Slider(0.0, 3.0, value=0.9, step=0.1, label="Pause between paragraphs (s)")
                output_m4b = gr.Checkbox(True, label="Single M4B file (recommended)",
                                         info="One file with chapter markers and cover, instead of an MP3 per chapter.")
            chapters_info = gr.Markdown()
            chapter_table = gr.Dataframe(headers=CHAPTER_COLUMNS, datatype=["number", "bool", "str", "str", "str"],
                                         interactive=True, static_columns=[0, 2, 3, 4], wrap=True, visible=False,
                                         label="Chapters: untick anything you don't want narrated",
                                         column_widths=["6%", "9%", "30%", "40%", "15%"], max_height=520)
            chapter_stats_state = gr.State([])
            with gr.Row(equal_height=True):
                tick_all_button = gr.Button("Tick all", size="sm")
                auto_tick_button = gr.Button("Auto-select (skip front/back matter)", size="sm")
                skip_existing = gr.Checkbox(False, label="Skip chapters already made",
                                            info="Resume an interrupted book in the same folder.")
            with gr.Accordion("Advanced", open=False):
                with gr.Row(equal_height=True):
                    title_mode = gr.Dropdown(["auto", "tag_text", "first_few"], value="auto", label="Chapter titles")
                    newline_mode = gr.Dropdown(["double", "single", "none"], value="double",
                                               label="Paragraph detection")
                    log_level = gr.Dropdown(["INFO", "DEBUG", "WARNING", "ERROR"], value="INFO", label="Log level")
                with gr.Row(equal_height=True):
                    remove_endnotes = gr.Checkbox(False, label="Remove endnote numbers")
                    remove_reference_numbers = gr.Checkbox(False, label="Remove [1]-style references")
                    output_text = gr.Checkbox(False, label="Also save each chapter's text")
                paced_unit_mode = gr.Dropdown(
                    [("One sentence per request (default)", "sentence"),
                     ("Whole paragraphs (about 8% faster, experimental)", "paragraph")],
                    value="sentence", label="Narration units",
                    info="Paragraphs send fewer, longer requests; the pauses inside a paragraph are placed "
                         "at the gaps Chatterbox leaves between sentences. Kokoro always narrates by "
                         "sentence: its gap detector was tuned on Chatterbox audio, and it saves nothing "
                         "on a server this fast.")
                tone_match = gr.Checkbox(
                    True, label="Match each voice to its clip",
                    info="Chatterbox makes some voices brighter than the recording they are copied from: a "
                         "sharp, fizzy top end. This turns each voice down only where it comes out brighter "
                         "than its own clip, never up. Chatterbox only.")
                search_and_replace_file = gr.File(label="Search & replace file (optional, e.g. fix pronunciations)",
                                                  file_types=[".txt"], file_count="single")
            with gr.Column(visible=initial_mode == VOICE_MODE_CAST) as cast_panel:
                gr.Markdown("**Cast workflow:** Choose chapters above → analyse cast. The narrator, every "
                            "character's voice and their delivery are picked for you, and the book joins the "
                            "queue once its cast is ready. Repeat for another book, then start the queued books. "
                            "To review a cast before queuing, untick Auto-pick under *Adjust the cast*.")
                with gr.Row(equal_height=True):
                    analyse_button = gr.Button("🎭 Analyse selected chapters", scale=0, min_width=210)
                    cast_status = gr.Markdown("Choose chapters above, then press **Analyse selected chapters**.")
                cast_table = gr.Dataframe(headers=CAST_COLUMNS,
                                          datatype=["str", "str", "number", "str", "str", "str", "str", "str"],
                                          interactive=False, wrap=True, visible=False,
                                          label="Cast: click a character to see who they are",
                                          column_widths=["15%", "10%", "6%", "8%", "7%", "14%", "24%", "16%"],
                                          max_height=400)
                cast_profile = gr.Markdown("")
                with gr.Accordion("Adjust the cast (advanced)", open=False):
                    with gr.Row(equal_height=True):
                        auto_pick_voices = gr.Checkbox(
                            True, label="Auto-pick suggested voices", scale=0, min_width=230,
                            info="Picks the narrator and delivery from the book's tone, queues the book as soon "
                                 "as its cast is ready, and re-analysing keeps only voices you saved here.")
                        resuggest_button = gr.Button("🎯 Suggest voices again", scale=0, min_width=190)
                    resuggest_confirmed = gr.Checkbox(False, visible=False)
                    with gr.Row(equal_height=True):
                        cast_editing = gr.Markdown("Click a character in the table.")
                        cast_gender = gr.Dropdown(_GENDER_CHOICES, value="unknown", label="Gender", scale=1)
                        cast_voice = gr.Dropdown(choices, value=None, label="Voice", allow_custom_value=True, scale=2)
                        cast_delivery = gr.Dropdown(DELIVERY_CHOICES, value="auto", label="Delivery", scale=1)
                        with gr.Column(scale=0, min_width=100):
                            cast_sample_button = gr.Button("▶ Sample", size="sm")
                            cast_apply_button = gr.Button("Save", size="sm")
                    with gr.Row(equal_height=True):
                        cast_merge_into = gr.Dropdown([], value=None, label="Same person as", scale=2,
                                                      info="One person listed twice (a first name and a surname, "
                                                           "two spellings): merge them.")
                        cast_merge_button = gr.Button("Merge", size="sm", scale=0, min_width=100)
                    cast_merge_confirmed = gr.Checkbox(False, visible=False)
                cast_key_state = gr.State(None)
                cast_keys_state = gr.State([])
                cast_selected = gr.State(None)
                cast_seen = gr.State(None)
            enqueue_button = gr.Button("➕ Add this book to queue", variant="primary",
                                       visible=initial_mode != VOICE_MODE_CAST)  # Auto-pick starts on
            with gr.Accordion("Queue", open=True):
                queue_status = gr.Markdown()
                start_books_button = gr.Button("▶ Start queued books", variant="primary",
                                               visible=start_available(queue))
                queue_table = gr.Dataframe(headers=QUEUE_COLUMNS, interactive=False, wrap=True, label="Books",
                                           column_widths=["5%", "33%", "14%", "9%", "27%", "12%"])
                queue_ids = gr.State([])
                selected_job = gr.State(None)
                selected_info = gr.Markdown()
                with gr.Accordion("More queue actions", open=False):
                    with gr.Row():
                        remove_button = gr.Button("Remove selected", size="sm")
                        retry_button = gr.Button("Retry selected", size="sm")
                        clear_button = gr.Button("Clear finished", size="sm")
                        pause_button = gr.Button("Pause queue", size="sm")
                        resume_button = gr.Button("Resume queue", size="sm")
                        stop_button = gr.Button("Stop current book", variant="stop", size="sm")
            queue_timer = gr.Timer(3)
            Log(str(web_ui.webui_log_file.absolute()), dark=True, xterm_font_size=12)

        with gr.Tab("Voice lab"):
            gr.Markdown("Try voices and tune delivery. **Save** applies the sliders to every book "
                        "(including one in progress). While a book is generating, a preview or "
                        "**Save** waits for the current chunk to finish (up to about a minute): "
                        "Chatterbox handles one request at a time.")
            with gr.Row(equal_height=True):
                lab_voice = gr.Dropdown(choices, value=default_voice, label="Voice", allow_custom_value=True)
                phrase = gr.Textbox(PREVIEW_PHRASE, lines=2, label="Phrase")
            with gr.Row(equal_height=True):
                exaggeration = gr.Slider(0.25, 2.0, value=saved["exaggeration"], step=0.01, label="Exaggeration",
                                         info="Emotion and emphasis")
                cfg_weight = gr.Slider(0.1, 1.0, value=saved["cfg_weight"], step=0.05, label="CFG weight",
                                       info="Lower = slower, more deliberate")
                temperature = gr.Slider(0.05, 1.5, value=saved["temperature"], step=0.05, label="Temperature",
                                        info="Higher = more varied")
            with gr.Row():
                play_button = gr.Button("▶ Play", variant="primary")
                play_delivery_button = gr.Button("▶ Play soft / normal / excited")
                save_button = gr.Button("Save for books")
                reset_button = gr.Button("Reset to saved")
            preview_audio = gr.Audio(label="Preview", autoplay=True, interactive=False)
            lab_status = gr.Markdown()

            gr.Markdown("### Add a voice")
            with gr.Row(equal_height=True):
                sample = gr.Audio(sources=["upload"], type="filepath",
                                  label="Voice sample: ~10 s of one person speaking clearly")
                with gr.Column():
                    new_voice_name = gr.Textbox(label="Voice name")
                    remove_pauses = gr.Checkbox(True, label="Remove long pauses (recommended)",
                                                info="Pauses in a sample make Chatterbox pause mid-sentence.")
                    replace = gr.Checkbox(False, label="Replace a voice with the same name")
                    add_button = gr.Button("Add voice")
            add_status = gr.Markdown()

            gr.Markdown("### Voice gender and sound (for cast suggestions)")
            with gr.Row(equal_height=True):
                lab_gender = gr.Dropdown(VOICE_GENDER_CHOICES, value="", label="Gender of the voice selected above",
                                         info="Multi-voice books suggest voices for characters by gender. "
                                              "Chatterbox voice files carry no gender, so set it here.")
                save_gender_button = gr.Button("Save gender")
            gender_status = gr.Markdown()
            lab_sound = gr.Markdown()
            with gr.Row(equal_height=True):
                measure_button = gr.Button("Measure voices", scale=0, min_width=160)
                measure_status = gr.Markdown(
                    "Cast suggestions match each character's profile to how voices sound: pitch, huskiness and "
                    "liveliness. **Measure voices** speaks one sentence with every voice not measured yet (about "
                    "3 s each); a voice you add is measured straight away.")

            gr.Markdown("### Delete a voice")
            with gr.Row(equal_height=True):
                delete_voice_dropdown = gr.Dropdown(own_voice_choices(), value=None, label="Your voices",
                                                    info="Chatterbox's built-in voices can't be deleted.")
                delete_voice_button = gr.Button("Delete voice", variant="stop")
            delete_voice_status = gr.Markdown()

        # adaptive_delivery, exaggeration, cfg_weight, temperature (the Voice lab sliders) come near the
        # end: enqueue captures the Voice lab's current values as this book's delivery baseline.
        settings = [output_dir, voice, speed, sentence_pause, paragraph_pause, output_m4b, skip_existing,
                    output_text, title_mode, newline_mode, remove_endnotes, remove_reference_numbers,
                    search_and_replace_file, log_level, paced_unit_mode, engine, voice_mode, dialogue_voice,
                    cast_key_state, adaptive_delivery, exaggeration, cfg_weight, temperature, tone_match]
        # The chapter list follows the book and the options that change how it's split (parsing is <0.5 s);
        # this re-runs the auto-selection. Ticks, speed, pauses, engine and "Tick all" only touch the table.
        timing = [speed, sentence_pause, paragraph_pause, engine]
        overview_inputs = [library_book, input_file, *timing, title_mode, newline_mode, remove_endnotes,
                           remove_reference_numbers, search_and_replace_file]
        overview_outputs = [chapter_table, chapter_stats_state, chapters_info]

        library_book.change(library_output_dir, inputs=library_book, outputs=output_dir)
        # An upload takes over from the library pick (its handler also clears library_book); chain
        # with .then() rather than wiring chapter_overview to input_file too, so the table is built
        # once, from the upload, instead of once from the stale library pick and again from the
        # upload (F-42b).
        input_file.change(uploaded_book_selected, inputs=input_file, outputs=[output_dir, library_book]) \
            .then(chapter_overview, inputs=overview_inputs, outputs=overview_outputs) \
            .then(book_cast_key, inputs=[library_book, input_file], outputs=cast_key_state)
        queue_outputs = [queue_table, queue_ids, queue_status, start_books_button]

        # Cast mode: the cast key follows the picked book; the panel follows the cast file.
        # Auto-pick sets the narrator: the Make tab's Voice and the Voice lab sliders (Add to queue reads both).
        narrator_outputs = [voice, exaggeration, cfg_weight, temperature]
        cast_view_inputs = [cast_key_state, engine, voice, cast_seen, auto_pick_voices]
        cast_view_outputs = [cast_table, cast_keys_state, cast_status, cast_seen, *narrator_outputs]
        library_book.change(book_cast_key, inputs=[library_book, input_file], outputs=cast_key_state)
        cast_key_state.change(cast_panel_update, inputs=cast_view_inputs, outputs=cast_view_outputs)
        voice_mode.change(voice_mode_changed, inputs=voice_mode, outputs=[dialogue_voice, cast_panel]) \
            .then(refresh_cast, inputs=[voice_mode, *cast_view_inputs], outputs=cast_view_outputs)
        for control in (voice_mode, auto_pick_voices):
            control.change(enqueue_button_update, inputs=[voice_mode, auto_pick_voices], outputs=enqueue_button)
        queue_timer.tick(refresh_cast, inputs=[voice_mode, *cast_view_inputs], outputs=cast_view_outputs)
        analyse_button.click(queue_analysis,
                             inputs=[library_book, input_file, chapter_table, chapter_stats_state, *settings,
                                     auto_pick_voices],
                             outputs=[*queue_outputs, cast_key_state, cast_status])
        # A browser-only confirm sets the hidden checkbox, then the handler reads it: state values
        # don't pass through a js step reliably, and a cancel must change nothing.
        resuggest_button.click(
            None, inputs=None, outputs=resuggest_confirmed,
            js="() => confirm('Suggest voices again? Voices you saved with Save voice are kept; every other "
               "character gets a fresh suggestion.')",
        ).then(resuggest_cast_voices, inputs=[cast_key_state, engine, voice, resuggest_confirmed],
               outputs=[cast_table, cast_keys_state, cast_status, *narrator_outputs])
        cast_table.select(select_cast_row, inputs=[cast_key_state, cast_keys_state, engine],
                          outputs=[cast_selected, cast_editing, cast_gender, cast_voice, cast_profile, cast_delivery]) \
            .then(merge_choices, inputs=[cast_key_state, cast_selected], outputs=cast_merge_into)
        cast_merge_button.click(
            None, inputs=None, outputs=cast_merge_confirmed,
            js="() => confirm('Merge this character into the one picked? Their lines, names and aliases move "
               "over, and they read in that character\\'s voice.')",
        ).then(merge_cast_character,
               inputs=[cast_key_state, cast_selected, cast_merge_into, engine, cast_merge_confirmed],
               outputs=[cast_table, cast_keys_state, cast_status, cast_selected, cast_editing])
        cast_apply_button.click(apply_cast_edit,
                                inputs=[cast_key_state, cast_selected, cast_gender, cast_voice, engine, cast_delivery],
                                outputs=[cast_table, cast_keys_state, cast_status])
        cast_sample_button.click(lambda: gr.update(visible=True), inputs=None, outputs=sample_audio) \
            .then(sample_character, inputs=[cast_key_state, cast_selected, engine, cast_voice, cast_delivery, speed,
                                            exaggeration, cfg_weight, temperature, voice], outputs=sample_audio)
        selection_outputs = [*queue_outputs, selected_job, selected_info]
        enqueue_button.click(enqueue, inputs=[library_book, input_file, chapter_table, chapter_stats_state, *settings],
                             outputs=queue_outputs)
        queue_timer.tick(refresh_queue, inputs=None, outputs=queue_outputs)
        queue_table.select(select_job, inputs=queue_ids, outputs=[selected_job, selected_info])
        remove_button.click(remove_selected, inputs=selected_job, outputs=selection_outputs)
        retry_button.click(retry_selected, inputs=selected_job, outputs=selection_outputs)
        clear_button.click(clear_finished, inputs=None, outputs=queue_outputs)
        start_books_button.click(start_books, inputs=None, outputs=queue_outputs)
        pause_button.click(pause_queue, inputs=None, outputs=queue_outputs)
        resume_button.click(resume_queue, inputs=None, outputs=queue_outputs)
        stop_button.click(stop_current, inputs=None, outputs=queue_outputs)

        for trigger in (library_book, title_mode, newline_mode, remove_endnotes,
                        remove_reference_numbers, search_and_replace_file):
            trigger.change(chapter_overview, inputs=overview_inputs, outputs=overview_outputs)
        auto_tick_button.click(chapter_overview, inputs=overview_inputs, outputs=overview_outputs)
        tick_all_button.click(tick_all_chapters, inputs=chapter_table, outputs=chapter_table)
        for timing_control in timing:
            timing_control.change(retime_chapters, inputs=[chapter_table, chapter_stats_state, *timing],
                                  outputs=chapter_table)
        chapter_table.change(chapter_summary, inputs=[chapter_table, chapter_stats_state, *timing],
                             outputs=chapters_info)

        engine.change(engine_changed, inputs=engine,
                     outputs=[voice, dialogue_voice, cast_voice, adaptive_delivery, delivery_baseline_info])
        voice.input(_sync_if_chatterbox, inputs=[voice, engine], outputs=lab_voice)
        lab_voice.input(_sync_if_chatterbox, inputs=[lab_voice, engine], outputs=voice)
        lab_voice.change(voice_gender_of, inputs=lab_voice, outputs=lab_gender)
        lab_voice.change(voice_sound_text, inputs=lab_voice, outputs=lab_sound)
        # A voice's words are relative to its gender, so they change with it.
        save_gender_button.click(save_voice_gender, inputs=[lab_voice, lab_gender], outputs=gender_status) \
            .then(voice_sound_text, inputs=lab_voice, outputs=lab_sound)
        measure_button.click(measure_voices, inputs=None, outputs=measure_status) \
            .then(voice_sound_text, inputs=lab_voice, outputs=lab_sound)
        # The player stays hidden until the first sample, then shows before the audio arrives.
        sample_button.click(lambda: gr.update(visible=True), inputs=None, outputs=sample_audio) \
            .then(sample_voice, inputs=[engine, voice, speed], outputs=sample_audio)
        play_button.click(preview_voice, inputs=[lab_voice, phrase, exaggeration, cfg_weight, temperature, speed],
                          outputs=preview_audio)
        play_delivery_button.click(preview_delivery_range,
                                   inputs=[lab_voice, phrase, exaggeration, cfg_weight, temperature, speed],
                                   outputs=preview_audio)
        for delivery_control in (exaggeration, cfg_weight, temperature):
            delivery_control.change(delivery_baseline_text, inputs=[exaggeration, cfg_weight, temperature],
                                    outputs=delivery_baseline_info)
        save_button.click(save_settings, inputs=[exaggeration, cfg_weight, temperature], outputs=lab_status)
        reset_button.click(load_saved_settings, inputs=None, outputs=[exaggeration, cfg_weight, temperature])
        sample.change(voice_name_from_sample, inputs=sample, outputs=new_voice_name)
        add_button.click(add_voice, inputs=[sample, new_voice_name, remove_pauses, replace, engine],
                         outputs=[add_status, lab_voice, voice, delete_voice_dropdown])
        delete_voice_button.click(
            delete_voice,
            inputs=[delete_voice_dropdown, lab_voice, voice, engine],
            outputs=[delete_voice_status, lab_voice, voice, delete_voice_dropdown],
            # Nothing picked: skip the dialog and send "" so the handler can say so; a cancelled
            # dialog sends null, which the handler treats as "change nothing".
            js="(name, lab, mk, eng) => !name ? ['', lab, mk, eng]"
               " : confirm(`Delete the voice '${name.replace(/\\.(wav|mp3)$/i, '')}'? This can't be undone.`)"
               " ? [name, lab, mk, eng] : [null, lab, mk, eng]",
        )

        ui.load(refresh_voices, inputs=None, outputs=[voice, lab_voice, delete_voice_dropdown, dialogue_voice])
        ui.load(voice_gender_of, inputs=lab_voice, outputs=lab_gender)
        ui.load(voice_sound_text, inputs=lab_voice, outputs=lab_sound)
        ui.load(refresh_library, inputs=None, outputs=library_book)
        ui.load(load_saved_settings, inputs=None, outputs=[exaggeration, cfg_weight, temperature])
        ui.load(delivery_baseline_text, inputs=[exaggeration, cfg_weight, temperature], outputs=delivery_baseline_info)
        ui.load(refresh_queue, inputs=None, outputs=queue_outputs)
    return ui


def host_ui(config) -> None:
    if library_index.load_index():
        library_index.warm_up_in_background()
    else:
        library_index.refresh_index()  # first run: build the list before the page is served
    sweep_voice_previews()
    # Job processes are started from this worker thread while Gradio's request threads run; forking
    # (the platform default on Linux) can copy another thread's lock mid-hold and hang the child on
    # its first log line (F-19). Spawn starts each job in a fresh interpreter instead.
    queue = JobQueue(QUEUE_FILE, build_config, lambda: str(web_ui.webui_log_file.absolute()),
                     process_factory=multiprocessing.get_context("spawn").Process, uploads_dir=QUEUE_UPLOADS)
    queue.on_done = lambda job: queue_book_after_cast(queue, job)
    sweep_orphaned_uploads(queue)
    ui = build_ui(queue)
    queue.start_worker()
    ui.launch(server_name=config.host, server_port=config.port)
