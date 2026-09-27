"""Personal web UI: EPUB -> audiobook through a local Chatterbox server, plus a voice lab.

Only the OpenAI-compatible provider is exposed (pointed at Chatterbox via OPENAI_BASE_URL).
Upstream's multi-provider UI stays in web_ui.py, whose process helpers are reused here.

Environment:
    OPENAI_BASE_URL      Chatterbox OpenAI endpoint, e.g. http://chatterbox:8004/v1
    CHATTERBOX_URL       Chatterbox root (default: OPENAI_BASE_URL without /v1)
    CHATTERBOX_CONFIG    Chatterbox config.yaml (read-only mount) for the saved delivery settings
    TTS_VOICES_DIR       Chatterbox voices folder (writable mount, for adding voices)
    OPENAI_DEFAULT_VOICE Voice selected by default
    EBOOK_LIBRARY_DIR    Ebook library (read-only mount) for the searchable book picker
"""
import glob
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from typing import List, Optional

import gradio as gr
import yaml
from gradio_log import Log

from audiobook_generator.book_parsers.base_book_parser import get_book_parser
from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.core.chapter_selection import preselect_chapters
from audiobook_generator.tts_providers.openai_tts_provider import PARAGRAPH_MARK
from audiobook_generator.ui import library_index, web_ui
from audiobook_generator.ui.job_queue import DONE, FAILED, QUEUED, RUNNING, STOPPED, JobQueue
from audiobook_generator.ui.web_ui import (
    OUTPUT_ROOT,
    default_openai_voice,
    openai_voice_choices,
    safe_folder_name,
    suggest_output_dir,
    timestamped_output_dir,
)
from audiobook_generator.utils.log_handler import generate_unique_log_path

PREVIEW_PHRASE = (
    "The rain had stopped by the time they reached the old bridge. "
    "\"We should have turned back an hour ago,\" she said, pulling her coat tighter."
)
FALLBACK_SETTINGS = {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8}
SHORT_SAMPLE_SECONDS = 6.0
PAUSE_FILTER = ("silenceremove=start_periods=1:start_threshold=-40dB:stop_periods=-1:"
                "stop_duration=0.3:stop_threshold=-40dB:stop_silence=0.15")


def chatterbox_url() -> str:
    """Chatterbox root URL (no /v1)."""
    explicit = os.environ.get("CHATTERBOX_URL", "").rstrip("/")
    if explicit:
        return explicit
    base = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


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


def add_voice(sample: Optional[str], name: str, remove_pauses: bool, replace: bool) -> tuple:
    """Save an uploaded sample into the Chatterbox voices folder as <name>.wav."""
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
    choices = openai_voice_choices()
    return message, gr.update(choices=choices, value=file_name), gr.update(choices=choices)


# ---- Audiobook generation ----

def build_config(input_file, output_dir: str, voice: str, speed: float, chapter_selection: list,
                 sentence_pause: float, paragraph_pause: float, output_m4b: bool, skip_existing: bool,
                 output_text: bool, title_mode: str, newline_mode: str, remove_endnotes: bool,
                 remove_reference_numbers: bool, search_and_replace_file, log_level: str) -> GeneralConfig:
    """GeneralConfig for the OpenAI provider pointed at Chatterbox (pauses in seconds)."""
    config = GeneralConfig(None)
    config.input_file = input_file.name if hasattr(input_file, "name") else input_file
    config.output_folder = output_dir
    config.preview = False
    config.output_text = output_text
    config.skip_existing = skip_existing
    config.log = log_level
    config.worker_count = 1  # Chatterbox handles one request at a time
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
    config.model_name = "chatterbox"
    config.instructions = None
    config.speed = float(speed)
    config.sentence_pause_ms = int(round(float(sentence_pause) * 1000))
    config.paragraph_pause_ms = int(round(float(paragraph_pause) * 1000))
    config.output_m4b = bool(output_m4b)
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
        for key in ("input_file", "search_and_replace_file"):
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
                   active_jobs: Optional[List[dict]] = None) -> dict:
    """Validate the form and turn it into build_config keyword arguments for a queued book."""
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
    return {
        "input_file": queued_input_file,
        "output_dir": output_dir, "voice": voice, "speed": float(speed),
        "chapter_selection": selection, "sentence_pause": float(sentence_pause),
        "paragraph_pause": float(paragraph_pause), "output_m4b": bool(output_m4b),
        "skip_existing": bool(skip_existing), "output_text": bool(output_text), "title_mode": title_mode,
        "newline_mode": newline_mode, "remove_endnotes": bool(remove_endnotes),
        "remove_reference_numbers": bool(remove_reference_numbers),
        "search_and_replace_file": queued_replace_file,
        "log_level": log_level,
    }


def generation_estimate(table, stats: list) -> float:
    """Seconds Chatterbox needs for the ticked chapters."""
    stats = stats or []
    chosen = [stats[n - 1] for n in selected_chapter_numbers(table) if 0 < n <= len(stats)]
    return sum(s[0] for s in chosen) / CHARS_PER_AUDIO_SECOND / GENERATION_SPEED * PACED_GENERATION_OVERHEAD


def _status_label(job: dict) -> str:
    note = f" · {job['note']}" if job.get("note") else ""
    if job["status"] == RUNNING:
        return f"▶ generating · {JobQueue.chapters_done(job)} of {job['chapters']} chapters done"
    if job["status"] == QUEUED:
        return f"waiting{note}"
    if job["status"] == DONE:
        return f"✓ done {job['finished']}"
    if job["status"] == FAILED:
        return f"✗ failed{note}"
    return f"■ stopped{note}"


QUEUE_COLUMNS = ["#", "Book", "Voice", "Chapters", "Status", "Generating time"]


def queue_view(queue: JobQueue) -> tuple:
    """(table rows, job ids in row order, one-line queue status)."""
    jobs = queue.jobs()
    rows = [[n, job["title"], os.path.splitext(job["voice"])[0], job["chapters"], _status_label(job),
             _duration(job["estimate_seconds"])] for n, job in enumerate(jobs, start=1)]
    ids = [job["id"] for job in jobs]
    left = 0.0
    for job in jobs:
        if job["status"] == QUEUED:
            left += job["estimate_seconds"]
        elif job["status"] == RUNNING:
            left += job["estimate_seconds"] * (1 - JobQueue.chapters_done(job) / max(1, job["chapters"]))
    to_go = sum(1 for job in jobs if job["status"] in (QUEUED, RUNNING))
    if not jobs:
        status = "The queue is empty. Pick a book, tick chapters, choose a voice, then **Add to queue**."
    elif queue.paused:
        status = f"⏸ **Queue paused** · {to_go} book(s) waiting. Press **Resume queue** to continue."
    elif to_go:
        status = f"▶ **Working** · {to_go} book(s) to go · {_about(left)} of generating left."
    else:
        status = "✓ **All done.** Finished books are in the audiobook library."
    return rows, ids, status


# ---- Chapter list ----

# Measured on a finished book (Elena, speed 1.0): 20.2 characters of text per second of speech,
# generated at ~1.8x real time; sentence-sized requests add ~14% generation time (A/B, 2026-09-27).
# Other voices read at somewhat different paces.
CHARS_PER_AUDIO_SECOND = 20.2
GENERATION_SPEED = 1.8
PACED_GENERATION_OVERHEAD = 1.14
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
    """[characters, sentences, paragraphs] of a chapter parsed with paragraph marks."""
    paragraphs = [p for p in (" ".join(part.split()) for part in text.split(PARAGRAPH_MARK)) if p]
    plain = " ".join(paragraphs)
    return [len(plain), max(1, len(_SENTENCE_END.findall(plain))), max(1, len(paragraphs))]


def listening_seconds(stats: list, speed, sentence_pause, paragraph_pause) -> float:
    """Speech plus the inserted pauses (both shrink with speed)."""
    characters, sentences, paragraphs = stats
    pauses = (max(0, sentences - paragraphs) * float(sentence_pause or 0)
              + max(0, paragraphs - 1) * float(paragraph_pause or 0))
    return (characters / CHARS_PER_AUDIO_SECOND + pauses) / float(speed or 1.0)


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


def chapter_summary(table, stats: list, speed, sentence_pause, paragraph_pause) -> str:
    """One-line summary of the ticked chapters."""
    rows = _table_rows(table)
    if not rows:
        return ""
    picked = selected_chapter_numbers(rows)
    if not picked:
        return "⚠️ No chapters ticked."
    stats = stats or []
    chosen = [stats[n - 1] for n in picked if 0 < n <= len(stats)]
    audio_seconds = sum(listening_seconds(s, speed, sentence_pause, paragraph_pause) for s in chosen)
    speech_seconds = sum(s[0] for s in chosen) / CHARS_PER_AUDIO_SECOND
    skipped = len(rows) - len(picked)
    unticked = f" ({skipped} unticked)" if skipped else ""
    return (f"**{len(picked)} of {len(rows)} chapters** ticked{unticked} · "
            f"**{_about(audio_seconds)}** of audio at {float(speed or 1.0):g}× · "
            f"{_about(speech_seconds / GENERATION_SPEED * PACED_GENERATION_OVERHEAD)} to generate. "
            f"They'll be numbered 1–{len(picked)} in the finished book.")


def chapter_overview(library_book, input_file, speed, sentence_pause, paragraph_pause, title_mode: str,
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
             _duration(listening_seconds(stat, speed, sentence_pause, paragraph_pause))]
            for number, ((title, text), ticked, stat) in enumerate(zip(plain, include, stats), start=1)]
    return (gr.update(value=rows, visible=True), stats,
            chapter_summary(rows, stats, speed, sentence_pause, paragraph_pause))


def retime_chapters(table, stats: list, speed, sentence_pause, paragraph_pause) -> dict:
    """Speed or pauses changed: update the Listening column, keeping the ticks."""
    rows = _table_rows(table)
    stats = stats or []
    for row in rows:
        number = int(row[0])
        if 0 < number <= len(stats):
            row[4] = _duration(listening_seconds(stats[number - 1], speed, sentence_pause, paragraph_pause))
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
    choices = openai_voice_choices()
    return (gr.update(choices=choices, value=default_openai_voice(choices)),
            gr.update(choices=choices, value=default_openai_voice(choices)))


# ---- Layout ----

def build_ui(queue: Optional[JobQueue] = None) -> gr.Blocks:
    web_ui.webui_log_file = generate_unique_log_path("EtA_WebUI")
    web_ui.webui_log_file.touch()
    if queue is None:  # standalone build (tests): a throwaway queue that never starts on its own
        queue = JobQueue(os.path.join(tempfile.mkdtemp(), QUEUE_FILE), build_config,
                         lambda: str(web_ui.webui_log_file.absolute()))
    choices = openai_voice_choices()
    default_voice = default_openai_voice(choices)

    def refresh_queue() -> tuple:
        rows, ids, status = queue_view(queue)
        return gr.update(value=rows), ids, status

    def enqueue(library_book, input_file, chapter_table, stats, *settings) -> tuple:
        job_settings = queue_settings(library_book, input_file, chapter_table, *settings, active_jobs=queue.jobs())
        title = os.path.basename(job_settings["output_dir"].rstrip("/\\")) or "Book"
        position = queue.add(title, job_settings, len(job_settings["chapter_selection"]),
                             generation_estimate(chapter_table, stats), job_settings["voice"])
        queue.tick()
        gr.Info(f"Added '{title}' to the queue" + ("." if position <= 1 else f" (#{position} in line)."))
        return refresh_queue()

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
                    voice = gr.Dropdown(choices, value=default_voice, label="Voice", allow_custom_value=True)
                    speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed",
                                      info="1.0 recommended (other speeds are stretched after generation).")
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
                search_and_replace_file = gr.File(label="Search & replace file (optional, e.g. fix pronunciations)",
                                                  file_types=[".txt"], file_count="single")
            enqueue_button = gr.Button("➕ Add to queue", variant="primary")
            with gr.Accordion("Queue", open=True):
                queue_status = gr.Markdown()
                queue_table = gr.Dataframe(headers=QUEUE_COLUMNS, interactive=False, wrap=True, label="Books",
                                           column_widths=["5%", "33%", "14%", "9%", "27%", "12%"])
                queue_ids = gr.State([])
                selected_job = gr.State(None)
                selected_info = gr.Markdown()
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
            saved = read_saved_settings()
            with gr.Row(equal_height=True):
                exaggeration = gr.Slider(0.25, 2.0, value=saved["exaggeration"], step=0.05, label="Exaggeration",
                                         info="Emotion and emphasis")
                cfg_weight = gr.Slider(0.1, 1.0, value=saved["cfg_weight"], step=0.05, label="CFG weight",
                                       info="Lower = slower, more deliberate")
                temperature = gr.Slider(0.05, 1.5, value=saved["temperature"], step=0.05, label="Temperature",
                                        info="Higher = more varied")
            with gr.Row():
                play_button = gr.Button("▶ Play", variant="primary")
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

        settings = [output_dir, voice, speed, sentence_pause, paragraph_pause, output_m4b, skip_existing,
                    output_text, title_mode, newline_mode, remove_endnotes, remove_reference_numbers,
                    search_and_replace_file, log_level]
        # The chapter list follows the book and the options that change how it's split (parsing is <0.5 s);
        # this re-runs the auto-selection. Ticks, speed, pauses and "Tick all" only touch the table.
        timing = [speed, sentence_pause, paragraph_pause]
        overview_inputs = [library_book, input_file, *timing, title_mode, newline_mode, remove_endnotes,
                           remove_reference_numbers, search_and_replace_file]
        overview_outputs = [chapter_table, chapter_stats_state, chapters_info]

        library_book.change(library_output_dir, inputs=library_book, outputs=output_dir)
        # An upload takes over from the library pick (its handler also clears library_book); chain
        # with .then() rather than wiring chapter_overview to input_file too, so the table is built
        # once, from the upload, instead of once from the stale library pick and again from the
        # upload (F-42b).
        input_file.change(uploaded_book_selected, inputs=input_file, outputs=[output_dir, library_book]) \
            .then(chapter_overview, inputs=overview_inputs, outputs=overview_outputs)
        queue_outputs = [queue_table, queue_ids, queue_status]
        selection_outputs = [*queue_outputs, selected_job, selected_info]
        enqueue_button.click(enqueue, inputs=[library_book, input_file, chapter_table, chapter_stats_state, *settings],
                             outputs=queue_outputs)
        queue_timer.tick(refresh_queue, inputs=None, outputs=queue_outputs)
        queue_table.select(select_job, inputs=queue_ids, outputs=[selected_job, selected_info])
        remove_button.click(remove_selected, inputs=selected_job, outputs=selection_outputs)
        retry_button.click(retry_selected, inputs=selected_job, outputs=selection_outputs)
        clear_button.click(clear_finished, inputs=None, outputs=queue_outputs)
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

        voice.input(lambda v: v, inputs=voice, outputs=lab_voice)
        lab_voice.input(lambda v: v, inputs=lab_voice, outputs=voice)
        play_button.click(preview_voice, inputs=[lab_voice, phrase, exaggeration, cfg_weight, temperature, speed],
                          outputs=preview_audio)
        save_button.click(save_settings, inputs=[exaggeration, cfg_weight, temperature], outputs=lab_status)
        reset_button.click(load_saved_settings, inputs=None, outputs=[exaggeration, cfg_weight, temperature])
        sample.change(voice_name_from_sample, inputs=sample, outputs=new_voice_name)
        add_button.click(add_voice, inputs=[sample, new_voice_name, remove_pauses, replace],
                         outputs=[add_status, lab_voice, voice])

        ui.load(refresh_voices, inputs=None, outputs=[voice, lab_voice])
        ui.load(refresh_library, inputs=None, outputs=library_book)
        ui.load(load_saved_settings, inputs=None, outputs=[exaggeration, cfg_weight, temperature])
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
    sweep_orphaned_uploads(queue)
    ui = build_ui(queue)
    queue.start_worker()
    ui.launch(server_name=config.host, server_port=config.port)
