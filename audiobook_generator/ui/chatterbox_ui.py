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
import json
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from typing import Optional

import gradio as gr
import yaml
from gradio_log import Log

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.ui import library_index, web_ui
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

def read_saved_settings() -> dict:
    """Chatterbox's saved delivery settings, read from its config file (never blocks on a busy server)."""
    path = os.environ.get("CHATTERBOX_CONFIG")
    settings = dict(FALLBACK_SETTINGS)
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            defaults = (yaml.safe_load(f) or {}).get("generation_defaults", {})
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

def preview_voice(voice: str, phrase: str, exaggeration: float, cfg_weight: float,
                  temperature: float, speed: float) -> str:
    """Speak the phrase with the slider values (not saved) and return the audio file path."""
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

def build_config(input_file, output_dir: str, voice: str, speed: float, chapter_start, chapter_end,
                 skip_existing: bool, output_text: bool, title_mode: str, newline_mode: str,
                 remove_endnotes: bool, remove_reference_numbers: bool, search_and_replace_file,
                 log_level: str, preview: bool) -> GeneralConfig:
    """GeneralConfig for the OpenAI provider pointed at Chatterbox."""
    config = GeneralConfig(None)
    config.input_file = input_file.name if hasattr(input_file, "name") else input_file
    config.output_folder = output_dir
    config.preview = preview
    config.output_text = output_text
    config.skip_existing = skip_existing
    config.log = log_level
    config.worker_count = 1  # Chatterbox handles one request at a time
    config.no_prompt = True
    config.title_mode = title_mode
    config.newline_mode = newline_mode
    config.chapter_start = int(chapter_start or 1)
    config.chapter_end = int(chapter_end if chapter_end not in (None, 0) else -1)
    config.remove_endnotes = remove_endnotes
    config.remove_reference_numbers = remove_reference_numbers
    config.search_and_replace_file = (search_and_replace_file.name if hasattr(search_and_replace_file, "name")
                                      else search_and_replace_file)
    config.tts = "openai"
    config.output_format = "mp3"
    config.voice_name = voice
    config.model_name = "chatterbox"
    config.instructions = None
    config.speed = float(speed)
    return config


def _start(preview: bool, library_book, input_file, *settings) -> None:
    book = library_book or input_file
    if not book:
        raise gr.Error("Pick a book from the library or upload an EPUB first.")
    if web_ui.running_process is not None and web_ui.running_process.is_alive():
        raise gr.Error("A book is already being generated. Stop it first or wait for it to finish.")
    web_ui.launch_audiobook_generator(build_config(book, *settings, preview=preview))
    gr.Info("Previewing chapters (no audio)..." if preview else "Generating audiobook...")


def start_generation(library_book, input_file, *settings) -> None:
    _start(False, library_book, input_file, *settings)


def preview_chapters(library_book, input_file, *settings) -> None:
    _start(True, library_book, input_file, *settings)


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


def stop_generation() -> None:
    web_ui.terminate_audiobook_generator()
    gr.Info("Stopped.")


def refresh_voices() -> tuple:
    choices = openai_voice_choices()
    return (gr.update(choices=choices, value=default_openai_voice(choices)),
            gr.update(choices=choices, value=default_openai_voice(choices)))


# ---- Layout ----

def build_ui() -> gr.Blocks:
    choices = openai_voice_choices()
    default_voice = default_openai_voice(choices)
    with gr.Blocks(analytics_enabled=False, title="Audiobook Maker") as ui:
        with gr.Tab("Make audiobook"):
            with gr.Row(equal_height=True):
                with gr.Column():
                    library_book = gr.Dropdown(library_index.book_choices(library_index.load_index()), value=None,
                                               label="Book", filterable=True,
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
                chapter_start = gr.Number(1, precision=0, minimum=1, label="Start at chapter")
                chapter_end = gr.Number(-1, precision=0, minimum=-1, label="End at chapter",
                                        info="-1 = through the last chapter")
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
            with gr.Row():
                preview_button = gr.Button("Preview chapters")
                start_button = gr.Button("Start", variant="primary")
                stop_button = gr.Button("Stop", variant="stop")
            web_ui.webui_log_file = generate_unique_log_path("EtA_WebUI")
            web_ui.webui_log_file.touch()
            Log(str(web_ui.webui_log_file.absolute()), dark=True, xterm_font_size=12)

        with gr.Tab("Voice lab"):
            gr.Markdown("Try voices and tune delivery. **Save** applies the sliders to every book "
                        "(including one in progress). While a book is generating, a preview waits for "
                        "the current chunk to finish (up to about a minute).")
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
                                  label="Voice sample: 10-15 s of one person speaking clearly")
                with gr.Column():
                    new_voice_name = gr.Textbox(label="Voice name")
                    remove_pauses = gr.Checkbox(True, label="Remove long pauses (recommended)",
                                                info="Pauses in a sample make Chatterbox pause mid-sentence.")
                    replace = gr.Checkbox(False, label="Replace a voice with the same name")
                    add_button = gr.Button("Add voice")
            add_status = gr.Markdown()

        settings = [output_dir, voice, speed, chapter_start, chapter_end, skip_existing, output_text,
                    title_mode, newline_mode, remove_endnotes, remove_reference_numbers,
                    search_and_replace_file, log_level]
        library_book.change(library_output_dir, inputs=library_book, outputs=output_dir)
        input_file.change(uploaded_book_selected, inputs=input_file, outputs=[output_dir, library_book])
        start_button.click(start_generation, inputs=[library_book, input_file, *settings], outputs=None)
        preview_button.click(preview_chapters, inputs=[library_book, input_file, *settings], outputs=None)
        stop_button.click(stop_generation, inputs=None, outputs=None)

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
    return ui


def host_ui(config) -> None:
    library_index.warm_up_in_background()
    build_ui().launch(server_name=config.host, server_port=config.port)
