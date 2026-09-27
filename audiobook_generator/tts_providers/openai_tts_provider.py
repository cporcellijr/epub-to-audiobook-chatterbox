import io
import logging
import math
import tempfile
import os
from typing import List, Tuple

from pydub import AudioSegment
from sentencex import segment

from openai import OpenAI

from audiobook_generator.core.audio_tags import AudioTags
from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.utils.utils import split_text, set_audio_tags, merge_audio_segments
from audiobook_generator.tts_providers.base_tts_provider import BaseTTSProvider


logger = logging.getLogger(__name__)

PARAGRAPH_MARK = "@BRK#"
MIN_UNIT_CHARS = 40   # shorter sentences join the next one: tiny inputs make TTS models stumble
MAX_UNIT_CHARS = 400  # a trailing short sentence joins the previous unit only if it stays under this
_PYDUB_EXPORT = {"aac": ("adts", "aac"), "opus": ("opus", "libopus")}


def paced_units(text: str, language: str) -> List[Tuple[int, str]]:
    """(paragraph number, text) units of one or more whole sentences, in reading order."""
    units: List[Tuple[int, str]] = []
    paragraphs = [" ".join(p.split()) for p in text.split(PARAGRAPH_MARK)]
    for number, paragraph in enumerate(p for p in paragraphs if p):
        pending, paragraph_units = "", []
        for sentence in segment(language, paragraph):
            sentence = str(sentence).strip()
            if not sentence:
                continue
            pending = f"{pending} {sentence}".strip()
            if len(pending) >= MIN_UNIT_CHARS:
                paragraph_units.append(pending)
                pending = ""
        if pending:
            if paragraph_units and len(paragraph_units[-1]) + len(pending) < MAX_UNIT_CHARS:
                paragraph_units[-1] = f"{paragraph_units[-1]} {pending}"
            else:
                paragraph_units.append(pending)
        units.extend((number, unit) for unit in paragraph_units)
    return units


def get_openai_supported_output_formats():
    return ["mp3", "aac", "flac", "opus", "wav"]

def get_openai_supported_voices():
    return ["alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"]

def get_openai_supported_models():
    return ["gpt-4o-mini-tts", "tts-1", "tts-1-hd"]

def get_openai_instructions_example():
    return """Voice Affect: Calm, composed, and reassuring. Competent and in control, instilling trust.
Tone: Sincere, empathetic, with genuine concern for the customer and understanding of the situation.
Pacing: Slower during the apology to allow for clarity and processing. Faster when offering solutions to signal action and resolution.
Emotions: Calm reassurance, empathy, and gratitude.
Pronunciation: Clear, precise: Ensures clarity, especially with key details. Focus on key words like 'refund' and 'patience.' 
Pauses: Before and after the apology to give space for processing the apology."""

def get_price(model):
    # https://platform.openai.com/docs/pricing#transcription-and-speech-generation
    if model == "tts-1": # $15 per 1 mil chars
        return 0.015
    elif model == "tts-1-hd": # $30 per 1 mil chars
        return 0.03
    elif model == "gpt-4o-mini-tts": # $12 per 1 mil tokens (not chars, as 1 token is ~4 chars)
        return 0.003 # TODO: this could be very wrong for Chinese. Not sure how openai calculates the audio token count.
    else:
        logger.warning(f"OpenAI: Unsupported model name: {model}, unable to retrieve the price")
        return 0.0


class OpenAITTSProvider(BaseTTSProvider):
    def __init__(self, config: GeneralConfig):
        config.model_name = config.model_name or "gpt-4o-mini-tts" # default to this model as it's the cheapest
        config.voice_name = config.voice_name or "alloy"
        config.speed = config.speed or 1.0
        config.instructions = config.instructions or None
        config.output_format = config.output_format or "mp3"

        self.price = get_price(config.model_name)
        super().__init__(config)

        self.client = OpenAI(max_retries=4)  # User should set OPENAI_API_KEY environment variable

    def __str__(self) -> str:
        return super().__str__()

    def pacing_enabled(self) -> bool:
        return any(isinstance(value, (int, float)) and not isinstance(value, bool)
                   for value in (self.config.sentence_pause_ms, self.config.paragraph_pause_ms))

    def text_to_speech(self, text: str, output_file: str, audio_tags: AudioTags):
        if self.pacing_enabled():
            self._paced_text_to_speech(text, output_file, audio_tags)
            return
        text = " ".join(text.replace(PARAGRAPH_MARK, " ").split())
        # Reason: The max num of input tokens is 2000 for gpt-4o-mini-tts https://platform.openai.com/docs/models/gpt-4o-mini-tts. One token is ~4 chars in English but ~1 word/char in Chinese.
        # So we reduce the max num of chars from 4000 to 1800 to avoid the input tokens limit.
        # TODO: detect the language and set the max num of chars accordingly.
        max_chars = 1800

        text_chunks = split_text(text, max_chars, self.config.language)

        audio_segments = []
        chunk_ids = []

        for i, chunk in enumerate(text_chunks, 1):
            chunk_id = f"chapter-{audio_tags.idx}_{audio_tags.title}_chunk_{i}_of_{len(text_chunks)}"
            logger.info(
                f"Processing {chunk_id}, length={len(chunk)}"
            )
            logger.debug(
                f"Processing {chunk_id}, length={len(chunk)}, text=[{chunk}]"
            )

            # NO retry for OpenAI TTS because SDK has built-in retry logic
            response = self.client.audio.speech.create(
                model=self.config.model_name,
                voice=self.config.voice_name,
                speed=self.config.speed,
                instructions=self.config.instructions,
                input=chunk,
                response_format=self.config.output_format,
            )

            # Log response details
            logger.debug(f"Remote server response: status_code={response.response.status_code}, "
                         f"size={len(response.content)} bytes, "
                         f"content={response.content[:128]}...")

            audio_segments.append(io.BytesIO(response.content))
            chunk_ids.append(chunk_id)

        # Use utility function to merge audio segments
        merge_audio_segments(audio_segments, output_file, self.config.output_format, chunk_ids, self.config.use_pydub_merge)

        set_audio_tags(output_file, audio_tags)

    def _paced_text_to_speech(self, text: str, output_file: str, audio_tags: AudioTags) -> None:
        """Speak sentence-sized units and insert real pauses between sentences and paragraphs.

        Long requests only get the server's own short gaps between sentences, and paragraph
        breaks are lost entirely, so narration runs together. Pauses shrink with speed.
        """
        speed = float(self.config.speed or 1.0)
        sentence_gap_ms = int((self.config.sentence_pause_ms or 0) / speed)
        paragraph_gap_ms = int((self.config.paragraph_pause_ms or 0) / speed)
        units = paced_units(text, self.config.language or "en")
        if not units:
            raise ValueError("No speakable text in this chapter")

        pieces: List[bytes] = []
        audio_format = None
        previous_paragraph = None
        for number, (paragraph, unit) in enumerate(units, 1):
            chunk_id = f"chapter-{audio_tags.idx}_{audio_tags.title}_chunk_{number}_of_{len(units)}"
            logger.info(f"Processing {chunk_id}, length={len(unit)}")
            logger.debug(f"Processing {chunk_id}, length={len(unit)}, text=[{unit}]")
            response = self.client.audio.speech.create(
                model=self.config.model_name,
                voice=self.config.voice_name,
                speed=self.config.speed,
                instructions=self.config.instructions,
                input=unit,
                response_format="wav",
            )
            audio = AudioSegment.from_file(io.BytesIO(response.content), format="wav")
            if audio_format is None:
                audio_format = (audio.frame_rate, audio.channels, audio.sample_width)
            else:
                audio = (audio.set_frame_rate(audio_format[0]).set_channels(audio_format[1])
                         .set_sample_width(audio_format[2]))
                gap_ms = paragraph_gap_ms if paragraph != previous_paragraph else sentence_gap_ms
                gap_frames = int(audio_format[0] * gap_ms / 1000)
                pieces.append(b"\0" * gap_frames * audio_format[1] * audio_format[2])
            pieces.append(audio.raw_data)
            previous_paragraph = paragraph

        combined = AudioSegment(data=b"".join(pieces), frame_rate=audio_format[0],
                                channels=audio_format[1], sample_width=audio_format[2])
        export_format, codec = _PYDUB_EXPORT.get(self.config.output_format, (self.config.output_format, None))
        combined.export(output_file, format=export_format, codec=codec,
                        bitrate="64k" if self.config.output_format in ("mp3", "aac", "opus") else None)
        if self.config.output_format == "mp3":
            set_audio_tags(output_file, audio_tags)

    def get_break_string(self):
        # Non-whitespace so paragraph breaks survive the parser's whitespace collapsing.
        return f" {PARAGRAPH_MARK}"

    def get_output_file_extension(self):
        return self.config.output_format

    def validate_config(self):
        if self.config.output_format not in get_openai_supported_output_formats():
            raise ValueError(f"OpenAI: Unsupported output format: {self.config.output_format}")
        if self.config.speed < 0.25 or self.config.speed > 4.0:
            raise ValueError(f"OpenAI: Unsupported speed: {self.config.speed}")
        if self.config.instructions and len(self.config.instructions) > 0 and self.config.model_name != "gpt-4o-mini-tts":
            raise ValueError(f"OpenAI: Instructions are only supported for 'gpt-4o-mini-tts' model")

    def estimate_cost(self, total_chars):
        return math.ceil(total_chars / 1000) * self.price
