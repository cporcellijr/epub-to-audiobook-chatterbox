import base64
import io
import logging
import math
import subprocess
import tempfile
import os
from typing import List, Tuple

from pydub import AudioSegment
from sentencex import segment

from mutagen.flac import FLAC, Picture as FlacPicture
from mutagen.oggopus import OggOpus
from mutagen.wave import WAVE
from mutagen.id3._frames import TIT2, TPE1, TALB, TRCK, APIC

from openai import OpenAI

from audiobook_generator.core.audio_tags import AudioTags
from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.utils.utils import split_text, split_long_sentence, set_audio_tags, merge_audio_segments
from audiobook_generator.tts_providers.base_tts_provider import BaseTTSProvider


logger = logging.getLogger(__name__)

PARAGRAPH_MARK = "@BRK#"
MIN_UNIT_CHARS = 40   # shorter sentences join the next one: tiny inputs make TTS models stumble
MAX_UNIT_CHARS = 400  # a trailing short sentence joins the previous unit only if it stays under this
# A unit over this goes through split_long_sentence (F-07): the server's ~1000-token cap silently
# truncates a single request around ~800 characters, and it is also the paragraph-mode (F-05)
# packing threshold, so one request never risks that cap either way.
MAX_REQUEST_CHARS = 450
_PYDUB_EXPORT = {"aac": ("adts", "aac"), "opus": ("opus", "libopus")}


def _is_speakable(unit: str) -> bool:
    """False for a unit with no letters or digits (F-18): scene breaks ("* * *", "...", "--")
    and other punctuation-only paragraphs should become silence, not a spoken request."""
    return any(char.isalnum() for char in unit)


def _split_oversized_unit(unit: str) -> List[str]:
    """Break a unit over MAX_REQUEST_CHARS into pieces (F-07), so none can reach the server's
    token cap. Pieces are meant to be sent as separate requests joined with NO pause: the cut
    is mid-sentence, not a real sentence or paragraph boundary."""
    if len(unit) <= MAX_REQUEST_CHARS:
        return [unit]
    return split_long_sentence(unit, MAX_REQUEST_CHARS)


_PCM_FORMATS = {1: "u8", 2: "s16le", 3: "s24le", 4: "s32le"}


def _atempo_filter_arg(speed: float) -> str:
    """Build an ffmpeg atempo filter chain for one speed factor.

    atempo accepts 0.5-100 per stage, so a speed under 0.5 is chained (the same approach
    chatterbox/utils.py's _ffmpeg_atempo uses server-side; reimplemented here, not shared,
    since the client and server are separate images).
    """
    stages = []
    remaining = speed
    while remaining < 0.5:
        stages.append(0.5)
        remaining /= 0.5
    stages.append(remaining)
    return ",".join(f"atempo={stage:.6f}" for stage in stages)


def _stretch_pcm(raw_pcm: bytes, frame_rate: int, channels: int, sample_width: int, speed: float) -> bytes:
    """Time-stretch raw PCM once for a whole chapter with ffmpeg atempo (F-27).

    Applying atempo per unit costs one ffmpeg process spawn (~57 ms measured) for every one of
    the ~330 units in a chapter; one pass over the finished chapter is a single spawn regardless
    of unit count, and stretches the client-inserted pauses along with the speech uniformly.
    """
    if speed == 1.0 or not raw_pcm:
        return raw_pcm
    pcm_format = _PCM_FORMATS.get(sample_width, "s16le")
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", pcm_format, "-ar", str(frame_rate), "-ac", str(channels), "-i", "pipe:0",
         "-filter:a", _atempo_filter_arg(speed),
         "-f", pcm_format, "-ar", str(frame_rate), "-ac", str(channels), "pipe:1"],
        input=raw_pcm, capture_output=True, timeout=120, check=True,
    )
    return proc.stdout


def paced_units(text: str, language: str) -> List[Tuple[int, str, bool]]:
    """(paragraph number, text, continues_previous) units of one or more whole sentences, in
    reading order. continues_previous is True only for a piece produced by splitting an
    oversized unit (F-07): it must follow the previous piece with NO pause.
    """
    units: List[Tuple[int, str, bool]] = []
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
        for unit in paragraph_units:
            if not _is_speakable(unit):
                continue
            pieces = _split_oversized_unit(unit)
            units.append((number, pieces[0], False))
            units.extend((number, piece, True) for piece in pieces[1:])
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


def _tag_wav(output_file: str, audio_tags: AudioTags) -> None:
    """Tag a loose WAV chapter (F-26). mutagen.wave.WAVE writes the same ID3 frames
    set_audio_tags uses for MP3, but into a proper RIFF chunk so the "RIFF....WAVE" header
    stays intact (a raw ID3.save() on a WAV, as set_audio_tags does, prepends bytes in front
    of that header instead)."""
    audio = WAVE(output_file)
    if audio.tags is None:
        audio.add_tags()
    audio.tags.add(TIT2(encoding=3, text=audio_tags.title))
    audio.tags.add(TPE1(encoding=3, text=audio_tags.author))
    audio.tags.add(TALB(encoding=3, text=audio_tags.book_title))
    audio.tags.add(TRCK(encoding=3, text=str(audio_tags.idx)))
    if audio_tags.cover:
        audio.tags.add(APIC(encoding=3, mime=audio_tags.cover.mime, type=3, desc="Cover",
                             data=audio_tags.cover.data))
    audio.save()


def _tag_flac(output_file: str, audio_tags: AudioTags) -> None:
    """Tag a loose FLAC chapter with native Vorbis comments and a picture block (F-26)."""
    audio = FLAC(output_file)
    audio["title"] = audio_tags.title
    audio["artist"] = audio_tags.author
    audio["album"] = audio_tags.book_title
    audio["tracknumber"] = str(audio_tags.idx)
    if audio_tags.cover:
        picture = FlacPicture()
        picture.data = audio_tags.cover.data
        picture.type = 3
        picture.mime = audio_tags.cover.mime
        picture.desc = "Cover"
        audio.clear_pictures()
        audio.add_picture(picture)
    audio.save()


def _tag_opus(output_file: str, audio_tags: AudioTags) -> None:
    """Tag a loose Opus chapter with native Vorbis comments (F-26). The cover goes in as a
    base64 METADATA_BLOCK_PICTURE comment, the same convention other Ogg-family taggers use."""
    audio = OggOpus(output_file)
    audio["title"] = audio_tags.title
    audio["artist"] = audio_tags.author
    audio["album"] = audio_tags.book_title
    audio["tracknumber"] = str(audio_tags.idx)
    if audio_tags.cover:
        picture = FlacPicture()
        picture.data = audio_tags.cover.data
        picture.type = 3
        picture.mime = audio_tags.cover.mime
        picture.desc = "Cover"
        audio["metadata_block_picture"] = [base64.b64encode(picture.write()).decode("ascii")]
    audio.save()


_LOOSE_FILE_TAGGERS = {"wav": _tag_wav, "flac": _tag_flac, "opus": _tag_opus}


def _tag_loose_file(output_file: str, output_format: str, audio_tags: AudioTags) -> None:
    """Tag a loose chapter file in its own native format (F-26).

    mp3 and aac (ADTS) both keep using set_audio_tags: mutagen's own docs say ADTS tagging
    is not supported and to use ID3 directly, which is what set_audio_tags already does, and
    it round-trips correctly (verified against ffprobe). wav/flac/opus get their native
    mutagen class instead, since a raw ID3 write is either inert (wav/flac: ffmpeg's demuxers
    skip a leading ID3v2 block without exposing it as format tags) or non-conformant (opus:
    it would sit in front of the required "OggS" capture pattern). A tagging failure is
    logged once and the (already-exported) audio file is left as is, never raised further.
    """
    if output_format in ("mp3", "aac"):
        set_audio_tags(output_file, audio_tags)
        return
    tagger = _LOOSE_FILE_TAGGERS.get(output_format)
    if tagger is None:
        logger.warning(f"OpenAI: no tagger for output format '{output_format}'; chapter file left untagged")
        return
    try:
        tagger(output_file, audio_tags)
    except Exception as e:
        logger.warning(f"OpenAI: could not tag {output_format} chapter file {output_file}: {e}", exc_info=True)


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

        _tag_loose_file(output_file, self.config.output_format, audio_tags)

    def _paced_text_to_speech(self, text: str, output_file: str, audio_tags: AudioTags) -> None:
        """Speak sentence-sized units and insert real pauses between sentences and paragraphs.

        Long requests only get the server's own short gaps between sentences, and paragraph
        breaks are lost entirely, so narration runs together.

        Every unit is requested at speed 1.0 and pauses are inserted at their full configured
        length (F-27): running the server's per-unit atempo ~330 times a chapter costs about
        19 s of ffmpeg process spawns alone. Instead, if speed != 1.0, one atempo pass stretches
        the whole finished chapter (speech and pauses together) at the end, which shrinks the
        pauses by the same proportion the server's per-unit approach did.
        """
        speed = float(self.config.speed or 1.0)
        sentence_gap_ms = int(self.config.sentence_pause_ms or 0)
        paragraph_gap_ms = int(self.config.paragraph_pause_ms or 0)
        units = paced_units(text, self.config.language or "en")
        if not units:
            raise ValueError("No speakable text in this chapter")

        pieces: List[bytes] = []
        audio_format = None
        previous_paragraph = None
        for number, (paragraph, unit, continues_previous) in enumerate(units, 1):
            chunk_id = f"chapter-{audio_tags.idx}_{audio_tags.title}_chunk_{number}_of_{len(units)}"
            logger.info(f"Processing {chunk_id}, length={len(unit)}")
            logger.debug(f"Processing {chunk_id}, length={len(unit)}, text=[{unit}]")
            response = self.client.audio.speech.create(
                model=self.config.model_name,
                voice=self.config.voice_name,
                speed=1.0,
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
                if continues_previous:
                    gap_ms = 0
                else:
                    gap_ms = paragraph_gap_ms if paragraph != previous_paragraph else sentence_gap_ms
                gap_frames = int(audio_format[0] * gap_ms / 1000)
                pieces.append(b"\0" * gap_frames * audio_format[1] * audio_format[2])
            pieces.append(audio.raw_data)
            previous_paragraph = paragraph

        self._combine_and_export(pieces, audio_format, speed, output_file, audio_tags)

    def _combine_and_export(self, pieces: List[bytes], audio_format: Tuple[int, int, int], speed: float,
                             output_file: str, audio_tags: AudioTags) -> None:
        """Join raw PCM pieces, apply one client-side atempo pass if speed != 1.0 (F-27), then
        export and tag the finished chapter."""
        raw_pcm = _stretch_pcm(b"".join(pieces), audio_format[0], audio_format[1], audio_format[2], speed)
        combined = AudioSegment(data=raw_pcm, frame_rate=audio_format[0],
                                channels=audio_format[1], sample_width=audio_format[2])
        export_format, codec = _PYDUB_EXPORT.get(self.config.output_format, (self.config.output_format, None))
        combined.export(output_file, format=export_format, codec=codec,
                        bitrate="64k" if self.config.output_format in ("mp3", "aac", "opus") else None)
        _tag_loose_file(output_file, self.config.output_format, audio_tags)

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
