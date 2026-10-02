import base64
import hashlib
import io
import json
import logging
import re
import math
import os
import secrets
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterator, List, Optional, Tuple

from pydub import AudioSegment
from sentencex import segment

from mutagen.flac import FLAC, Picture as FlacPicture
from mutagen.oggopus import OggOpus
from mutagen.wave import WAVE
from mutagen.id3._frames import TIT2, TPE1, TALB, TRCK, APIC

from openai import APIConnectionError, APIStatusError, OpenAI

from audiobook_generator.core.audio_tags import AudioTags
from audiobook_generator.core import breeze_client
from audiobook_generator.core import cast as cast_store
from audiobook_generator.core import delivery
from audiobook_generator.core import speech_check
from audiobook_generator.core import voice_transcripts
from audiobook_generator.core.m4b import LOSSY_BITRATE
from audiobook_generator.core.dialogue import DIALOGUE, NARRATION, PARAGRAPH_MARK, Segment, chapter_segments
from audiobook_generator.core.speech_tags import has_speech_tag
from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.utils.utils import split_text, split_long_sentence, set_audio_tags, merge_audio_segments
from audiobook_generator.tts_providers.base_tts_provider import BaseTTSProvider


logger = logging.getLogger(__name__)

__all__ = ["PARAGRAPH_MARK", "OpenAITTSProvider", "paced_units", "paragraph_mode_units", "voiced_units",
           "voiced_paragraph_units", "adaptive_units", "adaptive_paragraph_units"]

# Voice modes (config.voice_mode): one voice for everything (today's behaviour), a narrator plus one
# voice for every quoted line, or a cast where each attributed speaker has their own voice.
VOICE_MODE_SINGLE, VOICE_MODE_DIALOGUE, VOICE_MODE_CAST = "single", "dialogue", "cast"
VOICE_MODES = (VOICE_MODE_SINGLE, VOICE_MODE_DIALOGUE, VOICE_MODE_CAST)
MIN_UNIT_CHARS = 40   # shorter sentences join the next one: tiny inputs make TTS models stumble
MAX_UNIT_CHARS = 400  # a trailing short sentence joins the previous unit only if it stays under this
# A unit over this goes through split_long_sentence (F-07): the server's ~1000-token cap silently
# truncates a single request around ~800 characters, and it is also the paragraph-mode (F-05)
# packing threshold, so one request never risks that cap either way.
MAX_REQUEST_CHARS = 450
_PYDUB_EXPORT = {"aac": ("adts", "aac"), "opus": ("opus", "libopus")}
_LOSSY_FORMATS = ("mp3", "aac", "opus")
_BAD_CLIP_SILENCE_MS = 3000
_BAD_CLIP_SILENCE_DBFS = -50
_BAD_CLIP_RETRIES = 2
# Breeze is fast only when many sentences are generated together (7.1x real time at 32 per request,
# 0.45x at 1; 2026-10-01), so a chapter goes to its server in requests of this many units.
BREEZE_BATCH_SIZE = 32
_SHORT_UNIT_CHARS = 25
_SHORT_QUOTE_CONTEXT = "The room was quiet, and the window was open."

# F-02: after a restart the app can start before Chatterbox has finished loading its model
# (observed ~12 s); the OpenAI SDK's own retries give up after ~7 s. RETRYABLE_STATUS_CODES are
# the "temporarily unavailable" statuses worth waiting out; a plain APIConnectionError (including
# a timeout) is also retried. Anything else -- 4xx in particular -- is the caller's problem.
RETRYABLE_STATUS_CODES = (502, 503, 504)
SERVER_WAIT_TOTAL_SECONDS = 600  # ~10 minutes total budget across all retries of one request
SERVER_WAIT_INITIAL_DELAY_SECONDS = 2.0
SERVER_WAIT_MAX_DELAY_SECONDS = 30.0


def _is_server_unavailable(error: Exception) -> bool:
    """True for a connection failure or a 502/503/504: the server is down or still starting up
    and the request is worth retrying. False for everything else (4xx, a plain 500, ...): those
    are the caller's problem and must not be retried."""
    if isinstance(error, APIConnectionError):
        return True
    return isinstance(error, APIStatusError) and error.status_code in RETRYABLE_STATUS_CODES


def _is_speakable(unit: str) -> bool:
    """False for a unit with no letters or digits (F-18): scene breaks ("* * *", "...", "--")
    and other punctuation-only paragraphs should become silence, not a spoken request."""
    return any(char.isalnum() for char in unit)


def _long_silence_ms(audio: AudioSegment) -> int:
    """Longest near-silent run in one TTS response, before our own pauses are inserted."""
    if len(audio) < _BAD_CLIP_SILENCE_MS:
        return 0
    longest = run = 0
    for start in range(0, len(audio), 100):
        frame = audio[start:start + 100]
        run = run + len(frame) if frame.dBFS <= _BAD_CLIP_SILENCE_DBFS else 0
        longest = max(longest, run)
    return longest


_ELLIPSIS_MIN_MS = 950


def _overlong_ms(chars: int) -> int:
    """The longest plausible take of a short (up to 35 character) line."""
    return max(6000, chars * 300)


def _truncated_ms(chars: int) -> int:
    """The shortest plausible take of a long (45+ character) quoted line."""
    return chars * 40


def _truncated_ellipsis(audio: AudioSegment, text: str) -> bool:
    """A drawn-out short line should not end in under a second of audio."""
    spoken = text.strip().strip('"“”\'‘’').strip()
    return (len(spoken) <= 15 and spoken.endswith(("…", "..."))
            and len(spoken.rstrip("….")) >= 4 and len(audio) < _ELLIPSIS_MIN_MS)


def _implausible_quote_duration(audio: AudioSegment, text: str) -> Optional[str]:
    """Catch short dialogue loops and longer dialogue cut off before it is spoken."""
    quoted = text.strip()
    if not quoted.startswith(('“', '"', '‘', "'")):
        return None
    spoken = quoted.strip('"“”\'‘’').strip()
    chars = len(spoken)
    if 1 <= chars <= 35 and len(audio) > _overlong_ms(chars):
        return "overlong short dialogue"
    if chars >= 45 and len(audio) < _truncated_ms(chars):
        return "truncated dialogue"
    return None


def _implausible_short_narration_duration(audio: AudioSegment, text: str) -> Optional[str]:
    """Catch a short narration tag looping even though it is not a quoted line."""
    spoken = text.strip().strip('"“”\'‘’').strip()
    if text.lstrip().startswith(('“', '"', '‘', "'")):
        return None  # quoted lines have their own duration check
    chars = len(spoken)
    if 1 <= chars <= 35 and len(audio) > _overlong_ms(chars):
        return "overlong short narration"
    return None


def _duration_miss(audio: AudioSegment, text: str, reason: str) -> float:
    """How far a take rejected for `reason` (a length check above) is from its limit, as a
    fraction of the limit: the take nearest a plausible length is the one kept when every retry
    is rejected."""
    chars = len(text.strip().strip('"“”\'‘’').strip())
    if reason.startswith("overlong"):
        return len(audio) / _overlong_ms(chars) - 1
    limit = _ELLIPSIS_MIN_MS if reason == "too-short ellipsis audio" else _truncated_ms(chars)
    return 1 - len(audio) / limit


def _new_seed(previous: Optional[int] = None) -> int:
    """Give a short unit or rejected take its own reproducible Chatterbox sample."""
    seed = secrets.randbelow(2**31 - 1) + 1
    return seed if seed != previous else seed % (2**31 - 1) + 1


def _chatterbox_input(text: str) -> str:
    """Repair a mismatched end quote in the EPUB: an interrupted line ("You-“) loses its quotes and
    ends in a dash; any other line closed with an opening “ gets its proper ”."""
    quoted = text.strip()
    if quoted.startswith(('“', '"')) and quoted.endswith(('-“', '-"')):
        inner = quoted[1:-2].replace("…", "...").replace("...", "... ")
        return " ".join(inner.split()) + "—"
    if len(quoted) > 1 and quoted.startswith("“") and quoted.endswith("“"):
        return quoted[:-1] + "”"
    return text


def _breeze_batches(pending: List[int], items: List[dict]) -> List[List[int]]:
    """The units to send (indexes into items) as requests of up to BREEZE_BATCH_SIZE, each one model
    call that ends as early as it can. A batch generates until its longest take is done, so a small
    batch holding a long unit costs about what a full one does (2026-10-02: a call of 4 units took
    32 s, full calls 25-60 s). The server also runs directed and plain items as separate calls, so a
    few whispered lines in a book-order batch became a call of their own. So directed and plain
    units are batched apart, longest text first: units of similar length share a batch, and a
    group's leftover batch holds its shortest units. Equal lengths keep their order."""
    batches = []
    for directed in (False, True):
        group = sorted((i for i in pending if bool(items[i]["instruction"]) == directed),
                       key=lambda i: -len(items[i]["text"]))
        batches += [group[i:i + BREEZE_BATCH_SIZE] for i in range(0, len(group), BREEZE_BATCH_SIZE)]
    return batches


def _tiny_quote(text: str) -> bool:
    quoted = text.strip()
    return (2 < len(quoted) <= _SHORT_UNIT_CHARS
            and (quoted[0], quoted[-1]) in (("“", "”"), ('"', '"'))
            and re.fullmatch(r"[^\W\d_]+(?:['’][^\W\d_]+)*[!?.,]?", quoted[1:-1]) is not None)


def _new_sentence(text: str) -> str:
    """The unit with a capital first letter, as it follows the lead-in. Chatterbox capitalises a
    request's first letter itself, so "she said." was always spoken as "She said." before the
    lead-in; after it, lowercase reads as the same sentence running on, often with no pause to cut
    in (13 of 32 same-seed takes of hard tags unseparated as is, 8 capitalised; WORKLOG §26.5)."""
    for index, char in enumerate(text):
        if char.isalpha():
            return text[:index] + char.upper() + text[index + 1:]
    return text


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


def _sentence_units(text: str, language: str) -> List[Tuple[str, bool]]:
    """(text, continues_previous) sentence-sized units of one span of text: sentences under
    MIN_UNIT_CHARS join the next one, a short trailing sentence joins the previous unit while
    that stays under MAX_UNIT_CHARS, unspeakable units are dropped and oversized ones are split
    (F-07) into pieces marked continues_previous."""
    pending, packed = "", []
    for sentence in segment(language, text):
        sentence = str(sentence).strip()
        if not sentence:
            continue
        pending = f"{pending} {sentence}".strip()
        if len(pending) >= MIN_UNIT_CHARS:
            packed.append(pending)
            pending = ""
    if pending:
        if packed and len(packed[-1]) + len(pending) < MAX_UNIT_CHARS:
            packed[-1] = f"{packed[-1]} {pending}"
        else:
            packed.append(pending)
    units: List[Tuple[str, bool]] = []
    for unit in packed:
        if not _is_speakable(unit):
            continue
        pieces = _split_oversized_unit(unit)
        units.append((pieces[0], False))
        units.extend((piece, True) for piece in pieces[1:])
    return units


def paced_units(text: str, language: str) -> List[Tuple[int, str, bool]]:
    """(paragraph number, text, continues_previous) units of one or more whole sentences, in
    reading order. continues_previous is True only for a piece produced by splitting an
    oversized unit (F-07): it must follow the previous piece with NO pause.
    """
    units: List[Tuple[int, str, bool]] = []
    paragraphs = [" ".join(p.split()) for p in text.split(PARAGRAPH_MARK)]
    for number, paragraph in enumerate(p for p in paragraphs if p):
        units.extend((number, unit, continues) for unit, continues in _sentence_units(paragraph, language))
    return units


def _paragraph_units(text: str, language: str) -> List[Tuple[str, int, bool]]:
    """(text, sentence_count, continues_previous) units of one span of text for paragraph mode:
    whole sentences packed into one request each, splitting only at sentence boundaries when
    the span would otherwise exceed MAX_REQUEST_CHARS."""
    sentences = [s for s in (str(raw).strip() for raw in segment(language, text)) if s and _is_speakable(s)]
    units: List[Tuple[str, int, bool]] = []

    def flush(bin_sentences: List[str]) -> None:
        if bin_sentences:
            units.append((" ".join(bin_sentences), len(bin_sentences), False))

    current_bin: List[str] = []
    for sentence in sentences:
        if len(sentence) > MAX_REQUEST_CHARS:
            flush(current_bin)
            current_bin = []
            pieces = _split_oversized_unit(sentence)
            units.append((pieces[0], 1, False))
            units.extend((piece, 1, True) for piece in pieces[1:])
            continue
        candidate = current_bin + [sentence]
        if current_bin and len(" ".join(candidate)) > MAX_REQUEST_CHARS:
            flush(current_bin)
            current_bin = [sentence]
        else:
            current_bin = candidate
    flush(current_bin)
    return units


def paragraph_mode_units(text: str, language: str) -> List[Tuple[int, str, int, bool]]:
    """(paragraph number, text, sentence_count, continues_previous) units for paragraph mode
    (F-05): whole paragraphs are packed as one request each, splitting only at sentence
    boundaries when a paragraph would otherwise exceed MAX_REQUEST_CHARS. sentence_count says
    how many of the model's own inter-sentence gaps _stretch_sentence_gaps should look for and
    stretch to sentence_pause_ms; it is 1 (no internal gap to find) for a single-sentence piece
    and for every piece produced by splitting one oversized sentence (F-07), which are marked
    continues_previous like in paced_units.
    """
    units: List[Tuple[int, str, int, bool]] = []
    paragraphs = [" ".join(p.split()) for p in text.split(PARAGRAPH_MARK)]
    for number, paragraph in enumerate(p for p in paragraphs if p):
        units.extend((number, unit, count, continues) for unit, count, continues in _paragraph_units(paragraph, language))
    return units


VoiceOf = Callable[[Segment], str]


def _speech_segments(text: str) -> List[List[Segment]]:
    """Keep short quoted terms inside prose in the narrator's sentence.

    Cast line IDs come from chapter_segments and remain unchanged for real dialogue;
    this only changes the units sent to TTS. A quoted name such as
    `before he could, “Lena” materialized` is not a six-character speech line.
    """
    result = []
    for paragraph in chapter_segments(text):
        spoken = []
        i = 0
        while i < len(paragraph):
            piece = paragraph[i]
            if (piece.kind == DIALOGUE and not piece.continues and spoken
                    and spoken[-1].kind == NARRATION and i + 1 < len(paragraph)
                    and paragraph[i + 1].kind == NARRATION):
                quoted = piece.text.strip().strip('"“”„«»\'‘’').strip()
                after = paragraph[i + 1].text
                if (quoted and len(quoted) <= 30 and len(quoted.split()) <= 2
                        and quoted[-1] not in '.!?…,:;' and after[:1].islower()
                        and not has_speech_tag(spoken[-1].text, after)):
                    spoken[-1] = Segment(NARRATION, 0, f"{spoken[-1].text} {piece.text} {after}")
                    i += 2
                    continue
            spoken.append(piece)
            i += 1
        result.append(spoken)
    return result


def voiced_units(text: str, language: str, voice_of: VoiceOf) -> List[Tuple[int, str, bool, str]]:
    """(paragraph number, text, continues_previous, voice) sentence-mode units for multi-voice
    narration. Units are built inside each narration or dialogue segment, never across two, so a
    unit can never span a change of voice; the joining and splitting rules of paced_units apply
    within a segment. The first unit of a segment is a normal sentence boundary (sentence pause,
    or the paragraph pause when the paragraph changes)."""
    units: List[Tuple[int, str, bool, str]] = []
    for number, segments in enumerate(_speech_segments(text)):
        for piece in segments:
            voice = voice_of(piece)
            units.extend((number, unit, continues, voice) for unit, continues in _sentence_units(piece.text, language))
    return units


def voiced_paragraph_units(text: str, language: str, voice_of: VoiceOf) -> List[Tuple[int, str, int, bool, str]]:
    """(paragraph number, text, sentence_count, continues_previous, voice) paragraph-mode units
    for multi-voice narration: each segment is packed like a paragraph of its own."""
    units: List[Tuple[int, str, int, bool, str]] = []
    for number, segments in enumerate(_speech_segments(text)):
        for piece in segments:
            voice = voice_of(piece)
            units.extend((number, unit, count, continues, voice)
                         for unit, count, continues in _paragraph_units(piece.text, language))
    return units


MoodOf = Callable[[Segment], str]
SpeakerOf = Callable[[Segment], Optional[str]]


def adaptive_units(text: str, language: str, voice_of: VoiceOf, mood_of: MoodOf,
                   speaker_of: Optional[SpeakerOf] = None) -> List[Tuple[int, str, bool, str, str, Optional[str]]]:
    """(paragraph number, text, continues_previous, voice, mood, speaker) sentence-mode units for
    adaptive delivery: built inside each narration/dialogue segment exactly like voiced_units, with
    every unit of a segment also tagged with that segment's mood (core.delivery) and its cast
    speaker (None without speaker_of)."""
    units: List[Tuple[int, str, bool, str, str, Optional[str]]] = []
    for number, segments in enumerate(_speech_segments(text)):
        for piece in segments:
            voice, mood = voice_of(piece), mood_of(piece)
            speaker = speaker_of(piece) if speaker_of else None
            units.extend((number, unit, continues, voice, mood, speaker)
                         for unit, continues in _sentence_units(piece.text, language))
    return units


def adaptive_paragraph_units(text: str, language: str, voice_of: VoiceOf, mood_of: MoodOf,
                             speaker_of: Optional[SpeakerOf] = None
                             ) -> List[Tuple[int, str, int, bool, str, str, Optional[str]]]:
    """Paragraph-mode counterpart of adaptive_units: each segment is packed like a paragraph of
    its own, tagged with its voice, its mood and its cast speaker."""
    units: List[Tuple[int, str, int, bool, str, str, Optional[str]]] = []
    for number, segments in enumerate(_speech_segments(text)):
        for piece in segments:
            voice, mood = voice_of(piece), mood_of(piece)
            speaker = speaker_of(piece) if speaker_of else None
            units.extend((number, unit, count, continues, voice, mood, speaker)
                         for unit, count, continues in _paragraph_units(piece.text, language))
    return units


_GAP_FRAME_MS = 10
# A frame under this fraction of the clip's peak RMS counts as silence. Empirically checked
# against 5 short real Chatterbox (Original model) clips: 0.15 missed one true sentence gap
# (a low-energy consonant onset early in a sentence briefly outscored it); 0.20-0.30 found all
# 8 expected gaps, including in a comma-heavy sentence where the real gap was still correctly
# picked over both comma pauses. Not re-validated against longer or different-voice audio.
_GAP_SILENCE_RATIO = 0.20
# Inside a gap found that way, only frames at or below this level are true silence. The rest of the
# run is the quiet end of the word before it or the breathy start of the next one: measured on real
# Chatterbox audio, each gap began with 130-150 ms of it at -22 to -39 dBFS, which replacing the
# whole run erased (clipped word endings).
_TRUE_SILENCE_DBFS = -50.0


def _silence_runs(audio: AudioSegment) -> List[Tuple[int, int]]:
    """Contiguous (start_ms, end_ms) runs of low-RMS audio, in ~10 ms frames, excluding any run
    touching the very start or end of the clip (there is no sentence boundary to mark there)."""
    frame_count = len(audio) // _GAP_FRAME_MS
    if frame_count < 3:
        return []
    frame_rms = [audio[i * _GAP_FRAME_MS:(i + 1) * _GAP_FRAME_MS].rms for i in range(frame_count)]
    peak = max(frame_rms)
    if peak == 0:
        return []
    threshold = peak * _GAP_SILENCE_RATIO
    runs, start = [], None
    for i, rms in enumerate(frame_rms):
        if rms <= threshold:
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, frame_count))
    return [(s * _GAP_FRAME_MS, e * _GAP_FRAME_MS) for s, e in runs if s > 0 and e < frame_count]


def _stretch_sentence_gaps(audio: AudioSegment, gap_count: int, target_ms: int) -> AudioSegment:
    """Find the model's own inter-sentence pauses inside one multi-sentence paced unit and
    stretch the gap_count longest ones to target_ms (F-05).

    This is what lets a whole paragraph go in one request while still getting deliberate,
    configurable pauses between its sentences: the server places its own short, uneven gap at
    each sentence boundary, and this replaces each with exactly target_ms of silence.
    """
    if gap_count <= 0 or target_ms <= 0:
        return audio
    runs = _silence_runs(audio)
    if not runs:
        return audio
    chosen = sorted(runs, key=lambda r: r[1] - r[0], reverse=True)[:gap_count]
    chosen.sort(key=lambda r: r[0], reverse=True)  # splice back-to-front so earlier offsets hold
    silence = AudioSegment.silent(duration=target_ms, frame_rate=audio.frame_rate)
    silence = silence.set_channels(audio.channels).set_sample_width(audio.sample_width)
    for start, end in chosen:
        core_start, core_end = _silent_core(audio, start, end)
        audio = audio[:core_start] + silence + audio[core_end:]
    return audio


def _silent_core(audio: AudioSegment, start: int, end: int) -> Tuple[int, int]:
    """The part of a gap run to replace with the pause: its longest unbroken stretch of truly
    silent frames, so nothing audible is removed (the tail of the word before it, a consonant's
    release, a breath, the onset of the next word all stay). With no truly silent frame, an empty
    span at the run's quietest frame (the pause is inserted there and nothing is removed)."""
    frames = [audio[t:t + _GAP_FRAME_MS] for t in range(start, end, _GAP_FRAME_MS)]
    best, run_start = None, None
    for i, frame in enumerate(frames + [None]):
        if frame is not None and frame.dBFS <= _TRUE_SILENCE_DBFS:
            run_start = i if run_start is None else run_start
        elif run_start is not None:
            if best is None or i - run_start > best[1] - best[0]:
                best = (run_start, i)
            run_start = None
    if best:
        return start + best[0] * _GAP_FRAME_MS, min(end, start + best[1] * _GAP_FRAME_MS)
    quietest = min(range(len(frames)), key=lambda i: frames[i].rms)
    point = start + quietest * _GAP_FRAME_MS + _GAP_FRAME_MS // 2
    return point, point


_CONTEXT_GAP_MS = 80
# Where Whisper's word times locate the gap, a shorter pause is enough: 13 of 97 lead-in takes paused
# only 0-70 ms, and no true silence ever started between Whisper's end of "open." and the real gap
# (WORKLOG §26.5), so a word's own brief closure can't be mistaken for it.
_CARRIER_GAP_MS = 30
# Whisper's start of the unit's first word, plus this, is the latest the gap may start: measured on 150
# lead-in takes, the real gap started 470 ms before to 30 ms after it, and a pause inside the unit's
# first word ("he | grunts") no earlier than 90 ms after it (WORKLOG §26.5).
_UNIT_ONSET_SLACK_MS = 50


def _quote_context_cut(audio: AudioSegment) -> Optional[int]:
    # Single-word quotes only: a longer unit's own pause could be the one silence (_carrier_cut).
    cores = [_silent_core(audio, start, end) for start, end in _silence_runs(audio)
             if start > len(audio) / 2]
    cores = [(start, end) for start, end in cores if end - start >= _CONTEXT_GAP_MS]
    if len(cores) != 1:
        return None
    start, end = cores[0]
    cut = (start + end) // 20 * 10  # cut inside true silence, preserving the word's quiet onset
    return cut if 250 <= len(audio) - cut <= 2000 else None


def _carrier_cut(audio: AudioSegment, words: List[Tuple[str, int, int]]) -> Optional[int]:
    """Where the lead-in ends, from Whisper's word times: inside the first true silence that starts
    after the lead-in's last word ends and before the unit's first word begins (a two-word quote's
    own pause starts later), or after the lead-in when Whisper heard no word after it.

    Measured: the real gap starts after Whisper's end of the lead-in and before its start of the
    unit plus _UNIT_ONSET_SLACK_MS (WORKLOG §26.2, §26.5), so a pause before the lead-in's last word
    or inside the unit's first word doesn't qualify. A misplaced word time finds no silence, and
    the take is retried."""
    bounds = speech_check.carrier_bounds(words, _SHORT_QUOTE_CONTEXT)
    if bounds is None:
        return None
    last_end, next_start = bounds
    cores = [(start, end) for start, end in (_silent_core(audio, s, e) for s, e in _silence_runs(audio))
             if end - start >= _CARRIER_GAP_MS and start >= last_end
             and (next_start is None or start < next_start + _UNIT_ONSET_SLACK_MS)]
    if not cores:
        return None
    start, end = min(cores)
    cut = (start + end) // 20 * 10
    return cut if len(audio) - cut >= 250 else None


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
        config.paced_unit_mode = config.paced_unit_mode or "sentence"
        config.voice_mode = config.voice_mode or VOICE_MODE_SINGLE  # jobs from before multi-voice existed
        config.adaptive_delivery = bool(config.adaptive_delivery)  # jobs from before adaptive delivery existed

        self.price = get_price(config.model_name)
        super().__init__(config)

        # base_url=None falls back to OPENAI_BASE_URL exactly as before; a per-config URL (e.g.
        # Kokoro's) must win over that env var, since both can be set at once.
        # Breeze has its own client (core.breeze_client), so it needs no OpenAI key or endpoint.
        self.client = (None if config.model_name == "breeze"
                       else OpenAI(max_retries=4, base_url=config.openai_base_url))  # OPENAI_API_KEY env var still required
        self._tone = None  # tone_match.ToneMatcher, made on the first chapter that matches
        self.cast: Optional[dict] = None
        if config.voice_mode == VOICE_MODE_CAST:
            self.cast = cast_store.load_cast(config.cast_file) if config.cast_file else None
            if self.cast is None:
                raise ValueError(f"OpenAI: cast mode needs a readable cast file, got {config.cast_file!r}")

    def _voice_of(self, text: str) -> VoiceOf:
        """The per-segment voice rule for one chapter: narration is the narrator's voice; a
        quoted line is its attributed character's voice when the cast knows the speaker and has
        given them a voice (the narrator's own for a first-person book's narrating character),
        else the dialogue voice (else the narrator's, so a mode without a dialogue voice degrades
        to single voice rather than failing)."""
        narrator = self.config.voice_name
        if self.cast is None:
            dialogue_voice = self.config.dialogue_voice or narrator
            return lambda piece: dialogue_voice if piece.kind == DIALOGUE else narrator
        # Attributions are keyed by the chapter text's hash (the same hash the chapter manifest
        # uses), so they survive renumbering and a different chapter selection.
        chapter_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()
        # A collection's first-person story is narrated by its own teller's voice.
        narrator = cast_store.chapter_narrator_voice(self.cast, chapter_hash) or narrator
        dialogue_voice = self.config.dialogue_voice or narrator
        lines = cast_store.chapter_lines(self.cast, chapter_hash)
        if lines is None:
            logger.warning("OpenAI: this chapter is not in the cast (text changed or chapter not analysed); "
                           "every quoted line gets the dialogue voice")
            lines = {}

        # In a first-person chapter the "I" character's own lines are the narrator's too, as one
        # performer would read them (unless the owner gave that character a voice of their own). An
        # anthology can change narrator, or point of view, from story to story.
        narrating = cast_store.chapter_narrator(self.cast, chapter_hash)

        def voice_of(piece: Segment) -> str:
            if piece.kind != DIALOGUE:
                return narrator
            speaker = lines.get(piece.line_id)
            if speaker and speaker == narrating:
                return narrator
            return cast_store.character_voice(self.cast, speaker) or dialogue_voice
        return voice_of

    def _speaker_of(self, text: str) -> SpeakerOf:
        """The cast key behind each dialogue unit, kept even when voices are shared; None for the
        chapter's first-person narrator, whose lines take the narrator's own delivery."""
        if self.cast is None:
            return lambda piece: None
        chapter_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()
        lines = cast_store.chapter_lines(self.cast, chapter_hash) or {}
        narrating = cast_store.chapter_narrator(self.cast, chapter_hash)

        def speaker_of(piece: Segment) -> Optional[str]:
            speaker = lines.get(piece.line_id) if piece.kind == DIALOGUE else None
            return None if speaker and speaker == narrating else speaker
        return speaker_of

    def _mood_of(self, text: str) -> MoodOf:
        """The per-segment mood rule for one chapter (adaptive delivery): narration is always
        "normal"; a dialogue line's mood is the cast's saved mood when this is cast mode and the
        chapter was analysed, but a rule cue detected fresh from the current text always overrides
        it (rules win over the LLM's guess, whether that guess came from this pass or an earlier
        analysis) -- the same precedence core.cast_llm.attribute_chapter applies at analysis time.
        """
        paragraphs = chapter_segments(text)
        rule_moods, rule_cues = delivery.segment_moods_and_cues(paragraphs)
        cast_moods: dict = {}
        if self.cast is not None:
            chapter = cast_store.chapter_moods(self.cast, hashlib.sha1(text.encode("utf-8")).hexdigest())
            if chapter:
                cast_moods = chapter

        def mood_of(piece: Segment) -> str:
            if piece.kind != DIALOGUE:
                return delivery.MOOD_NORMAL
            rule_mood = rule_moods.get(piece.line_id, delivery.MOOD_NORMAL)
            if rule_mood != delivery.MOOD_NORMAL:
                # The speech tag's verb rides along for Breeze's spoken direction.
                return delivery.CuedMood(rule_mood, rule_cues.get(piece.line_id))
            return cast_moods.get(piece.line_id, delivery.MOOD_NORMAL)
        return mood_of

    def _is_chatterbox_engine(self) -> bool:
        """True when this provider is pointed at Chatterbox (not Kokoro or the real OpenAI API):
        adaptive delivery and the baseline sliders only mean anything to Chatterbox's own endpoint
        extensions, and chatterbox_ui.build_config is the only caller that sets this model name."""
        return self.config.model_name == "chatterbox"

    def _is_breeze_engine(self) -> bool:
        """True when this provider speaks through the Breeze server, in batches (build_config sets this
        model name)."""
        return self.config.model_name == "breeze"

    def _adaptive_active(self) -> bool:
        """Adaptive delivery is on for this book and the engine has a way to use it: Chatterbox's
        sliders and gain, or Breeze's spoken direction (Kokoro has neither)."""
        return bool(self.config.adaptive_delivery) and (self._is_chatterbox_engine() or self._is_breeze_engine())

    def _has_custom_baseline(self) -> bool:
        return any(value is not None for value in
                  (self.config.delivery_exaggeration, self.config.delivery_cfg_weight, self.config.delivery_temperature))

    def _delivery_baseline(self) -> delivery.Baseline:
        """The book's baseline sliders: this config's own delivery_* overrides, filled in from
        Chatterbox's saved generation defaults (or the approved fallback) for whichever of the
        three were left unset."""
        exaggeration = self.config.delivery_exaggeration
        cfg_weight = self.config.delivery_cfg_weight
        temperature = self.config.delivery_temperature
        if exaggeration is None or cfg_weight is None or temperature is None:
            saved = delivery.saved_chatterbox_defaults()
            exaggeration = saved.exaggeration if exaggeration is None else exaggeration
            cfg_weight = saved.cfg_weight if cfg_weight is None else cfg_weight
            temperature = saved.temperature if temperature is None else temperature
        return delivery.Baseline(exaggeration, cfg_weight, temperature)

    def _character_offsets(self) -> dict:
        """{character key: exaggeration offset} for adaptive cast delivery."""
        if getattr(self, "_offsets", None) is None:
            self._offsets = (cast_store.exaggeration_offsets(self.cast)
                             if self.cast is not None and self.config.voice_mode == VOICE_MODE_CAST
                             and self._adaptive_active() else {})
        return self._offsets

    def _voice_baseline(self, speaker: Optional[str], text: str,
                        baseline: Optional[delivery.Baseline] = None) -> delivery.Baseline:
        """`baseline` (the book's, read here when the caller hasn't already) shifted by the speaking
        character's delivery offset, eased in over the same short-line range as moods."""
        if baseline is None:
            baseline = self._delivery_baseline()
        offset = self._character_offsets().get(speaker or "", 0.0) * delivery.short_line_strength(text)
        if not offset:
            return baseline
        return baseline._replace(exaggeration=round(min(2.0, max(0.25, baseline.exaggeration + offset)), 2))

    def _delivery_extra_body(self, mood: str, speaker: Optional[str], text: str,
                             baseline: Optional[delivery.Baseline] = None) -> Optional[dict]:
        """The Chatterbox-only extra_body sliders for one unit, or None when neither adaptive
        delivery nor a per-book baseline applies (today's plain request, unchanged).

        Adaptive delivery sends the mood's own preset around the baseline of the unit's speaker (the
        book's, shifted for a character whose delivery is even or expressive); a baseline set with
        adaptive delivery off sends the plain book baseline (preset() at "normal" reproduces it
        unchanged, with 0 dB gain)."""
        if not self._is_chatterbox_engine() or not (self.config.adaptive_delivery or self._has_custom_baseline()):
            return None
        effective_mood = mood if self.config.adaptive_delivery else delivery.MOOD_NORMAL
        exaggeration, cfg_weight, temperature, _ = delivery.unit_preset(
            effective_mood, self._voice_baseline(speaker, text, baseline), text)
        # Same-seed listening checks: calmer tiny narration tags improved the marked takes.
        if (self.config.adaptive_delivery and len(text.strip()) <= _SHORT_UNIT_CHARS
                and has_speech_tag("", text)):
            exaggeration = min(exaggeration, 0.5)
        return {"exaggeration": exaggeration, "cfg_weight": cfg_weight, "temperature": temperature}

    def __str__(self) -> str:
        return super().__str__()

    def pacing_enabled(self) -> bool:
        return any(isinstance(value, (int, float)) and not isinstance(value, bool)
                   for value in (self.config.sentence_pause_ms, self.config.paragraph_pause_ms))

    def _create_speech(self, *, sleep: Callable[[float], None] = time.sleep,
                        clock: Callable[[], float] = time.monotonic, **kwargs):
        """Call the TTS endpoint, tolerating Chatterbox being temporarily unavailable (F-02).

        The SDK's own retries (max_retries=4) give up after ~7 s; after a restart the server can
        take longer than that to finish loading its model, which would otherwise fail a resumed
        book's first chapters every time. On a connection error or 502/503/504 this waits with
        backoff for up to SERVER_WAIT_TOTAL_SECONDS in total before giving up. A 4xx (or any
        other) error is the caller's problem and is never retried.

        `sleep`/`clock` are injectable so a test can drive this without a real ~10 minute wait.
        """
        deadline = clock() + SERVER_WAIT_TOTAL_SECONDS
        delay = SERVER_WAIT_INITIAL_DELAY_SECONDS
        while True:
            try:
                return self.client.audio.speech.create(**kwargs)
            except (APIConnectionError, APIStatusError) as e:
                if not _is_server_unavailable(e):
                    raise
                remaining = deadline - clock()
                if remaining <= 0:
                    logger.error(f"OpenAI: Chatterbox still unavailable after "
                                 f"{SERVER_WAIT_TOTAL_SECONDS}s, giving up: {e}")
                    raise
                wait_for = min(delay, remaining)
                logger.warning(f"OpenAI: waiting for Chatterbox to become available, "
                                f"retrying in {wait_for:.1f}s: {e}")
                sleep(wait_for)
                delay = min(delay * 2, SERVER_WAIT_MAX_DELAY_SECONDS)

    def text_to_speech(self, text: str, output_file: str, audio_tags: AudioTags):
        if self.pacing_enabled() or self._is_breeze_engine():
            # Paragraph mode's gap detector was tuned on Chatterbox audio: Breeze is sentence units only.
            if self.config.paced_unit_mode == "paragraph" and not self._is_breeze_engine():
                self._paced_text_to_speech_paragraph(text, output_file, audio_tags)
            else:
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

            # The SDK already retries a transient error within ~7s (max_retries=4);
            # _create_speech adds a much longer wait for the server still starting up (F-02).
            response = self._create_speech(
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

        Single voice mode sends exactly the units of paced_units() with the one configured voice;
        the other voice modes build units inside the narration/dialogue segments (voiced_units)
        and send each with its own voice. Adaptive delivery (Chatterbox and Breeze) always builds units
        inside segments (single voice mode included, with the narrator voice for every segment) so
        each unit can carry its segment's mood.
        """
        language = self.config.language or "en"
        if self._adaptive_active():
            voice_of = self._voice_of(text) if self.config.voice_mode != VOICE_MODE_SINGLE else (
                lambda piece: self.config.voice_name)
            speaker_of = self._speaker_of(text) if self.config.voice_mode == VOICE_MODE_CAST else None
            units = [(paragraph, unit, 1, continues, voice, mood, speaker)
                     for paragraph, unit, continues, voice, mood, speaker in adaptive_units(
                         text, language, voice_of, self._mood_of(text), speaker_of)]
        elif self.config.voice_mode == VOICE_MODE_SINGLE:
            units = [(paragraph, unit, 1, continues, self.config.voice_name, delivery.MOOD_NORMAL, None)
                     for paragraph, unit, continues in paced_units(text, language)]
        else:
            units = [(paragraph, unit, 1, continues, voice, delivery.MOOD_NORMAL, None)
                     for paragraph, unit, continues, voice in voiced_units(text, language, self._voice_of(text))]
        self._speak_units(units, output_file, audio_tags)

    def _paced_text_to_speech_paragraph(self, text: str, output_file: str, audio_tags: AudioTags) -> None:
        """Paragraph-unit mode (F-05, opt-in: paced_unit_mode="paragraph").

        One request per paragraph (or per packed group of sentences, when a paragraph would
        otherwise exceed MAX_REQUEST_CHARS) instead of one request per sentence: about 330-420
        requests per 30-minute chapter each pay a ~0.35 s fixed server cost regardless of how
        short the text is. The model's own inter-sentence gaps inside a multi-sentence request
        are found and stretched to sentence_pause_ms (_stretch_sentence_gaps); paragraph pauses
        and the F-27 one-shot speed change work exactly as in sentence mode. In a multi-voice
        mode each narration or dialogue segment is packed as a paragraph of its own.
        """
        language = self.config.language or "en"
        if self._adaptive_active():
            voice_of = self._voice_of(text) if self.config.voice_mode != VOICE_MODE_SINGLE else (
                lambda piece: self.config.voice_name)
            speaker_of = self._speaker_of(text) if self.config.voice_mode == VOICE_MODE_CAST else None
            units = adaptive_paragraph_units(text, language, voice_of, self._mood_of(text), speaker_of)
        elif self.config.voice_mode == VOICE_MODE_SINGLE:
            units = [(paragraph, unit, count, continues, self.config.voice_name, delivery.MOOD_NORMAL, None)
                     for paragraph, unit, count, continues in paragraph_mode_units(text, language)]
        else:
            units = [(paragraph, unit, count, continues, voice, delivery.MOOD_NORMAL, None) for paragraph, unit,
                     count, continues, voice in voiced_paragraph_units(text, language, self._voice_of(text))]
        self._speak_units(units, output_file, audio_tags)

    def _speak_units(self, units: List[tuple], output_file: str,
                     audio_tags: AudioTags) -> None:
        """Turn each (paragraph, text, sentence_count, continues_previous, voice, mood, speaker) unit
        into audio (_unit_takes), insert the configured pauses between them and export the chapter once.

        Producing a unit's audio is separate from assembling the chapter: Chatterbox and Kokoro make
        takes one request at a time, Breeze makes them in batches, and the assembly below is the same
        for all of them. Adaptive delivery for Chatterbox applies the unit's mood preset's gain plus
        the peak guard to the decoded audio before it is joined with the pauses (Breeze is directed
        in its request instead, see _breeze_takes).
        """
        speed = float(self.config.speed or 1.0)
        sentence_gap_ms = int(self.config.sentence_pause_ms or 0)
        paragraph_gap_ms = int(self.config.paragraph_pause_ms or 0)
        if not units:
            raise ValueError("No speakable text in this chapter")
        adaptive = self._adaptive_active()
        # Once per chapter: filling in unset sliders can mean reading Chatterbox's config file.
        baseline = (self._delivery_baseline() if self._is_chatterbox_engine()
                    and (self.config.adaptive_delivery or self._has_custom_baseline()) else None)
        pieces: List[bytes] = []
        spoken: List[Tuple[int, str]] = []  # (index in pieces, voice) of every unit's audio
        audio_format = None
        previous_paragraph = None
        timeline_frames = 0
        clip_entries = []
        mapped = self._is_chatterbox_engine() or self._is_breeze_engine()
        for number, (item, take) in enumerate(zip(units, self._unit_takes(units, audio_tags, baseline)), 1):
            paragraph, unit, sentence_count, continues_previous, voice, mood, speaker = item
            chunk_id = f"chapter-{audio_tags.idx}_{audio_tags.title}_chunk_{number}_of_{len(units)}"
            audio, params, attempts, flagged = take
            if sentence_count > 1 and sentence_gap_ms > 0:
                audio = _stretch_sentence_gaps(audio, sentence_count - 1, sentence_gap_ms)
            if adaptive and self._is_chatterbox_engine():  # Breeze's model does its own loudness
                audio = delivery.guarded_gain(audio, delivery.unit_preset(
                    mood, self._voice_baseline(speaker, unit, baseline), unit)[3])
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
                timeline_frames += gap_frames
            start_frame = timeline_frames
            spoken.append((len(pieces), voice))
            pieces.append(audio.raw_data)
            timeline_frames += len(audio.raw_data) // (audio_format[1] * audio_format[2])
            if mapped:
                clip_entries.append({
                    "chunk": number, "text_sha1": hashlib.sha1(unit.encode("utf-8")).hexdigest(),
                    "text_length": len(unit), "voice": voice, "mood": str(mood),
                    "start_ms": round(start_frame * 1000 / audio_format[0] / speed),
                    "end_ms": round(timeline_frames * 1000 / audio_format[0] / speed),
                    "seed": params.get("seed"), "attempts": attempts,
                    **({"context_trim_ms": params["_context_trim_ms"]}
                       if "_context_trim_ms" in params else {}),
                    **({"match": params["_match"]} if "_match" in params else {}),
                    **({"instruction": params["_instruction"]} if params.get("_instruction") else {}),
                    "settings": {key: params[key] for key in ("exaggeration", "cfg_weight", "temperature")
                                 if key in params},
                    **({"flagged": flagged} if flagged else {}),
                })
                logger.info("Clip %s: chapter %.3f–%.3fs, seed=%s, settings=%s",
                            chunk_id, clip_entries[-1]["start_ms"] / 1000,
                            clip_entries[-1]["end_ms"] / 1000, params.get("seed", "default"),
                            clip_entries[-1]["settings"])
            previous_paragraph = paragraph

        self._finish_units(pieces, spoken, audio_format)
        self._combine_and_export(pieces, audio_format, speed, output_file, audio_tags)
        if clip_entries:
            map_path = f"{output_file}.clips.json"
            try:
                with open(f"{map_path}.tmp", "w", encoding="utf-8") as output:
                    json.dump({"version": 1,
                               "duration_ms": round(timeline_frames * 1000 / audio_format[0] / speed),
                               "clips": clip_entries}, output, ensure_ascii=False)
                os.replace(f"{map_path}.tmp", map_path)
            except OSError as error:
                logger.warning("Could not save clip locations for %s: %s", output_file, error)

    def _unit_takes(self, units: List[tuple], audio_tags: AudioTags,
                    baseline: Optional[delivery.Baseline]) -> Iterator[Tuple[AudioSegment, dict, int, Optional[str]]]:
        """(audio, the parameters it was made with, attempts made, why the kept take is suspect) for
        every unit, in order: one request per unit for Chatterbox and Kokoro, batches for Breeze."""
        if self._is_breeze_engine():
            yield from self._breeze_takes(units, audio_tags)
            return
        adaptive = self._adaptive_active()
        for number, item in enumerate(units, 1):
            paragraph, unit, sentence_count, continues_previous, voice, mood, speaker = item
            chunk_id = f"chapter-{audio_tags.idx}_{audio_tags.title}_chunk_{number}_of_{len(units)}"
            detail = f", sentences={sentence_count}" if self.config.paced_unit_mode == "paragraph" else ""
            if self.config.voice_mode != VOICE_MODE_SINGLE:
                detail += f", voice={voice}"
            if adaptive:
                detail += f", mood={mood}"
            logger.info(f"Processing {chunk_id}, length={len(unit)}{detail}")
            logger.debug(f"Processing {chunk_id}, length={len(unit)}, text=[{unit}]")
            request_kwargs = dict(
                model=self.config.model_name,
                voice=voice,
                speed=1.0,
                instructions=self.config.instructions,
                input=_chatterbox_input(unit) if self._is_chatterbox_engine() else unit,
                response_format="wav",
            )
            extra_body = self._delivery_extra_body(mood, speaker, unit, baseline)
            if extra_body is not None:
                request_kwargs["extra_body"] = extra_body
            if self._is_chatterbox_engine() and len(unit.strip()) <= _SHORT_UNIT_CHARS:
                request_kwargs["extra_body"] = {**request_kwargs.get("extra_body", {}),
                                                "seed": _new_seed()}
            yield self._speak_take(request_kwargs, unit, chunk_id)

    def _breeze_takes(self, units: List[tuple],
                      audio_tags: AudioTags) -> List[Tuple[AudioSegment, dict, int, Optional[str]]]:
        """Every unit's take from the Breeze server, in unit order.

        The units go out in requests of BREEZE_BATCH_SIZE, grouped by _breeze_batches (directed apart
        from plain, similar lengths together); the takes still come back in unit order. The checks of
        one batch (near-silence, then the speech check's transcript through _verdict, for every take;
        no word times, which only Chatterbox's lead-in cut needs) run in a worker thread while the
        next batch generates: Whisper uses the CPU, generation the GPU. Generation never gets
        more than one batch ahead of the checks. Units that fail are sent again together, in new
        batches with a new seed, for up to _BAD_CLIP_RETRIES more rounds; the best take is kept
        when none passes (as _speak_take does), and a unit fails the chapter only when the server
        never made any audio for it.

        With adaptive delivery on, every unit whose mood is not normal is sent with its voice
        direction (delivery.breeze_instruction), still cloning its voice; normal units stay plain.
        Short lines are directed like any other: Breeze is autoregressive too, but nothing in its
        server's behaviour calls for a cut-off yet, and a live listening test decides."""
        total = len(units)
        adaptive = self._adaptive_active()
        ids = [f"chapter-{audio_tags.idx}_{audio_tags.title}_chunk_{n}_of_{total}" for n in range(1, total + 1)]
        items = []
        for number, (_, unit, _, _, voice, mood, _) in enumerate(units):
            ref_text = voice_transcripts.transcript(voice)
            if not ref_text:
                raise ValueError(f"Breeze needs the words spoken in the voice clip {voice} and could not get "
                                 "them (the speech check's Whisper model, SPEECH_CHECK_MODEL, transcribes it)")
            # The EPUB's quote repairs are not specific to Chatterbox.
            instruction = (delivery.breeze_instruction(mood, unit, getattr(mood, "cue", None))
                           if adaptive else None)
            items.append({"id": ids[number], "text": _chatterbox_input(unit), "voice": voice,
                          "ref_text": ref_text, "instruction": instruction, "cfg_scale": None})
        checker = speech_check.get()
        kept: dict = {}                            # unit index -> the take that passed
        rejected = {n: [] for n in range(total)}   # unit index -> (kind, badness, attempt, audio, params, reason)
        errors: dict = {}                          # unit index -> the server's last error for it
        pending = list(range(total))
        seed = None
        for attempt in range(_BAD_CLIP_RETRIES + 1):
            seed = _new_seed(seed)
            label = f"attempt {attempt + 1}/{_BAD_CLIP_RETRIES + 1}"
            batches = _breeze_batches(pending, items)
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="breeze-check") as checks:
                futures = []
                for number, batch in enumerate(batches, 1):
                    if len(futures) >= 2:
                        waited = time.perf_counter()
                        futures[-2].result()  # the checks of the batch before last are done
                        waited = time.perf_counter() - waited
                        if waited >= 1:  # the checks, not the GPU, are holding the chapter up
                            logger.info("Breeze %s waited %.1fs for the checks of batch %d", label, waited,
                                        number - 2)
                    logger.info("Breeze %s, batch %d of %d: %d units%s, seed=%d", label, number, len(batches),
                                len(batch), " (directed)" if items[batch[0]]["instruction"] else "", seed)
                    takes = breeze_client.synthesize_batch([items[i] for i in batch], seed)
                    futures.append(checks.submit(self._check_breeze_batch, batch, takes, units, items, ids, seed,
                                                 attempt, label, checker))
                results = [future.result() for future in futures]
            pending = []
            for batch_results in results:
                for index, outcome, detail in batch_results:
                    if outcome == "pass":
                        kept[index] = detail
                        continue
                    if outcome == "error":
                        errors[index] = detail
                    else:
                        rejected[index].append(detail)
                    pending.append(index)
            pending.sort()
            if not pending:
                break
        takes = []
        for index in range(total):
            if index in kept:
                takes.append(kept[index])
                continue
            if not rejected[index]:
                raise RuntimeError(f"Breeze made no audio for {ids[index]}: {errors.get(index, 'no take')}")
            _, _, attempt, audio, params, reason = min(rejected[index], key=lambda take: take[:3])
            logger.warning("Keeping attempt %d of %s (%.2fs, %s): no attempt passed its checks",
                           attempt + 1, ids[index], len(audio) / 1000, reason)
            takes.append((audio, params, _BAD_CLIP_RETRIES + 1, reason))
        return takes

    def _check_breeze_batch(self, batch: List[int], takes: list, units: List[tuple], items: List[dict],
                            ids: List[str], seed: int, attempt: int, label: str, checker) -> list:
        """(unit index, outcome, detail) for each take of one batch, run in the checking thread:
        "error" (detail: the server's message), "pass" (detail: the finished take) or "reject"
        (detail: the tuple _breeze_takes ranks the unit's rejected takes by). A near-silent take
        ranks last, so it is kept only when nothing else exists. A directed take's instruction goes
        into its params (the clip map records it), and a rejected one is logged with its mood so a
        listening test can see whether whispers or shouts fail the checks more often. Whisper hears
        speech_check.WORKERS takes at once: the checks, not the GPU, set the pace of a batch of
        short units."""
        started = time.perf_counter()
        silent = {index: _long_silence_ms(take) for index, take in zip(batch, takes) if not isinstance(take, str)}
        heard = {}
        if checker is not None:
            to_hear = [(index, take) for index, take in zip(batch, takes)
                       if index in silent and silent[index] < _BAD_CLIP_SILENCE_MS]
            with ThreadPoolExecutor(speech_check.WORKERS, thread_name_prefix="breeze-check-hear") as hearing:
                heard = dict(zip([index for index, _ in to_hear], hearing.map(
                    lambda pair: self._hear(checker, pair[1], ids[pair[0]], words=False), to_hear)))
        outcomes = []
        for index, take in zip(batch, takes):
            unit, chunk_id = units[index][1], ids[index]
            if isinstance(take, str):
                logger.warning("Breeze made no audio for %s (%s): %s", chunk_id, label, take)
                outcomes.append((index, "error", take))
                continue
            mood = units[index][5]
            params = {"seed": seed}
            if items[index]["instruction"]:
                params["_instruction"] = items[index]["instruction"]
            silent_ms = silent[index]
            if silent_ms >= _BAD_CLIP_SILENCE_MS:
                logger.warning("Breeze returned %.1fs of near-silence for %s (%s, mood=%s)",
                               silent_ms / 1000, chunk_id, label, mood)
                outcomes.append((index, "reject", (2, silent_ms, attempt, take, params, "near-silent audio")))
                continue
            params, verdict = self._verdict(take, unit, params, heard.get(index), chunk_id, label)
            if verdict is None:
                outcomes.append((index, "pass", (take, params, attempt + 1, None)))
            else:
                logger.info("Breeze rejected %s (%s, mood=%s): %s", chunk_id, label, mood, verdict[2])
                outcomes.append((index, "reject", (verdict[0], verdict[1], attempt, take, params, verdict[2])))
        logger.info("Breeze %s: checked %d takes in %.1fs", label, len(batch), time.perf_counter() - started)
        return outcomes

    def _speak_take(self, request_kwargs: dict, unit: str,
                    chunk_id: str) -> Tuple[AudioSegment, dict, int, Optional[str]]:
        """One unit's audio, requested again with a new seed while Chatterbox's take is bad:
        (audio, the extra_body it was made with, attempts made, why the kept take is suspect).

        A take with a long near-silent gap is never kept. When every attempt is rejected only for
        an implausible length, the one nearest a plausible length is kept instead of failing the
        chapter (and with it the whole M4B): a quick "Well…" can be real speech.

        Chatterbox garbles tiny requests, so a single-word English quote is spoken after a
        same-voice lead-in that is cut off again (WORKLOG §25). With the speech check on (§26),
        every short English unit gets the lead-in, cut by Whisper's word times unless it is a
        single word, and a take whose transcript doesn't match its text is retried like a bad
        length; the best match is kept. A take that may still hold the lead-in is never kept: if
        no attempt separates, the unit is spoken without it, judged the same way."""
        english = self._is_chatterbox_engine() and (self.config.language or "en") == "en"
        checker = speech_check.get() if english and len(unit.strip()) <= _SHORT_UNIT_CHARS else None
        single_word = english and _tiny_quote(request_kwargs["input"])
        contextual = single_word or checker is not None
        plain_kwargs = request_kwargs
        if contextual:
            request_kwargs = {**request_kwargs,
                              "input": f"{_SHORT_QUOTE_CONTEXT} {_new_sentence(request_kwargs['input'].strip())}"}
        rejected = []
        unsafe_reason = "near-silent audio"
        unseparated = False
        for attempt in range(_BAD_CLIP_RETRIES + 1):
            if attempt:
                # The server's audiobook preset fixes the seed, so repeating an
                # unchanged request reproduces the same silent clip every time.
                request_kwargs["extra_body"] = {
                    **request_kwargs.get("extra_body", {}),
                    "seed": _new_seed(request_kwargs.get("extra_body", {}).get("seed")),
                }
            response = self._create_speech(**request_kwargs)
            audio = AudioSegment.from_file(io.BytesIO(response.content), format="wav")
            params = request_kwargs.get("extra_body", {})
            if not self._is_chatterbox_engine():
                return audio, params, attempt + 1, None
            silent_ms = _long_silence_ms(audio)
            if silent_ms >= _BAD_CLIP_SILENCE_MS:
                unsafe_reason = "near-silent audio"
                logger.warning("Chatterbox returned %.1fs of near-silence for %s (attempt %d/%d)",
                               silent_ms / 1000, chunk_id, attempt + 1, _BAD_CLIP_RETRIES + 1)
                continue
            if contextual:
                cut = _quote_context_cut(audio) if single_word else None
                if cut is None and checker is not None:
                    heard = self._hear(checker, audio, chunk_id)
                    cut = _carrier_cut(audio, heard.words) if heard else None
                if cut is None:
                    unsafe_reason = "unseparated short-quote context"
                    unseparated = True
                    logger.warning("Chatterbox could not separate short-quote context for %s (attempt %d/%d)",
                                   chunk_id, attempt + 1, _BAD_CLIP_RETRIES + 1)
                    continue  # never export a take that could still contain the added lead-in
                audio = audio[cut:]
                params = {**params, "_context_trim_ms": cut}
                logger.info("Short-quote context %s: removed %.3fs, seed=%s",
                            chunk_id, cut / 1000, params.get("seed", "default"))
            # Heard before the length checks: a take kept for the least-wrong length is leak-checked too.
            heard = self._hear(checker, audio, chunk_id) if checker is not None else None
            if contextual and heard and (speech_check.leaked(unit, heard.text, _SHORT_QUOTE_CONTEXT)
                                         or speech_check.clipped(unit, heard.text)):
                unsafe_reason = "unseparated short-quote context"
                unseparated = True
                logger.warning("The cut missed the lead-in's end for %s (attempt %d/%d)",
                               chunk_id, attempt + 1, _BAD_CLIP_RETRIES + 1)
                continue
            params, verdict = self._verdict(audio, unit, params, heard, chunk_id,
                                            f"attempt {attempt + 1}/{_BAD_CLIP_RETRIES + 1}")
            if verdict is None:
                return audio, params, attempt + 1, None
            rejected.append((verdict[0], verdict[1], attempt, audio, params, verdict[2]))
        if not rejected and unseparated:
            return self._speak_plain(plain_kwargs, request_kwargs, checker, unit, chunk_id)
        if not rejected:
            raise RuntimeError(f"Chatterbox returned repeated {unsafe_reason} for {chunk_id}")
        _, _, attempt, audio, params, reason = min(rejected, key=lambda take: take[:3])
        logger.warning("Keeping attempt %d of %s (%.2fs, %s): no attempt passed its checks",
                       attempt + 1, chunk_id, len(audio) / 1000, reason)
        return audio, params, _BAD_CLIP_RETRIES + 1, reason

    def _speak_plain(self, plain_kwargs: dict, context_kwargs: dict, checker, unit: str,
                     chunk_id: str) -> Tuple[AudioSegment, dict, int, Optional[str]]:
        """The unit without the lead-in, after no attempt could separate it, judged and retried like
        any take: one short unit never fails the chapter, and the lead-in never reaches the book.
        Flagged unless the speech check passed it. Attempts count the lead-in attempts too."""
        seed = context_kwargs.get("extra_body", {}).get("seed")
        first = _BAD_CLIP_RETRIES + 1  # lead-in attempts already made
        rejected = []
        for attempt in range(_BAD_CLIP_RETRIES + 1):
            seed = _new_seed(seed)
            kwargs = {**plain_kwargs, "extra_body": {**plain_kwargs.get("extra_body", {}), "seed": seed}}
            audio = AudioSegment.from_file(io.BytesIO(self._create_speech(**kwargs).content), format="wav")
            label = f"attempt {attempt + 1}/{_BAD_CLIP_RETRIES + 1} without the lead-in"
            silent_ms = _long_silence_ms(audio)
            if silent_ms >= _BAD_CLIP_SILENCE_MS:
                logger.warning("Chatterbox returned %.1fs of near-silence for %s (%s)",
                               silent_ms / 1000, chunk_id, label)
                continue
            heard = self._hear(checker, audio, chunk_id) if checker is not None else None
            params, verdict = self._verdict(audio, unit, kwargs["extra_body"], heard, chunk_id, label)
            if verdict is None:
                logger.warning("Keeping %s without its lead-in (seed=%s): no attempt could be separated",
                               chunk_id, seed)
                return audio, params, first + attempt + 1, None if "_match" in params else "unseparated context"
            rejected.append((verdict[0], verdict[1], attempt, audio, params, verdict[2]))
        if not rejected:
            raise RuntimeError(f"Chatterbox returned repeated unseparated short-quote context for {chunk_id}")
        _, _, _, audio, params, reason = min(rejected, key=lambda take: take[:3])
        logger.warning("Keeping %s without its lead-in (seed=%s, %s): no take passed its checks",
                       chunk_id, params.get("seed"), reason)
        return audio, params, first + _BAD_CLIP_RETRIES + 1, reason

    def _verdict(self, audio: AudioSegment, unit: str, params: dict, heard, chunk_id: str,
                 label: str) -> Tuple[dict, Optional[Tuple[int, float, str]]]:
        """(params, None) to keep a take; (params, (kind, badness, reason)) to try again. Rejected
        takes sort by that: a plausible length Whisper doubts (kind 0) before a surely wrong length.
        params gain the transcript match (`_match`) when the speech check heard the take."""
        reason = (("too-short ellipsis audio" if _truncated_ellipsis(audio, unit) else None)
                  or _implausible_quote_duration(audio, unit)
                  or _implausible_short_narration_duration(audio, unit))
        if reason:
            logger.warning("Chatterbox returned %.2fs of %s for %s (%s)",
                           len(audio) / 1000, reason, chunk_id, label)
            return params, (1, _duration_miss(audio, unit, reason), reason)
        score = speech_check.match(unit, heard.text) if heard else None
        if score is None:
            return params, None
        logger.debug("Speech check %s heard [%s]", chunk_id, heard.text)
        params = {**params, "_match": round(score, 2)}
        if score >= speech_check.PASS_SCORE:
            return params, None
        logger.warning("Speech check %s: transcript matches %.2f (%s)", chunk_id, score, label)
        return params, (0, -score, "speech mismatch")

    @staticmethod
    def _hear(checker, audio: AudioSegment, chunk_id: str, **options) -> Optional["speech_check.Heard"]:
        """The checker's transcript of a take (`options` go to transcribe), or None if it failed."""
        try:
            return checker.transcribe(audio, **options)
        except Exception as error:  # a checker failure costs the check, never the chapter
            logger.warning("Speech check skipped for %s: %s", chunk_id, error)
            return None

    def _finish_units(self, pieces: List[bytes], spoken: List[Tuple[int, str]],
                      audio_format: Tuple[int, int, int]) -> None:
        """Last touches on each unit's audio before the chapter is joined (pauses are left alone).

        Tone matching (Chatterbox and Breeze, on unless the book turned it off): every voice is turned down
        wherever it comes out brighter than its own reference clip, measured over all of that
        voice's speech so far in this book (core/tone_match.py).

        Peak guard (lossy chapters): any unit still peaking above delivery.PEAK_GUARD_DBFS is turned
        down to it. Chatterbox hands back takes peaking near -0.45 dBFS and AAC decoding overshoots
        that: two finished 64 kb/s books decoded to +1.2 and +1.5 dBFS (2026-09-30), which a player
        decoding to 16 bits clips. Adaptive delivery already guarded its own units; now every unit is.
        """
        rate, channels, width = audio_format
        match = (self._is_chatterbox_engine() or self._is_breeze_engine()) and self.config.tone_match is not False
        guard = self.config.output_format in _LOSSY_FORMATS
        if channels != 1 or width != 2 or not (match or guard):
            return
        import numpy as np
        from audiobook_generator.core import tone_match

        takes = {index: np.frombuffer(pieces[index], dtype="<i2").astype(np.float32) / 32768
                 for index, _ in spoken}
        changed = set()
        if match:
            if self._tone is None:
                self._tone = tone_match.ToneMatcher(os.environ.get("TTS_VOICES_DIR"))
            for index, voice in spoken:
                self._tone.add(voice, takes[index])
            for voice in dict.fromkeys(voice for _, voice in spoken):
                cuts = self._tone.cuts(voice, rate)
                logger.info("Tone match %s: %s", voice,
                            "left as generated" if cuts is None else tone_match.describe(cuts, rate))
                if cuts is None or not np.any(cuts):
                    continue
                for index, unit_voice in spoken:
                    if unit_voice == voice:
                        takes[index] = tone_match.apply(takes[index], rate, cuts)
                        changed.add(index)
        if guard:
            limit = 10 ** (delivery.PEAK_GUARD_DBFS / 20)
            for index, samples in takes.items():
                peak = float(np.abs(samples).max()) if len(samples) else 0.0
                if peak > limit:
                    takes[index] = samples * (limit / peak)
                    changed.add(index)
        for index in changed:
            pieces[index] = np.clip(np.round(takes[index] * 32768), -32768, 32767).astype("<i2").tobytes()

    def _combine_and_export(self, pieces: List[bytes], audio_format: Tuple[int, int, int], speed: float,
                             output_file: str, audio_tags: AudioTags) -> None:
        """Join raw PCM pieces, apply one client-side atempo pass if speed != 1.0 (F-27), then
        export and tag the finished chapter."""
        raw_pcm = _stretch_pcm(b"".join(pieces), audio_format[0], audio_format[1], audio_format[2], speed)
        combined = AudioSegment(data=raw_pcm, frame_rate=audio_format[0],
                                channels=audio_format[1], sample_width=audio_format[2])
        export_format, codec = _PYDUB_EXPORT.get(self.config.output_format, (self.config.output_format, None))
        combined.export(output_file, format=export_format, codec=codec,
                        bitrate=LOSSY_BITRATE if self.config.output_format in _LOSSY_FORMATS else None)
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
        if self.config.paced_unit_mode not in ("sentence", "paragraph"):
            raise ValueError(f"OpenAI: Unsupported paced_unit_mode: {self.config.paced_unit_mode}")
        if self.config.voice_mode not in VOICE_MODES:
            raise ValueError(f"OpenAI: Unsupported voice_mode: {self.config.voice_mode}")

    def estimate_cost(self, total_chars):
        return math.ceil(total_chars / 1000) * self.price
