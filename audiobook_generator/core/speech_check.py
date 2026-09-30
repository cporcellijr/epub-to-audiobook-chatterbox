"""Machine listening for short Chatterbox takes: does the take say its text?

Chatterbox sometimes invents syllables around a very short request: "Kiss me!" came out as "Isn't
it, Lee?" and "I do," as "Hello?". The takes have ordinary lengths, so the duration and silence
checks can't hear this. Whisper small (faster-whisper, CPU, int8) transcribes a take, and `match`
scores the transcript against the text. On 2026-09-30 it agreed with all 17 takes the owner had
judged by ear, and a PASS_SCORE of 0.70 caught all 25 known-bad short takes (WORKLOG §26). Whisper
medium was worse: it heard two garbled takes as the intended word.

Whisper's word times also show where a spoken lead-in ends (`carrier_bounds`).

On when SPEECH_CHECK_MODEL names a model folder (the Docker setup's default); a missing model is
downloaded there once. Unset, or when faster-whisper can't load, `get()` is None and the provider
works as it did without it.
"""
import difflib
import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from pydub import AudioSegment

logger = logging.getLogger(__name__)

MODEL_ENV = "SPEECH_CHECK_MODEL"
MODEL_SIZE = "small"
PASS_SCORE = 0.70
_PAD_MS = 200  # silence around a take: Whisper hears a clipped first or last word better


@dataclass
class Heard:
    text: str
    words: List[Tuple[str, int, int]]  # (word, start_ms, end_ms) within the take


class SpeechChecker:
    def __init__(self, model):
        self._model = model

    def transcribe(self, audio: AudioSegment) -> Heard:
        import numpy as np

        pad = AudioSegment.silent(_PAD_MS, frame_rate=16000)
        pcm = pad + audio.set_channels(1).set_frame_rate(16000).set_sample_width(2) + pad
        samples = np.frombuffer(pcm.raw_data, dtype=np.int16).astype(np.float32) / 32768
        segments, _ = self._model.transcribe(
            samples, language="en", beam_size=5, temperature=0, condition_on_previous_text=False,
            vad_filter=False, word_timestamps=True)
        segments = list(segments)
        words = [(w.word.strip(), round(w.start * 1000) - _PAD_MS, round(w.end * 1000) - _PAD_MS)
                 for s in segments for w in (s.words or [])]
        return Heard("".join(s.text for s in segments).strip(), words)


_checker: Optional[SpeechChecker] = None
_loaded = False


def get() -> Optional[SpeechChecker]:
    """The process's checker, loaded on first use; None when off or unavailable."""
    global _checker, _loaded
    if _loaded:
        return _checker
    _loaded = True
    path = os.environ.get(MODEL_ENV, "").strip()
    if not path:
        return None
    try:
        from faster_whisper import WhisperModel
        if not os.path.isfile(os.path.join(path, "model.bin")):
            from faster_whisper.utils import download_model
            logger.info("Downloading the speech-check model (Whisper %s, about 480 MB) to %s", MODEL_SIZE, path)
            download_model(MODEL_SIZE, output_dir=path)
        _checker = SpeechChecker(WhisperModel(path, device="cpu", compute_type="int8",
                                              cpu_threads=min(8, os.cpu_count() or 1)))
        logger.info("Speech check on: Whisper %s from %s", MODEL_SIZE, path)
    except Exception as error:  # a missing package, a failed download or a bad model file
        logger.warning("Speech check off: could not load Whisper from %s (%s)", path, error)
    return _checker


def _words(text: str) -> List[str]:
    # Repeated letters collapse ("Mmm", "Jeeesuss"): Whisper spells drawn-out sounds freely.
    text = re.sub(r"(.)\1+", r"\1", text.lower().replace("’", "").replace("'", ""))
    return re.findall(r"[a-z0-9]+", text)


# Hesitation sounds (after repeated letters collapse). Whisper often leaves them out or spells them its
# own way ("Mm-hmm." as "Hmm"); a garbled take comes out as real words instead ("Do not tell!").
_FILLERS = {"um", "uh", "er", "erm", "hm", "m", "mh", "mhm", "ah", "eh", "huh"}


def match(expected: str, heard: str) -> Optional[float]:
    """How closely a transcript matches the text, 0-1; None when it can't be judged (numbers, which
    Whisper may spell out; no letters; nothing heard for a filler such as "Um..."). A filler heard as
    any filler matches."""
    if re.search(r"\d", expected) or not re.search(r"[^\W\d_]", expected):
        return None
    if set(_words(expected)) <= _FILLERS and set(_words(heard)) <= _FILLERS:
        return 1.0 if _words(heard) else None
    return difflib.SequenceMatcher(None, " ".join(_words(expected)), " ".join(_words(heard))).ratio()


def leaked(expected: str, heard: str, carrier: str) -> bool:
    """Whether a cut take starts with a lead-in word the text doesn't start with: the cut came too
    early. `match` alone can't tell: "Open. Kiss me!" still scores 0.71 against "Kiss me!"."""
    heard_words = _words(heard)
    return (bool(heard_words) and heard_words[0] in _words(carrier)
            and heard_words[:1] != _words(expected)[:1])


def carrier_bounds(words: List[Tuple[str, int, int]], carrier: str) -> Optional[Tuple[int, Optional[int]]]:
    """(end of the lead-in's last word, start of the next word or None) in ms, when Whisper heard
    nearly all of the lead-in. None for the next word: nothing heard after it, as for an "Um..." that
    Whisper leaves out."""
    want = _words(carrier)
    heard = ["".join(_words(word)) for word, _, _ in words]
    blocks = difflib.SequenceMatcher(None, want, heard, autojunk=False).get_matching_blocks()
    if sum(block.size for block in blocks) < len(want) - 2:
        return None
    last = next((b.b + len(want) - 1 - b.a for b in blocks if b.size and b.a <= len(want) - 1 < b.a + b.size),
                None)
    if last is None:
        return None
    return words[last][2], (words[last + 1][1] if last + 1 < len(words) else None)
