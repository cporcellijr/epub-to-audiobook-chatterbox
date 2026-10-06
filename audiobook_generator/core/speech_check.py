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


class _WhisperInfoAsDebug(logging.Filter):
    """faster-whisper logs "Processing audio with duration ..." at INFO for every take it hears, hundreds
    a chapter; its INFO lines go out as DEBUG, shown only when the app logs at DEBUG."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno != logging.INFO:
            return True
        record.levelno, record.levelname = logging.DEBUG, "DEBUG"
        return logging.getLogger().isEnabledFor(logging.DEBUG)


logging.getLogger("faster_whisper").addFilter(_WhisperInfoAsDebug())

MODEL_ENV = "SPEECH_CHECK_MODEL"
MODEL_SIZE = "small"
PASS_SCORE = 0.70
_PAD_MS = 200  # silence around a take: Whisper hears a clipped first or last word better
# Takes transcribe() can work on at once, each on _THREADS CPU threads. On the 24-thread host, 3 x 4
# heard 32 real Breeze takes in 17.3 s against 23.0 s for 1 x 8, with the same transcripts
# (2026-10-02, WORKLOG §50); Breeze's checks hear a batch this way, other callers one take at a time.
# 6 x 3 with BATCH_BEAM heard 32 short / medium / long takes in 11.5 / 13.4 / 25.1 s against 13.6 /
# 16.5 / 38.8 s for 3 x 4 with beam 5, with the same verdict on all 96 (2026-10-05, WORKLOG §53).
WORKERS = 6
_THREADS = 3
# Greedy decoding for Breeze's batch checks, which only need pass or fail; a voice clip's words, which
# Breeze then clones from, keep the default beam search.
BATCH_BEAM = 1


@dataclass
class Heard:
    text: str
    words: List[Tuple[str, int, int]]  # (word, start_ms, end_ms) within the take


class SpeechChecker:
    def __init__(self, model):
        self._model = model

    def transcribe(self, audio: AudioSegment, words: bool = True, beam_size: int = 5) -> Heard:
        """What the take says. Word times cost Whisper an extra alignment pass and only the lead-in
        cut needs them; words=False leaves Heard.words empty."""
        import numpy as np

        pad = AudioSegment.silent(_PAD_MS, frame_rate=16000)
        pcm = pad + audio.set_channels(1).set_frame_rate(16000).set_sample_width(2) + pad
        samples = np.frombuffer(pcm.raw_data, dtype=np.int16).astype(np.float32) / 32768
        segments, _ = self._model.transcribe(
            samples, language="en", beam_size=beam_size, temperature=0, condition_on_previous_text=False,
            vad_filter=False, word_timestamps=words)
        segments = list(segments)
        times = [(w.word.strip(), round(w.start * 1000) - _PAD_MS, round(w.end * 1000) - _PAD_MS)
                 for s in segments for w in (s.words or [])]
        return Heard("".join(s.text for s in segments).strip(), times)


_checker: Optional[SpeechChecker] = None
_loaded = False

# A quick first hearing for Breeze's batch checks: Whisper tiny.en hears every take, and only a take
# it doesn't pass at QUICK_PASS_SCORE goes on to Whisper small. On 472 real takes and 790 made-bad
# ones (wrong words, cut to 60%, a runaway tail of someone else's words), tiny at 0.85 sent 5% of the
# real takes on and passed none that small rejects; it hears 32 takes in 2.1-5.3 s against small's
# 11.2-25.3 s (2026-10-05, WORKLOG §55). base.en was slower and passed a runaway tail at every mark.
# The folder defaults to one beside the main model's; QUICK_MODEL_ENV=off hears everything with small.
QUICK_MODEL_ENV = "SPEECH_CHECK_QUICK_MODEL"
QUICK_MODEL_SIZE = "tiny.en"
QUICK_PASS_SCORE = 0.85
_quick: Optional[SpeechChecker] = None
_quick_loaded = False


def _load(path: str, size: str, megabytes: int, what: str) -> Optional[SpeechChecker]:
    """A checker on the Whisper model in `path`, downloaded there first if missing; None (logged) when
    it can't be loaded."""
    try:
        from faster_whisper import WhisperModel
        if not os.path.isfile(os.path.join(path, "model.bin")):
            from faster_whisper.utils import download_model
            logger.info("Downloading the %s model (Whisper %s, about %d MB) to %s", what, size, megabytes, path)
            download_model(size, output_dir=path)
        checker = SpeechChecker(WhisperModel(path, device="cpu", compute_type="int8",
                                             cpu_threads=min(_THREADS, os.cpu_count() or 1), num_workers=WORKERS))
        logger.info("%s on: Whisper %s from %s", what.capitalize(), size, path)
        return checker
    except Exception as error:  # a missing package, a failed download or a bad model file
        logger.warning("%s off: could not load Whisper from %s (%s)", what.capitalize(), path, error)
        return None


def get() -> Optional[SpeechChecker]:
    """The process's checker, loaded on first use; None when off or unavailable."""
    global _checker, _loaded
    if _loaded:
        return _checker
    _loaded = True
    path = os.environ.get(MODEL_ENV, "").strip()
    if path:
        _checker = _load(path, MODEL_SIZE, 480, "speech check")
    return _checker


def get_quick() -> Optional[SpeechChecker]:
    """The quick first-hearing checker, loaded on first use; None when the speech check is off, the
    quick hearing is switched off, or the model can't be loaded (then small hears everything)."""
    global _quick, _quick_loaded
    if _quick_loaded:
        return _quick
    _quick_loaded = True
    main = os.environ.get(MODEL_ENV, "").strip()
    path = os.environ.get(QUICK_MODEL_ENV, "").strip()
    if not main or path.lower() in ("off", "0", "false", "no"):
        return None
    path = path or os.path.join(os.path.dirname(os.path.normpath(main)), f"faster-whisper-{QUICK_MODEL_SIZE}")
    _quick = _load(path, QUICK_MODEL_SIZE, 75, "quick speech check")
    return _quick


def quick_pass(expected: str, heard: Optional[Heard]) -> bool:
    """Whether the quick hearing settles a take: a close enough transcript, or text no transcript can
    judge (match() is None whichever model heard it). A failed hearing settles nothing."""
    if heard is None:
        return False
    score = match(expected, heard.text)
    return score is None or score >= QUICK_PASS_SCORE


_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
         "sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_ORDINALS = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
             "nine": "ninth", "twelve": "twelfth"}


def _spoken(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + (f" {_ONES[n % 10]}" if n % 10 else "")
    for size, name in ((10**9, "billion"), (10**6, "million"), (1000, "thousand"), (100, "hundred")):
        if n >= size:
            return f"{_spoken(n // size)} {name}" + (f" {_spoken(n % size)}" if n % size else "")
    return str(n)


def _ordinal(words: str) -> str:
    head, _, last = words.rpartition(" ")
    last = _ORDINALS.get(last) or (last[:-1] + "ieth" if last.endswith("y") else last + "th")
    return f"{head} {last}".strip()


def _spell_numbers(text: str) -> str:
    """Digits as words: Whisper writes "3." for a spoken "Three,"; "21st" becomes "twenty first"."""
    def spell(number: re.Match) -> str:
        digits, suffix = number.group(1), number.group(2)
        if len(digits) > 12:
            return number.group(0)
        words = _spoken(int(digits))
        return _ordinal(words) if suffix else words
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)  # 2,000
    return re.sub(r"(\d+)(st|nd|rd|th)?\b", spell, text)


def _words(text: str) -> List[str]:
    # Repeated letters collapse ("Mmm", "Jeeesuss"): Whisper spells drawn-out sounds freely.
    text = re.sub(r"(.)\1+", r"\1", _spell_numbers(text).lower().replace("’", "").replace("'", ""))
    return re.findall(r"[a-z0-9]+", text)


# Hesitation sounds (after repeated letters collapse). Whisper often leaves them out or spells them its
# own way ("Mm-hmm." as "Hmm"); a garbled take comes out as real words instead ("Do not tell!").
_FILLERS = {"um", "uh", "er", "erm", "hm", "m", "mh", "mhm", "ah", "eh", "huh"}


def match(expected: str, heard: str) -> Optional[float]:
    """How closely a transcript matches the text, 0-1; None when it can't be judged (digits in the
    text, which a narrator may read several ways; no letters; nothing heard for a filler such as
    "Um..."). A filler heard as any filler matches. Digits Whisper writes are compared as words."""
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


# Whisper's spellings of a first word that is one sound (after repeated letters collapse): "Eye" for
# "I", "C." for "See!", "Ho!" for "Oh,".
_SAME_SOUND = {"eye": "i", "aye": "i", "ay": "i", "c": "se", "sea": "se", "ho": "oh", "o": "oh",
               "u": "you", "yu": "you", "r": "are", "y": "why", "b": "be"}


def clipped(expected: str, heard: str) -> bool:
    """Whether a cut take lost its first word: the cut came too late ("Grunts" for "he grunts.").
    Spelling variants and stutters still count as the word ("John" for "Jon", "Why? Yeah!" for
    "Y-- yeah!"), and a first word that is a filler isn't judged, since Whisper often drops those."""
    want, got = _words(expected)[:1], [_SAME_SOUND.get(w, w) for w in _words(heard)[:2]]
    if not want or not got or want[0] in _FILLERS:
        return False
    return not any(w.startswith(want[0]) or want[0].startswith(w)
                   or difflib.SequenceMatcher(None, want[0], w).ratio() >= 0.6 for w in got)


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
