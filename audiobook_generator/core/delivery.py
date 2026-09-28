"""Adaptive delivery: per-mood Chatterbox presets around a book's baseline sliders, the peak guard
every adaptive unit gets, and the rule-based mood cues used with no LLM involved (core.cast_llm
reuses these same cues and adds its own inheritance/LLM-merge rules for cast mode).

Moods: "soft" (whispered/quiet dialogue, read softer and quieter), "normal" (everything else, the
book's own baseline) and "excited" (shouted/urgent dialogue, read more excited and a little louder).
"""
import os
import re
import time
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import yaml
from pydub import AudioSegment

from audiobook_generator.core.dialogue import DIALOGUE, NARRATION, Segment
from audiobook_generator.core.speech_tags import has_speech_tag

MOOD_SOFT = "soft"
MOOD_NORMAL = "normal"
MOOD_EXCITED = "excited"
MOODS = (MOOD_SOFT, MOOD_NORMAL, MOOD_EXCITED)


class Baseline(NamedTuple):
    exaggeration: float
    cfg_weight: float
    temperature: float


# Approved by ear against the owner's own voice, 2026-09-28 (WORKLOG #14): the fallback baseline
# when a book has no per-book override and Chatterbox's own saved settings can't be read.
APPROVED_BASELINE = Baseline(0.73, 0.5, 0.61)

# Every adaptive unit is turned down to exactly this if its peak exceeds it after the mood's gain
# (a +2 dB excited take audibly clipped at the owner's baseline; -1 dB did not).
PEAK_GUARD_DBFS = -1.0


def preset(mood: str, baseline: Baseline) -> Tuple[float, float, float, float]:
    """(exaggeration, cfg_weight, temperature, gain_db) for one mood around a book's baseline
    sliders. The formulas keep the spacing the owner approved by ear at their own baseline
    (exaggeration 0.73, CFG 0.5, temperature 0.61): at exactly that baseline they reproduce the
    approved soft/normal/excited numbers (0.35/0.35/0.5/-6dB, baseline/0dB, 1.0/0.4/0.7/+1.5dB).
    An unrecognised mood is treated as "normal".
    """
    b_exag, b_cfg, b_temp = baseline
    if mood == MOOD_SOFT:
        return (max(0.25, round(b_exag * 0.48, 2)), max(0.2, round(b_cfg - 0.15, 2)),
                max(0.3, round(b_temp - 0.11, 2)), -6.0)
    if mood == MOOD_EXCITED:
        return (min(1.0, round(b_exag + 0.27, 2)), max(0.2, round(b_cfg - 0.1, 2)),
                min(0.9, round(b_temp + 0.09, 2)), 1.5)
    return (round(b_exag, 2), round(b_cfg, 2), round(b_temp, 2), 0.0)


def guarded_gain(audio: AudioSegment, gain_db: float) -> AudioSegment:
    """A mood's gain, but never more than takes the clip's peak to PEAK_GUARD_DBFS (a louder clip is
    turned down to it, as peak_guard does).

    Gain and guard must be one step. Measured 2026-09-28: Chatterbox returns clips already peaking
    near -0.4 dBFS, so the excited +1.5 dB applied first pushed peaks past full scale, where pydub's
    apply_gain clips the waveform flat, and turning the clip down afterwards can't undo that. A
    single shouted word, loud from end to end, came out audibly distorted (8 of 18 excited test clips
    had 20-85 samples flattened at the top; normal clips had none)."""
    peak = audio.max_dBFS
    if peak == float("-inf"):
        return audio
    return audio.apply_gain(min(gain_db, PEAK_GUARD_DBFS - peak))


def peak_guard(audio: AudioSegment) -> AudioSegment:
    """Turn a clip down to exactly PEAK_GUARD_DBFS if its peak is louder than that; silence (whose
    peak is -inf) and anything already at or under the guard are returned unchanged."""
    peak = audio.max_dBFS
    if peak == float("-inf") or peak <= PEAK_GUARD_DBFS:
        return audio
    return audio.apply_gain(PEAK_GUARD_DBFS - peak)


# ---- rule-based mood cues (no LLM) ----

_SOFT_VERBS = r"whispered|murmured|breathed|hissed|muttered|mumbled"
_SOFT_ADVERBS = r"softly|quietly|gently|under (?:his|her|their) breath|in a whisper|in a low voice"
_EXCITED_VERBS = r"shouted|yelled|screamed|shrieked|roared|bellowed|cried(?: out)?|exclaimed"
_EXCITED_ADVERBS = r"loudly|angrily|furiously"

_SOFT_CUE = re.compile(rf"\b(?:{_SOFT_VERBS}|{_SOFT_ADVERBS})\b", re.IGNORECASE)
_EXCITED_CUE = re.compile(rf"\b(?:{_EXCITED_VERBS}|{_EXCITED_ADVERBS})\b", re.IGNORECASE)
# Quote marks (straight and curly) that may trail a quotation's own closing mark before its '!'.
_TRAILING_QUOTE_CHARS = "\"'“”‘’"


def _has_soft_cue(text: str) -> bool:
    return bool(text) and bool(_SOFT_CUE.search(text))


def _has_excited_cue(text: str) -> bool:
    return bool(text) and bool(_EXCITED_CUE.search(text))


def _ends_with_exclaim(quote_text: str) -> bool:
    """True if the quotation's last sentence ends in '!', once its own trailing quote mark(s) are
    stripped ("Go now!" -> "Go now!" -> ends with '!')."""
    stripped = quote_text.rstrip()
    while stripped and stripped[-1] in _TRAILING_QUOTE_CHARS:
        stripped = stripped[:-1].rstrip()
    return stripped.endswith("!")


def mood_of(before: str, quote_text: str, after: str) -> str:
    """Rule-based mood for one dialogue line. Verb and adverb cues come only from the speech tag
    around it: the narration right before it (when that is a lead-in, i.e. ends in ',' or ':') and
    the narration right after it, never the spoken words ("I whispered it to him" is not a whisper).
    From the quotation itself only a closing '!' counts, as excited. A soft cue always wins."""
    lead_in = before if before.rstrip().endswith((",", ":")) else ""
    texts = (lead_in, after)
    if any(_has_soft_cue(t) for t in texts):
        return MOOD_SOFT
    if any(_has_excited_cue(t) for t in texts) or _ends_with_exclaim(quote_text):
        return MOOD_EXCITED
    return MOOD_NORMAL


def _cue_mood(before: str, after: str) -> Optional[str]:
    """The mood a speech tag's verb or adverb sets ("she whispered", "he shouted"), ignoring the
    quotation's own text and '!'; None when the tag carries no cue."""
    lead_in = before if before.rstrip().endswith((",", ":")) else ""
    if _has_soft_cue(lead_in) or _has_soft_cue(after):
        return MOOD_SOFT
    if _has_excited_cue(lead_in) or _has_excited_cue(after):
        return MOOD_EXCITED
    return None


def segment_moods(paragraphs: List[List[Segment]]) -> Dict[int, str]:
    """{line id: mood} from rule-based cues alone for every dialogue line of a chapter (no LLM
    involved): a continued line (Segment.continues) inherits the previous line's mood; narration is
    never included; anything else uses mood_of on the narration immediately around it and the
    quotation itself."""
    moods: Dict[int, str] = {}
    for segments in paragraphs:
        cues, untagged = set(), []
        for i, piece in enumerate(segments):
            if piece.kind != DIALOGUE:
                continue
            before = segments[i - 1].text if i > 0 and segments[i - 1].kind == NARRATION else ""
            after = segments[i + 1].text if i + 1 < len(segments) and segments[i + 1].kind == NARRATION else ""
            if piece.continues:
                # A speech continued over paragraphs keeps its mood, unless this paragraph gives a
                # clear new cue of its own ("... and run!" he shouted).
                own = mood_of(before, piece.text, after)
                moods[piece.line_id] = own if own != MOOD_NORMAL else moods.get(piece.line_id - 1, MOOD_NORMAL)
                continue
            moods[piece.line_id] = mood_of(before, piece.text, after)
            cue = _cue_mood(before, after)
            if cue:
                cues.add(cue)
            elif not has_speech_tag(before, after):
                untagged.append(piece.line_id)
        # One speaker's manner holds for the paragraph: "Tom, are you awake?" she whispered. "Don't
        # wake Mother." reads the second quotation softly too. Only quotations with no tag of their
        # own and no mood of their own borrow it, and only when the paragraph's cues agree.
        if len(cues) == 1:
            lent = next(iter(cues))
            for line_id in untagged:
                if moods[line_id] == MOOD_NORMAL:
                    moods[line_id] = lent
    return moods


# ---- the book's baseline: a config override, else Chatterbox's saved defaults, else APPROVED_BASELINE ----

# Mirrors ui.chatterbox_ui._read_generation_defaults()'s read of Chatterbox's config.yaml (one retry
# after this short a delay, since the server rewrites the file non-atomically -- F-52). Duplicated
# here, not imported, so this module (used from the TTS provider) never depends on the ui package:
# chatterbox_ui already imports the provider, so the reverse import would be circular.
SAVED_SETTINGS_RETRY_DELAY_SECONDS = 0.2


def _read_generation_defaults(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    defaults = data.get("generation_defaults")
    return defaults if isinstance(defaults, dict) else None


def saved_chatterbox_defaults(sleep: Callable[[float], None] = time.sleep) -> Baseline:
    """Chatterbox's own saved generation defaults (CHATTERBOX_CONFIG's generation_defaults), one
    retry after a short delay, else APPROVED_BASELINE when the file can't be read or a field is
    missing. `sleep` is injectable so a test never waits for real."""
    path = os.environ.get("CHATTERBOX_CONFIG")
    if not path:
        return APPROVED_BASELINE
    defaults = _read_generation_defaults(path)
    if defaults is None:
        sleep(SAVED_SETTINGS_RETRY_DELAY_SECONDS)
        defaults = _read_generation_defaults(path)
    if defaults is None:
        return APPROVED_BASELINE

    def _field(key: str, fallback: float) -> float:
        value = defaults.get(key)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else fallback

    return Baseline(_field("exaggeration", APPROVED_BASELINE.exaggeration),
                    _field("cfg_weight", APPROVED_BASELINE.cfg_weight),
                    _field("temperature", APPROVED_BASELINE.temperature))
