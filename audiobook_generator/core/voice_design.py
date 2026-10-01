"""Designed voices: Breeze TTS 2 invents a voice from a written description, and the clip it speaks
becomes an ordinary voice file in the voices folder.

A character of a cast book whose best library voice fits badly (every voice of their gender is
taken, or the nearest is in the wrong pitch band, a child with only adult voices) is marked
"pending" in the cast when the voices are suggested; the book's own process designs the pending
voices before it loads the cast (design_pending_voices), so nothing needs pressing. The owner can
also design one by hand in the cast editor, and the Voice lab designs a starter set.

A design request runs alone at about 0.2x real time (measured 2026-10-01: a ~10 s sample takes
30-45 s), so each voice is designed once and the result is kept in both casts (the book's snapshot
and the saved one). The sample's words are fixed (DESIGN_SAMPLE_TEXT), so they are also the clip's
exact transcript, which Breeze needs next to a clip it clones.
"""
import io
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

from pydub import AudioSegment

from audiobook_generator.core import breeze_client, speech_check, voice_measure, voice_transcripts
from audiobook_generator.core import cast as cast_store
from audiobook_generator.core.cast_profiles import PROFILE_MIN_LINES
from audiobook_generator.tts_providers.openai_tts_provider import _BAD_CLIP_SILENCE_MS, _long_silence_ms, _new_seed
from audiobook_generator.utils.safe_names import sanitize_display_name

logger = logging.getLogger(__name__)

# Two plain sentences with varied sounds, ~10 s spoken. It is the clip's transcript as well.
DESIGN_SAMPLE_TEXT = ("The old lighthouse stood at the edge of the harbor. Every evening the keeper climbed its "
                      "winding stairs, lit the lamp, and watched the boats come home.")
DESIGN_ATTEMPTS = 3
MIN_SAMPLE_SECONDS = 5.0
DESIGNED_SUFFIX = " (designed)"
MAX_AUTO_DESIGNS = 6        # per cast: each takes ~40 s of the book's start
MAX_DESCRIPTION_CHARS = 400

# A design must sound like who it was asked to be. Rough limits on the median pitch, set from the
# owner's library and the three designs measured on 2026-10-01 (girl 354 Hz, husky woman 265 Hz,
# medium woman 172 Hz); not tuned further. A child is judged by the child limit alone.
MALE_MAX_F0_HZ = 180
FEMALE_MIN_F0_HZ = 150
CHILD_MIN_F0_HZ = 250

PENDING, DONE, FAILED = "pending", "done", "failed"


class DesignError(RuntimeError):
    """A voice could not be designed (the reason is the message)."""


# ---- the description ----

def _article(phrase: str) -> str:
    return "an" if phrase[:1].lower() in "aeiou" else "a"


def _who(gender: str, age: str) -> str:
    noun = {"female": "woman", "male": "man"}.get(gender, "")
    if age == "child":
        return "young " + {"female": "girl", "male": "boy"}.get(gender, "child")
    if age == "elderly":
        return f"elderly {noun or 'person'}"
    if age == "adult":
        return f"adult {noun}" if noun else "adult"
    return noun


def _delivery_words(delivery: Optional[str]) -> str:
    return {"expressive": "with a lively, expressive delivery", "even": "with an even, measured delivery"}.get(
        delivery or "", "")


def describe(character: dict) -> str:
    """The voice a character's profile asks for, as a sentence or two for Breeze: "A young girl with a
    high, clear voice. Youthful, playful, sharp, with a lively, expressive delivery." Built only from
    what is known (gender, age, the profile's voice targets and voice note); "" when nothing is."""
    gender, age = character.get("gender", "unknown"), character.get("age", "unknown")
    targets = cast_store.voice_targets(character)
    qualities = [{"low": "low", "medium": "medium-pitched", "high": "high"}.get(targets["pitch"] or ""),
                 targets["quality"]]
    qualities = [q for q in qualities if q]
    sentences = []
    who = _who(gender, age)
    if who or qualities:
        subject = who or "speaker"
        sentences.append(f"{_article(subject).capitalize()} {subject}"
                         + (f" with a {', '.join(qualities)} voice" if qualities else "") + ".")
    note = ((character.get("profile") or {}).get("voice") or "").strip().strip(".;, ")
    note = ", ".join(part.strip() for part in note.replace(" and ", ", ").split(",") if part.strip())
    manner = ", ".join(p for p in (note, _delivery_words(targets["delivery"])) if p)
    if manner:
        sentences.append(f"{manner[0].upper()}{manner[1:]}.")
    return " ".join(sentences)[:MAX_DESCRIPTION_CHARS].strip()


# ---- the Voice lab's starter set ----

# (name, gender, age, description): a spread to fill a library with, not tuned by ear. The name becomes
# the file's ("Starter girl high clear (designed).wav"); a voice whose file exists is not designed again.
STARTER_VOICES: List[Tuple[str, str, str, str]] = [
    ("Starter girl high clear", "female", "child",
     "A young girl with a high, clear voice. Bright, curious and playful, with a lively, expressive delivery."),
    ("Starter boy high clear", "male", "child",
     "A young boy with a high, clear voice. Eager and cheerful, with a lively, expressive delivery."),
    ("Starter young woman high clear", "female", "adult",
     "A young woman with a high, clear voice. Bright and friendly, with a lively, expressive delivery."),
    ("Starter young woman medium clear", "female", "adult",
     "A young woman with a medium-pitched, clear voice. Warm and natural, with an even, relaxed delivery."),
    ("Starter young woman breathy", "female", "adult",
     "A young woman with a soft, breathy voice. Gentle and intimate, close to a whisper."),
    ("Starter woman medium clear", "female", "adult",
     "An adult woman with a medium-pitched, clear voice. Formal and cautious, with an even, measured delivery."),
    ("Starter woman husky", "female", "adult",
     "An adult woman with a husky, raspy voice. Tired and wry, with a slow, even delivery."),
    ("Starter woman low", "female", "adult",
     "An adult woman with a low, rich voice. Calm and commanding, with an even, measured delivery."),
    ("Starter elderly woman", "female", "elderly",
     "An elderly woman with a thin, slightly raspy voice. Kind and unhurried, with a gentle delivery."),
    ("Starter British woman", "female", "adult",
     "An adult woman with a medium-pitched, clear voice and a British accent. Precise and polite."),
    ("Starter Southern woman", "female", "adult",
     "An adult woman with a warm voice and a Southern US accent. Easygoing and friendly, with an expressive delivery."),
    ("Starter Irish woman", "female", "adult",
     "An adult woman with a medium-pitched voice and an Irish accent. Quick-witted and lively."),
    ("Starter young man medium clear", "male", "adult",
     "A young man with a medium-pitched, clear voice. Confident and easygoing, with a lively delivery."),
    ("Starter young man husky", "male", "adult",
     "A young man with a medium-pitched, husky voice. Restless and guarded, with an even delivery."),
    ("Starter man low clear", "male", "adult",
     "An adult man with a low, clear voice. Calm and authoritative, with an even, measured delivery."),
    ("Starter man medium clear", "male", "adult",
     "An adult man with a medium-pitched, clear voice. Friendly and straightforward, with a natural delivery."),
    ("Starter man husky", "male", "adult",
     "An adult man with a low, husky voice. Tired and gruff, with a slow, even delivery."),
    ("Starter man raspy", "male", "adult",
     "An adult man with a deep, raspy, gravelly voice. Menacing and dry, with a slow delivery."),
    ("Starter man breathy", "male", "adult",
     "An adult man with a soft, breathy voice. Quiet and thoughtful, almost whispering."),
    ("Starter elderly man low", "male", "elderly",
     "An elderly man with a low, husky voice. Weathered and slow, with a gravelly, warm delivery."),
    ("Starter elderly man gentle", "male", "elderly",
     "An elderly man with a medium-pitched, slightly shaky voice. Gentle and wise, with an unhurried delivery."),
    ("Starter British man", "male", "adult",
     "An adult man with a medium-pitched, clear voice and a British accent. Dry, polite and precise."),
    ("Starter Southern man", "male", "adult",
     "An adult man with a low voice and a Southern US accent. Slow, warm and easygoing."),
    ("Starter Irish man", "male", "adult",
     "An adult man with a medium-pitched voice and an Irish accent. Cheerful and lively, with an expressive delivery."),
]


# ---- who needs a designed voice ----

def is_main_character(character: dict) -> bool:
    """A character the profile pass described (it needs a few lines to be worth one)."""
    return bool((character.get("profile") or {}).get("description")) and int(character.get("lines", 0)) >= PROFILE_MIN_LINES


def poor_fit_reason(cast: dict, key: str, voice_genders: Dict[str, str], traits: Dict[str, dict]) -> str:
    """Why the voice a character has is a poor fit ("" when it is fine): the wrong gender, shared with a
    character who has more lines (or one the owner chose a voice for), or in the wrong pitch band
    (core.cast.match_cost; a voice nobody measured can't be judged, so it never counts)."""
    characters = cast["characters"]
    character = characters[key]
    voice = character.get("voice")
    if not voice:
        return ""
    wanted, have = character.get("gender"), voice_genders.get(voice)
    if wanted in ("female", "male") and have in ("female", "male") and wanted != have:
        return f"{voice} is a {have} voice"
    rank = {k: i for i, (k, _) in enumerate(cast_store.ranked_characters(cast))}
    for other_key, other in characters.items():
        if other_key != key and other.get("voice") == voice and (other.get("voice_picked") or rank[other_key] < rank[key]):
            return f"{voice} is shared with {other.get('name', other_key)}"
    if voice in traits and cast_store.match_cost(character, traits[voice]) >= cast_store.OUT_OF_BAND_COST:
        return f"{voice} is in the wrong pitch band"
    return ""


def mark_pending(cast: dict, keys: List[str], voice_genders: Dict[str, str], traits: Dict[str, dict]) -> List[str]:
    """Mark the characters among `keys` (just given a suggested voice) whose voice fits badly for a
    designed one: character["voice_design"] = {"status": "pending", "description": ...}. The suggestion
    stays as `voice`, which is what the book uses if designing fails. Never the narrating character, one
    the owner picked a voice for, or one the profile pass didn't describe; at most MAX_AUTO_DESIGNS
    pending per cast, most lines first. Returns the keys marked."""
    characters = cast["characters"]
    room = MAX_AUTO_DESIGNS - sum(1 for c in characters.values() if (c.get("voice_design") or {}).get("status") == PENDING)
    narrating = cast_store.narrating_character(cast)
    marked = []
    for key, character in cast_store.ranked_characters(cast):
        if len(marked) >= room:
            break
        if (key not in keys or key == narrating or character.get("voice_picked") or not is_main_character(character)
                or (character.get("voice_design") or {}).get("status") in (PENDING, DONE)):
            continue
        reason = poor_fit_reason(cast, key, voice_genders, traits)
        description = describe(character)
        if reason and description:
            character["voice_design"] = {"status": PENDING, "description": description}
            logger.info("Cast: %s gets a designed voice (%s): %s", character.get("name", key), reason, description)
            marked.append(key)
    return marked


# ---- designing one voice ----

def _voices_dir() -> str:
    folder = os.environ.get("TTS_VOICES_DIR")
    if not folder or not os.path.isdir(folder):
        raise DesignError("The voices folder is not mounted (TTS_VOICES_DIR).")
    return folder


def _base_name(name_hint: str) -> str:
    return sanitize_display_name(name_hint, max_length=60, fallback="Voice") + DESIGNED_SUFFIX


def file_name_for(name_hint: str) -> str:
    """The first file name a design of this name would get: "<name> (designed).wav"."""
    return f"{_base_name(name_hint)}.wav"


def unique_file_name(name_hint: str, folder: str) -> str:
    """"<name> (designed).wav" in the folder, or "<name> (designed) 2.wav" and on when taken."""
    base = _base_name(name_hint)
    name, number = f"{base}.wav", 1
    while os.path.exists(os.path.join(folder, name)):
        number += 1
        name = f"{base} {number}.wav"
    return name


def _pitch_problem(f0: float, gender: str, age: str) -> str:
    if age == "child":
        return f"pitch {f0:.0f} Hz is low for a child" if f0 < CHILD_MIN_F0_HZ else ""
    if gender == "male" and f0 > MALE_MAX_F0_HZ:
        return f"pitch {f0:.0f} Hz is high for a man"
    if gender == "female" and f0 < FEMALE_MIN_F0_HZ:
        return f"pitch {f0:.0f} Hz is low for a woman"
    return ""


def _check(audio: AudioSegment, gender: str, age: str) -> Tuple[Optional[dict], str]:
    """(measurement, "") for a usable design, else (None, why not): too short, near-silent, not the
    words asked for (Whisper, when the speech check is on), or the wrong pitch for who it is for."""
    if len(audio) < MIN_SAMPLE_SECONDS * 1000:
        return None, f"only {len(audio) / 1000:.1f} s of audio"
    silent_ms = _long_silence_ms(audio)
    if silent_ms >= _BAD_CLIP_SILENCE_MS:
        return None, f"{silent_ms / 1000:.1f} s of near-silence"
    checker = speech_check.get()
    if checker is not None:
        score = speech_check.match(DESIGN_SAMPLE_TEXT, checker.transcribe(audio).text)
        if score is not None and score < speech_check.PASS_SCORE:
            return None, f"the words were not the ones asked for (match {score:.2f})"
    try:
        measurement = voice_measure.measure_audio(_wav_bytes(audio))
    except ValueError as error:
        return None, str(error)
    problem = _pitch_problem(measurement["f0_median"], gender, age)
    return (None, problem) if problem else (measurement, "")


def _wav_bytes(audio: AudioSegment) -> bytes:
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    return buffer.getvalue()


def _save_clip(audio: AudioSegment, path: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        audio.set_channels(1).set_frame_rate(24000).set_sample_width(2).export(f, format="wav")
    os.replace(tmp, path)


def design_voice(name_hint: str, description: str, gender: str = "unknown", age: str = "unknown") -> Tuple[str, dict]:
    """Have Breeze design a voice from the description, keep the first take that passes _check (up to
    DESIGN_ATTEMPTS, each with a new seed) as a WAV in the voices folder, and record it like any voice:
    its transcript, its gender (when known) and its measurement. Returns (file name, measurement);
    raises DesignError with every attempt's reason when none passed."""
    description = " ".join((description or "").split())
    if not description:
        raise DesignError("There is no description to design a voice from.")
    if not breeze_client.configured():
        raise DesignError("Breeze is not configured (BREEZE_BASE_URL).")
    folder = _voices_dir()
    reasons, seed = [], None
    for attempt in range(1, DESIGN_ATTEMPTS + 1):
        seed = _new_seed(seed)
        item = {"id": "design", "text": DESIGN_SAMPLE_TEXT, "voice": None, "ref_text": None,
                "instruction": description, "cfg_scale": None}
        try:
            take = breeze_client.synthesize_batch([item], seed)[0]
        except Exception as error:
            raise DesignError(f"Breeze could not be reached: {error}")
        if isinstance(take, str):
            reasons.append(f"attempt {attempt}: Breeze made no audio ({take})")
            continue
        measurement, problem = _check(take, gender, age)
        if measurement is None:
            logger.warning("Voice design attempt %d/%d rejected (%s): %s", attempt, DESIGN_ATTEMPTS, problem, description)
            reasons.append(f"attempt {attempt}: {problem}")
            continue
        file_name = unique_file_name(name_hint, folder)
        path = os.path.join(folder, file_name)
        _save_clip(take, path)
        voice_transcripts.remember(file_name, DESIGN_SAMPLE_TEXT, folder)
        if gender in ("female", "male"):
            cast_store.save_voice_gender(file_name, gender)
        voice_measure.save_features(file_name, measurement, voice_measure.file_signature(path))
        logger.info("Voice designed: %s from %r (attempt %d, pitch %.0f Hz)", file_name, description, attempt,
                    measurement["f0_median"])
        return file_name, measurement
    raise DesignError("; ".join(reasons))


# ---- the book's start: design what the cast marked ----

def _design_of(character: dict) -> dict:
    return character.get("voice_design") or {}


def _record(character: dict, file_name: str, description: str) -> None:
    character["voice"] = file_name
    character["voice_design"] = {"status": DONE, "description": description, "file": file_name}


def _designed_clip_exists(character: dict) -> bool:
    folder = os.environ.get("TTS_VOICES_DIR") or ""
    return (_design_of(character).get("status") == DONE and bool(character.get("voice"))
            and os.path.isfile(os.path.join(folder, character["voice"])))


def design_pending_voices(cast_file: str, saved_path: Optional[str] = None,
                          designer: Callable[..., Tuple[str, dict]] = design_voice) -> int:
    """Design every voice the cast marked pending, in the book's own process after Breeze is loaded and
    before the provider reads the cast. The job's snapshot (cast_file) and the saved cast it came from
    (saved_path, default: the snapshot's key in the casts folder) both get the new voice, so the cast
    editor shows it and later books reuse it. A failed design leaves the matcher's suggestion as the
    voice (status "failed" and the reason are kept). A voice the saved cast already has designed (an
    earlier queued book got there first) is adopted, not designed again. Returns how many were designed."""
    cast = cast_store.load_cast(cast_file)
    if not cast:
        return 0
    narrating = cast_store.narrating_character(cast)
    pending = [(key, c) for key, c in cast_store.ranked_characters(cast)
               if _design_of(c).get("status") == PENDING and not c.get("voice_picked") and key != narrating]
    if not pending:
        return 0
    saved_path = saved_path or cast_store.cast_path(cast.get("key", ""))
    designed = 0
    for key, character in pending:
        saved = cast_store.load_cast(saved_path)
        theirs = ((saved or {}).get("characters") or {}).get(key)
        description = _design_of(character).get("description") or describe(character)
        if theirs and _designed_clip_exists(theirs) and not theirs.get("voice_picked"):
            _record(character, theirs["voice"], _design_of(theirs).get("description") or description)
        else:
            try:
                file_name, _ = designer(character.get("name", key), description,
                                        character.get("gender", "unknown"), character.get("age", "unknown"))
            except Exception as error:
                logger.warning("Could not design a voice for %s (the suggested voice %s stays): %s",
                               character.get("name", key), character.get("voice"), error)
                character["voice_design"] = {"status": FAILED, "description": description, "error": str(error)}
            else:
                _record(character, file_name, description)
                designed += 1
        cast_store.save_cast(cast_file, cast)
        if theirs and not theirs.get("voice_picked") and _design_of(theirs).get("status") != DONE:
            theirs["voice"], theirs["voice_design"] = character["voice"], dict(character["voice_design"])
            cast_store.save_cast(saved_path, saved)
    return designed
