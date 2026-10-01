"""The cast of a book: who speaks which line, and which voice each speaker gets.

A cast is one JSON file per book under CASTS_FOLDER (inside the app's data folder, next to
queue.json), keyed by a hash of the EPUB's bytes so the same book always finds its cast again and
can be re-analysed. It holds the characters (gender, age, line counts, the chosen voice, and a
"profile" from core.cast_profiles: first line, and for main characters role, description,
relationships and a voice note), the
narrator voice, and every chapter's per-line attributions keyed by the chapter text's SHA-1 (the
same hash the chapter manifest uses), so a different chapter selection or renumbering still finds
them.

Voice gender metadata: Kokoro ids carry it in their prefix; Chatterbox voice files don't, so the
owner records it in VOICE_GENDERS_FILE from the Voice lab. Nothing here guesses a gender from a
voice's name. How a voice sounds (pitch, huskiness, liveliness) is measured by core.voice_measure;
suggest_voices matches that against each character's profile.
"""
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from audiobook_generator.core.delivery import MOODS

CASTS_FOLDER = "casts"
VOICE_GENDERS_FILE = "voice_genders.json"
CAST_VERSION = 1

GENDERS = ("female", "male", "unknown")
AGES = ("child", "adult", "elderly", "unknown")
VOICE_GENDERS = ("female", "male", "neutral")  # what a voice can be recorded as
STATUS_RUNNING, STATUS_DONE, STATUS_FAILED = "running", "done", "failed"

_KOKORO_GENDER_BY_PREFIX = {"af": "female", "am": "male", "bf": "female", "bm": "male"}
_TITLES = ("mr", "mrs", "ms", "miss", "dr", "sir", "lady", "lord", "professor", "prof", "captain",
           "uncle", "aunt", "old", "young", "master", "madam", "madame", "mister", "reverend", "father",
           "mother", "brother", "sister")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


# ---- keys and paths ----

def cast_key(input_file: str) -> str:
    """Stable key for a book: a hash of the file's bytes, so an upload and a library copy of the
    same EPUB share one cast."""
    digest = hashlib.sha1()
    with open(input_file, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def cast_path(key: str, folder: str = CASTS_FOLDER) -> str:
    return os.path.join(folder, f"{key}.json")


def text_hash(text: str) -> str:
    """The chapter text hash attributions are keyed by (identical to the chapter manifest's)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ---- persistence ----

def load_cast(path: Optional[str]) -> Optional[dict]:
    """The saved cast at path, or None when it is missing or unreadable."""
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("characters"), dict):
        return None
    data.setdefault("chapters", {})
    return data


def save_cast(path: str, cast: dict) -> None:
    """Write the cast atomically (a crash mid-write must not leave a half file the UI would read)."""
    folder = os.path.dirname(path) or "."
    os.makedirs(folder, exist_ok=True)
    handle, tmp = tempfile.mkstemp(prefix=".cast_", suffix=".tmp", dir=folder)
    with os.fdopen(handle, "w", encoding="utf-8") as f:
        json.dump(cast, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def new_cast(key: str, input_file: str, title: str, author: str, engine: str, narrator_voice: Optional[str],
             chapter_selection: List[int]) -> dict:
    return {
        "version": CAST_VERSION, "key": key,
        "book": {"input_file": input_file, "title": title, "author": author},
        "engine": engine, "narrator_voice": narrator_voice,
        "status": STATUS_RUNNING, "error": "", "created": _now(), "finished": None,
        "chapter_selection": list(chapter_selection), "chapters_total": len(chapter_selection),
        "chapters_done": 0, "profiles_total": 0, "profiles_done": 0, "profile_error": "",
        "characters": {}, "chapters": {},
        "stats": {"windows": 0, "invalid_json": 0, "invalid_after_retry": 0, "lines": 0, "unknown_lines": 0,
                  "profiles": 0, "profiles_unusable": 0, "seconds": 0.0},
    }


# ---- lookups used while narrating ----

def chapter_lines(cast: dict, chapter_hash: str) -> Optional[Dict[int, Optional[str]]]:
    """{line id: speaker key or None} for the chapter with this text hash, or None when the
    chapter was never analysed."""
    chapter = cast.get("chapters", {}).get(chapter_hash)
    if not isinstance(chapter, dict):
        return None
    return {int(line_id): speaker for line_id, speaker in chapter.get("lines", {}).items()}


def chapter_moods(cast: dict, chapter_hash: str) -> Optional[Dict[int, str]]:
    """{line id: mood} for the chapter with this text hash, or None when the chapter was never
    analysed. A chapter analysed before adaptive delivery existed has no "moods" key, so every
    line comes back "normal" rather than missing."""
    chapter = cast.get("chapters", {}).get(chapter_hash)
    if not isinstance(chapter, dict):
        return None
    return {int(line_id): mood for line_id, mood in (chapter.get("moods") or {}).items()}


def mood_counts(cast: dict) -> Dict[str, int]:
    """{mood: line count} across every analysed chapter's saved moods."""
    counts = {mood: 0 for mood in MOODS}
    for chapter in cast.get("chapters", {}).values():
        for mood in (chapter.get("moods") or {}).values():
            if mood in counts:
                counts[mood] += 1
    return counts


def merge_characters(cast: dict, source: str, target: str) -> None:
    """One person under two names (the cast editor's Merge): source's lines in every chapter, name,
    aliases and line count go to target, which keeps its own voice, delivery and profile (taking
    source's voice only when it has none). Raises ValueError for an unknown key or the same key."""
    characters = cast.get("characters", {})
    if source == target or source not in characters or target not in characters:
        raise ValueError(f"cannot merge {source!r} into {target!r}")
    gone, keep = characters.pop(source), characters[target]
    known = {normalize_name(n) for n in [keep.get("name", target), *keep.get("aliases", [])]}
    for name in [gone.get("name", source), *gone.get("aliases", [])]:
        if normalize_name(name) and normalize_name(name) not in known:
            keep.setdefault("aliases", []).append(name)
            known.add(normalize_name(name))
    keep["lines"] = keep.get("lines", 0) + gone.get("lines", 0)
    for field in ("gender", "age"):
        if keep.get(field, "unknown") == "unknown":
            keep[field] = gone.get(field, "unknown")
    if not keep.get("voice") and gone.get("voice"):
        keep["voice"] = gone["voice"]
        keep["voice_picked"] = bool(gone.get("voice_picked"))
    for chapter in cast.get("chapters", {}).values():
        lines = chapter.get("lines", {})
        for line_id, key in lines.items():
            if key == source:
                lines[line_id] = target
        if chapter.get("narrator") == source:
            chapter["narrator"] = target
    tone = cast.get("book_tone") or {}
    if tone.get("pov_key") == source:
        tone["pov_key"] = target


def character_voice(cast: dict, speaker: Optional[str]) -> Optional[str]:
    """The voice chosen for a speaker key, or None (unknown speaker, or no voice picked yet)."""
    if not speaker:
        return None
    character = cast.get("characters", {}).get(speaker)
    if not isinstance(character, dict):
        return None
    return character.get("voice") or None


# ---- names ----

def _name_words(name: str) -> List[str]:
    return [w.strip("'-") for w in re.sub(r"[^\w\s'-]", " ", (name or "").lower()).split()]


def titled_name(name: str) -> str:
    """normalize_name with its titles kept: "Mrs. Smith" -> "mrs smith". Keys a second person who
    shares a surname with a character of the other gender (Roster)."""
    return " ".join(w for w in _name_words(name) if w)


_MALE_TITLES = frozenset({"mr", "mister", "sir", "lord", "master", "father", "brother", "uncle"})
_FEMALE_TITLES = frozenset({"mrs", "ms", "miss", "lady", "madam", "madame", "mother", "sister", "aunt"})


def title_gender(name: str) -> str:
    """"male" or "female" when a name's leading titles say so ("Mr. Smith", "Aunt Ruth", "Mother"),
    else "unknown"."""
    for word in _name_words(name):
        if word in _MALE_TITLES:
            return "male"
        if word in _FEMALE_TITLES:
            return "female"
        if word not in _TITLES:
            break
    return "unknown"


def normalize_name(name: str) -> str:
    """Key form of a character name: lower case, titles and punctuation dropped, single spaces.
    "Mr. Baker" -> "baker", "THOMAS  Baker" -> "thomas baker"."""
    words = _name_words(name)
    stripped = list(words)
    while stripped and stripped[0] in _TITLES:
        stripped = stripped[1:]
    # "Mother" or "Aunt" on its own is the whole name, not a title in front of one.
    return " ".join(w for w in (stripped or words) if w)


def display_name(name: str) -> str:
    return " ".join((name or "").split())


# ---- voice gender metadata ----

def kokoro_voice_gender(voice_id: str) -> str:
    """female/male from a Kokoro id's prefix (af_, am_, bf_, bm_), else "neutral"."""
    prefix = (voice_id or "").partition("_")[0]
    return _KOKORO_GENDER_BY_PREFIX.get(prefix, "neutral")


def load_voice_genders(path: Optional[str] = None) -> Dict[str, str]:
    """The owner's Chatterbox voice -> gender mapping (empty when none saved yet)."""
    try:
        with open(path or VOICE_GENDERS_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {voice: gender for voice, gender in data.items()
            if isinstance(voice, str) and gender in VOICE_GENDERS}


def save_voice_gender(voice: str, gender: Optional[str], path: Optional[str] = None) -> Dict[str, str]:
    """Record (or with gender None, forget) a Chatterbox voice's gender; returns the mapping."""
    path = path or VOICE_GENDERS_FILE
    genders = load_voice_genders(path)
    if gender in VOICE_GENDERS:
        genders[voice] = gender
    else:
        genders.pop(voice, None)
    folder = os.path.dirname(path) or "."
    os.makedirs(folder, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(genders, f, ensure_ascii=False, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return genders


def voice_gender(engine: str, voice: str, chatterbox_genders: Optional[Dict[str, str]] = None) -> str:
    """female/male/neutral for a voice of the given engine. A Chatterbox voice with no recorded
    gender is neutral: it can be suggested for anyone."""
    if engine == "kokoro":
        return kokoro_voice_gender(voice)
    genders = chatterbox_genders if chatterbox_genders is not None else load_voice_genders()
    return genders.get(voice, "neutral")


# ---- suggestions ----

def ranked_characters(cast: dict) -> List[Tuple[str, dict]]:
    """(key, character) pairs, most lines first, then by name for a stable order."""
    return sorted(cast.get("characters", {}).items(),
                  key=lambda item: (-int(item[1].get("lines", 0)), item[1].get("name", item[0])))


# What a profile's voice targets mean against a voice's traits (core.voice_measure.voice_traits,
# each 0..1 among voices of the same gender).
PITCH_BANDS = {"low": (0.0, 1 / 3), "medium": (1 / 3, 2 / 3), "high": (2 / 3, 1.0)}
OUT_OF_BAND_COST = 2.0  # more than huskiness, liveliness and centring can add (1.1 at most)
# A voice nobody measured, for a character who wants something: any measured voice in the wanted
# pitch band beats it, and it ties with a measured voice just outside the band.
UNMEASURED_COST = OUT_OF_BAND_COST


def voice_targets(character: dict) -> Dict[str, Optional[str]]:
    """The kind of voice a character's profile asks for: {"pitch": low|medium|high,
    "quality": husky|clear, "delivery": expressive|even}, each None when it doesn't matter. Without
    a pitch target, a child wants a high voice and an elderly character a low one."""
    targets = dict((character.get("profile") or {}).get("voice_targets") or {})
    if not targets.get("pitch"):
        targets["pitch"] = {"child": "high", "elderly": "low"}.get(character.get("age", ""))
    return {key: targets.get(key) for key in ("pitch", "quality", "delivery")}


def match_cost(character: dict, traits: Optional[dict]) -> float:
    """How far a voice is from what the character wants (0 fits perfectly; 0 for every voice when
    the character wants nothing in particular). Pitch comes first: any voice in the wanted third of
    its gender's range beats any voice outside it (OUT_OF_BAND_COST is more than everything else can
    add up to), and outside it the nearest wins. Within the band, huskiness and liveliness decide,
    then closeness to the band's middle, so the most extreme voice isn't everyone's first choice.

    Measured 2026-09-28: with pitch merely weighted twice as heavily, a character asking for a
    medium, clear, even voice got a high one that was clear and even."""
    targets = voice_targets(character)
    if not any(targets.values()):
        return 0.0
    if not traits:
        return UNMEASURED_COST
    cost = 0.0
    band = PITCH_BANDS.get(targets["pitch"] or "")
    if band:
        low, high = band
        outside = max(0.0, low - traits["pitch"], traits["pitch"] - high)
        if outside > 0:
            cost += OUT_OF_BAND_COST + 2 * outside
        cost += 0.2 * abs(traits["pitch"] - (low + high) / 2)
    if targets["quality"] in ("husky", "clear"):
        cost += 0.5 * (1 - traits["husky"] if targets["quality"] == "husky" else traits["husky"])
    if targets["delivery"] in ("expressive", "even"):
        cost += 0.5 * (1 - traits["expressive"] if targets["delivery"] == "expressive" else traits["expressive"])
    return cost


# ---- delivery per character and for the narrator (adaptive delivery, Chatterbox) ----

# An "even" character reads a little flatter and an "expressive" one a little livelier than the
# book's baseline, before the line's own mood preset (core.delivery) applies: about half of what
# "excited" adds. Set by reasoning, not tuned by ear.
CHARACTER_EXAGGERATION_STEP = 0.12
# The narrator: restrained or dramatic narration moves exaggeration, slow or brisk prose moves CFG
# (lower is slower and more deliberate), around the owner's saved sliders. Also not tuned by ear.
NARRATOR_EXAGGERATION_STEP = 0.1
NARRATOR_CFG_STEP = 0.05
DELIVERIES = ("auto", "even", "book", "expressive")  # a character's delivery setting in the cast editor


def character_delivery(character: dict) -> Optional[str]:
    """"even", "expressive", or None for the book's own delivery: the owner's setting
    (character["delivery"]) when it isn't "auto", else the profile's delivery target."""
    chosen = character.get("delivery") or "auto"
    if chosen in ("even", "expressive"):
        return chosen
    if chosen == "book":
        return None
    return ((character.get("profile") or {}).get("voice_targets") or {}).get("delivery")


def exaggeration_offset(character: dict) -> float:
    """One character's offset on its own (-step even, 0 as the book, +step expressive), uncentred."""
    return {"even": -CHARACTER_EXAGGERATION_STEP, "expressive": CHARACTER_EXAGGERATION_STEP}.get(
        character_delivery(character) or "", 0.0)


def exaggeration_offsets(cast: dict) -> Dict[str, float]:
    """{character key: exaggeration offset}. A delivery the owner set is applied as it is (+/-
    CHARACTER_EXAGGERATION_STEP, or 0 for "as the book"). One from the profile is partly centred
    on the cast's line-weighted average and capped at the step. Removing half the average keeps an
    "expressive" cast a little livelier than the book without adding the full step to every line.
    A character with no delivery at all (no profile), and a first-person book's narrating
    character, stay as the book and don't count in the average."""
    characters = cast.get("characters", {})
    narrating = narrating_character(cast)  # reads in the narrator's voice and delivery: no offset, not averaged
    owner_set = {key for key, c in characters.items() if (c.get("delivery") or "auto") != "auto" and key != narrating}
    scores = {key: exaggeration_offset(c) / CHARACTER_EXAGGERATION_STEP for key, c in characters.items()}
    centred = [key for key in characters
               if key not in owner_set and key != narrating and character_delivery(characters[key])]
    weights = {key: max(1, int(characters[key].get("lines", 0))) for key in centred}
    total = sum(weights.values())
    mean = sum(scores[k] * weights[k] for k in centred) / total if total else 0.0
    offsets = {}
    for key in characters:
        if key in owner_set:
            offsets[key] = round(scores[key] * CHARACTER_EXAGGERATION_STEP, 2)
        elif key in weights:
            step = CHARACTER_EXAGGERATION_STEP * (scores[key] - mean / 2)
            offsets[key] = round(max(-CHARACTER_EXAGGERATION_STEP, min(CHARACTER_EXAGGERATION_STEP, step)), 2)
        else:
            offsets[key] = 0.0
    return offsets


def narrator_delivery(book_tone: Optional[dict], base: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """(exaggeration, cfg_weight, temperature) for the narrator: the base sliders nudged by the
    book's intensity and pace; unchanged when the book tone says nothing about them."""
    tone = book_tone or {}
    exaggeration, cfg_weight, temperature = base
    exaggeration += {"restrained": -NARRATOR_EXAGGERATION_STEP, "dramatic": NARRATOR_EXAGGERATION_STEP}.get(
        tone.get("intensity") or "", 0.0)
    cfg_weight += {"slow": -NARRATOR_CFG_STEP, "brisk": NARRATOR_CFG_STEP}.get(tone.get("pace") or "", 0.0)
    return (round(min(2.0, max(0.25, exaggeration)), 2), round(min(1.0, max(0.1, cfg_weight)), 2),
            round(temperature, 2))


def pov_character(cast: dict) -> Optional[str]:
    """The key of the character who says "I" in a first-person book (matched by
    cast_profiles.describe_book), or None."""
    tone = cast.get("book_tone") or {}
    key = tone.get("pov_key")
    return key if tone.get("point_of_view") == "first" and key in cast.get("characters", {}) else None


def narrating_character(cast: dict) -> Optional[str]:
    """The first-person narrator whose dialogue is read in the narrator's voice, as one performer
    would read the book: pov_character, unless the owner gave that character a voice of their own in
    the cast editor (voice_picked)."""
    key = pov_character(cast)
    return None if key is None or cast["characters"][key].get("voice_picked") else key


def chapter_narrator(cast: dict, chapter_hash: str) -> Optional[str]:
    """The character whose dialogue this chapter reads in the narrator's voice: the chapter's own
    first-person narrator (cast_profiles.chapter_narrators; none in a third-person chapter), or the
    book's narrating_character for a cast analysed before chapters had one. Never a character the
    owner gave a voice of their own."""
    entry = (cast.get("chapters") or {}).get(chapter_hash) or {}
    if "narrator" not in entry:
        return narrating_character(cast)
    character = cast.get("characters", {}).get(entry["narrator"] or "")
    return entry["narrator"] if character and not character.get("voice_picked") else None


def chapter_narrator_voice(cast: dict, chapter_hash: str) -> Optional[str]:
    """The voice that narrates this chapter when it isn't the book's narrator voice: in a
    first-person story told by someone other than the book's own "I" (a collection's other
    stories), the teller's own character voice reads the whole story, narration and lines alike,
    so a man's story isn't narrated by a voice chosen for the book's women. None: the book's
    narrator voice (third person, the book's own "I", or a teller with no voice yet)."""
    key = ((cast.get("chapters") or {}).get(chapter_hash) or {}).get("narrator")
    if not key or key == pov_character(cast):
        return None
    return character_voice(cast, key)


def release_narrating_voice(cast: dict) -> bool:
    """Free a voice the narrating character still holds from an earlier suggestion (their lines use
    the narrator's voice), so another character can have it; True when one was freed."""
    key = narrating_character(cast)
    if key and cast["characters"][key].get("voice"):
        cast["characters"][key]["voice"] = None
        return True
    return False


def suggest_narrator(cast: dict, voices: List[Tuple[str, str]], traits: Dict[str, dict],
                     exclude: Tuple[str, ...] = (), current: Optional[str] = None) -> Optional[str]:
    """The measured voice that best fits the narrator the book's tone asks for (cast["book_tone"]),
    or None when there is no tone, nothing measured, or no gender/voice targets (the owner's voice
    then stays). A first-person book's narrator is its viewpoint character: their known gender, and
    their profile's voice targets where they have one, come before the tone's. When only a gender
    is asked for, the owner's `current` narrator is kept if it has that gender: every voice of the
    gender fits equally, so switching would only swap in the first one listed."""
    tone = cast.get("book_tone") or {}
    wants = tone.get("narrator") or {}
    targets = {key: wants.get(key) for key in ("pitch", "quality", "delivery")}
    gender = wants.get("gender")
    pov = cast.get("characters", {}).get(pov_character(cast) or "")
    if pov:
        own = (pov.get("profile") or {}).get("voice_targets") or {}
        targets.update({key: own[key] for key in targets if own.get(key)})
        if pov.get("gender") in ("female", "male"):
            gender = pov["gender"]
    if not traits or not any(targets.values()) and gender not in ("female", "male"):
        return None
    if not any(targets.values()) and current not in exclude and dict(voices).get(current) == gender:
        return current
    candidates = [voice for voice, voice_gender in voices if voice not in exclude and voice in traits
                  and (gender not in ("female", "male") or voice_gender in (gender, "neutral"))]
    if not candidates:
        return None
    narrator = {"profile": {"voice_targets": targets}}
    voice_genders = dict(voices)
    return min(candidates, key=lambda voice: (
        gender in ("female", "male") and voice_genders[voice] != gender,
        match_cost(narrator, traits[voice]), candidates.index(voice)))


def _match_main_voices(characters: List[Tuple[str, dict]], available: List[Tuple[str, str]],
                       traits: Dict[str, dict]) -> Dict[str, str]:
    """Find distinct voices for up to eight prominent characters together.

    The state is a bitmask of characters already assigned. Each voice is considered once, so the
    search stays small even with a large voice library. A character no available voice fits by
    gender is left out, so they can't block the others. If the whole group still cannot be matched
    by gender, try a smaller leading group and leave the rest to the usual sharing fallback.
    """
    def fits(character: dict, gender: str) -> bool:
        wanted = character.get("gender", "unknown")
        return wanted not in ("female", "male") or gender in (wanted, "neutral")

    matchable = [(key, c) for key, c in characters if any(fits(c, gender) for _, gender in available)]
    group = matchable[:min(8, len(available))]
    if len(group) < 2 or not traits or not any(any(voice_targets(c).values()) for _, c in group):
        return {}
    for size in range(len(group), 1, -1):
        selected = group[:size]
        most_lines = max(1, *(int(c.get("lines", 0)) for _, c in selected))
        weights = [1 + max(0, int(c.get("lines", 0))) / most_lines for _, c in selected]
        states = {0: (0.0, (-1,) * size)}
        for voice_index, (voice, gender) in enumerate(available):
            next_states = states.copy()  # leaving this voice unused is always allowed
            for mask, (score, choices) in states.items():
                for i, (_, character) in enumerate(selected):
                    if mask & (1 << i):
                        continue
                    if not fits(character, gender):
                        continue
                    wanted = character.get("gender", "unknown")
                    # Exact gender stays preferred; a neutral voice can free an exact match for
                    # another character when that gives the group a better result.
                    neutral_cost = 5.0 if wanted in ("female", "male") and gender == "neutral" else 0.0
                    candidate = (score + weights[i] * (neutral_cost + match_cost(character, traits.get(voice))),
                                 choices[:i] + (voice_index,) + choices[i + 1:])
                    new_mask = mask | (1 << i)
                    if new_mask not in next_states or candidate < next_states[new_mask]:
                        next_states[new_mask] = candidate
            states = next_states
        complete = states.get((1 << size) - 1)
        if complete:
            return {key: available[voice_index][0] for (key, _), voice_index in zip(selected, complete[1])}
    return {}


def suggest_voices(cast: dict, voices: List[Tuple[str, str]], narrator_voice: Optional[str],
                   traits: Optional[Dict[str, dict]] = None) -> Dict[str, str]:
    """Pick a voice for every character that has none yet: {character key: voice}.

    voices are (voice, gender) pairs for the job's engine. Characters with the most lines are
    served first and get a voice nobody else has, matching their gender (a neutral voice fits
    anyone; an unknown gender takes any voice). Among those, the voice whose measured traits
    best fit the character's profile (match_cost) wins; the prominent characters are matched as a
    group so one early choice cannot leave another with a poor fit. With no traits or no profile
    targets, the first fitting voice in the list wins. The narrator's voice is never suggested. Only when every
    suitable voice is taken does a character share one, with the least-used voice first, so the
    main characters always sound distinct. Voices already chosen by the owner are kept and count
    as taken. A first-person book's narrating character (narrating_character) gets none: their
    lines are read in the narrator's voice.
    """
    traits = traits or {}
    candidates = [(voice, gender) for voice, gender in voices if voice and voice != narrator_voice]
    if not candidates:
        return {}
    use_count = {voice: 0 for voice, _ in candidates}
    for _, character in cast.get("characters", {}).items():
        if character.get("voice") in use_count:
            use_count[character["voice"]] += 1
    suggestions: Dict[str, str] = {}
    # Characters with a known gender choose first, so a prominent character of unknown gender (who
    # can take any voice) never takes the only fitting voice from one who can't.
    ranked = ranked_characters(cast)
    ordered = ([item for item in ranked if item[1].get("gender", "unknown") != "unknown"]
               + [item for item in ranked if item[1].get("gender", "unknown") == "unknown"])
    narrating = narrating_character(cast)
    unassigned = [(key, c) for key, c in ordered if not c.get("voice") and key != narrating]
    available = [(voice, gender) for voice, gender in candidates if use_count[voice] == 0]
    suggestions.update(_match_main_voices(unassigned, available, traits))
    for voice in suggestions.values():
        use_count[voice] += 1
    voice_genders = dict(candidates)
    for key, character in ordered:
        if character.get("voice") or key == narrating or key in suggestions:
            continue
        wanted = character.get("gender", "unknown")

        def fits(gender: str) -> bool:
            return wanted == "unknown" or gender == "neutral" or gender == wanted

        pool = [voice for voice, gender in candidates if fits(gender)] or [voice for voice, _ in candidates]
        # Prefer an exact gender match over a neutral voice among the unused ones.
        exact = [voice for voice in pool if voice_genders[voice] == wanted and use_count[voice] == 0]
        unused = exact or [voice for voice in pool if use_count[voice] == 0]

        def cost(voice: str) -> tuple:
            return match_cost(character, traits.get(voice)), pool.index(voice)
        chosen = (min(unused, key=cost) if unused
                  else min(pool, key=lambda voice: (use_count[voice], *cost(voice))))
        use_count[chosen] += 1
        suggestions[key] = chosen
    return suggestions


def clear_suggested_voices(cast: dict) -> int:
    """Take away every voice the owner didn't pick (character["voice_picked"], set when a voice is
    saved in the cast editor), so suggest_voices can choose them again; returns how many."""
    cleared = 0
    for character in cast.get("characters", {}).values():
        if character.get("voice") and not character.get("voice_picked"):
            character["voice"] = None
            cleared += 1
    return cleared


def carry_voice_choices(previous: Optional[dict], characters: Dict[str, dict], picked_only: bool = False) -> int:
    """Give characters of a fresh analysis the voice (and a known gender) an earlier analysis of the
    same book had for the same person, so re-analysing never throws away the owner's picks.

    A character matches by key, else by exactly one earlier character sharing a name or alias with
    it; an ambiguous match is left for the owner to review. Characters that already have a voice are
    left alone. picked_only carries only the voices the owner saved in the cast editor
    ("voice_picked"), leaving the rest to fresh suggestions. Returns how many characters got a
    carried voice."""
    if not previous:
        return 0
    old = {key: c for key, c in previous.get("characters", {}).items()
           if c.get("voice") and (c.get("voice_picked") or not picked_only)}

    def forms(key: str, character: dict) -> set:
        names = [character.get("name", ""), *character.get("aliases", [])]
        return ({key} | {normalize_name(n) for n in names}) - {""}

    old_forms = {key: forms(key, c) for key, c in old.items()}
    carried = 0
    for key, character in characters.items():
        if character.get("voice"):
            continue
        if key in old:
            match = key
        else:
            mine = forms(key, character)
            hits = [old_key for old_key, names in old_forms.items() if names & mine]
            match = hits[0] if len(hits) == 1 else None
        if match is None:
            continue
        character["voice"] = old[match]["voice"]
        if old[match].get("voice_picked"):
            character["voice_picked"] = True
        if old[match].get("gender") in GENDERS and old[match]["gender"] != "unknown":
            character["gender"] = old[match]["gender"]
        carried += 1
    return carried


def voices_belong_to_engine(cast: dict, engine: str, known_voices: Optional[List[str]] = None) -> List[str]:
    """Voices chosen in the cast that cannot belong to the engine: a Kokoro id under Chatterbox, a
    file name under Kokoro, or (when known_voices is given) anything not in that list."""
    wrong = []
    for key, character in cast.get("characters", {}).items():
        voice = character.get("voice")
        if not voice:
            continue
        if known_voices is not None:
            if voice not in known_voices:
                wrong.append(voice)
            continue
        looks_like_file = voice.lower().endswith((".wav", ".mp3"))
        if (engine == "kokoro") == looks_like_file:
            wrong.append(voice)
    return sorted(set(wrong))


def analysis_progress(cast: Optional[dict]) -> Tuple[int, int]:
    """(chapters analysed, chapters total) for a running or finished analysis."""
    if not cast:
        return 0, 0
    return int(cast.get("chapters_done", 0)), int(cast.get("chapters_total", 0))


def profile_progress(cast: Optional[dict]) -> Tuple[int, int]:
    """(character profiles written or skipped, profiles to write); (0, 0) before the profile
    stage starts and for casts analysed before profiles existed."""
    if not cast:
        return 0, 0
    return int(cast.get("profiles_done", 0)), int(cast.get("profiles_total", 0))
