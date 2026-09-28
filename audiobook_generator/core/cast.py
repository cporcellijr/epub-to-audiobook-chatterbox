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
voice's name.
"""
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

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
    """{mood: line count} across every analysed chapter's saved moods (soft/normal/excited)."""
    counts = {"soft": 0, "normal": 0, "excited": 0}
    for chapter in cast.get("chapters", {}).values():
        for mood in (chapter.get("moods") or {}).values():
            if mood in counts:
                counts[mood] += 1
    return counts


def character_voice(cast: dict, speaker: Optional[str]) -> Optional[str]:
    """The voice chosen for a speaker key, or None (unknown speaker, or no voice picked yet)."""
    if not speaker:
        return None
    character = cast.get("characters", {}).get(speaker)
    if not isinstance(character, dict):
        return None
    return character.get("voice") or None


# ---- names ----

def normalize_name(name: str) -> str:
    """Key form of a character name: lower case, titles and punctuation dropped, single spaces.
    "Mr. Baker" -> "baker", "THOMAS  Baker" -> "thomas baker"."""
    words = re.sub(r"[^\w\s'-]", " ", (name or "").lower()).split()
    words = [w.strip("'-") for w in words]
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


def suggest_voices(cast: dict, voices: List[Tuple[str, str]], narrator_voice: Optional[str]) -> Dict[str, str]:
    """Pick a voice for every character that has none yet: {character key: voice}.

    voices are (voice, gender) pairs for the job's engine. Characters with the most lines are
    served first and get a voice nobody else has, matching their gender (a neutral voice fits
    anyone; an unknown gender takes any voice). The narrator's voice is never suggested. Only
    when every suitable voice is taken does a character share one, with the least-used voice
    first, so the main characters always sound distinct. Voices already chosen by the owner are
    kept and count as taken.
    """
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
    for key, character in ordered:
        if character.get("voice"):
            continue
        wanted = character.get("gender", "unknown")

        def fits(gender: str) -> bool:
            return wanted == "unknown" or gender == "neutral" or gender == wanted

        pool = [voice for voice, gender in candidates if fits(gender)] or [voice for voice, _ in candidates]
        # Prefer an exact gender match over a neutral voice among the unused ones.
        exact = [voice for voice in pool if dict(candidates)[voice] == wanted and use_count[voice] == 0]
        unused = exact or [voice for voice in pool if use_count[voice] == 0]
        chosen = unused[0] if unused else min(pool, key=lambda voice: (use_count[voice], pool.index(voice)))
        use_count[chosen] += 1
        suggestions[key] = chosen
    return suggestions


def carry_voice_choices(previous: Optional[dict], characters: Dict[str, dict]) -> int:
    """Give characters of a fresh analysis the voice (and a known gender) an earlier analysis of the
    same book had for the same person, so re-analysing never throws away the owner's picks.

    A character matches by key, else by exactly one earlier character sharing a name or alias with
    it; an ambiguous match is left for the owner to review. Characters that already have a voice are
    left alone. Returns how many characters got a carried voice."""
    if not previous:
        return 0
    old = {key: c for key, c in previous.get("characters", {}).items() if c.get("voice")}

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
