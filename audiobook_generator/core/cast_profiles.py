"""Character profiles for the cast table: who each main character is and how they might sound,
written by the local LLM from the book's own text, to help pick their voices.

After the attribution pass (core.cast_llm) every dialogue line has a speaker, so a character's
passages can be found without guessing: paragraphs where they speak (shown to the model as
[Name] "...", as in the attribution windows) and paragraphs whose narration names them. Up to
PROFILE_MAX_CHARS of those, the first few plus an even spread over the rest, go to the model in one
request per character; it answers with a role, a short description, relationships, a casting
note for the voice and voice targets (pitch, huskiness, liveliness) that voice suggestions match
against measured voices. The first line each character speaks is quoted by code, never by the model.

KOReader's X-Ray plugins send only the title and author and rely on a large cloud model having read
the book. Nothing here depends on the model knowing the book: a local 14B model mostly doesn't, and
invents characters when asked to.
"""
import logging
import re
from typing import Callable, Dict, List, NamedTuple, Optional

from audiobook_generator.core.cast import AGES, GENDERS, normalize_name
from audiobook_generator.core.cast_llm import AttributionError, Chat, _extract_json, render_paragraph
from audiobook_generator.core.dialogue import DIALOGUE, NARRATION, Segment
from audiobook_generator.core.speech_tags import NOT_NAMES

logger = logging.getLogger(__name__)

PROFILE_MIN_LINES = 3         # characters with fewer lines get no profile (their first line is still kept)
PROFILE_MAX_CHARACTERS = 15   # the most-spoken characters profiled per book: one request each
PROFILE_MAX_CHARS = 6000      # excerpt text per request, the attribution windows' size
PROFILE_LEAD_PASSAGES = 3     # the character's first passages always go in: that's where they're introduced
OTHERS_IN_PROMPT = 15         # other characters named in the prompt, for "relationships"
FIRST_LINE_MAX_CHARS = 200
FIELD_MAX_CHARS = {"description": 400, "relationships": 300, "voice": 160}

MINOR_SHARE_OF_TOP = 0.1      # fewer lines than this share of the most-spoken character's: a minor part

ROLES = ("protagonist", "antagonist", "supporting", "minor", "unknown")
# Words a model uses for each voice target, checked in this order ("medium-high" is medium).
_TARGET_WORDS = {
    "pitch": (("low", "low"), ("deep", "low"), ("medium", "medium"), ("mid", "medium"), ("high", "high")),
    "quality": (("husky", "husky"), ("breathy", "husky"), ("rough", "husky"), ("raspy", "husky"),
                ("gravel", "husky"), ("smoky", "husky"), ("clear", "clear"), ("crisp", "clear")),
    "delivery": (("expressive", "expressive"), ("lively", "expressive"), ("animated", "expressive"),
                 ("even", "even"), ("flat", "even"), ("calm", "even"), ("monoton", "even")),
}
# Accent and origin words a voice note may only use when the excerpts do (measured 2026-09-28: the
# model gave a character "a slight southern drawl" that appears nowhere in the book).
ACCENT_WORDS = (
    "accent", "drawl", "twang", "brogue", "lilt", "burr", "dialect", "southern", "northern", "british",
    "english", "scottish", "irish", "welsh", "american", "texan", "cockney", "french", "german", "italian",
    "spanish", "russian", "australian", "canadian", "european", "foreign", "posh",
)
# Words a model uses for a role instead of the four asked for, checked in this order.
_ROLE_WORDS = (("antagonist", "antagonist"), ("villain", "antagonist"), ("protagonist", "protagonist"),
               ("main", "protagonist"), ("hero", "protagonist"), ("lead", "protagonist"),
               ("supporting", "supporting"), ("secondary", "supporting"), ("minor", "minor"))

PROMPTS = {
    "system": (
        "You write short character notes that help an audiobook producer choose a voice for each character. "
        "Use only the book excerpts you are given. "
        "Answer with a single JSON object and nothing else: no prose, no markdown fences."
    ),
    "character": (
        "Character: {name}{aka}\n"
        "Other characters in the book: {others}\n\n"
        "Excerpts from the book in reading order; [...] marks skipped text. A name in brackets before a "
        "quotation says who speaks it: [{name}] \"...\" is {name} speaking.\n\n"
        "{excerpts}\n\n"
        "Reply with exactly this shape:\n"
        "{{\"role\": \"protagonist|antagonist|supporting|minor\", \"gender\": \"female|male|unknown\", "
        "\"age\": \"child|adult|elderly|unknown\", \"description\": \"...\", \"relationships\": \"...\", "
        "\"voice\": \"...\", \"pitch\": \"low|medium|high\", \"quality\": \"husky|clear|either\", "
        "\"delivery\": \"expressive|even|either\"}}\n"
        "Rules:\n"
        "- use only what the excerpts show; if you recognise the book, add nothing from elsewhere;\n"
        "- description: one or two sentences on who {name} is and their lasting personality traits, not a "
        "list of events; mention appearance only where it bears on how they sound;\n"
        "- relationships: a few words each on how {name} relates to the characters the excerpts connect them "
        "with; leave out everyone else, and write \"\" if there is no one;\n"
        "- voice: 4 to 12 words on how {name} should sound, from their apparent age, manner and energy in "
        "these excerpts; never an accent, dialect or place of origin unless the excerpts state it;\n"
        "- pitch, quality and delivery: the kind of voice that fits {name}, pitch relative to other voices of "
        "the same gender: low for older, commanding, stern or gruff characters, high for young, light or "
        "playful ones; husky for rough, sultry or tired characters, clear for crisp, gentle or precise ones; "
        "expressive for lively or emotional characters, even for calm or dry ones;\n"
        "- write \"unknown\" or \"\" rather than guess."
    ),
    "aka": " (also called {aliases})",
    "others_none": "(none)",
}


class ChapterText(NamedTuple):
    """One analysed chapter as the profiles need it."""
    number: int
    paragraphs: List[List[Segment]]
    lines: Dict[int, Optional[str]]  # line id -> character key (None: unknown speaker)


class Passage(NamedTuple):
    chapter: int    # index into the chapters list
    paragraph: int  # paragraph index within the chapter
    text: str       # rendered with the known speakers in place


class ProfileError(ValueError):
    """The model's reply cannot be used as a profile."""


# ---- which characters, and their passages ----

def profile_candidates(characters: Dict[str, dict], min_lines: int = PROFILE_MIN_LINES,
                       limit: int = PROFILE_MAX_CHARACTERS) -> List[str]:
    """Keys of the characters worth a profile: at least min_lines lines, the most spoken first."""
    ranked = sorted(characters.items(), key=lambda kv: (-int(kv[1].get("lines", 0)), kv[1].get("name", kv[0])))
    return [key for key, character in ranked if int(character.get("lines", 0)) >= min_lines][:limit]


def name_forms(character: dict) -> List[str]:
    """The written forms narration may use for a character: the name, its aliases, and the first
    name of a full name ("Ada" for "Mrs. Ada Marsh"). Never a bare surname, which a family shares,
    nor the first word of a description ("the man", "The Doctor")."""
    name = character.get("name", "")
    forms = [name, *character.get("aliases", [])]
    words = normalize_name(name).split()
    if len(words) >= 2:
        shown = next((w for w in name.split() if w.lower().strip(".,'’") == words[0]), None)
        if shown and shown[0].isupper() and shown not in NOT_NAMES:
            forms.append(shown)
    return sorted({f.strip() for f in forms if len(f.strip()) >= 2}, key=len, reverse=True)


def _mention_pattern(character: dict) -> Optional["re.Pattern"]:
    forms = name_forms(character)
    if not forms:
        return None
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(f) for f in forms) + r")(?!\w)")


def _known_names(chapter: ChapterText, characters: Dict[str, dict]) -> Dict[int, str]:
    return {line_id: characters[key].get("name", key) for line_id, key in chapter.lines.items()
            if key and key in characters}


def character_passages(chapters: List[ChapterText], key: str, characters: Dict[str, dict]) -> List[Passage]:
    """Every paragraph in which the character speaks or the narration names them, in reading
    order, rendered with each line's known speaker as [Name]."""
    pattern = _mention_pattern(characters[key])
    passages = []
    for c, chapter in enumerate(chapters):
        known = None
        for p, paragraph in enumerate(chapter.paragraphs):
            speaks = any(s.kind == DIALOGUE and chapter.lines.get(s.line_id) == key for s in paragraph)
            named = pattern is not None and any(s.kind == NARRATION and pattern.search(s.text) for s in paragraph)
            if speaks or named:
                if known is None:
                    known = _known_names(chapter, characters)
                passages.append(Passage(c, p, render_paragraph(paragraph, False, known)))
    return passages


def _spread(n: int) -> List[int]:
    """0..n-1 in an order whose every prefix is spread evenly over the range (0, 8, 4, 2, 6, 1, ...)."""
    step = 1
    while step * 2 < n:
        step *= 2
    order, seen = [], set()
    while step >= 1:
        for i in range(0, n, step):
            if i not in seen:
                seen.add(i)
                order.append(i)
        step //= 2
    return order


def select_passages(passages: List[Passage], max_chars: int = PROFILE_MAX_CHARS,
                    lead: int = PROFILE_LEAD_PASSAGES) -> List[Passage]:
    """Up to max_chars of passages, in reading order: the first `lead` ones (introductions), then
    an even spread over the rest. A passage that doesn't fit is skipped for a shorter one; a single
    passage longer than the whole budget is cut to it."""
    order = list(range(min(lead, len(passages)))) + [lead + i for i in _spread(max(0, len(passages) - lead))]
    chosen, used = [], 0
    for index in order:
        passage = passages[index]
        if used + len(passage.text) <= max_chars:
            chosen.append(passage)
            used += len(passage.text) + 2
        elif not chosen:
            chosen.append(passage._replace(text=_clip(passage.text, max_chars)))
            used = max_chars
    return sorted(chosen, key=lambda item: (item.chapter, item.paragraph))


def render_excerpts(passages: List[Passage], chapters: List[ChapterText]) -> str:
    """The chosen passages as one text: a chapter heading where the chapter changes, [...] where
    paragraphs were skipped."""
    parts, previous = [], None
    for passage in passages:
        if previous is None or passage.chapter != previous.chapter:
            parts.append(f"(Chapter {chapters[passage.chapter].number})")
        elif passage.paragraph != previous.paragraph + 1:
            parts.append("[...]")
        parts.append(passage.text)
        previous = passage
    return "\n\n".join(parts)


def first_lines(chapters: List[ChapterText]) -> Dict[str, dict]:
    """{character key: {"chapter": number, "text": the first line they speak}}, quoted from the
    book (a quotation that runs on over several paragraphs is represented by its first one)."""
    found: Dict[str, dict] = {}
    for chapter in chapters:
        for paragraph in chapter.paragraphs:
            for segment in paragraph:
                key = chapter.lines.get(segment.line_id) if segment.kind == DIALOGUE else None
                if key and key not in found and not segment.continues:
                    found[key] = {"chapter": chapter.number, "text": _clip(segment.text, FIRST_LINE_MAX_CHARS)}
    return found


# ---- the request and its reply ----

def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    cut = text[:limit - 1].rsplit(" ", 1)[0] or text[:limit - 1]
    return cut.rstrip(",;: ") + "…"


def profile_messages(character: dict, others: List[str], excerpts: str) -> List[dict]:
    name = character.get("name", "")
    aliases = [a for a in character.get("aliases", []) if a != name]
    return [
        {"role": "system", "content": PROMPTS["system"]},
        {"role": "user", "content": PROMPTS["character"].format(
            name=name, aka=PROMPTS["aka"].format(aliases=", ".join(aliases)) if aliases else "",
            others=", ".join(others) if others else PROMPTS["others_none"], excerpts=excerpts)},
    ]


def _role(value) -> str:
    text = str(value or "").lower()
    return next((role for word, role in _ROLE_WORDS if word in text), "unknown")


def _target(value, kind: str) -> Optional[str]:
    """One voice target from the reply (pitch, quality or delivery), or None for "either",
    "unknown" or anything unrecognised."""
    text = str(value or "").lower()
    return next((target for word, target in _TARGET_WORDS[kind] if word in text), None)


def _field(data: dict, name: str) -> str:
    value = data.get(name)
    if not isinstance(value, str) or value.strip().lower().strip(".") in ("", "unknown", "none", "n/a"):
        return ""
    return _clip(value, FIELD_MAX_CHARS[name])


def parse_profile(reply: str) -> dict:
    """Validate one profile reply: {role, gender, age, description, relationships, voice,
    voice_targets: {pitch, quality, delivery}} (the targets core.cast.suggest_voices matches
    measured voices against).

    Raises ProfileError unless it is a JSON object with a usable description. A bad role, gender or
    age becomes "unknown"; a missing relationships or voice note becomes ""; a target that is
    missing, "either" or unrecognised becomes None."""
    try:
        data = _extract_json(reply)
    except AttributionError as e:
        raise ProfileError(str(e))
    profile = {
        "role": _role(data.get("role")),
        "gender": data.get("gender") if data.get("gender") in GENDERS else "unknown",
        "age": data.get("age") if data.get("age") in AGES else "unknown",
        "description": _field(data, "description"),
        "relationships": _field(data, "relationships"),
        "voice": _field(data, "voice"),
        "voice_targets": {kind: _target(data.get(kind), kind) for kind in ("pitch", "quality", "delivery")},
    }
    if not profile["description"]:
        raise ProfileError("no description in the reply")
    profile["relationships"] = "; ".join(
        part.strip() for part in profile["relationships"].split(";")
        if part.strip() and not re.search(r"\bnot (mentioned|described|shown)\b|\bno (interaction|relationship)"
                                          r"|(^|:)\s*unknown\W*$", part.strip(), re.I))
    return profile


def drop_unsupported_accents(voice: str, excerpts: str) -> str:
    """The voice note without any comma-separated clause naming an accent or origin (ACCENT_WORDS)
    that the excerpts never mention: the model adds them despite the prompt."""
    source = excerpts.lower()
    kept = [clause for clause in voice.split(",")
            if not any(re.search(rf"\b{word}\b", clause, re.I) and not re.search(rf"\b{word}\b", source)
                       for word in ACCENT_WORDS)]
    return ",".join(kept).strip(" ,;")


# ---- the pass ----

def profile_cast(cast: dict, chapters: List[ChapterText], chat: Chat, log: logging.Logger = logger,
                 save: Callable[[], None] = lambda: None) -> None:
    """Give the cast's characters a "profile": every character its first line; the most spoken
    ones (profile_candidates) a role, description, relationships and voice note from the LLM, plus
    a gender or age the attribution left unknown. Two corrections by code: a voice-note clause
    naming an accent the excerpts never mention is dropped, and a character with under
    MINOR_SHARE_OF_TOP of the most-spoken character's lines is "minor" (an antagonist stays one).

    Each character is asked once and, when the reply is unusable, once more; then it keeps just its
    first line. Any other error (the LLM unreachable, a timeout) ends the pass there and is noted in
    cast["profile_error"]: profiles are an aid, so they never fail an analysis whose attribution
    succeeded. cast["profiles_done"] / ["profiles_total"] and stats are updated as it goes, and
    save() is called after each character so the UI can show progress.
    """
    characters = cast["characters"]
    stats = cast.setdefault("stats", {})
    stats.setdefault("profiles", 0)
    stats.setdefault("profiles_unusable", 0)
    for key, line in first_lines(chapters).items():
        if key in characters:
            characters[key]["profile"] = {"first_line": line}
    keys = profile_candidates(characters, PROFILE_MIN_LINES, PROFILE_MAX_CHARACTERS)
    cast["profiles_total"], cast["profiles_done"], cast["profile_error"] = len(keys), 0, ""
    save()
    names = [characters[k].get("name", k) for k in profile_candidates(characters, min_lines=1, limit=OTHERS_IN_PROMPT + 1)]
    top_lines = max((int(c.get("lines", 0)) for c in characters.values()), default=0)
    for number, key in enumerate(keys, 1):
        character = characters[key]
        name = character.get("name", key)
        others = [n for n in names if n != name][:OTHERS_IN_PROMPT]
        passages = select_passages(character_passages(chapters, key, characters), PROFILE_MAX_CHARS,
                                   PROFILE_LEAD_PASSAGES)
        excerpts = render_excerpts(passages, chapters)
        messages = profile_messages(character, others, excerpts)
        profile = None
        try:
            for attempt in (1, 2):
                try:
                    profile = parse_profile(chat(messages))
                    break
                except ProfileError as e:
                    log.warning(f"Cast: profile {number}/{len(keys)} ({name}) reply unusable ({e})"
                                + ("; asking again" if attempt == 1 else "; skipped"))
        except Exception as e:
            cast["profile_error"] = str(e) or type(e).__name__
            log.warning(f"Cast: character profiles stopped at {number}/{len(keys)} ({name}): {cast['profile_error']}")
            save()
            return
        if profile is None:
            stats["profiles_unusable"] += 1
        else:
            gender, age = profile.pop("gender"), profile.pop("age")
            profile["voice"] = drop_unsupported_accents(profile["voice"], excerpts)
            # The model sees one character's passages, so it can't judge how big a part is.
            if (profile["role"] in ("protagonist", "supporting", "unknown")
                    and int(character.get("lines", 0)) < MINOR_SHARE_OF_TOP * top_lines):
                profile["role"] = "minor"
            if character.get("gender", "unknown") == "unknown" and gender != "unknown":
                character["gender"] = gender
            if character.get("age", "unknown") == "unknown" and age != "unknown":
                character["age"] = age
            character["profile"] = {**character.get("profile", {}), **profile}
            stats["profiles"] += 1
        cast["profiles_done"] = number
        save()
        log.info(f"Cast: profile {number}/{len(keys)} ({name}) {'done' if profile else 'skipped'}")


# ---- the book's tone, for the narrator ----

TONE_PROMPTS = {
    "system": (
        "You describe how a novel is narrated, to help an audiobook producer choose its narrator. Use only "
        "the excerpts you are given. Answer with a single JSON object and nothing else: no prose, no markdown fences."
    ),
    "book": (
        "Main characters: {characters}\n\n"
        "Narration from the novel in reading order; [...] marks skipped text.\n\n"
        "{excerpts}\n\n"
        "Reply with exactly this shape:\n"
        "{{\"point_of_view\": \"first|third|second\", \"pov_character\": \"...\", \"tone\": \"...\", "
        "\"pace\": \"slow|measured|brisk\", \"intensity\": \"restrained|moderate|dramatic\", "
        "\"narrator_gender\": \"female|male|either\", \"narrator_pitch\": \"low|medium|high\", "
        "\"narrator_quality\": \"husky|clear|either\", \"narrator_delivery\": \"expressive|even|either\"}}\n"
        "Rules:\n"
        "- use only what the excerpts show; if you recognise the book, add nothing from elsewhere;\n"
        "- pov_character: in a first-person book, the name of the character who says \"I\" (as written in the "
        "list above when they are in it); otherwise \"\";\n"
        "- tone: 3 to 6 words on the mood of the writing;\n"
        "- pace: how quickly the prose moves; intensity: how emotionally heightened the narration is;\n"
        "- narrator_gender: in a first-person book, the gender of the character who says \"I\"; otherwise "
        "\"either\" unless the book clearly calls for one;\n"
        "- narrator_pitch, narrator_quality and narrator_delivery: the narrator voice that suits the book, "
        "pitch relative to other voices of the same gender: low for dark, serious or weighty books, high for "
        "light or youthful ones; husky for gritty or sensual books, clear for crisp or gentle ones; expressive "
        "for dramatic or comic books, even for calm or literary ones;\n"
        "- write \"unknown\" or \"\" rather than guess."
    ),
}
TONE_MAX_CHARS = PROFILE_MAX_CHARS
TONE_LEAD_PASSAGES = 2
_TONE_WORDS = {
    "point_of_view": (("first", "first"), ("second", "second"), ("third", "third")),
    "pace": (("slow", "slow"), ("measured", "measured"), ("moderate", "measured"), ("brisk", "brisk"),
             ("fast", "brisk"), ("quick", "brisk")),
    "intensity": (("restrained", "restrained"), ("subdued", "restrained"), ("moderate", "moderate"),
                  ("dramatic", "dramatic"), ("intense", "dramatic")),
    "narrator_gender": (("female", "female"), ("woman", "female"), ("male", "male"), ("man", "male")),
}


def narration_passages(chapters: List[ChapterText]) -> List[Passage]:
    """Every paragraph without dialogue, in reading order: the narrator's own voice."""
    return [Passage(c, p, " ".join(s.text for s in paragraph))
            for c, chapter in enumerate(chapters) for p, paragraph in enumerate(chapter.paragraphs)
            if paragraph and all(s.kind == NARRATION for s in paragraph)]


def _word(value, kind: str) -> Optional[str]:
    text = str(value or "").lower()
    return next((word for key, word in _TONE_WORDS[kind] if key in text), None)


def parse_tone(reply: str) -> dict:
    """Validate the book-tone reply: {point_of_view, pov_character, tone, pace, intensity,
    narrator: {gender, pitch, quality, delivery}}; unrecognised values become None (or "" for the
    free text). Raises ProfileError unless it is a JSON object with a tone."""
    try:
        data = _extract_json(reply)
    except AttributionError as e:
        raise ProfileError(str(e))
    tone = data.get("tone")
    if not isinstance(tone, str) or tone.strip().lower() in ("", "unknown"):
        raise ProfileError("no tone in the reply")
    pov_character = data.get("pov_character") if isinstance(data.get("pov_character"), str) else ""
    point_of_view = _word(data.get("point_of_view"), "point_of_view")
    return {
        "point_of_view": point_of_view,
        "pov_character": _clip(pov_character, 60) if point_of_view == "first"
        and pov_character.strip().lower() not in ("", "unknown", "none") else "",
        "tone": _clip(tone, 80),
        "pace": _word(data.get("pace"), "pace"),
        "intensity": _word(data.get("intensity"), "intensity"),
        "narrator": {"gender": _word(data.get("narrator_gender"), "narrator_gender"),
                     "pitch": _target(data.get("narrator_pitch"), "pitch"),
                     "quality": _target(data.get("narrator_quality"), "quality"),
                     "delivery": _target(data.get("narrator_delivery"), "delivery")},
    }


def describe_book(cast: dict, chapters: List[ChapterText], chat: Chat, log: logging.Logger = logger) -> None:
    """Ask the LLM once (and once more if the reply is unusable) how the book is narrated, from up to
    TONE_MAX_CHARS of its dialogue-free paragraphs, and keep the answer in cast["book_tone"]. A
    first-person narrator's name is matched to a cast character (cast["book_tone"]["pov_key"]).
    Like the profiles, this never fails the analysis: any error leaves no book tone and is logged."""
    passages = select_passages(narration_passages(chapters), TONE_MAX_CHARS, TONE_LEAD_PASSAGES)
    if not passages:
        return
    characters = cast.get("characters", {})
    listed = [f"{characters[k].get('name', k)} ({characters[k].get('gender', 'unknown')})"
              for k in profile_candidates(characters, min_lines=1, limit=OTHERS_IN_PROMPT)]
    messages = [{"role": "system", "content": TONE_PROMPTS["system"]},
                {"role": "user", "content": TONE_PROMPTS["book"].format(
                    characters=", ".join(listed) or PROMPTS["others_none"],
                    excerpts=render_excerpts(passages, chapters))}]
    try:
        for attempt in (1, 2):
            try:
                tone = parse_tone(chat(messages))
                break
            except ProfileError as e:
                log.warning(f"Cast: book tone reply unusable ({e})" + ("; asking again" if attempt == 1 else "; skipped"))
        else:
            return
    except Exception as e:
        log.warning(f"Cast: book tone not described: {e}")
        return
    if tone["pov_character"]:
        from audiobook_generator.core.cast_llm import Roster
        tone["pov_key"] = Roster(characters).resolve(tone["pov_character"])
    cast["book_tone"] = tone
    log.info(f"Cast: book tone: {tone}")
