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

from audiobook_generator.core.cast import AGES, GENDERS, normalize_name, refresh_cast_counts
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
    for character in characters.values():
        character.pop("profile", None)
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


# ponytail: prose boundaries only; use EPUB document metadata if real books exceed this evidence.
_DOCUMENT_OPEN = re.compile(r"\b(?:browse|read|open|turn(?:ed)? to|start(?:ed)? reading)\b[^.?!]{0,100}\b(?:notebook|journal|diary|letter|entry)\b", re.I)
_DOCUMENT_CLOSE = re.compile(r"\b(?:close[sd]?|slam(?:med)?|finish(?:ed)?|stop(?:ped)? reading|look(?:ed)? up from)\b[^.?!]{0,100}\b(?:notebook|journal|diary|letter)\b", re.I)
_BACKMATTER = re.compile(r"^(?:acknowledg(?:e)?ments?|about the author|also by|author(?:'|’)s? note)$", re.I)


INSET_MAX_PARAGRAPHS = 8  # a framed document longer than this is more likely a stray "I open the letter"


def _narration_exclusions(chapter: ChapterText) -> set:
    """Paragraph indexes to leave out of POV evidence: backmatter, and the contents of a document
    explicitly opened and closed within INSET_MAX_PARAGRAPHS (an unclosed or over-long frame excludes nothing)."""
    texts = [" ".join(segment.text for segment in paragraph).strip() for paragraph in chapter.paragraphs]
    excluded = set()
    opened = None
    for index, text in enumerate(texts):
        if _BACKMATTER.fullmatch(text) and len(text) <= 80:
            excluded.update(range(index, len(texts)))
            break
        if opened is not None and index - opened > INSET_MAX_PARAGRAPHS:
            opened = None  # too far to be this frame's close
        if opened is None:
            if _DOCUMENT_OPEN.search(text):
                opened = index
        elif _DOCUMENT_CLOSE.search(text):
            excluded.update(range(opened + 1, index))
            opened = None
    return excluded


def narration_passages(chapters: List[ChapterText]) -> List[Passage]:
    """Eligible narrator paragraphs, excluding only documents explicitly framed and closed in text."""
    passages = []
    for c, chapter in enumerate(chapters):
        excluded = _narration_exclusions(chapter)
        passages.extend(Passage(c, p, " ".join(s.text for s in paragraph))
                        for p, paragraph in enumerate(chapter.paragraphs)
                        if p not in excluded and paragraph and all(s.kind == NARRATION for s in paragraph))
    return passages


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


def _ask_tone(characters: Dict[str, dict], chapters: List[ChapterText], chat: Chat, log: logging.Logger,
              label: str = "book") -> Optional[dict]:
    """The LLM's description of how these chapters are narrated (asked once more if the reply is
    unusable), its first-person narrator matched to a cast character as "pov_key"; None on failure."""
    passages = select_passages(narration_passages(chapters), TONE_MAX_CHARS, TONE_LEAD_PASSAGES)
    if not passages:
        return None
    # The roster's unnamed "The Narrator" placeholder is no answer to "who says I" (seen live: offered
    # it, the model picked it over the named narrator, splitting the book's narrator in two).
    named = {k: c for k, c in characters.items() if c.get("name") != ANONYMOUS_NARRATOR}
    listed = [f"{named[k].get('name', k)} ({named[k].get('gender', 'unknown')})"
              for k in profile_candidates(named, min_lines=1, limit=OTHERS_IN_PROMPT)]
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
                log.warning(f"Cast: {label} tone reply unusable ({e})" + ("; asking again" if attempt == 1 else "; skipped"))
        else:
            return None
    except Exception as e:
        log.warning(f"Cast: {label} tone not described: {e}")
        return None
    if tone["pov_character"]:
        from audiobook_generator.core.cast_llm import Roster
        tone["pov_key"] = Roster(named).resolve(tone["pov_character"])
    return tone


# ---- who says "I", chapter by chapter ----

# First-person words per 1,000 words of a chapter's narration (text outside quotation marks): the
# first-person chapters of a real book measured 80-120; third-person narration uses "I" only in the
# odd unquoted thought. A chapter with too little narration to tell follows its neighbours.
FIRST_PERSON_PER_1000 = 20
POV_MIN_NARRATION_WORDS = 150
_FIRST_PERSON = re.compile(r"\b(?:I|me|my|mine|myself)\b|\bI[’'](?:m|d|ve|ll)\b")


def chapter_point_of_view(chapter: ChapterText) -> Optional[str]:
    """"first" or "third" from how often the chapter's narration says I/me/my; None when there is
    too little narration to tell."""
    excluded = _narration_exclusions(chapter)
    narration = " ".join(s.text for i, paragraph in enumerate(chapter.paragraphs) if i not in excluded
                         for s in paragraph if s.kind == NARRATION)
    words = len(narration.split())
    if words < POV_MIN_NARRATION_WORDS:
        return None
    return "first" if len(_FIRST_PERSON.findall(narration)) * 1000 / words >= FIRST_PERSON_PER_1000 else "third"


def _narration_words(chapter: ChapterText) -> int:
    excluded = _narration_exclusions(chapter)
    return sum(len(s.text.split()) for i, paragraph in enumerate(chapter.paragraphs) if i not in excluded
               for s in paragraph if s.kind == NARRATION)


def _stories(chapters: List[ChapterText], views: List[Optional[str]]) -> List[List[int]]:
    """Runs of consecutive first-person chapters (indexes), each taken as one story with one "I". A
    chapter too short to tell joins the run around it."""
    filled = list(views)
    for i, view in enumerate(filled):  # an undecided chapter follows the one before it (else after)
        if view is None:
            filled[i] = filled[i - 1] if i else next((v for v in views if v), None)
    runs, current = [], []
    for i, view in enumerate(filled):
        if view == "first":
            current.append(i)
        elif current:
            runs.append(current)
            current = []
    return runs + ([current] if current else [])


def _voted_narrator(chapters: List[ChapterText]) -> Optional[str]:
    """The character the attribution gave these chapters' "I said" lines to, when clear: at least
    two such lines and more than half of them."""
    from collections import Counter
    from audiobook_generator.core.speech_tags import first_person_tagged
    votes = Counter(chapter.lines.get(line_id) for chapter in chapters
                    for line_id in first_person_tagged(chapter.paragraphs))
    votes.pop(None, None)
    if not votes:
        return None
    key, count = votes.most_common(1)[0]
    return key if count >= 2 and count * 2 > sum(votes.values()) else None


def _chapter_cast(chapter: ChapterText, characters: Dict[str, dict], speakers: bool = True) -> set:
    """Who is in a chapter: its speakers (unless `speakers` is False) and the characters its text
    names (never an unnamed "I")."""
    text = " ".join(s.text for paragraph in chapter.paragraphs for s in paragraph)
    present = {key for key in chapter.lines.values() if key in characters} if speakers else set()
    for key, character in characters.items():
        pattern = _mention_pattern(character)
        if pattern and pattern.search(text):
            present.add(key)
    return {key for key in present if characters[key].get("name") != ANONYMOUS_NARRATOR}


NARRATED_I_MAX = 4  # narration paragraphs of a chapter that may name its own "I" (see could_say_i)


def _named_in_narration(chapter: ChapterText, character: dict) -> int:
    """Narration paragraphs of the chapter that name this character, leaving out framed documents and
    message labels ("Ada: on my way"), where a first-person narrator's own name is written too."""
    pattern = _mention_pattern(character)
    if pattern is None:
        return 0
    excluded = _narration_exclusions(chapter)
    return sum(1 for i, paragraph in enumerate(chapter.paragraphs) if i not in excluded
               and any(s.kind == NARRATION and any(not s.text[m.end():].lstrip().startswith(":")
                                                   for m in pattern.finditer(s.text)) for s in paragraph))


def could_say_i(chapter: ChapterText, key: Optional[str], characters: Dict[str, dict]) -> bool:
    """Whether this character could be the chapter's "I": first-person narration calls its teller "I",
    not by name. Measured over 153 first-person chapters of eight books: real narrators were named in
    0-3 narration paragraphs of their chapter (125 in none: a self-introduction, a nickname); the people
    the model or the book-wide guess mistook for one (seen live: a woman he talks to, the stepsister he
    talks about) in 9-54."""
    character = characters.get(key or "")
    if character is None:
        return False
    return character.get("name") == ANONYMOUS_NARRATOR or _named_in_narration(chapter, character) <= NARRATED_I_MAX


def _same_story(chapter: ChapterText, narrator: str, theirs: List[ChapterText], characters: Dict[str, dict]) -> bool:
    """Whether a chapter is part of the story `narrator` tells in `theirs`: besides the narrator, it
    shares someone with those chapters (seen live: a collection's next story, sharing nobody with
    the last one, had been handed the last story's narrator). No one else to compare: yes."""
    mine = _chapter_cast(chapter, characters) - {narrator}
    others = set().union(*(_chapter_cast(c, characters) for c in theirs if c is not chapter)) - {narrator}
    return not mine or not others or bool(mine & others)


TELLER_MIN_ADDRESSES = 3  # times a story's "I" must be addressed by name before that names them...
TELLER_MARGIN = 2         # ...and how many times more often than anyone else the narration doesn't name


def _is_description(character: dict) -> bool:
    """An unnamed "The Narrator" or a chapter-local description ("Man", "the girl"): no one's name."""
    return character.get("name") == ANONYMOUS_NARRATOR or character.get("reference_scope") == "chapter"


def addressed_tellers(members: List[ChapterText], characters: Dict[str, dict],
                      log: logging.Logger = logger, votes: Optional[Dict[int, Optional[str]]] = None) -> Dict[int, str]:
    """{chapter number: narrator} for the stories of a first-person run whose "I" the other people
    plainly call by name: the person addressed by name most often ("..., Vic?") by anyone else, among
    those the narration never names (could_say_i), at least TELLER_MIN_ADDRESSES times and
    TELLER_MARGIN times the next; never a description. A run is split into stories where a chapter's
    text names STORY_BREAK_MIN_CAST people and none the story so far named; who the model said speaks
    doesn't count (seen live: lines of the next story went to the last one's narrator, joining them
    up). A chapter whose "I said" lines (`votes`) went to another named person is theirs (seen live: a
    novel's "Nora's PoV" chapters): it neither counts for a teller nor gets one. Measured over 13
    stories of eight books: 9 named right, 4 left alone, none wrong; looser thresholds named wrong
    people (a collection story's other lead)."""
    from collections import Counter
    votes = votes or {}

    def told_by(chapter: ChapterText) -> Optional[str]:
        vote = votes.get(chapter.number)
        return vote if vote in characters and not _is_description(characters[vote]) else None

    stories, people = [], set()
    for chapter in members:
        mine = _chapter_cast(chapter, characters, speakers=False)
        if not stories or (len(mine) >= STORY_BREAK_MIN_CAST and not mine & people):
            stories.append([])
            people = set()
        stories[-1].append(chapter)
        people |= mine
    patterns = {key: pattern for key, pattern in ((key, _address_pattern(character)) for key, character
                                                  in characters.items() if not _is_description(character)) if pattern}
    found: Dict[int, str] = {}
    for story in stories:
        def theirs(key: str) -> List[ChapterText]:  # the story's chapters this person could tell
            return [chapter for chapter in story if told_by(chapter) in (None, key)]

        counts = Counter()
        for chapter in story:
            owner = told_by(chapter)
            for paragraph in chapter.paragraphs:
                for segment in paragraph:
                    if segment.kind == DIALOGUE:
                        quote, speaker = _quote(segment.text), chapter.lines.get(segment.line_id)
                        counts.update(key for key, pattern in patterns.items()
                                      if key != speaker and owner in (None, key) and pattern.search(quote))
        ranked = [(n, key) for key, n in counts.most_common()
                  if all(could_say_i(chapter, key, characters) for chapter in theirs(key))][:2]
        if ranked and ranked[0][0] >= TELLER_MIN_ADDRESSES and \
                ranked[0][0] >= TELLER_MARGIN * (ranked[1][0] if len(ranked) > 1 else 0):
            key = ranked[0][1]
            told = theirs(key)
            found.update({chapter.number: key for chapter in told})
            log.info(f"Cast: {len(told)} chapters from {told[0].number} to {told[-1].number} told by {key}, "
                     f"addressed by name {ranked[0][0]} times (next {ranked[1][0] if len(ranked) > 1 else 0})")
    return found


def chapter_narrators(cast: dict, chapters: List[ChapterText], book_tone: Optional[dict],
                      chat: Optional[Chat] = None, log: logging.Logger = logger) -> Dict[int, dict]:
    """{chapter number: {"point_of_view", "narrator"}} for the chapters whose point of view could be
    told (a chapter too short to tell, with no neighbour to follow, is left out, so the book's own
    narrator still applies to it). "I said" lines vote for a chapter's narrator; when that vote differs
    from a single first-person run's book narrator, narration-only chapter evidence must confirm the
    alternate or the book narrator wins. A chapter with too few votes takes a neighbour's narrator in
    the same run, the nearest first, but only one who speaks in it; otherwise the LLM is asked about
    that chapter alone when available. Last resort: the single run's book narrator. Nobody becomes a
    chapter's narrator whom its narration names (could_say_i). The person a story's others plainly
    address as its "I" (addressed_tellers) comes before the book narrator, the LLM's guess from
    narration alone, and alone may name a story whose "I said" lines only ever went to an unnamed
    "I"; otherwise that story keeps its unnamed narrator."""
    views = [chapter_point_of_view(chapter) for chapter in chapters]
    stories = _stories(chapters, views)
    characters = cast.get("characters", {})
    narrators: Dict[int, dict] = {c.number: {"point_of_view": view, "narrator": None}
                                  for c, view in zip(chapters, views) if view}
    single_run_tone = (book_tone or {}).get("pov_key") if (
        len(stories) == 1 and (book_tone or {}).get("point_of_view") == "first") else None
    if single_run_tone not in characters or _is_description(characters[single_run_tone]):
        single_run_tone = None  # a key nobody in the cast has, or a label ("Man"), names no narrator
    for story in stories:
        members = [chapters[i] for i in story]
        # Each chapter votes on its own: two first-person stories can sit side by side in a
        # collection (seen live), each with its own "I".
        own = []
        for chapter in members:
            vote = _voted_narrator([chapter])
            if vote and not could_say_i(chapter, vote, characters):
                log.info(f"Cast: chapter {chapter.number} narrator vote {vote} dropped: the narration names them")
                vote = None
            own.append(vote)
        pooled = _voted_narrator(members)
        unnamed = {key for key in own if (characters.get(key) or {}).get("name") == ANONYMOUS_NARRATOR}
        named_in_run = any(key and key not in unnamed for key in own)
        tellers = addressed_tellers(members, characters, log, {c.number: key for c, key in zip(members, own)})
        for i, chapter in enumerate(members):
            def fits(key: Optional[str]) -> bool:
                return bool(key) and could_say_i(chapter, key, characters)

            teller = tellers.get(chapter.number)
            book_narrator = teller or (single_run_tone if fits(single_run_tone) else None)
            found = own[i]
            tone = None
            if teller and (found is None or _is_description(characters.get(found) or {})):
                # The others call this chapter's "I" by name: they tell it, and the lines the model gave
                # the "I" under a label are theirs (seen live: a novel's narrator labelled "Man" in most
                # chapters, read in another voice than his own name's).
                narrators[chapter.number] = {"point_of_view": "first", "narrator": teller}
                if found and found != teller:
                    narrators[chapter.number]["narrator_reference"] = found
                own[i] = None  # not lent: another story's untagged chapter must not take this teller
                continue
            if found in unnamed and named_in_run:
                # The model left this chapter's "I" unnamed while others in the run are named: which
                # story it belongs to is settled by who else is in it (_share_anonymous_narrator),
                # not by asking again -- asked, the model named whoever it was offered (seen live).
                narrators[chapter.number] = {"point_of_view": "first", "narrator": found}
                own[i] = None  # nothing to lend to an untagged chapter
                continue
            if unnamed and not named_in_run and found in (None, *unnamed):
                # The model never named this story's "I": the person the others call by name tells
                # it, or else it keeps one unnamed narrator, an untagged chapter included
                # (_share_anonymous_narrator joins them). Seen live: the book-wide guess named a
                # woman he talks to instead, and her voice read his story.
                lend = [(abs(j - i), j) for j, key in enumerate(own) if key]
                if found or lend:
                    found = found or own[min(lend)[1]]
                    narrators[chapter.number] = {"point_of_view": "first", "narrator": found}
                    continue
            if not found:
                speaking = set(chapter.lines.values())
                nearest = sorted((abs(j - i), j) for j, key in enumerate(own) if key)
                found = next((own[j] for _, j in nearest if own[j] in speaking and fits(own[j])), None)
                if not found and pooled in speaking and fits(pooled):
                    found = pooled
                if not found and book_narrator in speaking:  # the same question, already answered
                    found = book_narrator
            if not found and chat is not None:
                tone = _ask_tone(characters, [chapter], chat, log, label=f"chapter {chapter.number}")
                found = tone.get("pov_key") if tone and tone.get("point_of_view") == "first" else None
                found = found if fits(found) else None
            if found and book_narrator and found != book_narrator:
                if tone is None and chat is not None:
                    tone = _ask_tone(characters, [chapter], chat, log, label=f"chapter {chapter.number}")
                verified = tone.get("pov_key") if tone and tone.get("point_of_view") == "first" else None
                if verified != found:
                    log.info(f"Cast: chapter {chapter.number} narrator vote {found} not confirmed; "
                             f"using book narrator {book_narrator}")
                    found = book_narrator
            own[i] = found  # lend the final narrator to later untagged chapters, like a vote
            narrators[chapter.number] = {"point_of_view": "first", "narrator": found or book_narrator}
    return narrators


def _drop_unused(cast: dict, keys, entries: dict) -> None:
    """Remove characters nobody speaks as any more, unless the owner gave them a voice or design."""
    used = {speaker for entry in entries.values() for speaker in entry.get("lines", {}).values() if speaker}
    characters = cast.get("characters", {})
    for key in set(keys) - used:
        character = characters.get(key) or {}
        if not character.get("voice_picked") and (character.get("voice_design") or {}).get("status") != "done":
            characters.pop(key, None)


ANONYMOUS_NARRATOR = "The Narrator"  # cast_llm.Roster's name for a chapter's never-named "I"


TURN_SCENE = 12  # paragraphs around an exchange searched for the other person when it collapsed onto one


def _address_pattern(character: dict) -> Optional["re.Pattern"]:
    forms = name_forms(character)
    if not forms:
        return None
    f = "(?:" + "|".join(re.escape(form) for form in forms) + ")"  # as written: "Oh man!" is no address
    return re.compile(rf"(?:^|[,;—–]\s*|\b(?i:hey|oh|okay|listen|look)\s+){f}\s*[,.?!…—]|,\s*{f}\b")


def _quote(text: str) -> str:
    return text.strip("“”\"'‘’ ")


def addresses(text: str, character: dict) -> bool:
    """Whether a quotation speaks to this character by name ("..., Vic?", "Vic, ...", "Hey Vic.")."""
    pattern = _address_pattern(character)
    return bool(pattern and pattern.search(_quote(text)))


def turn_taking(paragraphs: List[List["Segment"]], lines: Dict[int, Optional[str]],
                characters: Dict[str, dict]) -> Dict[int, str]:
    """{line id: speaker} for bare quotations (a paragraph that is one quotation and nothing else) that
    break strict turn-taking between two people. Seen live: untagged exchanges shifted by one line, or
    collapsed onto one person. A run of bare quotations is fixed only when it is anchored by a tagged
    line right before it, or by one that opens the next paragraph (a new beat first is no next turn),
    the anchors at both ends fit strict alternation, nobody else speaks in it, and no line would go
    to the person it addresses by name. Measured on three labelled books: 5-8 lines fixed per run,
    none broken; long untagged interviews have no anchors and are left to the model."""
    from audiobook_generator.core.speech_tags import first_person_tagged, tagged_speakers
    anchored = set(tagged_speakers(paragraphs)) | set(first_person_tagged(paragraphs))
    certain = {i: lines[i] for i in anchored if lines.get(i)}

    def bare(paragraph) -> bool:
        return len(paragraph) == 1 and paragraph[0].kind == DIALOGUE and not paragraph[0].continues

    fixes: Dict[int, str] = {}
    n = 0
    while n < len(paragraphs):
        if not bare(paragraphs[n]):
            n += 1
            continue
        start = n
        while n < len(paragraphs) and bare(paragraphs[n]):
            n += 1
        run = [paragraphs[k][0].line_id for k in range(start, n)]
        before = [s.line_id for s in paragraphs[start - 1] if s.kind == DIALOGUE] if start else []
        after = ([s.line_id for s in paragraphs[n] if s.kind == DIALOGUE]
                 if n < len(paragraphs) and paragraphs[n][0].kind == DIALOGUE else [])
        first = certain.get(before[-1]) if before else None
        last = certain.get(after[0]) if after else None
        if not (first or last):
            continue
        anchor = first or last
        others = {lines.get(i) for i in run} - {anchor, None}
        if len(others) > 1:
            continue  # a third person: not a two-person exchange
        other = next(iter(others), None)
        if other is None:  # collapsed onto one person: the other is the nearest tagged speaker in the scene
            near = sorted((abs(k - start), s.line_id) for k in range(max(0, start - TURN_SCENE),
                                                                    min(len(paragraphs), n + TURN_SCENE))
                          for s in paragraphs[k] if s.kind == DIALOGUE and certain.get(s.line_id) not in (None, anchor))
            other = certain[near[0][1]] if near else None
        if other is None:
            continue
        if first:
            expected = [other if k % 2 == 0 else first for k in range(len(run))]
            if last and (other if len(run) % 2 == 0 else first) != last:
                continue  # the far end does not fit strict alternation: keep the model's answers
        else:
            expected = [other if (len(run) - k) % 2 == 1 else last for k in range(len(run))]
        if any(addresses(paragraphs[k][0].text, characters.get(e) or {}) for k, e in zip(range(start, n), expected)):
            continue
        fixes.update({i: e for i, e in zip(run, expected) if lines.get(i) != e})
    return fixes


STORY_BREAK_MIN_CAST = 3  # other named people a chapter needs before sharing none of them can split a story


def _new_teller(characters: Dict[str, dict], number: int) -> str:
    """A new unnamed "The Narrator" for the story that starts at this chapter."""
    key = f"narrator of chapter {number}"
    characters[key] = {"name": ANONYMOUS_NARRATOR, "aliases": [], "gender": "unknown", "age": "unknown", "lines": 0}
    return key


def _split_story_breaks(cast: dict, chapters: List[ChapterText], narrators: Dict[int, dict], entries: dict) -> set:
    """A chapter "narrated" by someone who belongs to another story gets its own unnamed narrator.
    Seen live: in a collection, the model gave the next story's "I said" lines to the previous story's
    narrator, a man who is never named in it and whose people never appear in it. The signs, all
    needed: at least STORY_BREAK_MIN_CAST other named people in the chapter, none of them in the
    narrator's other chapters, and the narrator's name nowhere in its text. Their lines there move to
    a new "The Narrator" of that chapter. Returns the keys whose lines changed."""
    characters = cast.get("characters", {})
    changed = set()
    for chapter in chapters:
        found = narrators.get(chapter.number) or {}
        key, entry = found.get("narrator"), entries.get(chapter.number)
        if found.get("point_of_view") != "first" or key not in characters or entry is None:
            continue
        if characters[key].get("name") == ANONYMOUS_NARRATOR:
            continue
        theirs = [c for c in chapters if c is not chapter and (narrators.get(c.number) or {}).get("narrator") == key]
        mine = _chapter_cast(chapter, characters) - {key}
        text = " ".join(s.text for paragraph in chapter.paragraphs for s in paragraph)
        pattern = _mention_pattern(characters[key])
        if not theirs or len(mine) < STORY_BREAK_MIN_CAST or (pattern and pattern.search(text)):
            continue
        if not _same_story(chapter, key, theirs, characters):
            new = _new_teller(characters, chapter.number)
            for lines in (chapter.lines, entry.get("lines", {})):
                for line_id, speaker in lines.items():
                    if speaker == key:
                        lines[line_id] = new
            found["narrator"] = entry["narrator"] = entry["narrator_reference"] = new
            changed |= {key, new}
    return changed


def _share_anonymous_narrator(cast: dict, chapters: List[ChapterText], narrators: Dict[int, dict],
                              entries: dict) -> set:
    """One first-person story has one "I": a chapter whose narrator the model never named (an anonymous
    "The Narrator" the roster made per chapter) takes the nearest named narrator of its run whose story
    it fits, or else the unnamed narrator of the chapters before it, so the story keeps one voice; a
    chapter that plainly starts another story (the signs _split_story_breaks needs) gets its own.
    Returns the keys whose lines changed."""
    characters = cast.get("characters", {})
    anonymous = {e["narrator_reference"] for e in entries.values() if e.get("narrator_reference")}
    anonymous |= {k for k, c in characters.items() if c.get("name") == ANONYMOUS_NARRATOR and not c.get("voice_picked")}
    runs = _stories(chapters, [(narrators.get(c.number) or {}).get("point_of_view") for c in chapters])
    changed, folded = set(), set()
    for run in runs:
        members = [(chapters[i], entries[chapters[i].number]) for i in run if chapters[i].number in entries]
        final = {}
        for chapter, entry in members:
            key = (narrators.get(chapter.number) or {}).get("narrator") or entry.get("narrator_reference")
            if key in anonymous:
                final[chapter.number] = key
        if not final:
            continue
        # A run that names its narrator anywhere is that narrator's story: an unnamed "I" chapter
        # takes the nearest named one (seen live: two chapters named Polly, seven only "I", and the
        # book's narrator split in two). Only a run that never names its "I" keeps an unnamed one.
        named = [(n, (narrators.get(c.number) or {}).get("narrator")) for n, (c, _) in enumerate(members)
                 if c.number not in final and (narrators.get(c.number) or {}).get("narrator") in characters]
        theirs = {key: [members[n][0] for n, k in named if k == key] for _, key in named}
        tellers = []  # the run's unnamed narrators that fit no named story: [key, their chapters], in order
        for n, (chapter, entry) in enumerate(members):
            old = final.get(chapter.number)
            if old is None:
                continue
            # Never someone this chapter's narration names: in a novel whose chapters alternate
            # between two first-person tellers, the other teller is named all through it.
            fitting = [item for item in named if could_say_i(chapter, item[1], characters)
                       and _same_story(chapter, item[1], theirs[item[1]], characters)]
            if fitting:
                shared = min(fitting, key=lambda item: (abs(item[0] - n), item[0] > n))[1]
            else:
                # A collection's neighbouring stories can both leave their "I" unnamed (seen live);
                # each keeps its own unnamed narrator, not one voice for both.
                teller = tellers[-1] if tellers else None
                if teller is None or (len(_chapter_cast(chapter, characters)) >= STORY_BREAK_MIN_CAST
                                      and not _same_story(chapter, teller[0], teller[1], characters)):
                    key = old if all(old != k for k, _ in tellers) else _new_teller(characters, chapter.number)
                    anonymous.add(key)
                    teller = [key, []]
                    tellers.append(teller)
                teller[1].append(chapter)
                shared = teller[0]
            if old != shared:
                for line_id, speaker in chapter.lines.items():
                    if speaker == old:
                        chapter.lines[line_id] = shared
                for line_id, speaker in entry.get("lines", {}).items():
                    if speaker == old:
                        entry["lines"][line_id] = shared
                folded.add(old)
                changed.add(shared)
            entry["narrator"] = shared
            entry["narrator_reference"] = shared if shared in anonymous else old
            narrators.setdefault(chapter.number, {"point_of_view": "first"})["narrator"] = shared
    _drop_unused(cast, folded, entries)
    return changed


def apply_chapter_narrators(cast: dict, chapters: List[ChapterText], narrators: Dict[int, dict]) -> None:
    """Keep each chapter's point of view and narrator with its attributions (cast["chapters"]), and
    let them settle the book's own: first person when most of the narration is, told by whoever
    narrates most of it. For a known narrator, first-person-tagged dialogue (and its quote continuations)
    follows them in the saved lines and character counts. The book's narrator decides the narrator
    voice's gender and fit."""
    from audiobook_generator.core.speech_tags import first_person_tagged

    entries = {entry.get("number"): entry for entry in cast.get("chapters", {}).values()}
    for entry in cast.get("chapters", {}).values():
        if entry.get("number") in narrators:
            entry.update(narrators[entry["number"]])
        else:  # undecided: no chapter-level narrator, so the book's own applies (cast.chapter_narrator)
            entry.pop("point_of_view", None)
            entry.pop("narrator", None)
    characters = cast.get("characters", {})
    changed = set()
    for chapter in chapters:
        narrator = (narrators.get(chapter.number) or {}).get("narrator")
        if not narrator:
            continue
        line_ids = set(first_person_tagged(chapter.paragraphs))
        reference = (entries.get(chapter.number) or {}).get("narrator_reference")
        if reference:
            line_ids.update(line_id for line_id, speaker in chapter.lines.items() if speaker == reference)
        for paragraph in chapter.paragraphs:
            for piece in paragraph:
                if piece.kind == DIALOGUE and piece.continues and piece.line_id - 1 in line_ids:
                    line_ids.add(piece.line_id)
        entry = entries.get(chapter.number)
        stored_lines = entry.setdefault("lines", {}) if entry is not None else None
        for line_id in line_ids:
            previous = chapter.lines.get(line_id)
            if previous == narrator:
                continue
            if previous in characters:
                character = characters[previous]
                character["lines"] = max(0, int(character.get("lines", 0)) - 1)
                changed.add(previous)
            chapter.lines[line_id] = narrator
            if stored_lines is not None:
                stored_lines[str(line_id)] = narrator
            if narrator in characters:
                characters[narrator]["lines"] = int(characters[narrator].get("lines", 0)) + 1
                changed.add(narrator)
    for entry in entries.values():
        entry["unknown"] = sum(speaker is None for speaker in entry.get("lines", {}).values())
    if "stats" in cast:
        cast["stats"]["unknown_lines"] = sum(entry["unknown"] for entry in entries.values())
    changed |= _split_story_breaks(cast, chapters, narrators, entries)
    changed |= _share_anonymous_narrator(cast, chapters, narrators, entries)
    for key in changed:
        characters.get(key, {}).pop("profile", None)
    replaced_references = {entry.get("narrator_reference") for entry in entries.values()
                           if entry.get("narrator") and entry.get("narrator") != entry.get("narrator_reference")}
    _drop_unused(cast, replaced_references, entries)
    refresh_cast_counts(cast)
    words = {c.number: _narration_words(c) for c in chapters}
    first = sum(words[n] for n, found in narrators.items() if found["point_of_view"] == "first")
    third = sum(words[n] for n, found in narrators.items() if found["point_of_view"] == "third")
    if not first and not third:
        return  # too little narration to tell: the LLM's reading stands
    tone = cast.setdefault("book_tone", {})
    if first < third:
        tone.update(point_of_view="third", pov_character="")
        tone.pop("pov_key", None)
        return
    told = {}
    for number, found in narrators.items():
        if found["narrator"]:
            told[found["narrator"]] = told.get(found["narrator"], 0) + words[number]
    tone["point_of_view"] = "first"
    if told:
        key = max(told, key=told.get)
        tone["pov_key"] = key
        tone["pov_character"] = cast.get("characters", {}).get(key, {}).get("name", key)


def describe_book(cast: dict, chapters: List[ChapterText], chat: Chat, log: logging.Logger = logger) -> None:
    """Ask the LLM how the book is narrated, from up to TONE_MAX_CHARS of its dialogue-free
    paragraphs, and keep the answer in cast["book_tone"]; then settle, chapter by chapter, whether
    it is told in the first person and by whom (chapter_narrators), which overrides the LLM's guess
    at the book's "I": narration alone often never names them. Like the profiles, this never fails
    the analysis: an LLM error leaves the tone out and is logged."""
    tone = _ask_tone(cast.get("characters", {}), chapters, chat, log)
    if tone:
        cast["book_tone"] = tone
    narrators = chapter_narrators(cast, chapters, tone, chat, log)
    apply_chapter_narrators(cast, chapters, narrators)
    if cast.get("book_tone"):
        log.info(f"Cast: book tone: {cast['book_tone']}")
    told = sorted({(found["narrator"] or "?") for found in narrators.values() if found["point_of_view"] == "first"})
    if told:
        log.info(f"Cast: first-person narrators by chapter: {told}")
