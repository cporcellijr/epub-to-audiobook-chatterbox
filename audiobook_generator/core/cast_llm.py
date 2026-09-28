"""Speaker attribution with a local OpenAI-compatible chat model.

A chapter's dialogue lines (from core.dialogue) are sent in windows: WINDOW_LINES numbered lines
with the narration around them and the preceding paragraphs for context, plus the running list of
characters already met. Lines a speech tag already names (core.speech_tags: "said Tom") are not
asked; they, and every line decided in earlier windows, are shown to the model as [Name] "..." so
it can follow the turn-taking of the untagged lines. The model answers with JSON mapping each line id to a speaker and
listing new characters (gender, rough age, aliases). Every reply is validated strictly; a window
whose reply is unusable is asked once more, then its lines are left unknown (they get the
dialogue voice).

Everything the model is told lives in PROMPTS so it can be tuned in one place. Nothing in here
touches the network except ChatClient, which talks to LLM_BASE_URL only.
"""
import json
import logging
import os
import re
import time
from typing import Callable, Collection, Dict, List, NamedTuple, Optional, Tuple

from audiobook_generator.core.cast import AGES, GENDERS, display_name, normalize_name
from audiobook_generator.core.dialogue import DIALOGUE, Segment
from audiobook_generator.core.speech_tags import tagged_speakers

logger = logging.getLogger(__name__)

WINDOW_LINES = 20          # dialogue lines asked about per request
WINDOW_MAX_CHARS = 6000    # passage text per request (about 1,500 tokens): a 7-9B model's comfort zone
CONTEXT_PARAGRAPHS = 6     # paragraphs repeated before a window, with their known speakers, for context
LLM_TIMEOUT_SECONDS = 300  # one request; a small local model on a busy GPU can be slow
LLM_TEMPERATURE = 0.0

UNKNOWN_SPEAKER_WORDS = frozenset({"", "unknown", "narrator", "none", "n/a", "?", "nobody", "unclear"})

PROMPTS = {
    "system": (
        "You identify who speaks each line of dialogue in a passage from a novel. "
        "Answer with a single JSON object and nothing else: no prose, no markdown fences."
    ),
    "window": (
        "Known characters so far (use these exact names when the speaker is one of them):\n"
        "{roster}\n\n"
        "Passage. Lines to attribute are marked like [#N] in front of the quotation, N being the line's id. "
        "Lines shown as [Name] in front of the quotation already have a known speaker: use them to follow "
        "who is talking, but do not answer for them. Quotations without a mark need no answer:\n\n"
        "{passage}\n\n"
        "Reply with exactly this shape, one entry per id:\n"
        "{{\"speakers\": {{\"N\": \"Full Name\"}}, "
        "\"characters\": [{{\"name\": \"Full Name\", \"gender\": \"female|male|unknown\", "
        "\"age\": \"child|adult|elderly|unknown\", \"aliases\": [\"other names used for this person\"]}}]}}\n"
        "Rules:\n"
        "- give every marked id exactly one speaker, using the ids from the passage and no others;\n"
        "- write \"unknown\" only when the passage gives no clue who is speaking;\n"
        "- use a person's most complete name; a title alone (Mr. Baker) or a first name is an alias of "
        "the same person, not a new character;\n"
        "- \"characters\" lists only speakers who are not in the known list, plus known characters whose "
        "gender or age the passage now reveals.\n"
        "Ids to answer: {ids}"
    ),
    "roster_empty": "(none yet)",
}


# ---- settings (read per call, never at import) ----

def llm_base_url() -> str:
    return os.environ.get("LLM_BASE_URL", "").strip().rstrip("/")


def llm_model() -> str:
    return os.environ.get("LLM_MODEL", "").strip()


def llm_api_key() -> str:
    return os.environ.get("LLM_API_KEY", "").strip()


def llm_configured() -> bool:
    """Cast mode is offered only when a chat endpoint is configured."""
    return bool(llm_base_url())


# ---- the chat endpoint ----

Chat = Callable[[List[dict]], str]


class ChatClient:
    """One local OpenAI-compatible chat model. Uses the endpoint's JSON mode (response_format) until
    the server rejects it, then goes on without it and relies on parse_reply's tolerance."""

    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = LLM_TIMEOUT_SECONDS):
        from openai import OpenAI  # imported here so tests of the pure parts never build a client
        if not base_url or not model:
            raise ValueError("LLM_BASE_URL and LLM_MODEL must both be set for cast analysis.")
        self.client = OpenAI(base_url=base_url, api_key=api_key or "not-needed", timeout=timeout, max_retries=1)
        self.model = model
        self.json_mode = True

    def __call__(self, messages: List[dict]) -> str:
        from openai import BadRequestError
        kwargs = dict(model=self.model, messages=messages, temperature=LLM_TEMPERATURE)
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = self.client.chat.completions.create(**kwargs)
        except BadRequestError as e:
            if not self.json_mode:
                raise
            logger.warning(f"Cast: the chat endpoint refused JSON mode ({e}); continuing without response_format")
            self.json_mode = False
            kwargs.pop("response_format", None)
            response = self.client.chat.completions.create(**kwargs)
        return (response.choices[0].message.content or "") if response.choices else ""


# ---- windows ----

class Window(NamedTuple):
    ids: List[int]          # line ids to attribute, in order
    passage: str            # rendered text: context paragraphs, then the window's paragraphs
    continued: List[int]    # line ids in this window that continue the previous line (not asked)
    anchored: List[int] = []  # line ids in this window whose speaker a speech tag names (not asked)
    context_start: int = 0  # paragraph index where the context begins
    start: int = 0          # paragraph index of the window's first paragraph
    end: int = 0            # paragraph index after the window's last paragraph


def render_paragraph(segments: List[Segment], ask_ids: bool, known: Optional[Dict[int, str]] = None) -> str:
    """One paragraph as the model sees it: dialogue whose speaker is already known shown as
    [Name], dialogue whose speaker is wanted marked [#id]."""
    parts = []
    for piece in segments:
        if piece.kind == DIALOGUE and known and piece.line_id in known:
            parts.append(f"[{known[piece.line_id]}] {piece.text}")
        elif piece.kind == DIALOGUE and ask_ids and not piece.continues:
            parts.append(f"[#{piece.line_id}] {piece.text}")
        else:
            parts.append(piece.text)
    return " ".join(parts)


def render_window(paragraphs: List[List[Segment]], window: Window, known: Optional[Dict[int, str]] = None) -> str:
    """The window's passage with the speakers known right now shown in place."""
    return "\n\n".join([render_paragraph(p, False, known) for p in paragraphs[window.context_start:window.start]]
                       + [render_paragraph(p, True, known) for p in paragraphs[window.start:window.end]])


def build_windows(paragraphs: List[List[Segment]], window_lines: int = WINDOW_LINES,
                  max_chars: int = WINDOW_MAX_CHARS, context_paragraphs: int = CONTEXT_PARAGRAPHS,
                  known: Optional[Dict[int, str]] = None) -> List[Window]:
    """Cut a chapter into windows of consecutive paragraphs holding up to window_lines dialogue
    lines to ask about (continued lines and lines in known aside) and up to max_chars of text,
    each preceded by the previous context_paragraphs paragraphs as unasked context. Paragraphs
    without asked lines ride along with the window they fall in; a stretch with none makes no
    window."""
    known = known or {}
    windows: List[Window] = []
    start = 0
    while start < len(paragraphs):
        end, ids, continued, anchored, chars = start, [], [], [], 0
        while end < len(paragraphs):
            paragraph = paragraphs[end]
            asked = [s.line_id for s in paragraph if s.kind == DIALOGUE and not s.continues
                     and s.line_id not in known]
            tagged = [s.line_id for s in paragraph if s.kind == DIALOGUE and s.line_id in known]
            cont = [s.line_id for s in paragraph if s.kind == DIALOGUE and s.continues]
            length = sum(len(s.text) + 1 for s in paragraph)
            if ids and (len(ids) + len(asked) > window_lines or chars + length > max_chars):
                break
            ids.extend(asked)
            continued.extend(cont)
            anchored.extend(tagged)
            chars += length
            end += 1
            if len(ids) >= window_lines:
                break
        if end == start:  # a single paragraph over the budget still goes out on its own
            end = start + 1
        if ids:
            window = Window(ids, "", continued, anchored, max(0, start - context_paragraphs), start, end)
            windows.append(window._replace(passage=render_window(paragraphs, window, known)))
        start = end
    return windows


# ---- replies ----

class AttributionError(ValueError):
    """The model's reply cannot be used for this window."""


def _extract_json(reply: str) -> dict:
    text = (reply or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()
    first, last = text.find("{"), text.rfind("}")
    if first < 0 or last <= first:
        raise AttributionError("no JSON object in the reply")
    try:
        data = json.loads(text[first:last + 1])
    except ValueError as e:
        raise AttributionError(f"invalid JSON: {e}")
    if not isinstance(data, dict):
        raise AttributionError("the reply is not a JSON object")
    return data


def _line_id(key) -> Optional[int]:
    match = re.fullmatch(r"\s*\[?#?\s*(\d+)\s*\]?\s*", str(key))
    return int(match.group(1)) if match else None


def _speaker_or_none(value) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise AttributionError(f"speaker is not a name: {value!r}")
    name = display_name(value)
    return None if name.lower().strip(".!? ") in UNKNOWN_SPEAKER_WORDS else name


def parse_reply(reply: str, expected_ids: List[int],
                ignore_ids: Collection[int] = ()) -> Tuple[Dict[int, Optional[str]], List[dict]]:
    """Validate one window's reply: ({line id: speaker name or None}, new/updated characters).

    Raises AttributionError for anything but a JSON object whose "speakers" cover exactly the
    expected ids (a missing id, an invented id, a non-string name). Answers for ignore_ids (lines
    shown with a known speaker) are dropped rather than refused. Characters with a bad gender or
    age are kept with "unknown" there; a character without a usable name is dropped.
    """
    data = _extract_json(reply)
    raw = data.get("speakers")
    if not isinstance(raw, dict):
        raise AttributionError("\"speakers\" is missing or not an object")
    speakers: Dict[int, Optional[str]] = {}
    for key, value in raw.items():
        line_id = _line_id(key)
        if line_id is not None and line_id in ignore_ids:
            continue
        if line_id is None or line_id not in expected_ids:
            raise AttributionError(f"unknown line id {key!r}")
        speakers[line_id] = _speaker_or_none(value)
    missing = [i for i in expected_ids if i not in speakers]
    if missing:
        raise AttributionError(f"missing line ids {missing}")
    characters = []
    for item in data.get("characters") or []:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not normalize_name(item["name"]):
            continue
        aliases = [display_name(a) for a in item.get("aliases") or [] if isinstance(a, str) and normalize_name(a)]
        characters.append({
            "name": display_name(item["name"]),
            "gender": item.get("gender") if item.get("gender") in GENDERS else "unknown",
            "age": item.get("age") if item.get("age") in AGES else "unknown",
            "aliases": aliases,
        })
    return speakers, characters


# ---- the running character list ----

# Common English short forms that are not prefixes of the full name (Tom / Thomas is the brief's own
# example). Prefix forms (Ben / Benjamin, Chris / Christopher) need no table.
_NICKNAMES = {
    "tom": "thomas", "tommy": "thomas", "bill": "william", "billy": "william", "will": "william",
    "bob": "robert", "bobby": "robert", "rob": "robert", "robin": "robert", "dick": "richard", "rick": "richard",
    "jack": "john", "johnny": "john", "jim": "james", "jimmy": "james", "jamie": "james", "peggy": "margaret",
    "meg": "margaret", "maggie": "margaret", "betty": "elizabeth", "beth": "elizabeth", "liz": "elizabeth",
    "lizzie": "elizabeth", "eliza": "elizabeth", "bess": "elizabeth", "bessie": "elizabeth", "ned": "edward",
    "ted": "edward", "teddy": "edward", "eddie": "edward", "harry": "henry", "hal": "henry", "hank": "henry",
    "kate": "katherine", "kitty": "katherine", "katie": "katherine", "kathy": "katherine", "cathy": "catherine",
    "nell": "eleanor", "nellie": "eleanor", "ellie": "eleanor", "sally": "sarah", "molly": "mary", "polly": "mary",
    "nancy": "anne", "nan": "anne", "annie": "anne", "charlie": "charles", "chuck": "charles", "frank": "francis",
    "fred": "frederick", "freddie": "frederick", "joe": "joseph", "joey": "joseph", "mike": "michael",
    "dan": "daniel", "danny": "daniel", "dave": "david", "davy": "david", "steve": "stephen", "sam": "samuel",
    "tony": "anthony", "andy": "andrew", "drew": "andrew", "nick": "nicholas", "pat": "patrick", "paddy": "patrick",
    "matt": "matthew", "pete": "peter", "greg": "gregory", "jenny": "jennifer", "sue": "susan", "susie": "susan",
    "becky": "rebecca", "abby": "abigail", "vicky": "victoria", "ginny": "virginia", "tilly": "matilda",
    "mattie": "matilda", "hattie": "harriet", "carrie": "caroline", "dolly": "dorothy", "dot": "dorothy",
    "josie": "josephine", "kit": "christopher", "bert": "albert", "bertie": "albert", "archie": "archibald",
    "gus": "augustus", "reg": "reginald", "ron": "ronald", "len": "leonard", "lou": "louis", "louie": "louis",
    "walt": "walter", "wally": "walter", "phil": "philip", "ray": "raymond", "stan": "stanley", "vince": "vincent",
    "vic": "victor", "abe": "abraham", "ike": "isaac", "nate": "nathaniel", "nat": "nathaniel", "ben": "benjamin",
    "benny": "benjamin", "alex": "alexander", "sandy": "alexander", "theo": "theodore", "tim": "timothy",
    "gabe": "gabriel", "larry": "lawrence", "jerry": "gerald", "gerry": "gerald", "terry": "terence",
    "bart": "bartholomew", "ollie": "oliver", "maddie": "madeline", "millie": "millicent", "winnie": "winifred",
    "flo": "florence", "flossie": "florence", "rosie": "rose", "effie": "euphemia", "jo": "josephine",
    "amy": "amelia", "letty": "letitia", "lettie": "letitia", "lotty": "charlotte", "lottie": "charlotte",
    "toby": "tobias", "geoff": "geoffrey", "jeff": "jeffrey", "sid": "sidney", "syd": "sydney", "max": "maximilian",
}


def _same_person(word: str, other: str) -> bool:
    """The same first name in two forms: identical, one the start of the other (Ben / Benjamin),
    or a listed short form (Tom / Thomas, Bill / Will / William)."""
    if word == other:
        return True
    if (len(word) >= 3 and other.startswith(word)) or (len(other) >= 3 and word.startswith(other)):
        return True
    full_a, full_b = _NICKNAMES.get(word, word), _NICKNAMES.get(other, other)
    return full_a == full_b


class Roster:
    """Characters met so far, with alias merging. Keys are stable (the normalized first-seen
    name), so saved attributions keep pointing at the right character while its display name
    grows more complete ("Tom" -> "Thomas Baker"); when two entries turn out to be one person,
    canonical() maps the retired key to the surviving one.

    Merging is deliberately conservative: a bare surname ("Mrs. Marsh" next to "Ada Marsh") is
    never merged by code, since a family shares it, and neither are two people of different known
    genders; a wrong merge gives a main character the wrong voice, while a split just shows two
    rows the owner can give the same voice. The prompt asks the model for aliases, which do merge."""

    def __init__(self, characters: Optional[Dict[str, dict]] = None):
        self.characters: Dict[str, dict] = {}
        self.aliases: Dict[str, str] = {}  # normalized alias -> key
        for key, character in (characters or {}).items():
            self.characters[key] = dict(character)
            self.aliases[key] = key
            for alias in character.get("aliases", []):
                self.aliases[normalize_name(alias)] = key

    def names_for_prompt(self) -> List[str]:
        return [c["name"] for _, c in sorted(self.characters.items(), key=lambda kv: -kv[1].get("lines", 0))]

    def _candidates(self, norm: str) -> List[str]:
        """Existing characters this normalized name may refer to (see the class docstring), the
        ones whose first name is written identically first."""
        words = norm.split()
        found = []
        for key, character in self.characters.items():
            key_words = normalize_name(character["name"]).split() or key.split()
            if len(words) == 1:
                # "Tom" / "Thomas Baker" (first name), never "Baker" / "Thomas Baker" (surname)
                if _same_person(words[0], key_words[0]) and (len(key_words) == 1 or words[0] != key_words[-1]):
                    found.append(key)
            elif len(key_words) == 1:
                if _same_person(words[0], key_words[0]):  # "Thomas Baker" absorbs an earlier "Tom"
                    found.append(key)
            elif words[-1] == key_words[-1] and _same_person(words[0], key_words[0]):
                found.append(key)  # "Tom Baker" / "Thomas Baker"
        first = words[0]
        return sorted(found, key=lambda k: (normalize_name(self.characters[k]["name"]).split() or [k])[0] != first)

    def resolve(self, name: str, gender: str = "unknown") -> Optional[str]:
        """The key of the existing character this name refers to, or None when it is new: the same
        normalized name, a recorded alias, or (for a first name / fuller name pair) exactly one
        candidate of a compatible gender. Several candidates mean an ambiguous first name, which
        stays separate unless exactly one of them spells the first name the same way."""
        norm = normalize_name(name)
        if not norm:
            return None
        if norm in self.aliases:
            return self.aliases[norm]
        candidates = [k for k in self._candidates(norm)
                      if "unknown" in (gender, self.characters[k]["gender"]) or gender == self.characters[k]["gender"]]
        if len(candidates) == 1:
            return candidates[0]
        first = norm.split()[0]
        exact = [k for k in candidates if (normalize_name(self.characters[k]["name"]).split() or [k])[0] == first]
        return exact[0] if len(exact) == 1 else None

    def add(self, name: str, gender: str = "unknown", age: str = "unknown", aliases: Optional[List[str]] = None) -> str:
        """Register a character (or merge into the one this name refers to); returns its key."""
        key = self.resolve(name, gender)
        if key is None:
            key = normalize_name(name)
            self.characters[key] = {"name": display_name(name), "aliases": [], "gender": "unknown",
                                    "age": "unknown", "lines": 0}
            self.aliases[key] = key
        character = self.characters[key]
        shown = display_name(name)
        if len(normalize_name(shown).split()) > len(normalize_name(character["name"]).split()):
            previous, character["name"] = character["name"], shown  # the fuller form becomes the display name
            self._alias(key, previous)
        else:
            self._alias(key, shown)
        for alias in aliases or []:
            self._alias(key, alias)
        if gender in GENDERS and gender != "unknown" and character["gender"] == "unknown":
            character["gender"] = gender
        if age in AGES and age != "unknown" and character["age"] == "unknown":
            character["age"] = age
        return key

    def _alias(self, key: str, alias: str) -> None:
        norm = normalize_name(alias)
        if not norm or self.aliases.get(norm) not in (None, key):
            return  # an alias already owned by another character stays theirs
        self.aliases[norm] = key
        character = self.characters[key]
        listed = {normalize_name(character["name"])} | {normalize_name(a) for a in character["aliases"]}
        if norm not in listed:
            character["aliases"].append(display_name(alias))

    def count_line(self, key: str) -> None:
        self.characters[key]["lines"] = self.characters[key].get("lines", 0) + 1


# ---- attribution ----

def _messages(window: Window, roster: Roster) -> List[dict]:
    names = roster.names_for_prompt()
    return [
        {"role": "system", "content": PROMPTS["system"]},
        {"role": "user", "content": PROMPTS["window"].format(
            roster=", ".join(names) if names else PROMPTS["roster_empty"], passage=window.passage,
            ids=", ".join(str(i) for i in window.ids))},
    ]


def attribute_chapter(paragraphs: List[List[Segment]], roster: Roster, chat: Chat, stats: dict,
                      log: logging.Logger = logger, label: str = "") -> Dict[int, Optional[str]]:
    """Attribute every dialogue line of one chapter: {line id: character key or None}.

    Lines a speech tag names are not asked (core.speech_tags); the model sees them, and every line
    decided so far, as [Name] "...". Each window is asked once and, when its reply fails
    validation, once more; a second failure leaves its lines unknown. New names become roster characters (aliases merged). A line that
    continues the previous paragraph's quotation takes the previous line's speaker. stats is
    updated in place: windows asked, invalid_json (windows whose first reply was unusable),
    invalid_after_retry (windows whose retry was unusable too), lines, tagged_lines, unknown_lines,
    seconds.
    """
    result: Dict[int, Optional[str]] = {}
    for field in ("windows", "invalid_json", "invalid_after_retry", "lines", "tagged_lines", "unknown_lines"):
        stats.setdefault(field, 0)
    stats.setdefault("seconds", 0.0)
    anchors = tagged_speakers(paragraphs)
    windows = build_windows(paragraphs, known=anchors)
    started = time.monotonic()

    def known_now() -> Dict[int, str]:
        """Tag names, overridden by the display name of whoever each decided line resolved to."""
        known = dict(anchors)
        known.update({line_id: roster.characters[key]["name"] for line_id, key in result.items() if key})
        return known

    for number, window in enumerate(windows, 1):
        window = window._replace(passage=render_window(paragraphs, window, known_now()))
        speakers, characters = None, []
        for attempt in (1, 2):
            reply = chat(_messages(window, roster))
            try:
                speakers, characters = parse_reply(reply, window.ids, ignore_ids=anchors)
                break
            except AttributionError as e:
                if attempt == 1:
                    stats["invalid_json"] = stats.get("invalid_json", 0) + 1
                    log.warning(f"Cast{label}: window {number}/{len(windows)} reply unusable ({e}); asking again")
                else:
                    stats["invalid_after_retry"] = stats.get("invalid_after_retry", 0) + 1
                    log.warning(f"Cast{label}: window {number}/{len(windows)} unusable twice ({e}); "
                                f"{len(window.ids)} lines left unknown")
        stats["windows"] = stats.get("windows", 0) + 1
        if speakers is None:
            speakers = {line_id: None for line_id in window.ids}
        for character in characters:
            roster.add(character["name"], character["gender"], character["age"], character["aliases"])
        for line_id in window.ids:
            name = speakers.get(line_id)
            key = roster.add(name) if name else None
            result[line_id] = key
        # Tag names resolve after the model's character list, so "Mother" can land on the
        # character the model gave that alias instead of becoming a character of its own.
        for line_id in window.anchored:
            result[line_id] = roster.add(anchors[line_id])
        for line_id in window.continued:
            result[line_id] = result.get(line_id - 1)
        log.info(f"Cast{label}: window {number}/{len(windows)} done, {len(roster.characters)} characters so far")
    for line_id, name in anchors.items():  # tagged lines in stretches that needed no window
        if line_id not in result:
            result[line_id] = roster.add(name)
    # Lines the windows never covered (none expected) and continued lines' counts
    all_lines = [s for p in paragraphs for s in p if s.kind == DIALOGUE]
    for line in all_lines:
        result.setdefault(line.line_id, result.get(line.line_id - 1) if line.continues else None)
        if result[line.line_id]:
            roster.count_line(result[line.line_id])
    stats["lines"] = stats.get("lines", 0) + len(all_lines)
    stats["tagged_lines"] = stats.get("tagged_lines", 0) + len(anchors)
    stats["unknown_lines"] = stats.get("unknown_lines", 0) + sum(1 for v in result.values() if v is None)
    stats["seconds"] = round(stats.get("seconds", 0.0) + time.monotonic() - started, 2)
    return result
