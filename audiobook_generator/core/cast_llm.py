"""Speaker attribution with a local OpenAI-compatible chat model.

A chapter's dialogue lines (from core.dialogue) are sent in windows: WINDOW_LINES numbered lines
with the narration around them and the preceding paragraphs for context, plus the running list of
characters already met. Lines a speech tag already names (core.speech_tags: "said Tom") are not
asked; they, and every line decided in earlier windows, are shown to the model as [Name] "..." so
it can follow the turn-taking of the untagged lines. The model answers with JSON mapping each line id to a speaker and
listing new characters (gender, rough age, aliases). Every reply is validated strictly; a window
whose reply is unusable is asked once more, then its lines are left unknown (they get the
dialogue voice). Lines the text says need another look (core.cast_review: no speaker, "she said"
given to a man, a speaker change mid-paragraph, ...) are then asked once more in small groups with
wider context and the narrator named, before anything is counted.

Everything the model is told lives in PROMPTS so it can be tuned in one place. Nothing in here
touches the network except ChatClient, which talks to LLM_BASE_URL only.
"""
import json
import logging
import os
import re
import time
from typing import Callable, Collection, Dict, List, NamedTuple, Optional, Tuple

from audiobook_generator.core import cast_review
from audiobook_generator.core.cast import AGES, GENDERS, display_name, normalize_name, title_gender, titled_name
from audiobook_generator.core.delivery import MOOD_NORMAL, MOODS, segment_moods
from audiobook_generator.core.dialogue import DIALOGUE, Segment, split_paragraph
from audiobook_generator.core.speech_tags import contradicted, first_person_tagged, tagged_speakers

logger = logging.getLogger(__name__)

WINDOW_LINES = 20          # dialogue lines asked about per request
WINDOW_MAX_CHARS = 6000    # passage text per request (about 1,500 tokens): a 7-9B model's comfort zone
CONTEXT_PARAGRAPHS = 6     # paragraphs repeated before a window, with their known speakers, for context
LLM_TIMEOUT_SECONDS = 300  # one request; a small local model on a busy GPU can be slow
LLM_TEMPERATURE = 0.0

# Ask the LLM to guess each line's delivery mood alongside its speaker (adaptive delivery, see
# core.delivery). Measured against the labelled multivoice fixture 2026-09-28 (WORKLOG #14):
# speaker accuracy fell from 221/270 to 198/270 with the moods clause in the prompt, well past the
# 219/270 floor, so this stays False (rules-only moods) until a differently-worded prompt is
# measured not to cost accuracy.
ASK_LLM_FOR_MOODS = False

# Ask again about the lines core.cast_review flags (WORKLOG §28). Off only to measure without it.
REVIEW_FLAGGED_LINES = True

# A pronoun names no one: "he" as a speaker is an unknown speaker ("I" stays a speaker: in a
# first-person book the model uses it for the narrator, whom the attribution then keeps apart).
_PRONOUNS = frozenset({"he", "she", "they", "him", "her", "them", "you", "we", "us", "it", "his", "hers",
                       "their", "theirs", "your", "yours", "our", "ours", "one", "someone", "somebody",
                       "himself", "herself", "themselves", "yourself", "myself", "itself"})
UNKNOWN_SPEAKER_WORDS = frozenset({"", "unknown", "none", "n/a", "?", "nobody", "unclear"}) | _PRONOUNS
_NARRATOR_REFERENCES = frozenset({"i", "narrator", "the narrator"})
# Never an alias: what anyone may be called (seen live: "he", "honey" and "child" as aliases of one
# character), which would hand that character every line the model answers with the word. Family
# words (Mom, Dad) stay: within a story they name one person.
_NOT_ALIASES = _PRONOUNS | frozenset({
    "the", "a", "an", "honey", "hon", "baby", "babe", "sweetie", "sweetheart", "darling", "dear", "love",
    "sugar", "child", "kid", "kids", "boy", "girl", "man", "woman", "guy", "lady", "gentleman", "stranger",
    "friend", "buddy", "dude", "everyone", "everybody", "sir", "ma'am", "maam", "miss",
})
_POSSESSIVES = frozenset({"his", "her", "their", "my", "your", "our", "its"})


# What a family calls its members: a name only within one story (Roster keeps it per chapter).
_FAMILY_WORDS = frozenset({
    "mom", "mum", "mother", "ma", "mama", "mamma", "mommy", "mummy", "momma", "dad", "daddy", "father", "pa",
    "papa", "pop", "pops", "grandma", "grandpa", "granny", "gran", "grandmother", "grandfather", "nana",
    "grandad", "granddad", "step", "stepmom", "stepmother", "stepdad", "stepfather", "son", "daughter",
    "sis", "bro", "brother", "sister", "aunt", "auntie", "uncle", "cousin", "wife", "husband", "hubby",
    "little", "big", "baby", "older", "younger", "elder", "twin", "in-law", "in-laws", "son-in-law",
    "daughter-in-law", "mother-in-law", "father-in-law", "brother-in-law", "sister-in-law",
})


def family_word(alias: str) -> bool:
    """True for "Mom", "Step Mom", "Grandpa": what a family calls someone, not their name."""
    words = normalize_name(alias).split() or (alias or "").lower().split()
    return bool(words) and all(w in _FAMILY_WORDS for w in words)


def usable_alias(alias: str) -> bool:
    """False for a word anyone may be called (a pronoun, a pet name, "the woman") and for a
    description by someone else's relation to them ("his mom"): only names become aliases."""
    words = normalize_name(alias).split()
    return bool(words) and words[0] not in _POSSESSIVES and not all(w in _NOT_ALIASES for w in words)


def _relationship_owner(name: str) -> Optional[str]:
    """The person before a possessive relationship label, e.g. "Jimmy" in "Jimmy's companion"."""
    for marker in ("'s ", "’s "):
        if marker in name:
            return name.split(marker, 1)[0]
    return None


_LOCAL_REFERENCE_WORDS = frozenset({
    "adult", "attacker", "boy", "child", "companion", "customer", "doctor", "driver", "elder",
    "elderly", "female", "friend", "guy", "guys", "girl", "guard", "individual", "kid", "lady",
    "man", "male", "mystery", "neighbor", "nurse", "old", "one", "officer", "patient", "people",
    "person", "policeman", "policewoman", "soldier", "stranger", "teen", "teenager", "unknown",
    "unnamed", "visitor", "waiter", "woman", "women", "young",
})
_REFERENCE_FILLER = frozenset({"a", "an", "at", "in", "of", "the", "with"})


def _reference_key(name: str) -> str:
    """Key a chapter-local description without stripping descriptor words such as "young"."""
    words = titled_name(name).split()
    while words and words[0] in {"a", "an", "the"}:
        words.pop(0)
    return " ".join(words)


def _local_reference(name: str) -> bool:
    """Descriptions and relationship labels identify someone only in their current chapter."""
    relation = re.search(r"(?:'s|’s)\s+(.+)$", name)
    label = relation.group(1) if relation else name
    words = _reference_key(label).split()
    words = [word for word in words if word not in _REFERENCE_FILLER]
    return bool(words) and all(word in _LOCAL_REFERENCE_WORDS for word in words)

PROMPTS = {
    "system": (
        "You identify who speaks each line of dialogue in a passage from a novel. "
        "Answer with a single JSON object and nothing else: no prose, no markdown fences."
    ),
    "window": (
        "Known characters so far (use these exact names when the speaker is one of them):\n"
        "{roster}\n\n"
        "{narrator}"
        "Passage. Lines to attribute are marked like [#N] in front of the quotation, N being the line's id. "
        "Lines shown as [Name] in front of the quotation already have a known speaker: use them to follow "
        "who is talking, but do not answer for them. Quotations without a mark need no answer:\n\n"
        "{passage}\n\n"
        "Reply with exactly this shape, one entry per id:\n"
        "{{\"speakers\": {{\"N\": \"Full Name\"}}{moods_shape}, "
        "\"characters\": [{{\"name\": \"Full Name\", \"gender\": \"female|male|unknown\", "
        "\"age\": \"child|adult|elderly|unknown\", \"aliases\": [\"other names used for this person\"]}}]}}\n"
        "Rules:\n"
        "- give every marked id exactly one speaker, using the ids from the passage and no others;\n"
        "- a first-person tag such as \"I say\" belongs to the narrator; an \"I\" inside a quotation alone "
        "does not identify the narrator;\n"
        "- track who is present and able to speak in this scene. Someone merely mentioned or addressed "
        "by name is not thereby the speaker;\n"
        "- an identifiable unnamed speaker needs a consistent descriptive name and a separate character "
        "entry; do not assign their words to the narrator or another named person without evidence;\n"
        "- write \"unknown\" only when the passage gives no clue who is speaking;\n"
        "- use a person's most complete name; a title alone (Mr. Baker) or a first name is an alias of "
        "the same person, not a new character;\n"
        "- \"characters\" lists only speakers who are not in the known list, plus known characters whose "
        "gender or age the passage now reveals.\n"
        "{moods_rule}"
        "Ids to answer: {ids}"
    ),
    "review": (
        "Known characters (use these exact names when the speaker is one of them):\n"
        "{roster}\n"
        "{narrator}\n"
        "Passage. An earlier pass could not settle the lines marked [#N] (N is the line's id): it left them "
        "without a speaker, or its answer disagrees with the text around them. [Name] in front of a quotation "
        "is certain, from a speech tag. [Name?] is the earlier pass's guess and may be wrong. Quotations "
        "without a mark need no answer:\n\n"
        "{passage}\n\n"
        "Reply with exactly this shape, one entry per id:\n"
        "{{\"speakers\": {{\"N\": \"Full Name\"}}, \"characters\": [{{\"name\": \"Full Name\", "
        "\"gender\": \"female|male|unknown\", \"age\": \"child|adult|elderly|unknown\", "
        "\"aliases\": [\"other names used for this person\"]}}]}}\n"
        "Rules:\n"
        "- give every marked id exactly one speaker, using the ids from the passage and no others;\n"
        "- follow the text: \"she said\" is a woman speaking and \"he said\" a man; a quotation split by a "
        "tag (\"...,\" she said, \"...\") is one speaker's; a paragraph usually holds one speaker's words;\n"
        "- track who is present and able to speak in this scene. Someone merely mentioned or addressed "
        "by name is not thereby the speaker;\n"
        "- an identifiable unnamed speaker needs a consistent descriptive name and a separate character "
        "entry; do not assign their words to the narrator or another named person without evidence;\n"
        "- write \"unknown\" only when the passage gives no clue who is speaking;\n"
        "- \"characters\" lists only speakers who are not in the known list.\n"
        "Ids to answer: {ids}"
    ),
    "narrator": "In this chapter, the narrator is {name}{aliases}. Treat \"I\", \"narrator\", and "
                "\"the narrator\" as that same person for this chapter only.\n",
    "narrator_unnamed": "The narrator, who says \"I\" in the narration, is never named: answer \"I\" for their "
                        "lines.\n",
    "identity": (
        "Passage from a novel; a name in brackets before a quotation is who speaks it:\n\n{passage}\n\n"
        "The line \"{line}\" introduces a name, {name}, that the passage has not used before. Is {name} a "
        "new person, or another name for one of these characters who spoke just before: {candidates}?\n"
        "Reply with exactly this shape: {{\"same_as\": \"Name\"}} using one of those names, or "
        "{{\"same_as\": \"new\"}}."
    ),
    "roster_empty": "(none yet)",
    "moods_shape": ', "moods": {"N": "soft|normal|excited"}',
    "moods_rule": ("- moods is your best guess how each marked line sounds: \"soft\" (whispered or quiet), "
                  "\"excited\" (shouted or urgent), or \"normal\" otherwise;\n"),
}


# ---- settings (read per call, never at import) ----

def llm_base_url() -> str:
    return os.environ.get("LLM_BASE_URL", "").strip().rstrip("/")


def llm_model() -> str:
    return os.environ.get("LLM_MODEL", "").strip()


def llm_api_key() -> str:
    return os.environ.get("LLM_API_KEY", "").strip()


def llm_configured() -> bool:
    """Cast mode is offered only when a chat endpoint and a model are both configured (the
    analysis needs both)."""
    return bool(llm_base_url() and llm_model())


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


_JOINED = re.compile(r"\s+(?:and|&)\s+(?=[A-Z])")


def _one_person(name: str) -> str:
    """The first of two people answered as one speaker ("Jonathon and Jess" -> "Jonathon"): a pair is
    no character, and its fuller-looking name had become one twin's display name (§30)."""
    return _JOINED.split(name, maxsplit=1)[0]


def _speaker_or_none(value) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise AttributionError(f"speaker is not a name: {value!r}")
    name = display_name(_one_person(value))
    return None if normalize_name(name) in UNKNOWN_SPEAKER_WORDS else name


def parse_reply(reply: str, expected_ids: List[int],
                ignore_ids: Collection[int] = ()) -> Tuple[Dict[int, Optional[str]], List[dict], Dict[int, str]]:
    """Validate one window's reply: ({line id: speaker name or None}, new/updated characters,
    {line id: mood}).

    Raises AttributionError for anything but a JSON object whose "speakers" cover exactly the
    expected ids (a missing id, an invented id, a non-string name). Answers for ignore_ids (lines
    shown with a known speaker) are dropped rather than refused. Characters with a bad gender or
    age are kept with "unknown" there; a character without a usable name is dropped. "moods" is
    optional guidance, never required: a missing or invalid mood for an expected id becomes
    "normal", and an id outside expected_ids is ignored rather than raising (unlike "speakers").
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
        if _speaker_or_none(item["name"]) is None:  # "Unknown" listed as a character (seen live) is no one
            continue
        aliases = [display_name(a) for a in item.get("aliases") or [] if isinstance(a, str) and normalize_name(a)
                   and not _JOINED.search(a)]
        characters.append({
            "name": display_name(_one_person(item["name"])),
            "gender": item.get("gender") if item.get("gender") in GENDERS else "unknown",
            "age": item.get("age") if item.get("age") in AGES else "unknown",
            "aliases": aliases,
        })
    moods: Dict[int, str] = {}
    raw_moods = data.get("moods")
    if isinstance(raw_moods, dict):
        for key, value in raw_moods.items():
            line_id = _line_id(key)
            if line_id is None or line_id not in expected_ids:
                continue
            moods[line_id] = value if value in MOODS else MOOD_NORMAL
    for line_id in expected_ids:
        moods.setdefault(line_id, MOOD_NORMAL)
    return speakers, characters, moods


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
    """The same first name in two forms: identical, one the start of the other and at least two
    letters shorter (Ben / Benjamin, but not Ann / Anna or Paul / Paula, which are other names), or a
    listed short form (Tom / Thomas, Bill / Will / William)."""
    if word == other:
        return True
    short, long = sorted((word, other), key=len)
    if len(short) >= 3 and long.startswith(short) and len(long) - len(short) >= 2:
        return True
    full_a, full_b = _NICKNAMES.get(word, word), _NICKNAMES.get(other, other)
    return full_a == full_b


class Roster:
    """Characters met so far, with alias merging. Keys are stable (the normalized first-seen
    name), so saved attributions keep pointing at the right character while its display name
    grows more complete ("Tom" -> "Thomas Baker"); when two entries turn out to be one person,
    merge() folds the retired key into the surviving one.

    Merging is deliberately conservative: a bare surname ("Mrs. Marsh" next to "Ada Marsh") is
    never merged by code, since a family shares it, and neither are two people of different genders,
    known or said by a title ("Mr. Smith" and "Mrs. Smith" stay two, the second keyed "mrs smith"); a
    wrong merge gives a main character the wrong voice, while a split just shows two rows the owner
    can give the same voice. The prompt asks the model for aliases, which do merge.

    A family word ("Mom", "Dad", "Grandpa") names one person only within a chapter: in a collection
    every story has its own mother (seen live: every "Mom" of six stories merged into one), so such
    an alias lasts until new_chapter() and is never saved with the character."""

    def __init__(self, characters: Optional[Dict[str, dict]] = None):
        self.characters: Dict[str, dict] = {}
        self.aliases: Dict[str, str] = {}  # normalized alias -> key
        self.chapter_aliases: Dict[str, str] = {}  # narrator and family references, this chapter only
        self.local_aliases: Dict[str, str] = {}  # descriptive references, this scene only
        self.chapter_narrator: Optional[str] = None
        self._anonymous_narrators = set()
        self._replaced: Dict[str, str] = {}
        for key, character in (characters or {}).items():
            self.characters[key] = dict(character)
            if not _local_reference(character["name"]):
                self.aliases[key] = key
            owner = _relationship_owner(character["name"])
            for alias in character.get("aliases", []):
                norm = normalize_name(alias)
                if (not family_word(alias) and not _local_reference(alias) and norm not in _NARRATOR_REFERENCES
                        and norm != normalize_name(owner or "")):
                    self.aliases[normalize_name(alias)] = key

    def new_chapter(self) -> None:
        """Forget chapter-local narrator and family aliases."""
        self.chapter_aliases = {}
        self.local_aliases = {}
        self.chapter_narrator = None

    def new_scene(self) -> None:
        """Forget descriptive labels when the source explicitly starts a new scene/person."""
        self.local_aliases = {}

    def set_chapter_narrator(self, narrator: Optional[str], aliases: Collection[str] = ()) -> Optional[str]:
        """Bind first-person references and supplied aliases to this chapter's resolved narrator."""
        if not narrator:
            return None
        key = narrator if narrator in self.characters else self.resolve(narrator)
        if key is None:
            key = self.add(narrator)
        self.chapter_narrator = key
        for name in (*_NARRATOR_REFERENCES, *aliases):
            norm = normalize_name(name)
            if norm and self.chapter_aliases.get(norm) in (None, key):
                self.chapter_aliases[norm] = key
        return key

    def _anonymous_narrator(self) -> str:
        if self.chapter_narrator is None:
            key = self._new_key("Narrator")
            self.characters[key] = {"name": "The Narrator", "aliases": [], "gender": "unknown",
                                    "age": "unknown", "lines": 0}
            self._anonymous_narrators.add(key)
            self.chapter_narrator = key
        for name in _NARRATOR_REFERENCES:
            self.chapter_aliases[normalize_name(name)] = self.chapter_narrator
        return self.chapter_narrator

    def is_scoped_narrator(self, key: Optional[str]) -> bool:
        return bool(key and self.canonical_key(key) in self._anonymous_narrators)

    def canonical_key(self, key: str) -> str:
        while key in self._replaced:
            key = self._replaced[key]
        return key

    def names_for_prompt(self) -> List[str]:
        return [c["name"] for key, c in sorted(self.characters.items(), key=lambda kv: -kv[1].get("lines", 0))
                if key not in self._anonymous_narrators or key == self.chapter_narrator]

    def _gender(self, key: str) -> str:
        """A character's recorded gender, else the one its name's title says."""
        character = self.characters[key]
        known = character.get("gender", "unknown")
        return known if known != "unknown" else title_gender(character.get("name", ""))

    def _candidates(self, norm: str) -> List[str]:
        """Existing characters this normalized name may refer to (see the class docstring), the
        ones whose first name is written identically first."""
        words = norm.split()
        found = []
        for key, character in self.characters.items():
            key_words = normalize_name(character["name"]).split() or key.split()
            if _local_reference(character["name"]):
                continue  # a description is never evidence for a proper name
            owner = _relationship_owner(character["name"])
            if owner and _same_person(words[0], normalize_name(owner).split()[0]):
                continue  # "Jimmy's companion" is not the first-name character Jimmy
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
        if norm in _NARRATOR_REFERENCES:
            return self.chapter_aliases.get(norm) or self.chapter_narrator
        reference = _reference_key(name)
        if reference in self.local_aliases:
            return self.local_aliases[reference]
        if reference in self.chapter_aliases:
            return self.chapter_aliases[reference]
        if _local_reference(name):
            return None
        if gender == "unknown":
            gender = title_gender(name)
        titled = titled_name(name)
        if titled != norm and titled in self.aliases:  # "Mrs. Smith" keyed apart from a "Mr. Smith"
            key = self.aliases[titled]
            if "unknown" in (gender, self._gender(key)) or gender == self._gender(key):
                return key
        if norm in self.chapter_aliases:  # this chapter's "Mom" before anyone else's
            return self.chapter_aliases[norm]
        if norm in self.aliases:
            key = self.aliases[norm]
            if "unknown" in (gender, self._gender(key)) or gender == self._gender(key):
                return key
        candidates = [] if _relationship_owner(name) else [
            k for k in self._candidates(norm)
            if "unknown" in (gender, self._gender(k)) or gender == self._gender(k)]
        if len(candidates) == 1:
            return candidates[0]
        first = norm.split()[0]
        exact = [k for k in candidates if (normalize_name(self.characters[k]["name"]).split() or [k])[0] == first]
        return exact[0] if len(exact) == 1 else None

    def add(self, name: str, gender: str = "unknown", age: str = "unknown", aliases: Optional[List[str]] = None) -> str:
        """Register a character (or merge into the one this name refers to); returns its key."""
        if normalize_name(name) in _NARRATOR_REFERENCES:
            key = self.resolve(name)
            if key is None:
                key = self._anonymous_narrator()
            for alias in aliases or []:
                norm = normalize_name(alias)
                if norm in _NARRATOR_REFERENCES and self.chapter_aliases.get(norm) in (None, key):
                    self.chapter_aliases[norm] = key
            return key
        if _local_reference(name):
            reference = _reference_key(name)
            key = self.local_aliases.get(reference)
            if key and gender not in ("unknown", self._gender(key)) and self._gender(key) != "unknown":
                key = None
            if key is None:
                key = self._new_key(name)
                self.characters[key] = {"name": display_name(name), "aliases": [], "gender": "unknown",
                                        "age": "unknown", "lines": 0, "reference_scope": "chapter"}
                if reference not in self.local_aliases:
                    self.local_aliases[reference] = key
            character = self.characters[key]
            if gender in GENDERS and gender != "unknown" and character["gender"] == "unknown":
                character["gender"] = gender
            if age in AGES and age != "unknown" and character["age"] == "unknown":
                character["age"] = age
            for alias in aliases or []:
                self._alias(key, alias)
            return key
        key = self.resolve(name, gender)
        if gender not in GENDERS or gender == "unknown":
            gender = title_gender(name)
        if key is None:
            key = self._new_key(name)
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

    def name_reference(self, key: str, name: str) -> None:
        """A chapter-local description ("Rob's companion") turns out to have a name: the name becomes
        the display name and, unlike the old label (kept as a local alias), is known in later chapters."""
        key = self.canonical_key(key)
        character = self.characters[key]
        old, character["name"] = character["name"], display_name(name)
        character.pop("reference_scope", None)
        self._anonymous_narrators.discard(key)  # an unnamed "I" who gives their name is that person
        self._alias(key, old)
        self._alias(key, name)

    def _new_key(self, name: str) -> str:
        """The normalized name, or when someone else already has it (a "Mr. Smith" before this "Mrs.
        Smith"), the name with its titles, else a number."""
        for key in (normalize_name(name), titled_name(name)):
            if key not in self.characters:
                return key
        number = 2
        while f"{normalize_name(name)} {number}" in self.characters:
            number += 1
        return f"{normalize_name(name)} {number}"

    def _alias(self, key: str, alias: str) -> None:
        key = self.canonical_key(key)
        norm = normalize_name(alias)
        if not norm:
            return
        reference = _reference_key(alias)
        if _local_reference(alias):
            if reference in self.local_aliases or reference not in self.chapter_aliases:
                self.local_aliases.setdefault(reference, key)
            return
        if self.aliases.get(norm) not in (None, key):
            return  # an alias already owned by another character stays theirs
        owner = _relationship_owner(self.characters[key]["name"])
        if owner and norm == normalize_name(owner):
            return  # a relationship label's owner is a different character
        own_name = norm == normalize_name(self.characters[key]["name"])
        if norm in _NARRATOR_REFERENCES:
            previous = self.chapter_aliases.get(norm)
            if previous and previous != key and self.is_scoped_narrator(previous):
                self.merge(previous, key)
                previous = key
            if previous in (None, key):
                self.chapter_aliases[norm] = key
                if self.chapter_narrator in (None, previous):
                    self.chapter_narrator = key
            return
        if not usable_alias(alias) and not own_name:
            return  # "he", "honey", "his mom": not a name
        if family_word(alias) and not own_name:
            if self.chapter_aliases.get(norm) in (None, key):
                self.chapter_aliases[norm] = key
            return
        self.aliases[norm] = key
        character = self.characters[key]
        listed = {normalize_name(character["name"])} | {normalize_name(a) for a in character["aliases"]}
        if norm not in listed:
            character["aliases"].append(display_name(alias))

    def merge(self, source: str, target: str) -> None:
        """Fold source into target (one person under two names): its names become target's aliases
        and every alias of source points at target."""
        source, target = self.canonical_key(source), self.canonical_key(target)
        if source == target:
            return
        gone = self.characters.pop(source)
        self._replaced[source] = target
        self._anonymous_narrators.discard(source)
        if self.chapter_narrator == source:
            self.chapter_narrator = target
        for table in (self.aliases, self.chapter_aliases, self.local_aliases):
            for norm, key in list(table.items()):
                if key == source:
                    table[norm] = target
        for alias in [gone["name"], *gone.get("aliases", [])]:
            self._alias(target, alias)
        character = self.characters[target]
        character["lines"] = character.get("lines", 0) + gone.get("lines", 0)
        for field in ("gender", "age"):
            if character.get(field, "unknown") == "unknown":
                character[field] = gone.get(field, "unknown")

    def count_line(self, key: str) -> None:
        self.characters[key]["lines"] = self.characters[key].get("lines", 0) + 1


# ---- attribution ----

def _narrator_prompt(roster: Roster) -> str:
    key = roster.chapter_narrator
    if not key or key not in roster.characters:
        return ""
    name = roster.characters[key]["name"]
    aliases = sorted(alias for alias, owner in roster.chapter_aliases.items()
                     if owner == key and alias not in _NARRATOR_REFERENCES)
    suffix = f" (aliases: {', '.join(aliases)})" if aliases else ""
    return PROMPTS["narrator"].format(name=name, aliases=suffix)


def _messages(window: Window, roster: Roster) -> List[dict]:
    names = roster.names_for_prompt()
    moods_shape = PROMPTS["moods_shape"] if ASK_LLM_FOR_MOODS else ""
    moods_rule = PROMPTS["moods_rule"] if ASK_LLM_FOR_MOODS else ""
    return [
        {"role": "system", "content": PROMPTS["system"]},
        {"role": "user", "content": PROMPTS["window"].format(
            roster=", ".join(names) if names else PROMPTS["roster_empty"], narrator=_narrator_prompt(roster),
            passage=window.passage,
            ids=", ".join(str(i) for i in window.ids), moods_shape=moods_shape, moods_rule=moods_rule)},
    ]


# A speaker naming themselves: "please call me Lena", "you can call me Ann", "my name is Tom".
_SELF_NAMED = re.compile(r"(?:^|[.!?]\s+)(?:[Aa]nd\s+)?(?:[Pp]lease,?\s+)?(?:[Yy]ou (?:can|may)\s+|[Jj]ust\s+)?"
                         r"(?:[Cc]all me|[Mm]y name is|[Mm]y name['’]s|[Tt]he name['’]s)\s+"
                         r"((?:(?:Mrs?|Ms|Miss|Dr)\.?\s+)?[A-Z][\w'’-]+)")


_ASKS_NAME = re.compile(r"your name|who are you|what do (?:they|people|we|you|folks) call you", re.I)
_I_AM = re.compile(r"(?:I['’]m|I am)\s+([A-Z][\w'’-]+)")
_BARE_NAME = re.compile(r"[A-Z][\w'’]+")


def _bare_name(text: str) -> Optional[str]:
    """The name a whole line consists of ("Dex", "T-Tom", "D-D-Dex"), stutter prefixes removed."""
    whole = text.strip(" .,!?…\"'“”‘’")
    parts = whole.split("-")
    last = parts[-1]
    # A stutter repeats the name's start ("D-D-Dex"); "Jo-Ann" is a whole hyphenated name.
    stutter = all(re.fullmatch(r"[A-Za-z]{1,3}", p) and last.lower().startswith(p.lower()) for p in parts[:-1])
    if _BARE_NAME.fullmatch(last) and stutter:
        return last
    return whole if re.fullmatch(r"[A-Z][\w'’]+(?:-[A-Z][\w'’]+)+", whole) else None


INTRODUCTION_CONTEXT = 8  # paragraphs before a self-introduction whose speakers it may be a new name for


def _same_as(paragraphs: List[List[Segment]], result: Dict[int, Optional[str]], roster: Roster, line: Segment,
             name: str, new_key: str, chat: Chat, stats: dict) -> Optional[str]:
    """Ask the model whether a name introduced at its own first line ("...please call me Lena",
    given to Lena) is another name for someone who spoke just before; their key, or None."""
    where = next(i for i, p in enumerate(paragraphs) if any(s.line_id == line.line_id for s in p if s.kind == DIALOGUE))
    start = max(0, where - INTRODUCTION_CONTEXT)
    earlier = [s.line_id for p in paragraphs[start:where + 1] for s in p
               if s.kind == DIALOGUE and s.line_id < line.line_id]
    gender = roster._gender(new_key)
    candidates = [k for k in dict.fromkeys(result.get(i) for i in earlier)
                  if k and k != new_key and k in roster.characters
                  and ("unknown" in (gender, roster._gender(k)) or gender == roster._gender(k))]
    if not candidates:
        return None
    shown = {i: roster.characters[k]["name"] for i, k in result.items() if k in roster.characters and i < line.line_id}
    passage = cast_review.render(paragraphs, start, where + 1, [], shown, {})
    names = [roster.characters[k]["name"] for k in candidates]
    messages = [{"role": "system", "content": PROMPTS["system"]},
                {"role": "user", "content": PROMPTS["identity"].format(
                    passage=passage, line=line.text.strip("“”\" "), name=name, candidates=", ".join(names))}]
    stats["identity_questions"] = stats.get("identity_questions", 0) + 1
    try:
        answer = _extract_json(chat(messages)).get("same_as")
    except AttributionError:
        return None
    if not isinstance(answer, str):
        return None
    chosen = roster.resolve(answer)
    return chosen if chosen in candidates else None


def merge_self_introductions(paragraphs: List[List[Segment]], result: Dict[int, Optional[str]], roster: Roster,
                             stats: dict, log: logging.Logger = logger, label: str = "",
                             chat: Optional[Chat] = None) -> None:
    """A line in which its speaker names themselves ("please call me Lena") makes that name theirs:
    a separate character by that name who first speaks in this chapter is folded into the speaker,
    and a name not met yet becomes the speaker's alias. When the model already gave the line to the
    new name itself (seen live: a doctor's lines split between "Dr. Hale" and the first name she
    asks to be called, in two voices, §30), the model is asked whether that name belongs to someone
    who spoke just before; "new" (a newcomer introducing themselves) changes nothing. result is
    updated in place; family words never count."""
    dialogue = [s for p in paragraphs for s in p if s.kind == DIALOGUE]
    for n, line in enumerate(dialogue):
        speaker = result.get(line.line_id)
        if not speaker or speaker not in roster.characters:
            continue
        text = line.text.strip()
        if text and text[0] in "\"“„«'‘":
            closers = "'’" if text[0] in "'‘" else "\"”»“"
            text = text[1:]
            if text and text[-1] in closers:
                text = text[:-1]
        text = text.strip()
        # ponytail: mixed quoted introductions stay unmerged; ask the model if these matter later.
        if any(kind == DIALOGUE for style in ("single", "double")
               for kind, _ in split_paragraph(text, style)[0]):
            continue
        names = [match.group(1) for match in _SELF_NAMED.finditer(text)]
        previous = dialogue[n - 1] if n else None
        if (previous and result.get(previous.line_id) not in (None, speaker)
                and _ASKS_NAME.search(previous.text) and _bare_name(text)):
            names.append(_bare_name(text))  # an answer to "what's your name?" that is only the name
        elif len(text.split()) <= 4 and _I_AM.fullmatch(text.strip(" .,!")):
            names.append(_I_AM.fullmatch(text.strip(" .,!")).group(1))
        for name in names:
            if family_word(name) or not usable_alias(name):
                continue
            other = roster.resolve(name, roster.characters[speaker].get("gender", "unknown"))
            # A speaker known only as a description or as this chapter's unnamed "I" has no name of
            # their own to keep: the name they give wins, even when it is someone met earlier.
            nameless = (roster.is_scoped_narrator(speaker)
                        or roster.characters[speaker].get("reference_scope") == "chapter")
            if other is None:
                if nameless:
                    roster.name_reference(speaker, name)  # "Rob's companion" is Dex from now on
                else:
                    roster._alias(speaker, name)
                continue
            if nameless and other != speaker:
                other, speaker = speaker, other
            elif roster.characters[other].get("lines", 0):
                continue  # someone met in an earlier chapter
            if other == speaker:
                first = min(i for i, key in result.items() if key == speaker)
                if chat is None or first != line.line_id:
                    continue  # the name was theirs before this line
                target = _same_as(paragraphs, result, roster, line, name, other, chat, stats)
                if target is None:
                    continue
                other, speaker = speaker, target
            roster.merge(other, speaker)
            if roster.characters[speaker].get("reference_scope") == "chapter":
                roster.name_reference(speaker, name)
            for line_id, key in result.items():
                if key == other:
                    result[line_id] = speaker
            stats["merged_introductions"] = stats.get("merged_introductions", 0) + 1
            log.info(f"Cast{label}: line {line.line_id} introduces its speaker as {name}; merged them")


def review_lines(paragraphs: List[List[Segment]], result: Dict[int, Optional[str]], anchors: Dict[int, str],
                 roster: Roster, chat: Chat, stats: dict, log: logging.Logger = logger, label: str = "",
                 narrator: Optional[str] = None, narrator_aliases: Collection[str] = ()) -> None:
    """Ask once more, with wider context, about the lines core.cast_review flags. A reply that names
    someone replaces the first pass's answer unless the line's own text rules that character out
    (pronoun tag of the other gender, "I said" given to a non-narrator: stats["review_rejected"]);
    "unknown" keeps it; an unusable reply changes nothing (logged, counted in stats["review_unusable"]). result is updated in place."""
    if narrator:
        roster.set_chapter_narrator(narrator, narrator_aliases)
    flags = cast_review.flag_lines(paragraphs, result, anchors, roster.characters,
                                   narrator=roster.chapter_narrator)
    if not flags:
        return
    narrator_key = roster.chapter_narrator or cast_review.narrator_by_tags(paragraphs, result)
    narrator_line = ""
    if narrator_key and narrator_key in roster.characters:
        name = roster.characters[narrator_key]["name"]
        narrator_line = (PROMPTS["narrator_unnamed"].format(name=name) if normalize_name(name) == "i"
                         else PROMPTS["narrator"].format(name=name, aliases=""))
    names = roster.names_for_prompt()
    for start, end, ids in cast_review.groups(paragraphs, list(flags)):
        certain = {i: roster.characters[result[i]]["name"] if result.get(i) else anchors[i] for i in anchors}
        guessed = {i: roster.characters[key]["name"] for i, key in result.items()
                   if key and i not in anchors and i not in ids}
        passage = cast_review.render(paragraphs, start, end, ids, certain, guessed)
        messages = [
            {"role": "system", "content": PROMPTS["system"]},
            {"role": "user", "content": PROMPTS["review"].format(
                roster=", ".join(names) if names else PROMPTS["roster_empty"],
                narrator=narrator_line,
                passage=passage, ids=", ".join(str(i) for i in ids))},
        ]
        stats["review_requests"] = stats.get("review_requests", 0) + 1
        stats["review_lines"] = stats.get("review_lines", 0) + len(ids)
        try:
            speakers, characters, _ = parse_reply(chat(messages), ids, ignore_ids=anchors)
        except AttributionError as e:
            stats["review_unusable"] = stats.get("review_unusable", 0) + 1
            log.warning(f"Cast{label}: review of lines {ids} unusable ({e}); first answers kept")
            continue
        for character in characters:
            roster.add(character["name"], character["gender"], character["age"], character["aliases"])
        changed = []
        for line_id in ids:
            if speakers.get(line_id):
                key = roster.add(speakers[line_id])
                broken = cast_review.hard_violation(paragraphs, line_id, key, roster.characters, narrator_key)
                if broken:
                    stats["review_rejected"] = stats.get("review_rejected", 0) + 1
                    log.info(f"Cast{label}: review answer {roster.characters[key]['name']} for line {line_id} "
                             f"rejected ({broken}); first answer kept")
                    continue
                if key != result.get(line_id):
                    result[line_id] = key
                    changed.append(line_id)
        stats["review_changed"] = stats.get("review_changed", 0) + len(changed)
        log.info(f"Cast{label}: reviewed lines {ids} ({', '.join(sorted({r for i in ids for r in flags[i]}))}); "
                 f"{len(changed)} changed")


def attribute_chapter(paragraphs: List[List[Segment]], roster: Roster, chat: Chat, stats: dict,
                      log: logging.Logger = logger, label: str = "", narrator: Optional[str] = None,
                      narrator_aliases: Collection[str] = ()) -> Tuple[Dict[int, Optional[str]], Dict[int, str]]:
    """Attribute every dialogue line of one chapter: ({line id: character key or None},
    {line id: mood}).

    Lines a speech tag names are not asked (core.speech_tags); the model sees them, and every line
    decided so far, as [Name] "...". Each window is asked once and, when its reply fails
    validation, once more; a second failure leaves its lines unknown. New names become roster characters (aliases merged). A line that
    continues the previous paragraph's quotation takes the previous line's speaker (and, for moods,
    the previous line's final mood).

    Moods combine core.delivery's rule-based cues (whispered/shouted-style cues, a line ending in
    "!") with the LLM's own guess for lines it was asked about (module constant ASK_LLM_FOR_MOODS;
    when off, or for a line never asked, the LLM contributes nothing): a rule cue always overrides
    the LLM's guess, never the reverse.

    stats is updated in place: windows asked, invalid_json (windows whose first reply was
    unusable), invalid_after_retry (windows whose retry was unusable too), lines, tagged_lines,
    unknown_lines, seconds.
    """
    roster.new_chapter()  # family and narrator references belong only to this chapter
    roster.set_chapter_narrator(narrator, narrator_aliases)
    result: Dict[int, Optional[str]] = {}
    llm_moods: Dict[int, str] = {}
    for field in ("windows", "invalid_json", "invalid_after_retry", "lines", "tagged_lines", "unknown_lines"):
        stats.setdefault(field, 0)
    stats.setdefault("seconds", 0.0)
    anchors = tagged_speakers(paragraphs)
    narrator_name = roster.characters[roster.chapter_narrator]["name"] if roster.chapter_narrator else "I"
    contradictions = set(contradicted(paragraphs))
    for line_id in first_person_tagged(paragraphs):
        if line_id not in contradictions:
            anchors.setdefault(line_id, narrator_name)
    rule_moods = segment_moods(paragraphs)
    windows = build_windows(paragraphs, known=anchors)
    started = time.monotonic()

    def known_now() -> Dict[int, str]:
        """Tag names, overridden by the display name of whoever each decided line resolved to."""
        known = dict(anchors)
        known.update({line_id: roster.characters[key]["name"] for line_id, key in result.items() if key})
        return known

    for number, window in enumerate(windows, 1):
        window = window._replace(passage=render_window(paragraphs, window, known_now()))
        speakers, characters, window_moods = None, [], {}
        for attempt in (1, 2):
            reply = chat(_messages(window, roster))
            try:
                speakers, characters, window_moods = parse_reply(reply, window.ids, ignore_ids=anchors)
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
            window_moods = {}
        for character in characters:
            roster.add(character["name"], character["gender"], character["age"], character["aliases"])
        for line_id, key in result.items():
            if key:
                result[line_id] = roster.canonical_key(key)
        for line_id in window.ids:
            name = speakers.get(line_id)
            key = roster.add(name) if name else None
            result[line_id] = key
        if ASK_LLM_FOR_MOODS:  # rules-only means rules only, even if a reply volunteers moods
            llm_moods.update(window_moods)
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
    all_lines = [s for p in paragraphs for s in p if s.kind == DIALOGUE]
    merge_self_introductions(paragraphs, result, roster, stats, log, label, chat)
    if REVIEW_FLAGGED_LINES:
        review_lines(paragraphs, result, anchors, roster, chat, stats, log, label,
                     narrator, narrator_aliases)
        for line in all_lines:  # a continued line follows its (possibly corrected) first part
            if line.continues and line.line_id - 1 in result:
                result[line.line_id] = result[line.line_id - 1]
    # Lines the windows never covered (none expected) and continued lines' counts
    moods: Dict[int, str] = {}
    for line in all_lines:
        result.setdefault(line.line_id, result.get(line.line_id - 1) if line.continues else None)
        if result[line.line_id]:
            roster.count_line(result[line.line_id])
        if line.continues:
            moods[line.line_id] = moods.get(line.line_id - 1, MOOD_NORMAL)
        else:
            rule_mood = rule_moods.get(line.line_id, MOOD_NORMAL)
            moods[line.line_id] = rule_mood if rule_mood != MOOD_NORMAL else llm_moods.get(line.line_id, MOOD_NORMAL)
    stats["lines"] = stats.get("lines", 0) + len(all_lines)
    stats["tagged_lines"] = stats.get("tagged_lines", 0) + len(anchors)
    stats["unknown_lines"] = stats.get("unknown_lines", 0) + sum(1 for v in result.values() if v is None)
    stats["seconds"] = round(stats.get("seconds", 0.0) + time.monotonic() - started, 2)
    return result, moods
