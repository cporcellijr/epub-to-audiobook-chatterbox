"""Speakers named by a speech tag right next to their quotation, found without an LLM.

    "You left the gate open," said Ada.          -> Ada
    "I did not," Tom said.                       -> Tom
    Mrs. Marsh said quietly, "Out, both of you." -> Mrs. Marsh

Only a capitalised name (optionally behind a title: Mrs. Marsh, Captain Ferris, Old Hobb) or a
family word used as a name (Mother, Grandfather) counts. Pronoun tags ("she said") and
descriptions ("said the doctor") are left to the LLM, which sees the whole conversation.

A tag points one way. Narration after a quotation whose first sentence runs on to a colon introduces
the next quotation instead ("When can we meet?" Ann answered without a pause: "Monday." -> Monday is
Ann's; the question is not), and a quotation named differently before and after is left to the LLM.
An untagged quotation takes a named speaker only when it follows their line in the same paragraph with
nothing between that introduces or names anyone else ("Bring him up," said Mrs. Marsh. She turned.
"Ada, run ahead.") and the paragraph's tags name no one else. An earlier quotation never borrows.

These anchors are shown to the LLM as known speakers and are not asked about, which also gives it
fixed points to follow the turn-taking in untagged exchanges. A wrong anchor can't be corrected
later, so every rule here errs toward asking (WORKLOG §27). That includes a family word on its own
("Dad said"): in one story "Dad" was both the narrator's father and her husband, their son's "Daddy",
and the tag had locked the father's lines to the husband (§30).
"""
import re
from typing import Dict, List, Optional

from audiobook_generator.core.dialogue import DIALOGUE, NARRATION, Segment

# Past and present tense: novels are told in either ("she said", "she says").
SPEECH_VERBS = (
    "said", "says", "asked", "asks", "replied", "replies", "answered", "answers", "called", "calls",
    "cried", "cries", "shouted", "shouts", "yelled", "yells", "whispered", "whispers", "murmured",
    "murmurs", "muttered", "mutters", "added", "adds", "snapped", "snaps", "repeated", "repeats",
    "continued", "continues", "exclaimed", "exclaims", "demanded", "demands", "insisted", "insists",
    "admitted", "admits", "agreed", "agrees", "began", "begins", "told", "tells", "laughed", "laughs",
    "sighed", "sighs", "grunted", "grunts", "growled", "growls", "hissed", "hisses", "breathed",
    "breathes", "declared", "declares", "announced", "announces", "observed", "observes", "remarked",
    "remarks", "suggested", "suggests", "offered", "offers", "protested", "protests", "retorted",
    "retorts", "interrupted", "interrupts", "went on", "goes on", "put in", "puts in", "shot back",
    "shoots back", "wondered", "wonders", "pleaded", "pleads", "urged", "urges", "warned", "warns",
    "explained", "explains", "mumbled", "mumbles", "stammered", "stammers", "screamed", "screams",
    "called out", "calls out", "ordered", "orders", "begged", "begs", "implored", "implores",
    # Voiced ways of speaking that stand as tags ("he murmurs.", "she sobs.")
    "moaned", "moans", "groaned", "groans", "gasped", "gasps", "panted", "pants", "whimpered",
    "whimpers", "sobbed", "sobs", "wailed", "wails", "snarled", "snarls", "sniffed", "sniffs",
    "purred", "purrs", "giggled", "giggles", "chuckled", "chuckles", "squealed", "squeals",
    "shrieked", "shrieks", "barked", "barks", "mocked", "mocks", "teased", "teases", "grumbled",
    "grumbles", "whined", "whines", "stuttered", "stutters", "blurted", "blurts", "bellowed",
    "bellows", "roared", "roars", "seethed", "seethes",
)
# Titles that must be followed by a name, and family words that are names on their own.
TITLES = ("Mr.", "Mrs.", "Ms.", "Mr", "Mrs", "Ms", "Miss", "Dr.", "Dr", "Doctor", "Lady", "Lord",
          "Sir", "Dame", "Master", "Mistress", "Madam", "Captain", "Constable", "Sergeant", "Inspector",
          "Professor", "Aunt", "Uncle", "Old", "Young", "Little", "Father", "Mother", "Sister",
          "Brother", "King", "Queen", "Prince", "Princess", "Saint", "St.")
STANDALONE = ("Mother", "Father", "Mum", "Mom", "Mama", "Mamma", "Papa", "Pa", "Ma", "Dad", "Daddy",
              "Mummy", "Mommy", "Grandfather", "Grandmother", "Grandma", "Grandpa", "Granny", "Gran",
              "Nana")
# Capitalised words that start sentences or refer to someone without naming them.
NOT_NAMES = frozenset({
    "He", "She", "They", "It", "I", "We", "You", "The", "A", "An", "Then", "But", "And", "So", "Now",
    "Still", "Yet", "This", "That", "These", "Those", "There", "Here", "His", "Her", "Their", "Its",
    "My", "Our", "Your", "Someone", "Somebody", "Everyone", "Everybody", "Nobody", "No", "Yes", "Oh",
    "Well", "Just", "Again", "Finally", "At", "After", "Before", "When", "While", "If", "As", "Only",
    "Even", "Suddenly", "Quietly", "Softly", "Later", "Once", "One", "Another", "Somewhere", "What",
    "Who", "Why", "How", "Where", "Which", "Not", "Never", "Always", "All", "Both", "Each", "None",
})



def _base_form(verb: str) -> str:
    """ "asks" -> "ask", "cries" -> "cry", "hisses" -> "hiss", "goes on" -> "go on"."""
    first, _, rest = verb.partition(" ")
    if first == "goes":
        first = "go"
    elif first.endswith("ies"):
        first = first[:-3] + "y"
    elif first.endswith(("sses", "shes", "ches", "xes")):
        first = first[:-2]
    else:
        first = first[:-1]
    return f"{first} {rest}".strip()


# "I" and they/we/you take the base form in the present tense ("I ask", "they whisper"). Only pronoun
# tags use it: after a name, "tell" or "call" is rarely a tag.
BASE_SPEECH_VERBS = tuple(_base_form(v) for v in SPEECH_VERBS if v.split()[0].endswith("s"))


def _alternatives(verbs) -> str:
    return "(?:" + "|".join(re.escape(v).replace(r"\ ", r"\s+") for v in sorted(verbs, key=len, reverse=True)) + ")"


_VERB = _alternatives(SPEECH_VERBS)
_ANY_VERB = _alternatives(SPEECH_VERBS + BASE_SPEECH_VERBS)
_WORD = r"[A-Z][a-zA-Z'’\-]*"
_TITLE = "(?:" + "|".join(re.escape(t) for t in sorted(TITLES, key=len, reverse=True)) + ")"
_NAME = rf"(?:{_TITLE}\s+)*{_WORD}(?:\s+{_WORD}){{0,2}}"
_ADVERB = r"(?:\s+\w+ly)?"
_PRONOUN = r"(?:he|she|they|I|we|you)"

# Narration right after the quotation: ' said Tom.' / ' Tom said quietly.'
_AFTER_VERB_NAME = re.compile(rf"^\s*[,;]?\s*{_VERB}{_ADVERB}\s+(?P<name>{_NAME})")
_AFTER_NAME_VERB = re.compile(rf"^\s*[,;]?\s*(?P<name>{_NAME}){_ADVERB}\s+{_VERB}\b")
# Narration right before the quotation, ending at it: 'Tom said, ' / 'Mrs. Marsh said quietly: ' /
# 'Ann answered without a pause: ' (a short trailing phrase naming no one else)
_BEFORE_NAME_VERB = re.compile(rf"(?:^|[.!?]\s+|\s)(?P<name>{_NAME}){_ADVERB}\s+{_VERB}\b[^\"“”.!?A-Z]{{0,30}}[,:]\s*$")
# Pronoun tags: the quotation has a tag, just not a name (so a paragraph's named speaker isn't lent to it).
_AFTER_PRONOUN = re.compile(rf"^\s*[,;]?\s*(?:{_VERB}{_ADVERB}\s+{_PRONOUN}\b|{_PRONOUN}{_ADVERB}\s+{_ANY_VERB}\b)", re.I)
_BEFORE_PRONOUN = re.compile(rf"(?:^|[.!?]\s+|\s){_PRONOUN}{_ADVERB}\s+{_ANY_VERB}{_ADVERB}\s*[,:]\s*$", re.I)


def clean_name(raw: str) -> Optional[str]:
    """The captured name if it really is a name, else None: no pronoun or sentence word, no
    possessive (said Tom's mother), and a title only as a prefix unless it is a family word."""
    words = raw.split()
    while words and words[0] in NOT_NAMES:
        words.pop(0)  # a sentence word caught in front of the name ("Then Tom said")
    while words and words[-1] in TITLES and words[-1] not in STANDALONE:
        words.pop()  # a title with no name after it ("said Old")
    if not words:
        return None
    if any(w.endswith(("'s", "’s")) for w in words):
        return None
    if any(w in NOT_NAMES for w in words):
        return None
    if all(w in TITLES for w in words) and not (len(words) == 1 and words[0] in STANDALONE):
        return None
    return " ".join(words)


def _introduces(narration: str) -> bool:
    """Narration after a quotation that introduces the next one instead of tagging this one: its
    first sentence runs on to a colon ('Ann answered without a pause: "Monday."')."""
    # A title's period is not a sentence boundary ("Mrs. Marsh answered: ...").
    narration = re.sub(rf"\b{_TITLE}(?=\s+{_WORD})", lambda m: m.group().rstrip("."), narration)
    first = re.split(r"(?<=[.!?])\s+", narration.strip(), maxsplit=1)[0]
    return first.endswith(":")


def _tag_after(narration: str) -> Optional[str]:
    """The name tagging the quotation this narration follows."""
    return None if _introduces(narration) else _name_after(narration)


def _name_after(narration: str) -> Optional[str]:
    for pattern in (_AFTER_VERB_NAME, _AFTER_NAME_VERB):
        match = pattern.match(narration)
        if match:
            name = clean_name(match.group("name"))
            if name:
                return name
    return None


def _tag_before(narration: str) -> Optional[str]:
    match = _BEFORE_NAME_VERB.search(narration)
    return clean_name(match.group("name")) if match else None


def _has_pronoun_tag(before: str, after: str) -> bool:
    return bool(_AFTER_PRONOUN.match(after) or _BEFORE_PRONOUN.search(before))


def has_speech_tag(before: str, after: str) -> bool:
    """True if the narration right around a quotation has a speech tag at all, whichever quotation it
    belongs to: a named tag ("said Tom") or a pronoun tag ("she said")."""
    return bool(_name_after(after) or _tag_before(before) or _has_pronoun_tag(before, after))


# First-person tags: '"...," I said.' / '"...," said I.' / 'I told her, "..."'
_AFTER_I = re.compile(rf"^\s*[,;]?\s*(?:{_VERB}{_ADVERB}\s+I\b|I{_ADVERB}\s+{_ANY_VERB}\b)")
_BEFORE_I = re.compile(rf"(?:^|[.!?]\s+|\s)I{_ADVERB}\s+{_ANY_VERB}\b[^\"“”.!?]{{0,20}}[,:]\s*$")


def first_person_tagged(paragraphs: List[List[Segment]]) -> List[int]:
    """Ids of the dialogue lines a first-person tag ("I said") gives to the narrator: whoever the
    attribution names for them is the "I" of a first-person chapter."""
    found = []
    for segments in paragraphs:
        for i, piece in enumerate(segments):
            if piece.kind != DIALOGUE or piece.continues:
                continue
            before = segments[i - 1].text if i > 0 and segments[i - 1].kind == NARRATION else ""
            after = segments[i + 1].text if i + 1 < len(segments) and segments[i + 1].kind == NARRATION else ""
            if (_AFTER_I.match(after) and not _introduces(after)) or _BEFORE_I.search(before):
                found.append(piece.line_id)
    return found


_PRONOUN_WORD = re.compile(r"\b(he|she|they|I|we|you)\b", re.I)


def tag_pronoun(before: str, after: str) -> Optional[str]:
    """The pronoun of a quotation's own pronoun tag ('"...," she said.' -> "she"), lower-cased; None
    when it has none. Narration that introduces the next quotation is not this one's tag."""
    match = (None if _introduces(after) else _AFTER_PRONOUN.match(after)) or _BEFORE_PRONOUN.search(before)
    word = _PRONOUN_WORD.search(match.group(0)) if match else None
    return word.group(1).lower() if word else None


def has_own_tag(before: str, after: str) -> bool:
    """Whether anything tags this quotation itself: a name, a pronoun or "I", after it (unless that
    narration introduces the next quotation) or right before it."""
    return bool(_tag_after(after) or _tag_before(before) or tag_pronoun(before, after)
                or (_AFTER_I.match(after) and not _introduces(after)) or _BEFORE_I.search(before))


def contradicted(paragraphs: List[List[Segment]]) -> List[int]:
    """Ids of the lines named one way right before them and another right after (left unanchored)."""
    found = []
    for segments in paragraphs:
        for i, piece in enumerate(segments):
            if piece.kind != DIALOGUE or piece.continues:
                continue
            before = segments[i - 1].text if i > 0 and segments[i - 1].kind == NARRATION else ""
            after = segments[i + 1].text if i + 1 < len(segments) and segments[i + 1].kind == NARRATION else ""
            after_name, before_name = _tag_after(after), _tag_before(before)
            if after_name and before_name and after_name.lower() != before_name.lower():
                found.append(piece.line_id)
    return found


def keeps_speaker(narration: str, speaker: str) -> bool:
    """Whether narration between two quotations lets the first one's speaker run on: it introduces
    nobody (no colon at the end) and names nobody else ("She smiled." keeps them; "Tom smiled." and
    "Ada looked up:" don't)."""
    if narration.rstrip().endswith(":"):
        return False
    own = set(speaker.split())
    for word in re.findall(_WORD, narration):
        word = re.sub(r"['’]s$", "", word)
        if word not in own and word not in NOT_NAMES and word not in TITLES:
            return False
    return True


def tagged_speakers(paragraphs: List[List[Segment]]) -> Dict[int, str]:
    """{line id: name} for every dialogue line whose speaker a speech tag names, plus the untagged
    quotations that continue a named line's speech in a paragraph naming no one else.
    Lines that continue a quotation from the previous paragraph are left out (the attribution gives
    them the previous line's speaker)."""
    found: Dict[int, str] = {}
    for segments in paragraphs:
        named: Dict[int, str] = {}
        continuing: List[int] = []
        speaker, between = None, []  # whose speech may run on, and the narration since their line
        for i, piece in enumerate(segments):
            if piece.kind == NARRATION:
                between.append(piece.text)
                continue
            if piece.continues:
                speaker, between = None, []
                continue
            before = segments[i - 1].text if i > 0 and segments[i - 1].kind == NARRATION else ""
            after = segments[i + 1].text if i + 1 < len(segments) and segments[i + 1].kind == NARRATION else ""
            after_name, before_name = _tag_after(after), _tag_before(before)
            if after_name and before_name and after_name.lower() != before_name.lower():
                speaker = None  # contradictory tags: the model decides
            elif (after_name or before_name) in STANDALONE:
                speaker = None  # "Dad said": whose dad depends on who is telling it; the model decides
            elif after_name or before_name:
                speaker = named[piece.line_id] = after_name or before_name
            elif _has_pronoun_tag(before, after):
                speaker = None
            elif speaker and all(keeps_speaker(narration, speaker) for narration in between):
                continuing.append(piece.line_id)  # "...," said Ada. She smiled. "..." -- still Ada
            else:
                speaker = None
            between = []
        found.update(named)
        if continuing and len({n.lower() for n in named.values()}) == 1:
            for line_id in continuing:
                found[line_id] = next(iter(named.values()))
    return found
