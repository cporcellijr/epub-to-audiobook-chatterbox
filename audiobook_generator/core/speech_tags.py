"""Speakers named by a speech tag right next to their quotation, found without an LLM.

    "You left the gate open," said Ada.          -> Ada
    "I did not," Tom said.                       -> Tom
    Mrs. Marsh said quietly, "Out, both of you." -> Mrs. Marsh

Only a capitalised name (optionally behind a title: Mrs. Marsh, Captain Ferris, Old Hobb) or a
family word used as a name (Mother, Grandfather) counts. Pronoun tags ("she said") and
descriptions ("said the doctor") are left to the LLM, which sees the whole conversation. A
paragraph whose named tags all agree lends that speaker to its other quotations when they carry no
tag of their own, since a paragraph normally holds one speaker's words.

These anchors are shown to the LLM as known speakers and are not asked about, which also gives it
fixed points to follow the turn-taking in untagged exchanges.
"""
import re
from typing import Dict, List, Optional

from audiobook_generator.core.dialogue import DIALOGUE, NARRATION, Segment

SPEECH_VERBS = (
    "said", "says", "asked", "asks", "replied", "answered", "called", "cried", "shouted", "yelled",
    "whispered", "murmured", "muttered", "added", "snapped", "repeated", "continued", "exclaimed",
    "demanded", "insisted", "admitted", "agreed", "began", "told", "laughed", "sighed", "grunted",
    "growled", "hissed", "breathed", "declared", "announced", "observed", "remarked", "suggested",
    "offered", "protested", "retorted", "interrupted", "went on", "put in", "shot back", "wondered",
    "pleaded", "urged", "warned", "explained", "mumbled", "stammered", "screamed", "called out",
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

_VERB = "(?:" + "|".join(re.escape(v).replace(r"\ ", r"\s+") for v in sorted(SPEECH_VERBS, key=len, reverse=True)) + ")"
_WORD = r"[A-Z][a-zA-Z'’\-]*"
_TITLE = "(?:" + "|".join(re.escape(t) for t in sorted(TITLES, key=len, reverse=True)) + ")"
_NAME = rf"(?:{_TITLE}\s+)*{_WORD}(?:\s+{_WORD}){{0,2}}"
_ADVERB = r"(?:\s+\w+ly)?"
_PRONOUN = r"(?:he|she|they|I|we|you)"

# Narration right after the quotation: ' said Tom.' / ' Tom said quietly.'
_AFTER_VERB_NAME = re.compile(rf"^\s*[,;]?\s*{_VERB}{_ADVERB}\s+(?P<name>{_NAME})")
_AFTER_NAME_VERB = re.compile(rf"^\s*[,;]?\s*(?P<name>{_NAME}){_ADVERB}\s+{_VERB}\b")
# Narration right before the quotation, ending at it: 'Tom said, ' / 'Mrs. Marsh said quietly: '
_BEFORE_NAME_VERB = re.compile(rf"(?:^|[.!?]\s+|\s)(?P<name>{_NAME}){_ADVERB}\s+{_VERB}{_ADVERB}\s*[,:]\s*$")
# Pronoun tags: the quotation has a tag, just not a name (so a paragraph's named speaker isn't lent to it).
_AFTER_PRONOUN = re.compile(rf"^\s*[,;]?\s*(?:{_VERB}{_ADVERB}\s+{_PRONOUN}\b|{_PRONOUN}{_ADVERB}\s+{_VERB}\b)", re.I)
_BEFORE_PRONOUN = re.compile(rf"(?:^|[.!?]\s+|\s){_PRONOUN}{_ADVERB}\s+{_VERB}{_ADVERB}\s*[,:]\s*$", re.I)


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


def _tag_after(narration: str) -> Optional[str]:
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
    """True if the narration right around a quotation tags it at all: a named tag ("said Tom") or a
    pronoun tag ("she said")."""
    return bool(_tag_after(after) or _tag_before(before) or _has_pronoun_tag(before, after))


# First-person tags: '"...," I said.' / '"...," said I.' / 'I told her, "..."'
_AFTER_I = re.compile(rf"^\s*[,;]?\s*(?:{_VERB}{_ADVERB}\s+I\b|I{_ADVERB}\s+{_VERB}\b)")
_BEFORE_I = re.compile(rf"(?:^|[.!?]\s+|\s)I{_ADVERB}\s+{_VERB}\b[^\"“”.!?]{{0,20}}[,:]\s*$")


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
            if _AFTER_I.match(after) or _BEFORE_I.search(before):
                found.append(piece.line_id)
    return found


def tagged_speakers(paragraphs: List[List[Segment]]) -> Dict[int, str]:
    """{line id: name} for every dialogue line whose speaker a speech tag names, plus the untagged
    quotations of a paragraph whose named tags all name the same person. Lines that continue a
    quotation from the previous paragraph are left out (the attribution gives them the previous
    line's speaker)."""
    found: Dict[int, str] = {}
    for segments in paragraphs:
        named: Dict[int, str] = {}
        untagged: List[int] = []
        for i, piece in enumerate(segments):
            if piece.kind != DIALOGUE or piece.continues:
                continue
            before = segments[i - 1].text if i > 0 and segments[i - 1].kind == NARRATION else ""
            after = segments[i + 1].text if i + 1 < len(segments) and segments[i + 1].kind == NARRATION else ""
            name = _tag_after(after) or _tag_before(before)
            if name:
                named[piece.line_id] = name
            elif not _has_pronoun_tag(before, after):
                untagged.append(piece.line_id)
        found.update(named)
        if named and untagged and len({n.lower() for n in named.values()}) == 1:
            speaker = next(iter(named.values()))
            for line_id in untagged:
                found[line_id] = speaker
    return found
