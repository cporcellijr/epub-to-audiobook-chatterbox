"""Which attributed lines deserve a second look, and the passages to show for them (no LLM here).

The first pass asks each window once; a well-formed answer is kept even when the text contradicts it.
A review of the first real book's cast (WORKLOG §28) found the misattributions were of kinds the text
itself flags:
  - no speaker at all (the narrator's "I said" lines came back "Narrator", which names no one);
  - a "she said" line given to a man, or "he said" to a woman;
  - a line tagged "I said" given to someone other than the chapter's clear "I";
  - a speaker change inside one paragraph where nothing between the two quotations names or
    introduces anyone else and the second has no tag of its own ("...the best part," Ann looked at me.
    "Isn't that sweet?" -- still Ann), including a sentence split by a tag ("...a good girl," ...
    "but sometimes she talks too much.");
  - contradictory speech tags (core.speech_tags.contradicted).
A line calling its own speaker by name ("Dom." given to Dom) was tried as a signal too and dropped:
asked again, the model kept its answer (WORKLOG §28).
These are signals to ask again with more context, never corrections on their own.
"""
from collections import Counter
from typing import Dict, List, Optional, Tuple

from audiobook_generator.core.dialogue import DIALOGUE, NARRATION, Segment
from audiobook_generator.core import speech_tags

CONTEXT_BEFORE = 4   # paragraphs shown before a group's first flagged paragraph
CONTEXT_AFTER = 2    # and after its last: the answer often comes in the next paragraph
GROUP_GAP = 3        # flagged paragraphs this close share one request
GROUP_LINES = 12     # lines asked per request at most

UNKNOWN, GENDER, NARRATOR, CONTINUES, CONTRADICTED = (
    "unknown", "pronoun gender", "I-tag not the narrator", "speaker change mid-paragraph", "contradictory tags")
_GENDER_OF = {"he": "male", "she": "female"}


def narrator_by_tags(paragraphs: List[List[Segment]], result: Dict[int, Optional[str]]) -> Optional[str]:
    """The key the chapter's "I said" lines clearly point to: at least 2 of them, more than half of
    those with a speaker (as core.cast_profiles.chapter_narrators decides the narrator)."""
    votes = Counter(result.get(i) for i in speech_tags.first_person_tagged(paragraphs) if result.get(i))
    if not votes:
        return None
    key, count = votes.most_common(1)[0]
    return key if count >= 2 and count * 2 > sum(votes.values()) else None


def _around(segments: List[Segment], i: int) -> Tuple[str, str]:
    before = segments[i - 1].text if i > 0 and segments[i - 1].kind == NARRATION else ""
    after = segments[i + 1].text if i + 1 < len(segments) and segments[i + 1].kind == NARRATION else ""
    return before, after


def flag_lines(paragraphs: List[List[Segment]], result: Dict[int, Optional[str]], anchors: Dict[int, str],
               characters: Dict[str, dict]) -> Dict[int, List[str]]:
    """{line id: reasons} for the lines to ask again. Anchored (tag-named) and continued lines are
    never asked; they stay evidence. characters: the roster's {key: {"name", "gender", ...}}."""
    flags: Dict[int, List[str]] = {}

    def flag(line_id: int, reason: str) -> None:
        if line_id not in anchors and reason not in flags.setdefault(line_id, []):
            flags[line_id].append(reason)

    narrator = narrator_by_tags(paragraphs, result)
    first_person = set(speech_tags.first_person_tagged(paragraphs))
    for line_id in speech_tags.contradicted(paragraphs):
        flag(line_id, CONTRADICTED)
    for segments in paragraphs:
        quotes = [(i, s) for i, s in enumerate(segments) if s.kind == DIALOGUE and not s.continues]
        for n, (i, piece) in enumerate(quotes):
            speaker = result.get(piece.line_id)
            before, after = _around(segments, i)
            if speaker is None:
                flag(piece.line_id, UNKNOWN)
            pronoun = speech_tags.tag_pronoun(before, after)
            gender = (characters.get(speaker) or {}).get("gender")
            if pronoun in _GENDER_OF and gender in _GENDER_OF.values() and _GENDER_OF[pronoun] != gender:
                flag(piece.line_id, GENDER)
            if piece.line_id in first_person and narrator and speaker != narrator:
                flag(piece.line_id, NARRATOR)
            if n == 0 or speaker is None:
                continue
            j, previous = quotes[n - 1]
            earlier = result.get(previous.line_id)
            if earlier is None or earlier == speaker:
                continue
            between = [s.text for s in segments[j + 1:i]]
            name = (characters.get(earlier) or {}).get("name") or earlier
            split_sentence = (previous.text.rstrip("”\"’' ").endswith(",")
                              and piece.text.lstrip("“\"‘' ")[:1].islower())
            if split_sentence or (not speech_tags.has_own_tag(before, after)
                                  and all(speech_tags.keeps_speaker(text, name) for text in between)):
                flag(previous.line_id, CONTINUES)
                flag(piece.line_id, CONTINUES)
    return flags


def groups(paragraphs: List[List[Segment]], flagged: List[int]) -> List[Tuple[int, int, List[int]]]:
    """(first paragraph shown, paragraph after the last shown, line ids to ask) per request: flagged
    lines whose paragraphs lie within GROUP_GAP of each other share one, up to GROUP_LINES lines."""
    where = {s.line_id: p for p, segments in enumerate(paragraphs) for s in segments if s.kind == DIALOGUE}
    found: List[Tuple[int, int, List[int]]] = []
    ids: List[int] = []
    for line_id in sorted(flagged):
        if ids and (where[line_id] - where[ids[-1]] > GROUP_GAP or len(ids) >= GROUP_LINES):
            found.append(_span(paragraphs, where, ids))
            ids = []
        ids.append(line_id)
    if ids:
        found.append(_span(paragraphs, where, ids))
    return found


def _span(paragraphs, where, ids) -> Tuple[int, int, List[int]]:
    return (max(0, where[ids[0]] - CONTEXT_BEFORE), min(len(paragraphs), where[ids[-1]] + CONTEXT_AFTER + 1), ids)


def render(paragraphs: List[List[Segment]], start: int, end: int, ask: List[int], certain: Dict[int, str],
           guessed: Dict[int, str]) -> str:
    """The passage for one request: asked lines as [#N], tag-named lines as [Name] and the first
    pass's other answers as [Name?] (a guess that may be wrong)."""
    shown = []
    for segments in paragraphs[start:end]:
        parts = []
        for piece in segments:
            if piece.kind == DIALOGUE and piece.line_id in ask:
                parts.append(f"[#{piece.line_id}] {piece.text}")
            elif piece.kind == DIALOGUE and piece.line_id in certain:
                parts.append(f"[{certain[piece.line_id]}] {piece.text}")
            elif piece.kind == DIALOGUE and piece.line_id in guessed:
                parts.append(f"[{guessed[piece.line_id]}?] {piece.text}")
            else:
                parts.append(piece.text)
        shown.append(" ".join(parts))
    return "\n\n".join(shown)
