"""Pick which chapters to narrate: skip front/back matter (title pages, copyright, contents,
blurbs, acknowledgements, newsletter sign-ups, ...) while keeping the story.

Scoring adapted from abogen's chapter_classification (MIT, github.com/denizsafak/abogen), with
three changes: short sections only count against a chapter inside the leading/trailing runs of
the book (a mid-book "Part One" divider stays), "chapter/part/book" only marks real content when
the title *starts* with it ("Five Book Collection" is a blurb, "Book Two" is not), and text
keywords only count on short sections (a long chapter mentioning "index" or "copyright" is story).
"""
import re
from typing import List, Tuple

_SUPPLEMENT_TITLES = [
    (re.compile(r"\btitle\s+page\b"), 3.0),
    (re.compile(r"\bcopyright\b"), 2.4),
    (re.compile(r"\btable\s+of\s+contents\b"), 2.8),
    (re.compile(r"^contents\b"), 2.4),
    (re.compile(r"\backnowledg(e)?ments?\b"), 2.0),
    (re.compile(r"\bdedication\b"), 2.0),
    (re.compile(r"\babout\s+the\s+authors?\b"), 2.4),
    (re.compile(r"\balso\s+by\b"), 2.2),
    (re.compile(r"\bbooks\s+by\b"), 2.0),
    (re.compile(r"\bpraise\s+for\b"), 2.0),
    (re.compile(r"\bcolophon\b"), 2.2),
    (re.compile(r"\bnewsletter\b"), 2.4),
    (re.compile(r"\bsneak\s+peek\b"), 1.6),
    (re.compile(r"\bexcerpt\b"), 1.4),
    (re.compile(r"\bglossary\b"), 2.0),
    (re.compile(r"^index\b"), 2.0),
    (re.compile(r"\bbibliograph(y|ies)\b"), 2.0),
    (re.compile(r"\bappendix\b"), 1.2),
    (re.compile(r"\bcoming\s+soon\b"), 1.4),
    (re.compile(r"\b(meet|stalk|follow|contact)\s+the\s+authors?\b"), 1.4),
]

# Headings that mark narration when they START the title ("Chapter 3", "Part One", "Prologue").
_CONTENT_HEADING = re.compile(r"^(chapter|part|book|section|prologue|epilogue|interlude|act)\b")

_SUPPLEMENT_TEXT = [
    ("copyright", 1.2),
    ("all rights reserved", 1.1),
    ("isbn", 0.9),
    ("library of congress", 1.0),
    ("table of contents", 1.0),
    ("dedicated to", 0.8),
    ("acknowledg", 0.8),
    ("printed in", 0.6),
    ("praise for", 0.9),
    ("also by", 0.9),
    ("newsletter", 3.2),
    ("mailing list", 2.6),
    ("sign up", 1.6),
    ("sign-up", 2.2),
    ("leave a review", 1.6),
    ("thank you for reading", 1.2),
    ("thank you for buying", 1.2),
    ("we hope you enjoyed", 1.2),
    ("if you enjoyed", 1.2),
    ("if you liked", 1.0),
    ("discover your next", 1.2),
    ("sneak peek", 1.0),
    ("www.", 0.6),
]

STRONG_SUPPLEMENT = 1.9   # excluded anywhere in the book
EDGE_SUPPLEMENT = 1.0     # excluded inside the leading/trailing runs
SHORT_FRONT = 1000        # characters (~50 s): title pages, dedications, epigraphs, warnings
SHORT_BACK = 400          # stricter at the end, where a short final chapter is often an epilogue
KEYWORD_TEXT_LIMIT = 3000  # text keywords only count on sections shorter than this
ALWAYS_STORY = 20000       # characters (~17 min); some EPUBs put a whole novel, copyright line first,
                           # in one section -- never drop anything this long


def supplement_score(title: str, text: str) -> float:
    """Higher = more likely non-story material."""
    title = re.sub(r"[_\s]+", " ", title or "").strip().lower()
    body = (text or "").strip()
    if len(body) >= ALWAYS_STORY:
        return -10.0
    score = 0.0
    for pattern, weight in _SUPPLEMENT_TITLES:
        if pattern.search(title):
            score += weight
    if _CONTENT_HEADING.search(title):
        score -= 2.0
    if len(body) <= 150:
        score += 0.9
    elif len(body) <= 400:
        score += 0.6
    elif len(body) <= 800:
        score += 0.35
    if len(body) < KEYWORD_TEXT_LIMIT:
        lowered = body.lower()
        for keyword, weight in _SUPPLEMENT_TEXT:
            if keyword in lowered:
                score += weight
    return score


def _is_edge_matter(title: str, text: str, short_limit: int) -> bool:
    """Front/back matter test for sections at the start or end of the book."""
    score = supplement_score(title, text)
    return score >= EDGE_SUPPLEMENT or (len((text or "").strip()) < short_limit and score > -1.0)


def preselect_chapters(chapters: List[Tuple[str, str]]) -> List[bool]:
    """Include flags for (title, text) chapters: story in, front/back matter out."""
    count = len(chapters)
    if count <= 1:
        return [True] * count
    include = [supplement_score(title, text) < STRONG_SUPPLEMENT for title, text in chapters]

    for index in range(count):  # leading run
        if not _is_edge_matter(*chapters[index], SHORT_FRONT):
            break
        include[index] = False
    for index in range(count - 1, -1, -1):  # trailing run
        if not _is_edge_matter(*chapters[index], SHORT_BACK):
            break
        include[index] = False

    if not any(include):
        include[max(range(count), key=lambda i: len(chapters[i][1]))] = True
    return include
