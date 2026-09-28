"""Split chapter text into narration and dialogue without an LLM.

A chapter is a sequence of paragraphs (PARAGRAPH_MARK-separated, as the EPUB parser produces
them). Each paragraph becomes ordered segments: narration, or one quoted span of dialogue. Every
dialogue segment carries a line id that is stable within the chapter (1, 2, 3 ... in reading
order), which is what the LLM attribution and the saved cast refer to.

Quote styles handled:
  - double quotes, straight ("...") and curly (“...”), the common English style;
  - single-quote dialogue ('...' and ‘...’, the common British style), while apostrophes
    inside words (don't, o'clock) and possessives (James' hat) stay narration;
  - em-dash dialogue (a paragraph opening with an em dash), only in a chapter that uses no quote
    marks for dialogue at all;
  - a quotation that runs over several paragraphs: each paragraph opens with a quote mark and only
    the last one closes. The continuation paragraphs are separate lines marked `continues`, so the
    attribution can give them the previous line's speaker without asking.

One style per chapter: the style with more openings wins, and the other kind of mark is treated as
ordinary text (a single quote nested inside double-quoted speech stays inside the speech).
"""
from typing import List, NamedTuple, Optional, Tuple

PARAGRAPH_MARK = "@BRK#"  # the EPUB parser's paragraph separator (re-exported by the OpenAI provider)

NARRATION = "narration"
DIALOGUE = "dialogue"

_DOUBLE_OPEN = "\"“„«"
_DOUBLE_CLOSE = "\"”»"
_SINGLE_OPEN = "'‘"
_SINGLE_CLOSE = "'’"
_STRAIGHT = "\"'"  # marks that can be either an opening or a closing
_DASHES = "—–"  # em dash, en dash
# What may sit right before a straight opening quote mark: start of paragraph, whitespace, or an
# opening bracket / dash. A straight quote after a letter is an apostrophe, never an opening.
_BEFORE_OPEN = " \t(—–-["


class Segment(NamedTuple):
    kind: str          # NARRATION or DIALOGUE
    line_id: int       # dialogue line number within the chapter (0 for narration)
    text: str
    continues: bool = False  # dialogue continuing an unclosed quotation from the previous paragraph


def paragraphs_of(text: str) -> List[str]:
    """The chapter's non-empty paragraphs with whitespace collapsed: exactly the paragraphs
    paced_units() numbers, so paragraph numbers agree between the two."""
    return [p for p in (" ".join(part.split()) for part in text.split(PARAGRAPH_MARK)) if p]


def _is_letter(char: str) -> bool:
    return char.isalpha()


def _opens_here(paragraph: str, i: int, opens: str) -> bool:
    """True if the mark at position i can open a quotation: preceded by nothing or a separator and
    followed by something that is not whitespace."""
    char = paragraph[i]
    if char not in opens:
        return False
    before = paragraph[i - 1] if i > 0 else " "
    after = paragraph[i + 1] if i + 1 < len(paragraph) else " "
    if after.isspace():
        return False
    if char in _STRAIGHT:
        return before in _BEFORE_OPEN
    return not before.isalnum()  # a dedicated opening mark (curly) only needs not to sit inside a word


def _closes_here(paragraph: str, i: int, closes: str, single: bool) -> bool:
    """True if the mark at position i closes the open quotation. For single quotes a mark with a
    letter on both sides is an apostrophe (don't), and one followed by a letter is never a close."""
    char = paragraph[i]
    if char not in closes:
        return False
    if not single:
        return True
    after = paragraph[i + 1] if i + 1 < len(paragraph) else " "
    before = paragraph[i - 1] if i > 0 else " "
    if _is_letter(after):
        return False
    if before.isspace():  # a stray mark after a space is an opening, not a close
        return False
    return True


def _count_openings(paragraphs: List[str], opens: str) -> int:
    return sum(1 for paragraph in paragraphs for i in range(len(paragraph)) if _opens_here(paragraph, i, opens))


def detect_quote_style(paragraphs: List[str]) -> Optional[str]:
    """"double", "single", "dash", or None when the chapter has no dialogue marks at all."""
    doubles = _count_openings(paragraphs, _DOUBLE_OPEN)
    singles = _count_openings(paragraphs, _SINGLE_OPEN)
    if doubles == 0 and singles == 0:
        dashes = sum(1 for p in paragraphs if len(p) > 1 and p[0] in _DASHES)
        return "dash" if dashes else None
    return "double" if doubles >= singles else "single"


def _split_dash_paragraph(paragraph: str) -> Tuple[List[Tuple[str, str]], bool]:
    """Em-dash style: a paragraph opening with a dash is speech; a later " -- " starts narration
    (or speech again), alternating. Returns (kind, text) pairs and whether speech is left open
    (never, for this style)."""
    if not (len(paragraph) > 1 and paragraph[0] in _DASHES):
        return [(NARRATION, paragraph)], False
    parts, kind = [], DIALOGUE
    start = 1
    i = 1
    while i < len(paragraph):
        if paragraph[i] in _DASHES and i > 0 and paragraph[i - 1] == " ":
            piece = paragraph[start:i].strip()
            if piece:
                parts.append((kind, piece))
            kind = NARRATION if kind == DIALOGUE else DIALOGUE
            start = i + 1
        i += 1
    piece = paragraph[start:].strip()
    if piece:
        parts.append((kind, piece))
    return parts, False


def _split_quoted_paragraph(paragraph: str, style: str) -> Tuple[List[Tuple[str, str]], bool]:
    """Split one paragraph at quote marks of the chapter's style. Returns (kind, text) pairs, in
    order, and whether the paragraph ended inside an open quotation (a speech that continues in
    the next paragraph, or a missing closing mark)."""
    single = style == "single"
    opens = _SINGLE_OPEN if single else _DOUBLE_OPEN
    closes = _SINGLE_CLOSE if single else _DOUBLE_CLOSE
    parts: List[Tuple[str, str]] = []
    start, inside = 0, False
    for i, char in enumerate(paragraph):
        if not inside:
            if _opens_here(paragraph, i, opens):
                piece = paragraph[start:i].strip()
                if piece:
                    parts.append((NARRATION, piece))
                start, inside = i, True
        elif (_closes_here(paragraph, i, closes, single)
              or (not single and char in "“„«" and i == len(paragraph) - 1)):
            # Some EPUBs end a quotation with an opening curly mark by mistake. At the
            # paragraph's end it cannot open another quotation, so close this one.
            # A single-quote mark straight after a letter (James' hat) is a possessive when the
            # speech still has a closing mark further on; typeset speech closes after punctuation.
            if single and paragraph[i - 1].isalpha() and any(
                    _closes_here(paragraph, j, closes, single) for j in range(i + 1, len(paragraph))):
                continue
            parts.append((DIALOGUE, paragraph[start:i + 1].strip()))
            start, inside = i + 1, False
    tail = paragraph[start:].strip()
    if tail:
        parts.append((DIALOGUE if inside else NARRATION, tail))
    return parts, inside


def split_paragraph(paragraph: str, style: Optional[str]) -> Tuple[List[Tuple[str, str]], bool]:
    """(kind, text) pieces of one paragraph for the chapter's quote style, plus whether the
    paragraph ended with its quotation still open."""
    if style is None:
        return [(NARRATION, paragraph)], False
    if style == "dash":
        return _split_dash_paragraph(paragraph)
    return _split_quoted_paragraph(paragraph, style)


def chapter_segments(text: str) -> List[List[Segment]]:
    """Segments per paragraph for a whole chapter, with dialogue line ids numbered 1.. in reading
    order. The paragraph list matches paragraphs_of(text) one to one (paragraphs with no
    speakable text are kept; the unit builder drops them exactly as before)."""
    paragraphs = paragraphs_of(text)
    style = detect_quote_style(paragraphs)
    result: List[List[Segment]] = []
    next_id = 1
    left_open = False
    for paragraph in paragraphs:
        pieces, open_at_end = split_paragraph(paragraph, style)
        segments: List[Segment] = []
        for index, (kind, piece) in enumerate(pieces):
            if kind == DIALOGUE:
                continues = left_open and index == 0
                segments.append(Segment(DIALOGUE, next_id, piece, continues))
                next_id += 1
            else:
                segments.append(Segment(NARRATION, 0, piece))
        # Only a paragraph that opens with speech can continue the previous paragraph's quotation;
        # anything else means the previous paragraph simply lacked its closing mark.
        left_open = open_at_end and bool(pieces) and pieces[-1][0] == DIALOGUE
        result.append(segments)
    return result


def dialogue_lines(text: str) -> List[Segment]:
    """Just the dialogue segments of a chapter, in order."""
    return [segment for paragraph in chapter_segments(text) for segment in paragraph if segment.kind == DIALOGUE]
