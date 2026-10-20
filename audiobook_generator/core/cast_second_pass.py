"""Second-pass speaker refinement after the 14B analysis.

WORKLOG §81: after the 14B's analysis, a second model (a fine-tuned 7B) asks every
first-pass window again with the finished cast held fixed and a closed list
("every speaker is one of these"). An answer that names exactly one cast entry
replaces that line's speaker; anything else keeps the first pass's line. Tag-named
lines, first-person "I said" anchors and quoted terms are never asked. Chapter
narrators and the character list are never changed.
"""

import logging
import os
from collections import Counter
from typing import Callable, Dict, List, Optional

import requests

from audiobook_generator.core import cast_llm
from audiobook_generator.core import cast as cast_store
from audiobook_generator.core.cast import normalize_name
from audiobook_generator.core.cast_llm import AttributionError, Chat, parse_reply, render_window
from audiobook_generator.core.cast_profiles import ChapterText
from audiobook_generator.core.dialogue import DIALOGUE
from audiobook_generator.core.speech_tags import contradicted, first_person_tagged, quoted_terms, tagged_speakers

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen2.5-cast6:7b"
FIXED_HEADER = 'Characters (every speaker is one of these; answer with the exact name shown, or "unknown"):'
PRODUCTION_HEADER = "Known characters so far (answer with the exact name or @key shown here):"
NARRATOR_WORDS = frozenset({"i", "me", "narrator", "the narrator"})


def second_pass_model() -> str:
    """Model name for the second pass, read per call."""
    return os.environ.get("CAST_SECOND_PASS_MODEL", DEFAULT_MODEL).strip()


def fixed_window_prompt() -> str:
    """Window prompt with the production header replaced by the fixed header."""
    prompt = cast_llm.PROMPTS["window"]
    if not prompt.startswith(PRODUCTION_HEADER):
        raise ValueError("window prompt does not start with PRODUCTION_HEADER")
    return prompt.replace(PRODUCTION_HEADER, FIXED_HEADER, 1)


def model_available(base_url: str, model: str, timeout: float = 5) -> bool:
    """Return True if the model is listed by the Ollama-compatible endpoint."""
    if not base_url or not model:
        return False
    try:
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        url = f"{root}/api/tags"
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200:
            logger.info("Second pass skipped: model list returned %s", resp.status_code)
            return False
        data = resp.json()
        names = {m.get("name") or m.get("model") for m in data.get("models", [])}
        if model in names or f"{model}:latest" in names:
            return True
        logger.info("Second pass skipped: model %s not available", model)
        return False
    except Exception as exc:
        logger.info("Second pass skipped: %s", exc)
        return False


def _same(a: str, b: str) -> bool:
    """True if one normalized name is a single word matching first or last word of the other."""
    aw = a.split()
    bw = b.split()
    return (len(aw) == 1 and aw[0] in (bw[0], bw[-1])) or (len(bw) == 1 and bw[0] in (aw[0], aw[-1]))


def roster_names(characters: Dict[str, dict]) -> List[str]:
    """Display names of characters with lines >0, most lines first, unique by normalized name."""
    best = {}
    for key, ch in characters.items():
        lines = int(ch.get("lines", 0))
        if lines <= 0:
            continue
        name = ch.get("name", "")
        norm = normalize_name(name)
        if norm not in best or lines > best[norm][0]:
            best[norm] = (lines, name)
    ordered = sorted(best.values(), key=lambda x: -x[0])
    return [name for _, name in ordered]


def resolver(characters: Dict[str, dict]) -> Callable[[Optional[str], Optional[str], Dict[str, int]], Optional[str]]:
    """Build to_key resolver for speaker names."""
    displayed = {}
    forms = {}
    for key, ch in characters.items():
        if int(ch.get("lines", 0)) <= 0:  # not on the roster the model sees
            continue
        name = ch.get("name", "")
        norm = normalize_name(name)
        displayed.setdefault(norm, []).append(key)
        for form in [name] + ch.get("aliases", []):
            if not form:
                continue
            nf = normalize_name(form)
            forms.setdefault(nf, set()).add(key)

    def to_key(answer: Optional[str], narrator_key: Optional[str], local_counts: Dict[str, int]) -> Optional[str]:
        a = normalize_name(answer or "")
        if not a:
            return None
        if a in NARRATOR_WORDS:
            return narrator_key
        if a in displayed:
            candidates = displayed[a]
            best_key = None
            best_score = (-1, -1)
            for k in candidates:
                score = (local_counts.get(k, 0), int(characters[k].get("lines", 0)))
                if score > best_score:
                    best_score = score
                    best_key = k
            return best_key
        found = set()
        if a in forms:
            found.update(forms[a])
        else:
            for f, ks in forms.items():
                if f and _same(f, a):
                    found.update(ks)
        if len(found) > 1:
            filtered = {k for k in found if local_counts.get(k, 0) > 0}
            if filtered:
                found = filtered
        if len(found) == 1:
            return next(iter(found))
        return None

    return to_key


def second_pass(
    cast: dict,
    chapters: List[ChapterText],
    chat: Chat,
    model: str,
    log: logging.Logger = logger,
    save: Callable[[], None] = lambda: None,
) -> None:
    """Ask every chapter's windows again with the cast fixed; progress and counts in cast["second_pass"].
    Changes are applied only after the last chapter, so an exception from chat (server down) leaves the
    first pass's lines as they were."""
    chapters_by_number = {c.number: c for c in chapters}
    characters = cast.get("characters", {})
    roster_text = ", ".join(roster_names(characters))
    to_key = resolver(characters)
    template = fixed_window_prompt()

    cast["second_pass"] = {
        "model": model,
        "done": 0,
        "total": len(chapters),
        "windows": 0,
        "asked": 0,
        "changed": 0,
        "unusable": 0,
        "error": "",
    }
    save()

    chapter_replacements: Dict[int, Dict[int, str]] = {}
    total_asked = 0

    for chapter in chapters:
        entry = next((e for e in cast.get("chapters", {}).values() if e.get("number") == chapter.number), None)
        if not entry:
            continue

        anchors = tagged_speakers(chapter.paragraphs, include_continued=True)
        paragraphs = [[s._replace(continues=False) if s.line_id in anchors else s for s in p] for p in chapter.paragraphs]

        narrator_key = entry.get("narrator")
        if narrator_key not in characters:
            narrator_key = None
        narrator_name = characters[narrator_key]["name"] if narrator_key else "I"

        contradictions = set(contradicted(paragraphs))
        for line_id in first_person_tagged(paragraphs):
            if line_id not in contradictions:
                anchors.setdefault(line_id, narrator_name)

        terms = quoted_terms(paragraphs)
        for line_id in terms:
            anchors.pop(line_id, None)

        narrator_line = cast_llm.PROMPTS["narrator"].format(name=narrator_name, aliases="") if narrator_key else ""
        local = Counter(v for v in entry.get("lines", {}).values() if v)

        windows = cast_llm.build_windows(paragraphs, known=anchors, skip=terms)
        answers: Dict[int, Optional[str]] = {}
        replaced: Dict[int, str] = {}

        for window in windows:
            cast["second_pass"]["windows"] += 1
            known = dict(anchors)
            for line_id in answers:
                key = replaced.get(line_id) or entry.get("lines", {}).get(str(line_id))
                if key in characters:
                    known[line_id] = characters[key]["name"]
            content = template.format(
                roster=roster_text,
                narrator=narrator_line,
                passage=render_window(paragraphs, window, known, terms),
                ids=", ".join(str(i) for i in window.ids),
                moods_shape="",
                moods_rule="",
            )
            messages = [
                {"role": "system", "content": cast_llm.PROMPTS["system"]},
                {"role": "user", "content": content},
            ]
            speakers = {}
            for attempt in (1, 2):
                try:
                    reply = chat(messages)
                    speakers, _, _ = parse_reply(reply, window.ids, ignore_ids={*anchors, *terms})
                    break
                except AttributionError as e:
                    cast["second_pass"]["unusable"] += 1
                    if attempt == 2:
                        log.warning("Cast second pass window unusable twice (%s); lines kept", e)
                        speakers = {}
                    else:
                        log.warning("Cast second pass window unusable (%s); retrying", e)

            for line_id in window.ids:
                ans = speakers.get(line_id)
                answers[line_id] = ans
                if ans:
                    key = to_key(ans, narrator_key, local)
                    # Only someone the first pass already hears in this chapter (or its narrator): a name
                    # brought in from elsewhere spread over a whole scene (Six Wakes ch. 28, WORKLOG §81).
                    if key and (local.get(key) or key == narrator_key):
                        replaced[line_id] = key
        for paragraph in paragraphs:
            for segment in paragraph:
                if segment.kind == DIALOGUE and segment.continues and segment.line_id - 1 in replaced:
                    replaced[segment.line_id] = replaced[segment.line_id - 1]

        chapter_replacements[chapter.number] = replaced
        cast["second_pass"]["done"] += 1
        total_asked += len(answers)
        cast["second_pass"]["asked"] = total_asked
        save()
        changed_this = 0
        for lid, key in replaced.items():
            old = entry.get("lines", {}).get(str(lid))
            if old != key:
                changed_this += 1
        log.info("Cast: second pass chapter %s: %s asked, %s changed", chapter.number, len(answers), changed_this)

    changed_total = 0
    for number, repl in chapter_replacements.items():
        entry = next((e for e in cast.get("chapters", {}).values() if e.get("number") == number), None)
        if not entry:
            continue
        lines = entry.setdefault("lines", {})
        chapter_obj = chapters_by_number.get(number)
        for line_id, key in repl.items():
            line_key = str(line_id)
            old = lines.get(line_key)
            if old != key:
                changed_total += 1
                lines[line_key] = key
            if chapter_obj is not None:
                chapter_obj.lines[line_id] = key

    cast["second_pass"]["changed"] = changed_total
    cast_store.refresh_cast_counts(cast)
    save()
