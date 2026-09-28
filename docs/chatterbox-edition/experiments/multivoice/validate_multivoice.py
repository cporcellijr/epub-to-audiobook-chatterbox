"""Multi-voice validation: runs the production attribution code (core.cast_llm through a real local
LLM) over the hand-labelled fixture passages in ./fixture, prints the numbers the build brief asks
for, then narrates a two-minute multi-voice sample through the real pipeline (build_config ->
main -> OpenAITTSProvider in cast mode) for listening.

Runs in a throwaway container from the app image, on the same network as Chatterbox, like the other
scripts under experiments/. LLM_BASE_URL must be reachable from inside that container (the LLM
container's name on the network, or --add-host=host.docker.internal:host-gateway for a server on
the host):

    docker run --rm --network tts -e PYTHONPATH=/app_src \\
      -e OPENAI_BASE_URL=http://chatterbox:8004/v1 -e OPENAI_API_KEY=not-needed \\
      -e TTS_VOICES_DIR=/voices -v <CHATTERBOX_DATA>/voices:/voices:ro \\
      -v <APP_DATA>:/app \\
      -e LLM_BASE_URL=http://<llm host>:11434/v1 -e LLM_MODEL=<model name> \\
      -v <this folder>:/mv \\
      epub_to_audiobook:local python3 /mv/validate_multivoice.py --out /mv/validate_out

<APP_DATA> is mounted so the owner's voice genders (voice_genders.json) drive the sample's voice
suggestions; leave it out and every Chatterbox voice counts as neutral. Add -e LLM_API_KEY=... if the
endpoint wants one, -e LLM_UNLOAD_CHATTERBOX=off to keep Chatterbox loaded during the pass, and
--no-sample to skip the narration. Exit code 1 if the fixture and the splitter disagree, or if
Chatterbox did not come back after the pass.

What it prints:
  - attribution accuracy per passage and overall (a line is right when the predicted character is
    the labelled one under any of its listed aliases; an "unknown" label is right when the model
    left it unknown);
  - the confusion between the most frequent characters;
  - the rate of unusable replies before and after the one retry;
  - seconds per 1,000 dialogue lines;
  - whether Chatterbox reported its model unloaded during the pass and loaded again after it.
"""
import argparse
import glob
import json
import os
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, "/app_src")

from audiobook_generator.config.general_config import GeneralConfig  # noqa: E402
from audiobook_generator.core import cast as cast_store  # noqa: E402
from audiobook_generator.core import cast_llm  # noqa: E402
from audiobook_generator.core import chatterbox_control  # noqa: E402
from audiobook_generator.core import delivery  # noqa: E402
from audiobook_generator.core.cast_llm import ChatClient, Roster, _same_person, attribute_chapter, llm_api_key, \
    llm_base_url, llm_model  # noqa: E402
from audiobook_generator.core.dialogue import PARAGRAPH_MARK, chapter_segments, dialogue_lines  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE_DIR = os.path.join(HERE, "fixture")
TAG = re.compile(r"«([^»]+)»")
SAMPLE_CHARS = 2400  # about two minutes at 20.2 characters per second of speech
TOP_CHARACTERS = 6


class Passage:
    def __init__(self, path: str):
        self.name = os.path.basename(path)
        header, body = {}, []
        with open(path, encoding="utf-8") as f:
            for line in f.read().split("\n"):
                if line.startswith("#"):
                    key, _, value = line[1:].partition(":")
                    header[key.strip()] = value.strip()
                else:
                    body.append(line)
        self.title = header.get("title", self.name)
        self.genders = dict(item.split("=") for item in header.get("characters", "").split(",") if "=" in item)
        self.genders = {k.strip(): v.strip() for k, v in self.genders.items()}
        self.aliases: Dict[str, List[str]] = {}
        for item in header.get("aliases", "").split(";"):
            name, _, names = item.partition("=")
            if name.strip():
                self.aliases[name.strip()] = [n.strip() for n in names.split(",") if n.strip()]
        paragraphs = [p.strip() for p in "\n".join(body).split("\n\n") if p.strip()]
        self.labels = TAG.findall("\n".join(paragraphs))
        self.text = PARAGRAPH_MARK.join(TAG.sub("", p) for p in paragraphs)

    def forms(self, label: str) -> set:
        return {cast_store.normalize_name(n) for n in [label, *self.aliases.get(label, [])]}


def matches(passage: Passage, label: str, character: Optional[dict]) -> bool:
    """Is the predicted character the labelled speaker (under any listed alias)?"""
    if label == "unknown":
        return character is None
    if character is None:
        return False
    wanted = passage.forms(label)
    predicted = {cast_store.normalize_name(n) for n in [character["name"], *character.get("aliases", [])]}
    if wanted & predicted:
        return True
    for w in wanted:
        for p in predicted:
            w_words, p_words = w.split(), p.split()
            if w_words and p_words and _same_person(w_words[0], p_words[0]) and \
                    (len(w_words) == 1 or len(p_words) == 1 or w_words[-1] == p_words[-1]):
                return True
    return False


def label_of(passage: Passage, character: Optional[dict]) -> str:
    """The fixture label a predicted character stands for (for the confusion table)."""
    if character is None:
        return "unknown"
    for label in passage.genders:
        if label != "unknown" and matches(passage, label, character):
            return label
    return f"other ({character['name']})"


def run_attribution(passages: List[Passage], chat) -> Tuple[dict, List[dict]]:
    """Attribute every passage; returns (overall stats, per-passage results)."""
    stats: dict = {}
    results = []
    for passage in passages:
        paragraphs = chapter_segments(passage.text)
        lines = dialogue_lines(passage.text)
        if len(lines) != len(passage.labels):
            print(f"FAIL {passage.name}: the splitter found {len(lines)} dialogue lines but the fixture labels "
                  f"{len(passage.labels)}; fix the fixture or the splitter before measuring.")
            sys.exit(1)
        roster = Roster()
        started = time.monotonic()
        attributed, moods = attribute_chapter(paragraphs, roster, chat, stats, label=f" {passage.name}")
        seconds = time.monotonic() - started
        rule_moods = delivery.segment_moods(paragraphs)
        confusion = Counter()
        correct = 0
        mood_counts = Counter()
        llm_added_moods = Counter()
        for line, label in zip(lines, passage.labels):
            key = attributed.get(line.line_id)
            character = roster.characters.get(key) if key else None
            ok = matches(passage, label, character)
            correct += ok
            confusion[(label, label_of(passage, character))] += 1
            rule_mood = rule_moods.get(line.line_id, delivery.MOOD_NORMAL)
            final_mood = moods.get(line.line_id, delivery.MOOD_NORMAL)
            mood_counts[rule_mood] += 1
            if rule_mood == delivery.MOOD_NORMAL and final_mood != delivery.MOOD_NORMAL:
                llm_added_moods[final_mood] += 1
        results.append({"passage": passage, "correct": correct, "total": len(lines), "seconds": seconds,
                        "confusion": confusion, "roster": roster, "lines": attributed, "moods": moods,
                        "mood_counts": mood_counts, "llm_added_moods": llm_added_moods})
    return stats, results


def print_report(stats: dict, results: List[dict], unloaded_during: Optional[bool], loaded_after: Optional[bool]) -> None:
    print("\n=== Attribution accuracy ===")
    total_correct = total_lines = 0
    for r in results:
        pct = 100.0 * r["correct"] / max(1, r["total"])
        print(f"  {r['passage'].title:<36} {r['correct']:>3}/{r['total']:<3} {pct:5.1f}%   {r['seconds']:6.1f}s")
        total_correct += r["correct"]
        total_lines += r["total"]
    overall = 100.0 * total_correct / max(1, total_lines)
    print(f"  {'overall':<36} {total_correct:>3}/{total_lines:<3} {overall:5.1f}%")

    print(f"\n=== Delivery moods (adaptive delivery, ASK_LLM_FOR_MOODS={cast_llm.ASK_LLM_FOR_MOODS}) ===")
    rule_totals, llm_totals = Counter(), Counter()
    for r in results:
        rule_totals.update(r["mood_counts"])
        llm_totals.update(r["llm_added_moods"])
    print(f"  from rules: {dict(rule_totals)}")
    print(f"  added by the LLM (lines rules called normal): {dict(llm_totals)}")

    print("\n=== Confusion (rows: true speaker, columns: predicted), top characters per passage ===")
    for r in results:
        labels = [l for l, _ in Counter(r["passage"].labels).most_common(TOP_CHARACTERS)]
        columns = [l for l in labels if l != "unknown"] + ["unknown", "other"]
        table = defaultdict(Counter)
        for (true, predicted), n in r["confusion"].items():
            column = predicted if predicted in labels else ("unknown" if predicted == "unknown" else "other")
            table[true][column] += n
        print(f"  {r['passage'].title}")
        print("    " + " " * 18 + "".join(f"{c[:9]:>10}" for c in columns))
        for true in labels:
            print(f"    {true[:18]:<18}" + "".join(f"{table[true][c]:>10}" for c in columns))
        others = [(name, c["lines"]) for name, c in
                  ((c["name"], c) for c in r["roster"].characters.values())
                  if label_of(r["passage"], c).startswith("other")]
        if others:
            print("    characters the model invented or split off: " + ", ".join(f"{n} ({k})" for n, k in others))

    windows = max(1, stats.get("windows", 0))
    print("\n=== Replies ===")
    print(f"  windows asked: {stats.get('windows', 0)}")
    print(f"  windows whose first reply was unusable: {stats.get('invalid_json', 0)} "
          f"({100.0 * stats.get('invalid_json', 0) / windows:.1f}% of windows)")
    print(f"  windows still unusable after the retry: {stats.get('invalid_after_retry', 0)} "
          f"({100.0 * stats.get('invalid_after_retry', 0) / windows:.1f}% of windows)")
    seconds = sum(r["seconds"] for r in results)
    print(f"\n=== Speed ===\n  {seconds:.1f}s for {total_lines} lines = {1000.0 * seconds / max(1, total_lines):.0f}s per 1,000 lines")

    print("\n=== Chatterbox ===")
    if unloaded_during is None:
        print("  unload not attempted (LLM_UNLOAD_CHATTERBOX=off or Chatterbox unreachable)")
    else:
        print(f"  model reported unloaded during the pass: {'yes' if unloaded_during else 'NO'}")
    if loaded_after is not None:
        print(f"  model reported loaded after the pass: {'yes' if loaded_after else 'NO'}")


def narrate_sample(result: dict, out_dir: str) -> Optional[str]:
    """Write the first ~two minutes of the passage as a one-chapter EPUB, save its cast (from the
    attribution just made, with suggested voices), and narrate it in cast mode through main()."""
    from ebooklib import epub
    from main import main

    passage: Passage = result["passage"]
    paragraphs = passage.text.split(PARAGRAPH_MARK)
    chosen, size = [], 0
    for p in paragraphs:
        if size + len(p) > SAMPLE_CHARS and chosen:
            break
        chosen.append(p)
        size += len(p)
    os.makedirs(out_dir, exist_ok=True)
    book_path = os.path.join(out_dir, "sample.epub")
    book = epub.EpubBook()
    book.set_identifier("multivoice-sample")
    book.set_title("Multi-voice sample")
    book.add_author("Invented")
    chapter = epub.EpubHtml(title=passage.title, file_name="c1.xhtml", lang="en")
    chapter.content = f"<html><body><h1>{passage.title}</h1>" + "".join(f"<p>{p}</p>" for p in chosen) + "</body></html>"
    book.add_item(chapter)
    book.toc = [chapter]
    book.spine = ["nav", chapter]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(book_path, book)

    # The chapter text as the parser will hand it to the provider, so the cast's hash matches.
    from audiobook_generator.book_parsers.base_book_parser import get_book_parser
    config = GeneralConfig(None)
    config.input_file, config.title_mode, config.newline_mode = book_path, "auto", "double"
    config.remove_endnotes = config.remove_reference_numbers = False
    config.search_and_replace_file = None
    chapters = [(t, x) for t, x in get_book_parser(config).get_chapters(f" {PARAGRAPH_MARK}") if x.strip()]
    if not chapters:
        print("  could not read the sample EPUB back; no sample made")
        return None
    text = chapters[0][1]
    sample_lines = dialogue_lines(text)

    voices_dir = os.environ.get("TTS_VOICES_DIR", "")
    voices = sorted(n for n in os.listdir(voices_dir) if n.lower().endswith((".wav", ".mp3"))) if os.path.isdir(voices_dir) else []
    if not voices:
        print("  TTS_VOICES_DIR has no voices; no sample made")
        return None
    genders = cast_store.load_voice_genders()
    narrator = os.environ.get("OPENAI_DEFAULT_VOICE") if os.environ.get("OPENAI_DEFAULT_VOICE") in voices else voices[0]

    roster: Roster = result["roster"]
    cast = cast_store.new_cast(cast_store.cast_key(book_path), book_path, "Multi-voice sample", "Invented", "chatterbox",
                               narrator, [1])
    cast["characters"] = {k: dict(c) for k, c in roster.characters.items()}
    # The sample's line ids are the passage's first N ids (same text, same splitter).
    cast["chapters"][cast_store.text_hash(text)] = {
        "number": 1, "title": passage.title,
        "lines": {str(line.line_id): result["lines"].get(line.line_id) for line in sample_lines},
        "unknown": sum(1 for line in sample_lines if result["lines"].get(line.line_id) is None),
    }
    suggestions = cast_store.suggest_voices(cast, [(v, cast_store.voice_gender("chatterbox", v, genders)) for v in voices],
                                            narrator)
    for key, voice in suggestions.items():
        cast["characters"][key]["voice"] = voice
    cast["status"], cast["chapters_done"] = cast_store.STATUS_DONE, 1
    cast_path = os.path.join(out_dir, "sample_cast.json")
    cast_store.save_cast(cast_path, cast)
    dialogue_voice = next((v for v in voices if v != narrator and v not in suggestions.values()), narrator)
    print(f"  narrator {narrator}, dialogue voice {dialogue_voice}, cast: " +
          ", ".join(f"{c['name']} -> {c.get('voice')}" for c in cast["characters"].values()))

    from audiobook_generator.ui.chatterbox_ui import build_config
    job = build_config(book_path, os.path.join(out_dir, "sample_book"), narrator, 1.0, [1], 0.35, 0.9, False, False,
                       False, "auto", "double", False, False, None, "INFO", "sentence", "chatterbox", "cast",
                       dialogue_voice, cast_path)
    ok = main(job, os.path.join(out_dir, "sample.log"))
    files = glob.glob(os.path.join(out_dir, "sample_book", "*.mp3"))
    print(f"  narration {'succeeded' if ok else 'FAILED'}: {files[0] if files else 'no file'}")
    return files[0] if files else None


def main_cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=os.path.join(HERE, "validate_out"), help="folder for the sample and the JSON dump")
    parser.add_argument("--no-sample", action="store_true", help="skip the two-minute narration")
    parser.add_argument("--fixture", default=FIXTURE_DIR, help="folder of labelled passages")
    args = parser.parse_args()

    if not llm_base_url() or not llm_model():
        print("Set LLM_BASE_URL and LLM_MODEL (the local OpenAI-compatible chat endpoint) first.")
        return 1
    passages = [Passage(p) for p in sorted(glob.glob(os.path.join(args.fixture, "*.txt")))]
    print(f"{len(passages)} passages, {sum(len(p.labels) for p in passages)} labelled lines; "
          f"LLM {llm_model()} at {llm_base_url()}")

    unloaded_during = loaded_after = None
    unloaded = False
    if chatterbox_control.unload_enabled() and chatterbox_control.model_loaded() is not None:
        unloaded = chatterbox_control.unload()
        unloaded_during = chatterbox_control.model_loaded() is False if unloaded else None
    try:
        chat = ChatClient(llm_base_url(), llm_model(), llm_api_key())
        stats, results = run_attribution(passages, chat)
        print(f"  JSON mode accepted by the endpoint: {'yes' if chat.json_mode else 'no (parsed plain replies)'}")
    finally:
        if unloaded:
            loaded_after = chatterbox_control.reload()
    print_report(stats, results, unloaded_during, loaded_after)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "attribution.json"), "w", encoding="utf-8") as f:
        json.dump([{"passage": r["passage"].name, "correct": r["correct"], "total": r["total"], "seconds": r["seconds"],
                    "characters": r["roster"].characters,
                    "lines": {str(k): v for k, v in r["lines"].items()}} for r in results], f, indent=1, ensure_ascii=False)

    if not args.no_sample:
        print("\n=== Two-minute multi-voice sample ===")
        if loaded_after is False:
            print("  Chatterbox did not come back; no sample made")
        else:
            narrate_sample(results[0], args.out)
    return 1 if loaded_after is False else 0


if __name__ == "__main__":
    sys.exit(main_cli())
