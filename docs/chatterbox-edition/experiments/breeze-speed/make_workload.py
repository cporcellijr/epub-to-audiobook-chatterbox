"""Fixed Breeze benchmark workload from real book text: long, medium and short groups, four voices."""
import json
import random
import re
import types

from audiobook_generator.book_parsers.base_book_parser import get_book_parser
from audiobook_generator.tts_providers import openai_tts_provider as p

cfg = types.SimpleNamespace(input_file="/library/Robert Lubrican/Stranded (2011).epub", newline_mode="double",
                            title_mode="auto", remove_endnotes=False, remove_reference_numbers=False,
                            search_and_replace_file=None, chapter_start=1, chapter_end=-1, language="en",
                            tts="openai", engine="breeze", voice_mode="cast")
chapters = [(t, x) for t, x in get_book_parser(cfg).get_chapters(" ") if x.strip()]
texts = []
for _, x in chapters:
    texts += [p._chatterbox_input(u[1]) for u in p.paced_units(x, "en")]
texts = [t for t in dict.fromkeys(texts) if t.strip()]

ref = json.load(open("/app/voice_transcripts.json"))["voices"]
voices = ["Chloe.wav", "Adrian.wav", "Cora.wav", "Elena.wav"]
rng = random.Random(7)


def items(group, tag):
    return [{"id": f"{tag}{i}", "text": t, "speaker": "S0", "voice": voices[i % 4],
             "ref_text": ref[voices[i % 4]]["text"].strip()} for i, t in enumerate(group)]


# Cast mode splits dialogue from its tags, which is where most short units come from.
QUOTED = re.compile('["“][^"“”]{4,26}["”]')
by_len = sorted(texts, key=len, reverse=True)
long_ = by_len[:64]
medium = [t for t in texts if 45 <= len(t) <= 75]
rng.shuffle(medium)
medium = sorted(medium[:128], key=len, reverse=True)
short = list(dict.fromkeys([m.group(0) for t in texts for m in QUOTED.finditer(t)]
                           + [t for t in texts if 5 <= len(t) <= 30]))
rng.shuffle(short)
short = sorted(short[:128], key=len, reverse=True)
out = {"long": items(long_, "L"), "medium": items(medium, "M"), "short": items(short, "S")}
# Batch-size tests (WORKLOG §56): the lengths between medium and long.
for name, low, high, tag in (("mid", 90, 150, "D"), ("midlong", 150, 250, "G")):
    group = [t for t in texts if low <= len(t) <= high]
    rng.shuffle(group)
    out[name] = items(sorted(group[:128], key=len, reverse=True), tag)
json.dump(out, open("/tmp/workload.json", "w"), ensure_ascii=False, indent=0)
for k, v in out.items():
    ls = [len(i["text"]) for i in v]
    print(k, len(v), "units, chars", min(ls), "-", max(ls), "mean", round(sum(ls) / len(ls)))
