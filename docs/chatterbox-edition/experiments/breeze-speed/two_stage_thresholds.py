"""Pass mark for the quick model: two-stage passes a take if quick scores >= mark, else small decides."""
import json
import os

from audiobook_generator.core import speech_check

texts = {i["id"]: i["text"] for g in json.load(open("/tmp/bench/workload.json")).values() for i in g}
takes = []
for label in ("graph", "eager"):
    for group in sorted(os.listdir(f"/tmp/bench/out/{label}")):
        for name in sorted(os.listdir(f"/tmp/bench/out/{label}/{group}")):
            takes.append((group.rstrip("0123456789"), f"{label}/{group}", name[:-4]))
wrong = {}
for i, (kind, batch, tid) in enumerate(takes):
    j = i + 1 if i + 1 < len(takes) and takes[i + 1][1] == batch else i - 1
    wrong[f"{batch}|{tid}"] = texts[takes[j][2]]
heard = {m: json.load(open(f"/tmp/bench/heard_{m}.json")) for m in ("small", "tiny.en", "base.en")}
small = heard["small"]
PASS = speech_check.PASS_SCORE


def cases():
    """(test, kind, expected text, test key, transcript key) for every judged case."""
    for kind, batch, tid in takes:
        key = f"{batch}|{tid}"
        yield "true", kind, texts[tid], "true", key
        yield "wrong", kind, wrong[key], "true", key
        if key in small["cut"]:
            yield "cut", kind, texts[tid], "cut", key
        if key in small["tail"]:
            yield "tail", kind, texts[tid], "tail", key


for quick in ("tiny.en", "base.en"):
    for mark in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0):
        stats = {}
        for test, kind, text, part, key in cases():
            q = speech_check.match(text, heard[quick][part][key])
            s = speech_check.match(text, small[part][key])
            if q is None or s is None:
                continue
            st = stats.setdefault(test, {"n": 0, "sent": 0, "false_pass": 0, "small_rejects": 0,
                                         "sent_short": 0, "n_short": 0})
            st["n"] += 1
            st["small_rejects"] += s < PASS
            if q >= mark:
                st["false_pass"] += s < PASS  # passed quickly although small would reject it
            else:
                st["sent"] += 1
                st["sent_short"] += kind == "short"
            st["n_short"] += kind == "short"
        t = stats["true"]
        print(f"{quick} mark {mark:.2f}: real takes sent on to small {t['sent']}/{t['n']} "
              f"(short {t['sent_short']}/{t['n_short']}); passed although small rejects: "
              + ", ".join(f"{k} {v['false_pass']}/{v['small_rejects']}" for k, v in stats.items()))
