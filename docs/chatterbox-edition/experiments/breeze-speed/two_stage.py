"""Would a quick Whisper pass (tiny.en / base.en) let the speech check skip Whisper small on most takes?

Every saved take is heard by each model the way Breeze's batch checks hear it (SpeechChecker.transcribe,
words=False, beam 1, 6 workers x 3 threads). Bad takes are made two ways: a take scored against another
line's text of similar length (wrong words), and a take cut to its first 60% (missing words).
Two-stage verdict = the quick model passes it, else small's verdict. What matters: how much time it
saves, and whether the quick model ever passes a take small rejects.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from faster_whisper import WhisperModel
from pydub import AudioSegment

from audiobook_generator.core import speech_check

texts = {i["id"]: i["text"] for g in json.load(open("/tmp/bench/workload.json")).values() for i in g}
takes = []  # (kind, group, id, audio)
for label in ("graph", "eager"):
    for group in sorted(os.listdir(f"/tmp/bench/out/{label}")):
        for name in sorted(os.listdir(f"/tmp/bench/out/{label}/{group}")):
            audio = AudioSegment.from_wav(f"/tmp/bench/out/{label}/{group}/{name}")
            takes.append((group.rstrip("0123456789"), f"{label}/{group}", name[:-4], audio))
# wrong words: the next take's text in the same batch (similar length, sorted longest first)
wrong = {}
for i, (kind, batch, tid, _) in enumerate(takes):
    j = i + 1 if i + 1 < len(takes) and takes[i + 1][1] == batch else i - 1
    wrong[(batch, tid)] = texts[takes[j][2]]
cut = [(kind, batch, tid, audio[: int(len(audio) * 0.6)]) for kind, batch, tid, audio in takes[::3]]
# a runaway tail: another take's words after this one's, as when generation doesn't stop
tail = [(kind, batch, tid, audio + takes[(i * 3 + 7) % len(takes)][3]) for i, (kind, batch, tid, audio) in
        enumerate(takes[1::3])]
print(f"{len(takes)} takes, {len(cut)} cut takes", flush=True)

models = {"small": os.environ["SPEECH_CHECK_MODEL"], "base.en": "/app/models/faster-whisper-base.en",
          "tiny.en": "/app/models/faster-whisper-tiny.en"}
heard = {}
for name in sys.argv[1:] or models:
    checker = speech_check.SpeechChecker(WhisperModel(models[name], device="cpu", compute_type="int8",
                                                      cpu_threads=3, num_workers=6))
    hear = lambda t: checker.transcribe(t[3], words=False, beam_size=1).text
    with ThreadPoolExecutor(6) as pool:
        list(pool.map(hear, takes[:6]))  # warm-up
        per_kind = {}
        out = {}
        for kind in ("short", "medium", "long"):
            batch = [t for t in takes if t[0] == kind]
            s = time.perf_counter()
            for t, h in zip(batch, pool.map(hear, batch)):
                out[(t[1], t[2])] = h
            per_kind[kind] = (len(batch), time.perf_counter() - s)
        cut_heard = {(t[1], t[2]): h for t, h in zip(cut, pool.map(hear, cut))}
        tail_heard = {(t[1], t[2]): h for t, h in zip(tail, pool.map(hear, tail))}
    heard[name] = (out, cut_heard)
    json.dump({"true": {"|".join(k): v for k, v in out.items()}, "cut": {"|".join(k): v for k, v in cut_heard.items()},
               "tail": {"|".join(k): v for k, v in tail_heard.items()}}, open(f"/tmp/bench/heard_{name}.json", "w"))
    print(json.dumps({"model": name, "seconds_per_32": {k: round(32 * s / n, 1) for k, (n, s) in per_kind.items()}}),
          flush=True)


def passes(text, transcript):
    score = speech_check.match(text, transcript)
    return None if score is None else score >= speech_check.PASS_SCORE


small_true, small_cut = heard["small"]
for name in [m for m in heard if m != "small"]:
    q_true, q_cut = heard[name]
    rows = {"true": [], "wrong": [], "cut": []}
    for kind, batch, tid, _ in takes:
        key = (batch, tid)
        rows["true"].append((kind, passes(texts[tid], q_true[key]), passes(texts[tid], small_true[key])))
        rows["wrong"].append((kind, passes(wrong[key], q_true[key]), passes(wrong[key], small_true[key])))
    for kind, batch, tid, _ in cut:
        key = (batch, tid)
        rows["cut"].append((kind, passes(texts[tid], q_cut[key]), passes(texts[tid], small_cut[key])))
    for test, r in rows.items():
        judged = [x for x in r if x[1] is not None and x[2] is not None]
        quick_pass = sum(1 for x in judged if x[1])
        small_pass = sum(1 for x in judged if x[2])
        false_pass = sum(1 for x in judged if x[1] and not x[2])  # quick passes what small rejects
        rescued = sum(1 for x in judged if not x[1] and x[2])  # needs small: quick doubted it
        by_kind = {k: f"{sum(1 for x in judged if x[0] == k and not x[1])}/{sum(1 for x in judged if x[0] == k)}"
                   for k in ("short", "medium", "long")}
        print(json.dumps({"quick": name, "test": test, "judged": len(judged), "quick_pass": quick_pass,
                          "small_pass": small_pass, "quick_passes_small_rejects": false_pass,
                          "sent_on_to_small": len(judged) - quick_pass, "of_which_small_passes": rescued,
                          "sent_on_by_kind": by_kind}), flush=True)
