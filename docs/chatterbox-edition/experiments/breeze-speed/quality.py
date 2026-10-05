"""Whisper-check saved takes the way the app does: speech_check.match >= PASS_SCORE passes."""
import json, os, statistics, sys
from concurrent.futures import ThreadPoolExecutor
from pydub import AudioSegment
from audiobook_generator.core import speech_check
texts = {i["id"]: i["text"] for g in json.load(open("/tmp/bench/workload.json")).values() for i in g}
checker = speech_check.get()
rows = []
for label in sys.argv[1:]:
    for group in sorted(os.listdir(f"/tmp/bench/out/{label}")):
        folder = f"/tmp/bench/out/{label}/{group}"
        files = sorted(os.listdir(folder))
        def score(name):
            audio = AudioSegment.from_wav(f"{folder}/{name}")
            heard = checker.transcribe(audio, words=False).text
            return speech_check.match(texts[name[:-4]], heard), len(audio) / 1000
        with ThreadPoolExecutor(speech_check.WORKERS) as pool:
            results = list(pool.map(score, files))
        scores = [s for s, _ in results if s is not None]
        passed = sum(s >= speech_check.PASS_SCORE for s in scores)
        row = {"label": label, "group": group, "takes": len(files), "judged": len(scores), "pass": passed,
               "pass_rate": round(passed / len(scores), 3) if scores else None,
               "mean_match": round(statistics.mean(scores), 3) if scores else None,
               "seconds": round(sum(d for _, d in results), 1)}
        rows.append(row)
        print(json.dumps(row), flush=True)
