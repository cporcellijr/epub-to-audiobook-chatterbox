"""Speed and verdict agreement of speech-check settings on the same takes."""
import json, os, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from faster_whisper import WhisperModel
from pydub import AudioSegment
from audiobook_generator.core import speech_check
texts = {i["id"]: i["text"] for g in json.load(open("/tmp/bench/workload.json")).values() for i in g}
takes = []
for group in ("short32", "medium32", "long32"):
    folder = f"/tmp/bench/out/graph/{group}"
    for name in sorted(os.listdir(folder)):
        a = AudioSegment.from_wav(f"{folder}/{name}")
        pad = AudioSegment.silent(200, frame_rate=16000)
        pcm = pad + a.set_channels(1).set_frame_rate(16000).set_sample_width(2) + pad
        takes.append((group, name[:-4], np.frombuffer(pcm.raw_data, dtype=np.int16).astype(np.float32) / 32768))
path = os.environ["SPEECH_CHECK_MODEL"]
base = None
for workers, threads, beam in [(3, 4, 5), (3, 4, 1), (6, 3, 1), (4, 5, 1), (6, 3, 5)]:
    model = WhisperModel(path, device="cpu", compute_type="int8", cpu_threads=threads, num_workers=workers)
    def hear(t):
        segs, _ = model.transcribe(t[2], language="en", beam_size=beam, temperature=0, condition_on_previous_text=False,
                                   vad_filter=False, word_timestamps=False)
        return "".join(s.text for s in segs).strip()
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(hear, takes[:6]))  # warm-up
        per_group = {}
        verdicts = {}
        for group in ("short32", "medium32", "long32"):
            batch = [t for t in takes if t[0] == group]
            s = time.perf_counter()
            heard = list(pool.map(hear, batch))
            per_group[group] = round(time.perf_counter() - s, 1)
            for t, h in zip(batch, heard):
                m = speech_check.match(texts[t[1]], h)
                verdicts[t[1]] = None if m is None else m >= speech_check.PASS_SCORE
    if base is None:
        base = verdicts
    agree = sum(verdicts[k] == base[k] for k in base)
    passes = sum(1 for v in verdicts.values() if v)
    print(json.dumps({"workers": workers, "threads": threads, "beam": beam, "seconds_per_32": per_group,
                      "pass": passes, "agree_with_beam5": f"{agree}/{len(base)}"}), flush=True)
