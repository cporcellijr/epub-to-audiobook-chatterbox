"""Breeze speed bench on a fixed workload, in a throwaway GPU container.

usage: python harness.py <label> [--attn eager|sdpa] [--plan short:32,short:64,...] [--depth-timer]
The server's fast depth decoder is on unless BREEZE_FAST_DEPTH=0 (then --depth-timer times the stock one).
Each plan entry runs the first N units of a group as one chunk (cfg 1.0, fixed seed), after a warm-up.
Writes WAVs to /bench/out/<label>/<entry>/ and one JSON line per entry to /bench/results.jsonl.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, "/opt/breeze-infer")
import server  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("label")
p.add_argument("--attn", default="eager")
p.add_argument("--plan", default="long:32,medium:32,medium:64,short:32,short:64,short:96")
p.add_argument("--depth-timer", action="store_true")
p.add_argument("--seed", type=int, default=4242)
p.add_argument("--repeat", type=int, default=1)
args = p.parse_args()

import torch  # noqa: E402
import breeze_infer.runtime as rt  # noqa: E402

_load_runtime = rt.load_runtime
rt.load_runtime = lambda d, device, attn_implementation: _load_runtime(d, device=device, attn_implementation=args.attn)

synth = server.BreezeSynthesizer()
t0 = time.perf_counter()
synth.load()
print(f"loaded in {time.perf_counter() - t0:.1f}s attn={args.attn}", flush=True)
model = synth._runtime[1]

depth_time = [0.0, 0]
if args.depth_timer:
    _gen = model.depth_decoder.generate

    def timed(*a, **k):
        torch.cuda.synchronize()
        s = time.perf_counter()
        out = _gen(*a, **k)
        torch.cuda.synchronize()
        depth_time[0] += time.perf_counter() - s
        depth_time[1] += 1
        return out
    model.depth_decoder.generate = timed

work = json.load(open("/bench/workload.json", encoding="utf-8"))


def requests(group, n, offset=0):
    return [{"id": i["id"], "text": i["text"], "speaker": "S0", "ref_audio_path": f"/voices/{i['voice']}",
             "ref_text": i["ref_text"]} for i in work[group][offset:offset + n]]


def run(group, n, save=None, offset=0):
    chunk = requests(group, n, offset)
    tokens = server.max_tokens_for(synth.frame_rate, max(len(r["text"]) for r in chunk))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    depth_time[:] = [0.0, 0]
    s = time.perf_counter()
    audios = synth.generate(chunk, 1.0, args.seed, tokens)
    torch.cuda.synchronize()
    el = time.perf_counter() - s
    seconds = [len(a) / server.SAMPLE_RATE for a in audios]
    frames = round(max(seconds) * synth.frame_rate)
    if save:
        os.makedirs(save, exist_ok=True)
        for r, a in zip(chunk, audios):
            sf.write(f"{save}/{r['id']}.wav", a, server.SAMPLE_RATE, subtype="PCM_16")
    return {"label": args.label, "attn": args.attn, "fast_depth": os.environ.get("BREEZE_FAST_DEPTH", "1"), "group": group, "n": n,
            "offset": offset, "elapsed": round(el, 2), "audio_s": round(sum(seconds), 1), "rtf": round(sum(seconds) / el, 2),
            "frames": frames, "ms_per_frame": round(1000 * el / max(frames, 1), 1),
            "peak_mib": torch.cuda.max_memory_allocated() >> 20, "reserved_mib": torch.cuda.memory_reserved() >> 20,
            **({"depth_s": round(depth_time[0], 2), "depth_calls": depth_time[1]} if args.depth_timer else {})}


run("short", 8)  # warm-up: kernels, allocator, voice clip encodings
with open("/bench/results.jsonl", "a") as out:
    for entry in args.plan.split(","):  # group:n, or group:n@offset for the units after the first `offset`
        group, n = entry.split(":")
        n, offset = (n.split("@") + ["0"])[:2]
        n, offset = int(n), int(offset)
        for rep in range(args.repeat):
            try:
                res = run(group, n, save=f"/bench/out/{args.label}/{group}{n}@{offset}" if rep == 0 else None,
                          offset=offset)
            except Exception as exc:  # out of memory or an allocator fault: note it and go on
                res = {"label": args.label, "attn": args.attn, "fast_depth": os.environ.get("BREEZE_FAST_DEPTH", "1"), "group": group,
                       "n": n, "offset": offset, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
                torch.cuda.empty_cache()
            print(json.dumps(res), flush=True)
            out.write(json.dumps(res) + "\n")
