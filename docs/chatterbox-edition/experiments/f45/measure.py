"""F-45 step 1: how much of T3's token loop is GPU work, and how much is CPU/launch overhead that
CUDA graphs or torch.compile could remove. Runs the production engine (engine.py, BF16, autocast,
voice cache) in a throwaway container; never touches the live server."""
import json
import os
import statistics
import sys
import time

sys.path.insert(0, "/app")
os.chdir("/app")

import torch
from torch.profiler import ProfilerActivity, profile, record_function

import engine

TEXT = ("The keeper climbed the stairs at dusk and counted every step, as he always did. "
        "Outside, the wind had turned, and the first heavy drops of rain struck the glass. "
        "Far below, the sea was rising against the rocks.")
VOICE = "/app/voices/Elena.wav"
RUNS = int(os.environ.get("RUNS", "5"))

assert engine.load_model(), "model did not load"
model = engine.chatterbox_model
t3 = model.t3
stats = {}

_t3_inference = t3.inference
_s3_inference = model.s3gen.inference


def timed_t3(*args, **kwargs):
    torch.cuda.synchronize()
    started = time.perf_counter()
    with record_function("T3_TOKEN_LOOP"):
        out = _t3_inference(*args, **kwargs)
    torch.cuda.synchronize()
    stats["t3_s"] = time.perf_counter() - started
    stats["tokens"] = int(out.shape[-1])
    return out


def timed_s3(*args, **kwargs):
    torch.cuda.synchronize()
    started = time.perf_counter()
    with record_function("S3GEN_DECODE"):
        out = _s3_inference(*args, **kwargs)
    torch.cuda.synchronize()
    stats["s3gen_s"] = time.perf_counter() - started
    return out


t3.inference = timed_t3
model.s3gen.inference = timed_s3


def run(seed: int = 888) -> dict:
    stats.clear()
    started = time.perf_counter()
    wav, sr = engine.synthesize(TEXT, audio_prompt_path=VOICE, temperature=0.61, exaggeration=0.73,
                                cfg_weight=0.5, seed=seed)
    total = time.perf_counter() - started
    return dict(total_s=total, audio_s=wav.shape[-1] / sr, **stats)


for _ in range(2):
    run()  # warm-up: voice conditionals, cuDNN autotune, allocator

results = [run() for _ in range(RUNS)]
per_token_ms = [r["t3_s"] / r["tokens"] * 1000 for r in results]
summary = {
    "runs": RUNS,
    "tokens": [r["tokens"] for r in results],
    "audio_s": round(statistics.mean(r["audio_s"] for r in results), 2),
    "total_s": round(statistics.mean(r["total_s"] for r in results), 2),
    "t3_s": round(statistics.mean(r["t3_s"] for r in results), 2),
    "s3gen_s": round(statistics.mean(r["s3gen_s"] for r in results), 3),
    "t3_ms_per_token": round(statistics.mean(per_token_ms), 2),
    "tokens_per_s": round(1000 / statistics.mean(per_token_ms), 1),
    "realtime_x": round(statistics.mean(r["audio_s"] / r["total_s"] for r in results), 2),
}
print("BASELINE", json.dumps(summary))

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    profiled = run()
averages = {e.key: e for e in prof.key_averages()}


def device_us(event) -> float:
    for name in ("device_time_total", "cuda_time_total"):
        value = getattr(event, name, None)
        if value:
            return float(value)
    return 0.0


kernels = [e for e in prof.key_averages() if getattr(e, "device_type", None) is not None
           and str(e.device_type).endswith("CUDA")]
loop = averages.get("T3_TOKEN_LOOP")
decode = averages.get("S3GEN_DECODE")
gpu_loop_ms = device_us(loop) / 1000 if loop else None
kernel_launches = sum(e.count for e in kernels)
print("PROFILE", json.dumps({
    "tokens": profiled["tokens"],
    "t3_wall_ms_unprofiled_per_token": summary["t3_ms_per_token"],
    "t3_gpu_ms_total": round(gpu_loop_ms, 1) if gpu_loop_ms else None,
    "t3_gpu_ms_per_token": round(gpu_loop_ms / profiled["tokens"], 2) if gpu_loop_ms else None,
    "gpu_busy_fraction_of_loop": (round(gpu_loop_ms / (summary["t3_ms_per_token"] * profiled["tokens"]), 3)
                                  if gpu_loop_ms else None),
    "s3gen_gpu_ms": round(device_us(decode) / 1000, 1) if decode else None,
    "kernel_launches_whole_call": kernel_launches,
    "kernel_launches_per_token": round(kernel_launches / profiled["tokens"], 1),
}))
top = sorted(kernels, key=device_us, reverse=True)[:12]
for e in top:
    print("KERNEL", f"{device_us(e) / 1000:8.1f} ms", f"x{e.count:<6}", e.key[:110])
