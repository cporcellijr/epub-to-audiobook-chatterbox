"""F-45: A/B samples and per-text speed, stock token loop vs the compiled one (writes /f45/textN_*.wav)."""
import json
import statistics
import time

import torch

from f45lib import *

compile_started = time.perf_counter()
for _ in range(3):  # compile + CUDA-graph recording happen on the first calls
    timed(fast_tokens, TEXTS[1])
print("COMPILE_WARMUP_S", round(time.perf_counter() - compile_started, 1))
for _ in range(2):
    timed(stock_tokens, TEXTS[1])

rows = []
for n, text in enumerate(TEXTS):
    stock_runs = [timed(stock_tokens, text) for _ in range(3)]
    fast_runs = [timed(fast_tokens, text) for _ in range(3)]
    s_tok, f_tok = stock_runs[0][0], fast_runs[0][0]
    same_prefix = int((s_tok[:min(len(s_tok), len(f_tok))] == f_tok[:min(len(s_tok), len(f_tok))]).long().cumprod(0).sum())
    s_ms = statistics.mean(t / len(tok) * 1000 for tok, t in stock_runs)
    f_ms = statistics.mean(t / len(tok) * 1000 for tok, t in fast_runs)
    s_audio = to_wav(s_tok, f"/f45/text{n + 1}_stock.wav")
    f_audio = to_wav(f_tok, f"/f45/text{n + 1}_compiled.wav")
    rows.append({"text": n + 1, "chars": len(text), "stock_tokens": len(s_tok), "compiled_tokens": len(f_tok),
                 "identical_leading_tokens": same_prefix, "stock_ms_per_token": round(s_ms, 2),
                 "compiled_ms_per_token": round(f_ms, 2), "token_loop_speedup": round(s_ms / f_ms, 2),
                 "stock_audio_s": round(s_audio, 2), "compiled_audio_s": round(f_audio, 2)})
    print("RESULT", json.dumps(rows[-1]))
print("PEAK_VRAM_GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))
