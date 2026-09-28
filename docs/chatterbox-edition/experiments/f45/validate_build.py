"""F-45 build validation: exercises the production implementation (chatterbox/fast_t3.py through
engine.load_model() / engine.synthesize() with TTS_COMPILE=on), not the prototype, and prints PASS or
FAIL per check. Runs in a throwaway GPU container like the other scripts in this folder:

    docker run --rm --gpus all -e TTS_BF16=auto -e TTS_COMPILE=on -e PYTHONPATH=/f45 \\
      -e CHATTERBOX_CONFIG_PATH=/app/data/config.yaml -v <copy of your config.yaml>:/app/data/config.yaml \\
      -v <HF cache volume>:/app/hf_cache -v <this folder>:/f45 \\
      --entrypoint python3 chatterbox-tts-server:local /f45/validate_build.py

WAVs for listening land in /f45/validate_out (override with --out). About 40,000 characters are
synthesised in total; the 240-request memory check is most of it. Exit code 1 if any check fails.
"""
import argparse
import gc
import json
import logging
import os
import statistics
import sys
import threading
import time

os.environ.setdefault("TTS_COMPILE", "on")
os.environ.setdefault("TORCH_LOGS", "recompiles")
sys.path.insert(0, "/app")
os.chdir("/app")

import torch  # noqa: E402
import torchaudio  # noqa: E402
from transformers import DynamicCache  # noqa: E402

import engine  # noqa: E402
import fast_t3  # noqa: E402

# The three passages from f45lib.py (invented text).
TEXTS = [
    ("The keeper climbed the stairs at dusk and counted every step, as he always did. "
     "Outside, the wind had turned, and the first heavy drops of rain struck the glass. "
     "Far below, the sea was rising against the rocks."),
    "\"Is anyone there?\" she called. Nobody answered.",
    ("By midnight the rain was steady, and the lamp turned slowly above the rocks, throwing its long "
     "white arm across the water every eleven seconds, as it had done for a hundred years, while "
     "somewhere below a small boat with a broken mast was looking for the harbour and finding only "
     "the dark."),
]
SHORT = "The road ran on past the last of the houses and into the trees, where the light was already going."
POOL = [  # sentences of different lengths for the recompile check (invented)
    "Yes.",
    "The door was open.",
    "She waited at the top of the stairs and listened to the rain.",
    "By the time the lamp came round again the boat had gone, and the water was as flat and dark as slate.",
    "He counted the steps as he always did, forty-one to the gallery and nine more to the lamp room, and on the "
    "last one he stopped, because the light had not turned.",
]
VOICE = "/app/voices/Elena.wav"
SEED, TEMP, EXAG, CFG = 888, 0.61, 0.73, 0.5

results: list = []


def report(name: str, ok: bool, **info) -> None:
    results.append((name, ok))
    print(f"{'PASS' if ok else 'FAIL'} {name} {json.dumps(info)}", flush=True)


def synthesize(text: str) -> tuple:
    wav, sr = engine.synthesize(text, audio_prompt_path=VOICE, temperature=TEMP, exaggeration=EXAG,
                                cfg_weight=CFG, seed=SEED)
    if wav is None:
        raise RuntimeError("engine.synthesize returned None")
    return wav, sr


def timed_request(text: str) -> tuple:
    torch.cuda.synchronize()
    started = time.perf_counter()
    wav, sr = synthesize(text)
    torch.cuda.synchronize()
    return time.perf_counter() - started, wav.shape[-1] / sr, wav, sr


def realtime_factor(runs: int = 3) -> tuple:
    elapsed = audio = 0.0
    first_wavs = {}
    for n, text in enumerate(TEXTS):
        for run in range(runs):
            e, a, wav, sr = timed_request(text)
            elapsed += e
            audio += a
            if run == 0:
                first_wavs[n] = (wav, sr)
    return round(audio / elapsed, 2), round(elapsed, 1), round(audio, 1), first_wavs


class RecordCounter(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        self.count += 1


def compile_counts() -> tuple:
    """(dynamo frames compiled, recompile log records) so far."""
    from torch._dynamo.utils import counters
    return counters["frames"]["ok"], recompile_records.count


def cuda_tensor_count() -> int:
    gc.collect()
    return sum(1 for obj in gc.get_objects() if torch.is_tensor(obj) and obj.is_cuda)


def save_wav(path: str, wav: torch.Tensor, sr: int) -> None:
    torchaudio.save(path, wav.detach().float().cpu().reshape(1, -1), sr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/f45/validate_out")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    global recompile_records
    recompile_records = RecordCounter()
    logging.getLogger("torch._dynamo.guards.__recompiles").addHandler(recompile_records)
    warnings = RecordCounter()
    warnings.setLevel(logging.WARNING)
    fast_t3.logger.addHandler(warnings)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Start-up: load plus warm-up on the worker thread, and the fast path must be installed.
    started = time.perf_counter()
    loaded = engine.load_model()
    load_s = round(time.perf_counter() - started, 1)
    installed = engine.compiled_t3 is not None
    report("startup", loaded and installed and load_s < 300, load_s=load_s, compiled_installed=installed,
           compile_setting=engine.COMPILE_ENABLED, bf16=engine.BF16_ENABLED, model_type=engine.loaded_model_type)
    if not (loaded and installed):
        print("Cannot continue without the compiled loop installed.")
        return 1

    for _ in range(2):  # the voice conditionals and any remaining one-off work
        timed_request(TEXTS[1])
    frames_before, recompiles_before = compile_counts()

    # Speed: compiled on, then the stock loop, then reinstall (also exercises teardown/reinstall).
    try:
        on_rtf, on_s, on_audio, on_wavs = realtime_factor()
        for n, (wav, sr) in on_wavs.items():
            save_wav(os.path.join(args.out, f"text{n + 1}_compiled.wav"), wav, sr)
        engine.teardown_fast_path()
        assert engine.compiled_t3 is None
        off_rtf, off_s, off_audio, off_wavs = realtime_factor()
        for n, (wav, sr) in off_wavs.items():
            save_wav(os.path.join(args.out, f"text{n + 1}_stock.wav"), wav, sr)
        reinstall_started = time.perf_counter()
        engine.install_fast_path()
        reinstall_s = round(time.perf_counter() - reinstall_started, 1)
        speedup = round(on_rtf / off_rtf, 2)
        report("speed", engine.compiled_t3 is not None and speedup >= 2.0, compiled_realtime_x=on_rtf,
               stock_realtime_x=off_rtf, speedup=speedup, compiled_s=on_s, stock_s=off_s, reinstall_s=reinstall_s)
        report("listening", True, folder=args.out, files=sorted(os.listdir(args.out)))
    except Exception as e:  # noqa: BLE001
        report("speed", False, error=repr(e))
        if engine.compiled_t3 is None:
            engine.install_fast_path()

    # Same model: teacher-forced comparison of the stock eager path and the compiled decoder.
    try:
        model = engine.chatterbox_model
        fast = engine.compiled_t3
        t3 = model.t3

        def cfg_logits(step: torch.Tensor) -> torch.Tensor:
            cond, uncond = step[0:1, :].float(), step[1:2, :].float()
            return (cond + CFG * (cond - uncond)) / TEMP

        def teacher_forced(text: str) -> dict:
            tokens_text = fast_t3._text_tokens_for(model, text)
            engine.set_seed(SEED)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=engine.BF16_ENABLED):
                tokens = fast._stock_inference(t3_cond=model.conds.t3, text_tokens=tokens_text, max_new_tokens=1000,
                                               temperature=TEMP, cfg_weight=CFG, repetition_penalty=1.2,
                                               min_p=0.05, top_p=1.0)[0]
                inputs = fast.prompt_embeds(model.conds.t3, tokens_text, CFG)
                length = inputs.shape[1]
                ref_cache = DynamicCache()
                out = t3.tfmr(inputs_embeds=inputs, past_key_values=ref_cache, use_cache=True, return_dict=True)
                ref = [t3.speech_head(out.last_hidden_state)[:, -1, :]]
                compiled = [fast.prefill(inputs)]
                for i, token in enumerate(tokens[:-1]):
                    embed = t3.speech_emb(token.view(1, 1)) + t3.speech_pos_emb.get_fixed_embedding(i + 1)
                    pair = torch.cat([embed, embed])
                    out = t3.tfmr(inputs_embeds=pair, past_key_values=ref_cache, use_cache=True, return_dict=True)
                    ref.append(t3.speech_head(out.last_hidden_state)[:, -1, :])
                    compiled.append(fast.decode(pair, length + i))
            agree, top5, kls = 0, 0, []
            for r, c in zip(ref, compiled):
                lr, lc = cfg_logits(r), cfg_logits(c)
                agree += int(lr.argmax() == lc.argmax())
                top5 += int(lc.argmax() in lr.topk(5).indices)
                pr, pc = torch.log_softmax(lr, -1), torch.log_softmax(lc, -1)
                kls.append(float((pr.exp() * (pr - pc)).sum()))
            steps = len(ref)
            return {"steps": steps, "argmax_agreement": round(agree / steps, 3), "top5": round(top5 / steps, 3),
                    "kl_mean": round(statistics.mean(kls), 5), "kl_max": round(max(kls), 4)}

        per_text = [fast_t3.run(teacher_forced, text) for text in TEXTS]
        if engine.BF16_ENABLED:
            fast_t3.run(torch.clear_autocast_cache)
        ok = all(r["argmax_agreement"] >= 0.90 and r["top5"] == 1.0 and r["kl_mean"] <= 0.005 for r in per_text)
        report("same_model", ok, per_text=per_text)
    except Exception as e:  # noqa: BLE001
        report("same_model", False, error=repr(e))

    # No recompiles: 50 requests of varied lengths after warm-up.
    try:
        frames_before, recompiles_before = compile_counts()
        for i in range(50):
            text = " ".join(POOL[(i + k) % len(POOL)] for k in range(1 + i % 4))
            synthesize(text)
        frames_after, recompiles_after = compile_counts()
        report("no_recompiles", frames_after == frames_before and recompiles_after == recompiles_before,
               new_frames=frames_after - frames_before, recompile_records=recompiles_after - recompiles_before)
    except Exception as e:  # noqa: BLE001
        report("no_recompiles", False, error=repr(e))

    # Threads: requests from 8 different threads, one at a time (a pool would reuse one idle
    # thread, so each request gets its own); no re-recording, no latency spike.
    try:
        worker_before = fast_t3.worker.thread_ident
        frames_before, recompiles_before = compile_counts()
        latencies, idents = [], []

        def one_request() -> None:
            idents.append(threading.get_ident())
            latencies.append(timed_request(TEXTS[1])[0])

        for _ in range(8):
            thread = threading.Thread(target=one_request)
            thread.start()
            thread.join()
        median = statistics.median(latencies)
        frames_after, recompiles_after = compile_counts()
        ok = (len(set(idents)) == 8 and max(latencies[1:]) <= 1.5 * median and frames_after == frames_before
              and recompiles_after == recompiles_before and fast_t3.worker.thread_ident == worker_before)
        report("threads", ok, distinct_request_threads=len(set(idents)), latencies_s=[round(x, 2) for x in latencies],
               median_s=round(median, 2), new_frames=frames_after - frames_before,
               recompile_records=recompiles_after - recompiles_before)
    except Exception as e:  # noqa: BLE001
        report("threads", False, error=repr(e))

    # Memory: 240 requests; live CUDA tensors and allocated bytes must stay flat.
    try:
        for _ in range(3):
            synthesize(SHORT)
        tensors_before, bytes_before = cuda_tensor_count(), torch.cuda.memory_allocated()
        for _ in range(240):
            synthesize(SHORT)
        tensors_after, bytes_after = cuda_tensor_count(), torch.cuda.memory_allocated()
        delta_tensors, delta_mb = tensors_after - tensors_before, (bytes_after - bytes_before) / 2**20
        report("memory", delta_tensors <= 50 and delta_mb <= 64, cuda_tensors_before=tensors_before,
               cuda_tensors_after=tensors_after, allocated_mb_before=round(bytes_before / 2**20, 1),
               allocated_mb_after=round(bytes_after / 2**20, 1), delta_tensors=delta_tensors,
               delta_mb=round(delta_mb, 1))
    except Exception as e:  # noqa: BLE001
        report("memory", False, error=repr(e))

    # Fallback: force the compiled step to raise; the request must still succeed on the stock loop.
    try:
        fast = engine.compiled_t3
        original_step = fast.compiled_step

        def broken_step(*_args, **_kwargs):
            raise RuntimeError("forced failure for the fallback check")

        fast.compiled_step = broken_step
        fast._warned = False
        warnings_before, fallbacks_before = warnings.count, fast.fallback_calls
        wav, sr = synthesize(TEXTS[1])
        fast.compiled_step = original_step
        fast._warned = False
        ok = wav is not None and wav.shape[-1] > sr and warnings.count - warnings_before == 1 \
            and fast.fallback_calls == fallbacks_before + 1
        report("fallback", ok, audio_s=round(wav.shape[-1] / sr, 2), warnings_logged=warnings.count - warnings_before,
               fallback_calls=fast.fallback_calls - fallbacks_before)
        synthesize(TEXTS[1])  # back on the compiled path
    except Exception as e:  # noqa: BLE001
        report("fallback", False, error=repr(e))

    failed = [name for name, ok in results if not ok]
    print("SUMMARY", json.dumps({"passed": len(results) - len(failed), "failed": failed,
                                 "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                                 "fast_calls": engine.compiled_t3.fast_calls if engine.compiled_t3 else None,
                                 "fallback_calls": engine.compiled_t3.fallback_calls if engine.compiled_t3 else None}))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
