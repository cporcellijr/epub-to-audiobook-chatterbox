"""F-45 step 4: whole requests through the production engine.synthesize (voice cache, autocast, S3Gen
decode, watermark), stock token loop vs the compiled one swapped in as T3.inference."""
import json
import statistics
import time

import torch

from f45lib import *


def fast_inference(t3_cond, text_tokens, max_new_tokens, temperature, cfg_weight, repetition_penalty, min_p,
                   top_p, **_ignored):
    assert (temperature, cfg_weight, repetition_penalty, min_p, top_p) == (TEMP, CFG, REP, MIN_P, TOP_P)
    embeds, _ = t3.prepare_input_embeds(t3_cond=t3_cond, text_tokens=text_tokens,
                                        speech_tokens=t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1]),
                                        cfg_weight=cfg_weight)
    bos = torch.tensor([[t3.hp.start_speech_token]], dtype=torch.long, device=embeds.device)
    bos_embed = t3.speech_emb(bos) + t3.speech_pos_emb.get_fixed_embedding(0)
    embeds = torch.cat([embeds, torch.cat([bos_embed, bos_embed])], dim=1)
    cache.reset()
    length = embeds.shape[1]
    out = tfmr(inputs_embeds=embeds, past_key_values=cache, cache_position=torch.arange(length, device=model.device),
               use_cache=True, return_dict=True)
    first = t3.speech_head(out.last_hidden_state)[:, -1, :]

    def forward_next(embed, i):
        step_embeds.copy_(embed)
        step_position.fill_(length + i)
        return compiled_step(step_embeds, step_position).clone()

    return sample_loop(first, forward_next)


def request(text: str) -> tuple:
    torch.cuda.synchronize()
    started = time.perf_counter()
    wav, sr = engine.synthesize(text, audio_prompt_path=VOICE, temperature=TEMP, exaggeration=EXAG, cfg_weight=CFG,
                                seed=SEED)
    torch.cuda.synchronize()
    return time.perf_counter() - started, wav.shape[-1] / sr


def measure(label: str) -> dict:
    request(TEXTS[1])
    request(TEXTS[1])
    elapsed = audio = 0.0
    for text in TEXTS:
        for _ in range(3):
            e, a = request(text)
            elapsed += e
            audio += a
    return {"path": label, "requests": 9, "generate_s": round(elapsed, 1), "audio_s": round(audio, 1),
            "realtime_x": round(audio / elapsed, 2)}


stock_inference = t3.inference
stock = measure("stock")
print("E2E", json.dumps(stock))
t3.inference = fast_inference
fast = measure("compiled")
print("E2E", json.dumps(fast))
print("E2E_SPEEDUP", round(fast["realtime_x"] / stock["realtime_x"], 2))
print("PEAK_VRAM_GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))
