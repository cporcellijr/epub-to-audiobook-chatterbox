"""F-45 step 3: per-run timings, and a teacher-forced numeric check: the SAME token sequence is fed
through the stock eager path (DynamicCache) and the compiled static-cache path, and the next-token
distributions are compared at every step."""
import json, statistics, time
import torch
from transformers import DynamicCache
from f45lib import *

for _ in range(3):
    timed(fast_tokens, TEXTS[1])

for n, text in enumerate(TEXTS):
    stock = [round(t / len(tok) * 1000, 2) for tok, t in (timed(stock_tokens, text) for _ in range(3))]
    fast = [round(t / len(tok) * 1000, 2) for tok, t in (timed(fast_tokens, text) for _ in range(4))]
    print("RUNS", json.dumps({"text": n + 1, "stock_ms_per_token": stock, "compiled_ms_per_token": fast}))


def cfg_logits(step_logits):
    cond, uncond = step_logits[0:1, :].float(), step_logits[1:2, :].float()
    return (cond + CFG * (cond - uncond)) / TEMP


eager_cache = StaticCache(config=tfmr.config, batch_size=2, max_cache_len=CACHE_LEN, device=model.device,
                          dtype=next(tfmr.parameters()).dtype)


def teacher_forced(text, tokens):
    """Per-step CFG logits for the same forced token sequence from: the stock eager path (DynamicCache),
    the same model eager on a static cache (no compile), and the compiled static-cache step."""
    embeds = prompt_embeds(text_tokens_for(text))
    length = embeds.shape[1]
    ref_cache = DynamicCache()
    out = tfmr(inputs_embeds=embeds, past_key_values=ref_cache, use_cache=True, return_dict=True)
    ref = [t3.speech_head(out.last_hidden_state)[:, -1, :]]
    cache.reset()
    out = tfmr(inputs_embeds=embeds, past_key_values=cache, cache_position=torch.arange(length, device=model.device),
               use_cache=True, return_dict=True)
    fast = [t3.speech_head(out.last_hidden_state)[:, -1, :]]
    eager_cache.reset()
    out = tfmr(inputs_embeds=embeds, past_key_values=eager_cache,
               cache_position=torch.arange(length, device=model.device), use_cache=True, return_dict=True)
    eager_static = [t3.speech_head(out.last_hidden_state)[:, -1, :]]
    for i, token in enumerate(tokens[:-1]):
        embed = t3.speech_emb(token.view(1, 1)) + t3.speech_pos_emb.get_fixed_embedding(i + 1)
        pair = torch.cat([embed, embed])
        out = tfmr(inputs_embeds=pair, past_key_values=ref_cache, use_cache=True, return_dict=True)
        ref.append(t3.speech_head(out.last_hidden_state)[:, -1, :])
        step_embeds.copy_(pair)
        step_position.fill_(length + i)
        fast.append(compiled_step(step_embeds, step_position).clone())
        out = tfmr(inputs_embeds=pair, past_key_values=eager_cache,
                   cache_position=torch.tensor([length + i], device=model.device), use_cache=True, return_dict=True)
        eager_static.append(t3.speech_head(out.last_hidden_state)[:, -1, :])
    return ref, fast, eager_static


def compare(ref, other):
    agree, kls, top5 = 0, [], 0
    for r, f in zip(ref, other):
        lr, lf = cfg_logits(r), cfg_logits(f)
        agree += int(lr.argmax() == lf.argmax())
        pr, pf = torch.log_softmax(lr, -1), torch.log_softmax(lf, -1)
        kls.append(float((pr.exp() * (pr - pf)).sum()))
        top5 += int(lf.argmax() in lr.topk(5).indices)
    steps = len(ref)
    return {"steps": steps, "argmax_agreement": round(agree / steps, 3),
            "argmax_in_stock_top5": round(top5 / steps, 3), "kl_mean": round(statistics.mean(kls), 5),
            "kl_max": round(max(kls), 4)}


for n, text in enumerate(TEXTS):
    tokens, _ = timed(stock_tokens, text)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=engine.BF16_ENABLED):
        ref, fast, eager_static = teacher_forced(text, tokens)
    print("TEACHER_FORCED", json.dumps({"text": n + 1, "stock_vs_compiled": compare(ref, fast),
                                        "stock_vs_eager_static_cache": compare(ref, eager_static)}))
