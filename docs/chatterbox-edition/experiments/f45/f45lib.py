"""F-45 step 2: prototype. Same T3 token loop as chatterbox-v2's T3.inference (CFG, repetition
penalty, temperature, min_p, top_p, multinomial, EOS check), but the per-token transformer forward
runs from a static KV cache through torch.compile(mode="reduce-overhead") (CUDA graphs). Compares
speed and output against the stock loop on the same text and seed. Throwaway container only."""
import json
import os
import statistics
import sys
import time

sys.path.insert(0, "/app")
os.chdir("/app")

import torch
import torch.nn.functional as F
import torchaudio
from transformers import StaticCache
from transformers.generation.logits_process import (MinPLogitsWarper, RepetitionPenaltyLogitsProcessor,
                                                    TopPLogitsWarper)

import chatterbox.tts as ctts
import engine

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
assert sum(len(t) for t in TEXTS) < 1500
VOICE = "/app/voices/Elena.wav"
SEED, TEMP, EXAG, CFG, REP, MIN_P, TOP_P = 888, 0.61, 0.73, 0.5, 1.2, 0.05, 1.0
MAX_NEW = 1000
CACHE_LEN = 2048

assert engine.load_model(), "model did not load"
model = engine.chatterbox_model
t3 = model.t3
tfmr = t3.tfmr
print("MODEL", json.dumps({"backbone": type(tfmr).__name__, "attn": getattr(tfmr.config, "_attn_implementation", None),
                           "layers": tfmr.config.num_hidden_layers, "hidden": tfmr.config.hidden_size,
                           "dtype": str(next(tfmr.parameters()).dtype), "bf16": engine.BF16_ENABLED}))
model.prepare_conditionals(VOICE, exaggeration=EXAG)


def text_tokens_for(text: str) -> torch.Tensor:
    tokens = model.tokenizer.text_to_tokens(ctts.punc_norm(text)).to(model.device)
    tokens = torch.cat([tokens, tokens], dim=0)  # CFG pair
    tokens = F.pad(tokens, (1, 0), value=t3.hp.start_text_token)
    return F.pad(tokens, (0, 1), value=t3.hp.stop_text_token)


def prompt_embeds(text_tokens: torch.Tensor) -> torch.Tensor:
    embeds, _ = t3.prepare_input_embeds(t3_cond=model.conds.t3, text_tokens=text_tokens,
                                        speech_tokens=t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1]),
                                        cfg_weight=CFG)
    bos = torch.tensor([[t3.hp.start_speech_token]], dtype=torch.long, device=embeds.device)
    bos_embed = t3.speech_emb(bos) + t3.speech_pos_emb.get_fixed_embedding(0)
    return torch.cat([embeds, torch.cat([bos_embed, bos_embed])], dim=1)


def sample_loop(first_logits, forward_next) -> torch.Tensor:
    """The stock loop's sampling, verbatim in effect; forward_next(embed, step) -> logits [2, V]."""
    rep = RepetitionPenaltyLogitsProcessor(penalty=float(REP))
    min_p, top_p = MinPLogitsWarper(min_p=MIN_P), TopPLogitsWarper(top_p=TOP_P)
    generated = torch.tensor([[t3.hp.start_speech_token]], dtype=torch.long, device=first_logits.device)
    predicted = []
    logits_step = first_logits
    for i in range(MAX_NEW):
        cond, uncond = logits_step[0:1, :], logits_step[1:2, :]
        logits = cond + torch.as_tensor(CFG, device=cond.device, dtype=cond.dtype) * (cond - uncond)
        ids = generated[:1, ...]
        logits = rep(ids, logits)
        logits = logits / TEMP
        logits = min_p(ids, logits)
        logits = top_p(ids, logits)
        next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        predicted.append(next_token)
        generated = torch.cat([generated, next_token], dim=1)
        if next_token.view(-1) == t3.hp.stop_speech_token:
            break
        embed = t3.speech_emb(next_token) + t3.speech_pos_emb.get_fixed_embedding(i + 1)
        logits_step = forward_next(torch.cat([embed, embed]), i)
    return torch.cat(predicted, dim=1)


cache = StaticCache(config=tfmr.config, batch_size=2, max_cache_len=CACHE_LEN, device=model.device,
                    dtype=next(tfmr.parameters()).dtype)
step_embeds = torch.zeros(2, 1, tfmr.config.hidden_size, device=model.device, dtype=next(tfmr.parameters()).dtype)
step_position = torch.zeros(1, dtype=torch.long, device=model.device)


def decode_step(embeds: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    out = tfmr(inputs_embeds=embeds, past_key_values=cache, cache_position=position, use_cache=True,
               return_dict=True)
    return t3.speech_head(out.last_hidden_state)[:, -1, :]


compiled_step = torch.compile(decode_step, mode="reduce-overhead", fullgraph=True)


def fast_tokens(text: str) -> torch.Tensor:
    embeds = prompt_embeds(text_tokens_for(text))
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


def stock_tokens(text: str) -> torch.Tensor:
    return t3.inference(t3_cond=model.conds.t3, text_tokens=text_tokens_for(text), max_new_tokens=MAX_NEW,
                        temperature=TEMP, cfg_weight=CFG, repetition_penalty=REP, min_p=MIN_P, top_p=TOP_P)[0]


def timed(fn, text):
    engine.set_seed(SEED)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=engine.BF16_ENABLED):
        tokens = fn(text)
    torch.cuda.synchronize()
    return tokens.reshape(-1), time.perf_counter() - started


def to_wav(tokens: torch.Tensor, path: str) -> float:
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=engine.BF16_ENABLED):
        speech = ctts.drop_invalid_tokens(tokens)
        speech = speech[speech < 6561].to(model.device)
        wav, _ = model.s3gen.inference(speech_tokens=speech, ref_dict=model.conds.gen)
    wav = wav.squeeze(0).detach().float().cpu()
    torchaudio.save(path, wav.unsqueeze(0), model.sr)
    return wav.shape[-1] / model.sr


