"""Compiled T3 token loop (review finding F-45).

Chatterbox spends most of each request sampling speech tokens one at a time, and most of
that time is CPU-side kernel-launch overhead rather than GPU work. This module runs the
per-token transformer step from a static KV cache through
``torch.compile(mode="reduce-overhead")`` (CUDA graphs), which removes that overhead:
measured 2.35x faster whole requests on the owner's RTX 4070 (WORKLOG section 12).

It is installed by ``engine.py`` for the Original English model only, when ``TTS_COMPILE``
is on, ``TTS_BF16`` is on and the model runs on CUDA. It replaces ``T3.inference`` on the
loaded model with an equivalent that samples exactly as the stock loop does, and falls back
to the stock loop whenever the fast path does not apply or raises.

CUDA graphs are recorded and replayed per thread (``torch/_inductor/cudagraph_trees.py``
keeps its managers in a ``threading.local``), so everything that touches the compiled step
(warm-up, every generation, teardown) runs on one dedicated worker thread, see
``SynthesisWorker``. ``engine.synthesize`` submits whole generations to it.

torch is imported lazily so the decision logic can be unit-tested without it.
"""
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

CACHE_LEN = 2048   # positions in the static KV cache (~0.5 GB in bf16 for this backbone)
CFG_BATCH = 2      # the conditional / unconditional pair the stock loop always runs
STOCK_MAX_NEW_TOKENS = 1000  # what ChatterboxTTS.generate passes as max_new_tokens
WARMUP_TEXT = "The lamp turned slowly above the rocks, and the sea was quiet."


# ---- settings and decisions (pure Python, unit-tested) ----

def resolve_compile_setting(value: Optional[str] = None) -> bool:
    """TTS_COMPILE: ``on``/``1``/``true`` enable the compiled loop; anything else is off (the default)."""
    if value is None:
        value = os.environ.get("TTS_COMPILE", "off")
    return value.strip().lower() in ("on", "1", "true")


def install_reason(setting_on: bool, model_type: Optional[str], bf16_enabled: bool,
                   device: Optional[str]) -> Optional[str]:
    """None when the fast path should be installed for a loaded model, else why not."""
    if not setting_on:
        return "TTS_COMPILE is off"
    if model_type != "original":
        return f"model type is {model_type!r}, not 'original'"
    if not bf16_enabled:
        return "TTS_BF16 is off (the compiled loop is only validated for the bf16 backbone)"
    if not str(device or "").startswith("cuda"):
        return f"device is {device!r}, not CUDA"
    return None


def fallback_reason(batch: int, device_type: str, *, num_return_sequences: int = 1,
                    stop_on_eos: bool = True, do_sample: bool = True, initial_speech_tokens: Any = None,
                    prepend_prompt_speech_tokens: Any = None, needs_analyzer: bool = False) -> Optional[str]:
    """None when a request can take the fast path, else why the stock loop must run it.

    The cache-length check needs the prompt embeddings and is done inside the fast path
    (``FallbackNeeded``); everything that can be decided from the arguments is decided here.
    """
    if batch != CFG_BATCH:
        return f"batch size {batch} (the compiled step is recorded for the CFG pair of {CFG_BATCH})"
    if device_type != "cuda":
        return f"tokens are on {device_type!r}, not CUDA"
    if num_return_sequences != 1:
        return f"num_return_sequences={num_return_sequences}"
    if not stop_on_eos or not do_sample:
        return "stop_on_eos or do_sample is off"
    if initial_speech_tokens is not None or prepend_prompt_speech_tokens is not None:
        return "initial or prepended speech tokens were given"
    if needs_analyzer:
        return "an alignment stream analyzer is needed"
    return None


def cache_fit_reason(prompt_len: int, max_new_tokens: int, cache_len: int = CACHE_LEN) -> Optional[str]:
    """None when prompt plus generation fits the static cache, else why not."""
    if prompt_len + max_new_tokens > cache_len:
        return f"prompt of {prompt_len} + {max_new_tokens} new tokens exceeds the {cache_len}-position cache"
    return None


class FallbackNeeded(Exception):
    """Raised inside the fast path when the request must run on the stock loop (not an error)."""


# ---- the dedicated synthesis thread ----

class SynthesisWorker:
    """Runs callables on one long-lived thread, one at a time.

    CUDA graphs belong to the thread that recorded them, so every call that may replay
    the compiled step goes through here. Calls made from the worker thread itself run
    inline, so a callable may call ``run`` again without deadlocking.
    """

    def __init__(self, name: str = "t3-compiled") -> None:
        self._name = name
        self._executor: Optional[ThreadPoolExecutor] = None
        self._thread_ident: Optional[int] = None
        self._guard = threading.Lock()

    def _ensure_started(self) -> ThreadPoolExecutor:
        """Start the worker thread on first use and remember its identity."""
        with self._guard:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=self._name)
                self._thread_ident = self._executor.submit(threading.get_ident).result()
            return self._executor

    @property
    def thread_ident(self) -> Optional[int]:
        """The worker thread's ``threading.get_ident()``, or None before the first call."""
        return self._thread_ident

    def run(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run ``fn`` on the worker thread and return its result (or raise its exception)."""
        if self._thread_ident is not None and threading.get_ident() == self._thread_ident:
            return fn(*args, **kwargs)
        return self._ensure_started().submit(fn, *args, **kwargs).result()

    def close(self) -> None:
        """Stop the worker thread after any running call finishes; a later ``run`` starts a new one."""
        with self._guard:
            executor, self._executor, self._thread_ident = self._executor, None, None
        if executor is not None:
            executor.shutdown(wait=True)


worker = SynthesisWorker()


def run(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run ``fn`` on the module's synthesis worker thread."""
    return worker.run(fn, *args, **kwargs)


# ---- the compiled decoder ----

def _make_static_cache(config: Any, batch: int, cache_len: int, device: Any, dtype: Any) -> Any:
    """A zeroed StaticCache for ``batch`` sequences of ``cache_len`` positions in the backbone's dtype."""
    from transformers import StaticCache
    try:  # transformers 4.46 (the image) calls the argument batch_size; later versions max_batch_size
        return StaticCache(config=config, batch_size=batch, max_cache_len=cache_len, device=device, dtype=dtype)
    except TypeError:
        return StaticCache(config=config, max_batch_size=batch, max_cache_len=cache_len, device=device, dtype=dtype)


def _make_step_module(backbone: Any, head: Any, cache: Any) -> Any:
    """One decoder step as a module that references only the backbone and the head.

    Deliberately not ``t3`` itself: the stock loop registers a new ``patched_model``
    submodule on ``t3`` every call, and a compiled function that closed over ``t3`` would
    carry a guard on ``len(t3._modules)`` and recompile after every fallback.
    """
    import torch

    class DecodeStep(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.head = head
            self.cache = cache

        def forward(self, embeds: "torch.Tensor", position: "torch.Tensor") -> "torch.Tensor":
            out = self.backbone(inputs_embeds=embeds, past_key_values=self.cache, cache_position=position,
                                use_cache=True, return_dict=True)
            return self.head(out.last_hidden_state)[:, -1, :]

    return DecodeStep()


class CompiledT3:
    """The compiled token loop for one loaded T3 model.

    ``inference`` has the signature and return value of ``T3.inference`` (``[1, N]`` speech
    tokens, EOS included when reached) and is installed in its place by ``install``.
    Construct and use it on the synthesis worker thread.
    """

    def __init__(self, t3: Any, cache_len: int = CACHE_LEN) -> None:
        import torch
        import torch._dynamo

        self.t3 = t3
        self.backbone = t3.tfmr
        self.head = t3.speech_head
        self.cache_len = cache_len
        self.dtype = next(self.backbone.parameters()).dtype
        self.device = next(self.backbone.parameters()).device
        self.cache = _make_static_cache(self.backbone.config, CFG_BATCH, cache_len, self.device, self.dtype)
        self.step_embeds = torch.zeros(CFG_BATCH, 1, self.backbone.config.hidden_size,
                                       device=self.device, dtype=self.dtype)
        self.step_position = torch.zeros(1, dtype=torch.long, device=self.device)
        # These tensors keep their addresses for the life of the decoder, so tell the CUDA-graph
        # runtime not to copy them into placeholders on every replay.
        for tensor in [*getattr(self.cache, "key_cache", []), *getattr(self.cache, "value_cache", []),
                       self.step_embeds, self.step_position]:
            torch._dynamo.mark_static_address(tensor)
        self.step = _make_step_module(self.backbone, self.head, self.cache)
        self.compiled_step = torch.compile(self.step, mode="reduce-overhead", fullgraph=True)
        # Bound to the class, not the instance, so a reinstall never captures a previous patch.
        self._stock_inference = type(t3).inference.__get__(t3, type(t3))
        self._warned = False
        self.fast_calls = 0
        self.fallback_calls = 0
        self.closed = False

    # -- dispatch --

    def inference(self, *, t3_cond: Any, text_tokens: Any, initial_speech_tokens: Any = None,
                  prepend_prompt_speech_tokens: Any = None, num_return_sequences: int = 1,
                  max_new_tokens: Optional[int] = None, stop_on_eos: bool = True, do_sample: bool = True,
                  temperature: float = 0.8, top_p: float = 0.95, min_p: float = 0.05,
                  length_penalty: float = 1.0, repetition_penalty: float = 1.2,
                  cfg_weight: float = 0.5) -> Any:
        kwargs = dict(t3_cond=t3_cond, text_tokens=text_tokens, initial_speech_tokens=initial_speech_tokens,
                      prepend_prompt_speech_tokens=prepend_prompt_speech_tokens,
                      num_return_sequences=num_return_sequences, max_new_tokens=max_new_tokens,
                      stop_on_eos=stop_on_eos, do_sample=do_sample, temperature=temperature, top_p=top_p,
                      min_p=min_p, length_penalty=length_penalty, repetition_penalty=repetition_penalty,
                      cfg_weight=cfg_weight)
        batch = text_tokens.shape[0] if len(text_tokens.shape) > 1 else 1
        reason = fallback_reason(batch, text_tokens.device.type, num_return_sequences=num_return_sequences,
                                 stop_on_eos=stop_on_eos, do_sample=do_sample,
                                 initial_speech_tokens=initial_speech_tokens,
                                 prepend_prompt_speech_tokens=prepend_prompt_speech_tokens,
                                 needs_analyzer=bool(getattr(getattr(self.t3, "hp", None), "is_multilingual", False)))
        if reason is None and self.closed:
            reason = "the compiled loop was closed"
        if reason is not None:
            return self._stock(reason, kwargs)
        try:
            tokens = self._fast_inference(**kwargs)
        except FallbackNeeded as e:
            return self._stock(str(e), kwargs)
        except Exception:
            if not self._warned:
                self._warned = True
                logger.warning("Compiled T3 loop failed; this request and later failures fall back to the "
                               "stock loop (later failures are logged at DEBUG).", exc_info=True)
            else:
                logger.debug("Compiled T3 loop failed again; falling back to the stock loop.", exc_info=True)
            return self._stock("the compiled step raised", kwargs)
        self.fast_calls += 1
        return tokens

    def _stock(self, reason: str, kwargs: dict) -> Any:
        logger.debug(f"T3 request on the stock loop: {reason}")
        self.fallback_calls += 1
        return self._stock_inference(**kwargs)

    # -- the fast path --

    def prefill(self, inputs_embeds: Any) -> Any:
        """Reset the cache, run the whole prompt eagerly and return the first-step logits ``[2, V]``."""
        import torch

        length = inputs_embeds.shape[1]
        self.cache.reset()
        out = self.backbone(inputs_embeds=inputs_embeds, past_key_values=self.cache,
                            cache_position=torch.arange(length, device=self.device), use_cache=True,
                            return_dict=True)
        return self.head(out.last_hidden_state)[:, -1, :]

    def decode(self, embed_pair: Any, position: int) -> Any:
        """One compiled step for the CFG pair ``[2, 1, hidden]`` at ``position``; returns logits ``[2, V]``."""
        self.step_embeds.copy_(embed_pair)
        self.step_position.fill_(position)
        # CUDA-graph outputs are overwritten on the next replay, so hand out a copy.
        return self.compiled_step(self.step_embeds, self.step_position).clone()

    def prompt_embeds(self, t3_cond: Any, text_tokens: Any, cfg_weight: float) -> Any:
        """Conditioning + text + BOS embeddings for the CFG pair, exactly as the stock loop builds them."""
        import torch

        t3, hp = self.t3, self.t3.hp
        text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=t3.device)
        speech_start = hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        embeds, _ = t3.prepare_input_embeds(t3_cond=t3_cond, text_tokens=text_tokens, speech_tokens=speech_start,
                                            cfg_weight=cfg_weight)
        bos = torch.tensor([[hp.start_speech_token]], dtype=torch.long, device=embeds.device)
        bos_embed = t3.speech_emb(bos) + t3.speech_pos_emb.get_fixed_embedding(0)
        return torch.cat([embeds, torch.cat([bos_embed, bos_embed])], dim=1)

    def _fast_inference(self, *, t3_cond: Any, text_tokens: Any, max_new_tokens: Optional[int],
                        temperature: float, top_p: float, min_p: float, repetition_penalty: float,
                        cfg_weight: float, stop_on_eos: bool, **_unused: Any) -> Any:
        import torch
        from transformers.generation.logits_process import (MinPLogitsWarper, RepetitionPenaltyLogitsProcessor,
                                                            TopPLogitsWarper)

        t3, hp = self.t3, self.t3.hp
        max_new_tokens = max_new_tokens or hp.max_speech_tokens
        inputs = self.prompt_embeds(t3_cond, text_tokens, cfg_weight)
        prompt_len = inputs.shape[1]
        reason = cache_fit_reason(prompt_len, max_new_tokens, self.cache_len)
        if reason is not None:
            raise FallbackNeeded(reason)

        logits_step = self.prefill(inputs)
        repetition = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))
        min_p_warper = MinPLogitsWarper(min_p=min_p)
        top_p_warper = TopPLogitsWarper(top_p=top_p)
        generated = torch.tensor([[hp.start_speech_token]], dtype=torch.long, device=self.device)
        predicted = []
        for i in range(max_new_tokens):
            # The stock loop's sampling, step for step: CFG combine, repetition penalty,
            # temperature, min_p, top_p, softmax, multinomial, EOS check.
            cond, uncond = logits_step[0:1, :], logits_step[1:2, :]
            cfg = torch.as_tensor(cfg_weight, device=cond.device, dtype=cond.dtype)
            logits = cond + cfg * (cond - uncond)
            ids = generated[:1, ...]
            logits = repetition(ids, logits)
            if temperature != 1.0:
                logits = logits / temperature
            logits = min_p_warper(ids, logits)
            logits = top_p_warper(ids, logits)
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            predicted.append(next_token)
            generated = torch.cat([generated, next_token], dim=1)
            if stop_on_eos and next_token.view(-1) == hp.stop_speech_token:
                break
            embed = t3.speech_emb(next_token) + t3.speech_pos_emb.get_fixed_embedding(i + 1)
            logits_step = self.decode(torch.cat([embed, embed]), prompt_len + i)
        return torch.cat(predicted, dim=1)

    # -- lifecycle --

    def close(self) -> None:
        """Drop the compiled function, its CUDA graphs and the static cache. Run on the worker thread."""
        self.closed = True
        self.compiled_step = None
        self.step = None
        self.cache = None
        self.step_embeds = None
        self.step_position = None
        try:
            import torch
            import torch._dynamo
            torch._dynamo.reset()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


# ---- install / warm-up / uninstall, all on the worker thread ----

def _text_tokens_for(model: Any, text: str) -> Any:
    """Text tokens for the CFG pair with the start/stop markers, as ``ChatterboxTTS.generate`` builds them."""
    import torch
    import torch.nn.functional as F
    from chatterbox.tts import punc_norm

    tokens = model.tokenizer.text_to_tokens(punc_norm(text)).to(model.device)
    tokens = torch.cat([tokens, tokens], dim=0)
    tokens = F.pad(tokens, (1, 0), value=model.t3.hp.start_text_token)
    return F.pad(tokens, (0, 1), value=model.t3.hp.stop_text_token)


def _install_on_worker(model: Any, autocast_enabled: bool) -> CompiledT3:
    """Build the decoder, patch ``model.t3.inference`` and warm it up; undo everything on failure."""
    import torch

    fast = CompiledT3(model.t3)
    model.t3.inference = fast.inference
    try:
        # Warm up inside the contexts real requests use (inference mode + autocast are part
        # of dynamo's guards), so the first real request does not pay for compiling or for
        # recording the CUDA graphs. The model's built-in conditionals stand in for a voice.
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            for _ in range(2):  # reduce-overhead compiles on the first call and records graphs on the next
                fast.inference(t3_cond=model.conds.t3, text_tokens=_text_tokens_for(model, WARMUP_TEXT),
                               max_new_tokens=STOCK_MAX_NEW_TOKENS, cfg_weight=0.5)
        if autocast_enabled:
            torch.clear_autocast_cache()
        if fast.fast_calls == 0:
            raise RuntimeError("the warm-up generation did not run on the compiled loop")
    except Exception:
        _uninstall_on_worker(model, fast)
        raise
    return fast


def _uninstall_on_worker(model: Any, fast: Optional[CompiledT3]) -> None:
    """Put the class's ``inference`` back on ``model.t3`` and close the decoder, if any."""
    t3 = getattr(model, "t3", None)
    if t3 is not None and "inference" in vars(t3):
        del t3.inference  # back to the class method, i.e. the stock loop
    if fast is not None:
        fast.close()


def install(model: Any, autocast_enabled: bool) -> CompiledT3:
    """Build, install and warm up the compiled loop for ``model`` on the worker thread.

    Raises whatever went wrong after restoring the stock loop, so the caller can log it
    and keep serving.
    """
    started = time.perf_counter()
    fast = run(_install_on_worker, model, autocast_enabled)
    logger.info(f"Compiled T3 loop installed (torch.compile reduce-overhead, static cache of {fast.cache_len} "
                f"positions); warm-up took {time.perf_counter() - started:.1f} s.")
    return fast


def uninstall(model: Any, fast: Optional[CompiledT3]) -> None:
    """Restore the stock loop on ``model`` and release the compiled state, on the worker thread."""
    run(_uninstall_on_worker, model, fast)
