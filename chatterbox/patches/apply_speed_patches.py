"""Local speed patches for the pip-installed chatterbox package (run at image build).

1. T3.inference asks the transformer for attention weights on every sampling
   step. Only the multilingual model's AlignmentStreamAnalyzer reads them; for
   the English model they are discarded, yet requesting them forces transformers'
   SDPA attention onto the slower eager path. Request them only when an analyzer
   is attached.

Measured on the RTX 4070 (in-process A/B, identical tokens): 13-23% more tokens/s.
Also tried and rejected: EOS check every 8 steps (no gain), fp16/bf16 autocast (slower).

2. ChatterboxTTS.generate() hard-codes max_new_tokens=1000 (~40s of audio) with no
   signal when the sampling loop runs out instead of hitting EOS naturally — a long
   chunk is silently cut off mid-sentence (F-07/F-56; WORKLOG: 4 of 629 requests).
   Log a warning when a generation lands at or past that cap. Log only, no behaviour
   change: max_new_tokens itself is untouched.
"""
import pathlib
import sys

import chatterbox.models.t3.t3 as t3_module
import chatterbox.tts as tts_module

PATCHES = [
    (
        "output_attentions=True,",
        "output_attentions=self.patched_model.alignment_stream_analyzer is not None,",
        2,
    ),
]

TTS_PATCHES = [
    (
        "            speech_tokens = speech_tokens[0]",
        "            speech_tokens = speech_tokens[0]\n"
        "            if speech_tokens.shape[-1] >= 999:\n"
        "                import logging as _bb_logging\n"
        "                _bb_logging.getLogger(\"chatterbox.tts\").warning(\n"
        "                    \"Generation hit the max_new_tokens=1000 cap (%d tokens); \"\n"
        "                    \"output may be truncated mid-sentence.\",\n"
        "                    speech_tokens.shape[-1],\n"
        "                )",
        1,
    ),
]


def _apply(module, patches: list) -> None:
    path = pathlib.Path(module.__file__)
    src = path.read_text(encoding="utf-8")
    for needle, replacement, expected in patches:
        if replacement in src:
            print(f"already applied: {replacement.strip()[:60]}")
            continue
        found = src.count(needle)
        if found != expected:
            sys.exit(
                f"expected {expected} occurrence(s) of {needle.strip()!r}, found {found} "
                f"in {path}; upstream changed, review by hand"
            )
        src = src.replace(needle, replacement)
        print(f"applied: {replacement.strip()[:60]}")
    path.write_text(src, encoding="utf-8")


_apply(t3_module, PATCHES)
_apply(tts_module, TTS_PATCHES)
