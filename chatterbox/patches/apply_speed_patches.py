"""Local speed patches for the pip-installed chatterbox package (run at image build).

1. T3.inference asks the transformer for attention weights on every sampling
   step. Only the multilingual model's AlignmentStreamAnalyzer reads them; for
   the English model they are discarded, yet requesting them forces transformers'
   SDPA attention onto the slower eager path. Request them only when an analyzer
   is attached.

Measured on the RTX 4070 (in-process A/B, identical tokens): 13-23% more tokens/s.
Also tried and rejected: EOS check every 8 steps (no gain), fp16/bf16 autocast (slower).
"""
import pathlib
import sys

import chatterbox.models.t3.t3 as t3_module

PATCHES = [
    (
        "output_attentions=True,",
        "output_attentions=self.patched_model.alignment_stream_analyzer is not None,",
        2,
    ),
]

path = pathlib.Path(t3_module.__file__)
src = path.read_text(encoding="utf-8")
for needle, replacement, expected in PATCHES:
    if replacement in src:
        print(f"already applied: {replacement.strip()[:60]}")
        continue
    found = src.count(needle)
    if found != expected:
        sys.exit(f"expected {expected} occurrence(s) of {needle.strip()!r}, found {found}; upstream changed, review by hand")
    src = src.replace(needle, replacement)
    print(f"applied: {replacement.strip()[:60]}")
path.write_text(src, encoding="utf-8")
