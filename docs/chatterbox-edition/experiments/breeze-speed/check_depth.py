"""Correctness check: FastDepth (eager mode) against the stock depth_decoder.generate(), call by call.

For each depth call during one real batch, the stock generate() runs first; then the CUDA RNG is rewound
and FastDepth runs on the same inputs. Same math and same sampling ops should give the same tokens.
The stock output is what the batch keeps, so the run itself is unchanged. Run it with BREEZE_FAST_DEPTH=0,
or the server will already have replaced the stock decoder.
"""
import json
import sys

sys.path.insert(0, "/opt/breeze-infer")
import server  # noqa: E402
import torch  # noqa: E402

import fast_depth  # noqa: E402

synth = server.BreezeSynthesizer()
synth.load()
model = synth._runtime[1]
stock = model.depth_decoder.generate
fast = fast_depth.FastDepth(model, "eager")
stats = {"calls": 0, "rows": 0, "rows_equal": 0, "tokens": 0, "tokens_equal": 0, "shape": None}


def compare(*a, **k):
    before = torch.cuda.get_rng_state()
    ref = stock(*a, **k)
    ref = ref if isinstance(ref, torch.Tensor) else ref.sequences
    after = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(before)
    mine = fast.generate(*a, **k)
    torch.cuda.set_rng_state(after)
    stats["shape"] = [list(ref.shape), list(mine.shape)]
    width = min(ref.shape[1], mine.shape[1])
    same = ref[:, :width] == mine[:, :width]
    stats["calls"] += 1
    stats["rows"] += ref.shape[0]
    stats["rows_equal"] += int(same.all(dim=1).sum())
    stats["tokens"] += same.numel()
    stats["tokens_equal"] += int(same.sum())
    return ref


model.depth_decoder.generate = compare
work = json.load(open("/bench/workload.json", encoding="utf-8"))
for group in ("medium", "short"):
    chunk = [{"id": i["id"], "text": i["text"], "speaker": "S0", "ref_audio_path": f"/voices/{i['voice']}",
              "ref_text": i["ref_text"]} for i in work[group][:16]]
    synth.generate(chunk, 1.0, 4242, server.max_tokens_for(synth.frame_rate, max(len(r["text"]) for r in chunk)))
print(json.dumps(stats))
