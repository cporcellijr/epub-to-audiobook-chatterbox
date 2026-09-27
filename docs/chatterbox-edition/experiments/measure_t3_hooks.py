"""Measure two things about the installed chatterbox model package (F-49 and F-50 in
REVIEW_FINDINGS_CHATTERBOX.md): how many forward hooks attention layer 9 carries after N
generations, and how throughput changes from the first call to the N-th.

Run inside the Chatterbox container while no book is generating (it loads a second copy of
the model, about 4.5 GB of VRAM):

    docker cp docs/chatterbox-edition/experiments/measure_t3_hooks.py chatterbox:/app/
    docker exec -it chatterbox python3 /app/measure_t3_hooks.py --calls 200

Run it once on the current image and once after rebuilding with the patch changes, and
compare the last lines. Throughput is reported as seconds of audio produced per second of
wall time (the WORKLOG's 1.6-1.8x real time is this number's inverse view).
"""
import argparse
import time

import torch
from chatterbox.tts import ChatterboxTTS

TEXT = ("The rain had stopped by the time they reached the old bridge, and the river below "
        "was running high and brown after three days of weather.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=int, default=200)
    parser.add_argument("--voice", default="/app/voices/Emily.wav")
    parser.add_argument("--every", type=int, default=25, help="print a line every N calls")
    args = parser.parse_args()

    model = ChatterboxTTS.from_pretrained(device="cuda")
    layer = model.t3.tfmr.layers[9].self_attn
    print(f"hooks on layer 9 before any call: {len(layer._forward_hooks)}")

    model.prepare_conditionals(args.voice)  # voice cached, as the server does after the first request
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model.generate(TEXT)  # warm-up
    torch.cuda.synchronize()

    first = None
    for i in range(1, args.calls + 1):
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            wav = model.generate(TEXT)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        rate = (wav.shape[-1] / model.sr) / wall
        first = first or rate
        if i == 1 or i % args.every == 0:
            print(f"call {i:4d}: hooks on layer 9 = {len(layer._forward_hooks):4d}, "
                  f"{rate:.2f} s audio per s wall ({rate / first:.0%} of call 1)")


if __name__ == "__main__":
    main()
