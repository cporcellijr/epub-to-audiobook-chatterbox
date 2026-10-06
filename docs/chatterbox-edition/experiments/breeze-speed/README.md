# Breeze speed bench (WORKLOG §53)

A fixed workload of real book sentences, run through the real model in a throwaway GPU container,
so settings can be compared on identical input. Stop the queue first: the bench needs the whole GPU,
and the live `breeze` container should be unloaded (`curl -X POST localhost:8005/api/unload`).

1. **Workload** (in the app container; it reads a library EPUB): `make_workload.py` writes
   `/tmp/workload.json` with three groups, long (the book's 64 longest sentences), medium (45-75
   characters) and short (quoted dialogue fragments, as cast mode splits them), spread over four
   voices. Copy it next to these scripts. **It holds book text: keep it out of git** (this folder's
   `.gitignore` does).
2. **Speed**: `harness.py <label> --plan medium:32,short:64,...` runs each entry's first N units as one
   chunk after a warm-up and appends a JSON line per entry to `results.jsonl` (time, real-time factor,
   frames, ms per frame, peak memory). The server's fast depth decoder is on unless
   `BREEZE_FAST_DEPTH=0`; `--depth-timer` (with it off) times the stock decoder's share.
   ```
   docker run --rm --gpus all -v breeze-models:/models -v <voices>:/voices:ro -v <this folder>:/bench \
     -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --entrypoint python breeze-tts:local \
     /bench/harness.py fast --plan short:32,medium:32,long:32
   ```
   A fixed seed makes repeat runs produce the same takes (the same `frames`), so timings compare
   directly. The decode loop is bound to one CPU core: anything else busy on the CPU (a test suite,
   the speech check) slows it 10-25%.
3. **Same math**: `check_depth.py` (with `BREEZE_FAST_DEPTH=0`) runs the fast decoder's exact mode
   beside the stock one on every frame of two batches, rewinding the CUDA RNG, and counts matching
   tokens. 2026-10-05: 97.8% of frames and 99.6% of tokens identical; the rest are bf16 rounding
   differences that flip one draw and the rest of that frame after it.
4. **Same words**: copy `out/` and the workload into the app container and run `quality.py <labels>`:
   Whisper hears every saved take as the app's check does (`speech_check.match`, pass at 0.70).
5. **Check speed**: `whisper_bench.py` times speech-check settings (workers x threads, beam) on the
   same takes and reports whether their verdicts agree.
6. **Quick first hearing** (WORKLOG §55): `two_stage.py [small tiny.en base.en]` hears every take, plus
   takes made bad three ways (another line's text, cut to 60%, a runaway tail of another take), with
   each model and saves the transcripts. `two_stage_thresholds.py` then counts, for each quick model
   and pass mark, the real takes sent on to small and the bad takes passed that small would reject.
   The quick models live beside small: `download_model("tiny.en", output_dir=...)`.
