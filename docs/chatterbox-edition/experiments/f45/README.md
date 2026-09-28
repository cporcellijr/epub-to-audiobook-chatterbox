# F-45 experiment: compiled T3 token loop

Scripts behind `WORKLOG.md` section 12. They run the production engine in a throwaway GPU container
and never touch a running server. `f45lib.py` holds the prototype (static KV cache + `torch.compile`
of the per-token transformer step); the others import it.

| Script | Measures |
|---|---|
| `measure.py` | Stock baseline: where a request's time goes (token loop vs audio decode) |
| `ab_samples.py` | Per-text speed, stock vs compiled, and A/B WAVs of the same text |
| `verify.py` | Per-run timings; teacher-forced check that the compiled path predicts like stock |
| `e2e_fast.py` | Whole requests through `engine.synthesize`, stock vs compiled |

```
docker run --rm --gpus all -e TTS_BF16=auto -e PYTHONPATH=/f45 \
  -e CHATTERBOX_CONFIG_PATH=/app/data/config.yaml -v <copy of your config.yaml>:/app/data/config.yaml \
  -v <HF cache volume>:/app/hf_cache -v <this folder>:/f45 \
  --entrypoint python3 chatterbox-tts-server:local /f45/e2e_fast.py
```

Needs about 3.5 GB of free VRAM next to a running server.
