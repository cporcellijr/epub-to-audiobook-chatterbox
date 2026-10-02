"""Server contract tests with a fake synthesizer: no torch, no GPU."""
import base64
import io
import threading
import time
import weakref

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import server
from server import OutOfMemory, create_app


class FakeSynth:
    """Records every generate() call; speaks len(text)/100 seconds of silence per item."""

    def __init__(self, frame_rate=12.5, fail=None, delay=0.0):
        self.loaded = False
        self.loads = 0
        self.unloads = 0
        self.calls = []
        self.frame_rate = frame_rate
        self.fail = fail  # fail(chunk, cfg) -> exception or None
        self.delay = delay
        self.active = 0
        self.max_active = 0

    def load(self):
        self.loaded = True
        self.loads += 1

    def unload(self):
        self.loaded = False
        self.unloads += 1

    def generate(self, chunk, cfg, seed, max_new_tokens):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            self.calls.append({"ids": [r["id"] for r in chunk], "cfg": cfg, "seed": seed, "tokens": max_new_tokens,
                               "texts": [r["text"] for r in chunk], "template": server.template_name(chunk[0])})
            if self.delay:
                time.sleep(self.delay)
            exc = self.fail(chunk, cfg) if self.fail else None
            if exc:
                raise exc
            return [np.zeros(int(len(r["text"]) * 240), dtype=np.float32) for r in chunk]
        finally:
            self.active -= 1


@pytest.fixture
def voices(tmp_path):
    (tmp_path / "Ann.wav").write_bytes(b"x")
    (tmp_path / "Bob.wav").write_bytes(b"x")
    return tmp_path


def make(voices, max_batch=32, **kw):
    synth = FakeSynth(**kw)
    return synth, TestClient(create_app(synth, voices_dir=str(voices), max_batch=max_batch))


def item(id, text="hello there", voice="Ann.wav", **kw):
    d = {"id": id, "text": text, "voice": voice, "ref_text": "clip words" if voice else None}
    d.update(kw)
    return d


def post(client, items, **kw):
    return client.post("/v1/batch", json={"items": items, **kw})


def test_health_and_load_unload(voices):
    synth, c = make(voices, max_batch=7)
    assert c.get("/health").json() == {"status": "ok", "loaded": False, "max_batch": 7}
    assert c.post("/api/load").json() == {"loaded": True}
    assert c.post("/api/load").json() == {"loaded": True}
    assert synth.loads == 1
    assert c.get("/health").json()["loaded"] is True
    assert c.post("/api/unload").json() == {"loaded": False}
    assert c.post("/api/unload").json() == {"loaded": False}
    assert c.get("/health").json()["loaded"] is False


def test_batch_shape_and_wav(voices):
    synth, c = make(voices)
    r = post(c, [item("a", "x" * 100)])
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"items", "elapsed_s"}
    out = body["items"][0]
    assert set(out) == {"id", "wav_b64", "seconds", "error"}
    assert out["id"] == "a" and out["error"] is None
    data, rate = sf.read(io.BytesIO(base64.b64decode(out["wav_b64"])), dtype="int16")
    info = sf.info(io.BytesIO(base64.b64decode(out["wav_b64"])))
    assert rate == 24000 and info.channels == 1 and info.subtype == "PCM_16"
    assert out["seconds"] == pytest.approx(1.0)
    assert synth.calls[0]["seed"] == 1234


def test_auto_load(voices):
    synth, c = make(voices)
    assert not synth.loaded
    post(c, [item("a")])
    assert synth.loaded and synth.loads == 1


@pytest.mark.parametrize("voice", ["../Ann.wav", "sub/Ann.wav", "sub\\Ann.wav", ".."])
def test_path_traversal_rejected(voices, voice):
    synth, c = make(voices)
    assert post(c, [item("a", voice=voice)]).status_code == 400
    assert synth.calls == []


def test_voice_needs_ref_text(voices):
    _, c = make(voices)
    for ref in (None, "", "   "):
        r = post(c, [{"id": "a", "text": "hi", "voice": "Ann.wav", "ref_text": ref}])
        assert r.status_code == 400


def test_nothing_to_say_and_empty_text(voices):
    _, c = make(voices)
    assert post(c, [{"id": "a", "text": "hi"}]).status_code == 400
    assert post(c, [{"id": "a", "text": "hi", "instruction": "  "}]).status_code == 400
    assert post(c, [item("a", text="  ")]).status_code == 400


def test_unknown_voice_and_duplicate_ids(voices):
    _, c = make(voices)
    assert post(c, [item("a", voice="Nope.wav")]).status_code == 404
    assert post(c, [item("a"), item("a")]).status_code == 400


def test_voice_design_uses_instruction_cfg(voices):
    synth, c = make(voices)
    r = post(c, [{"id": "d", "text": "hello", "instruction": "A deep voice."}])
    assert r.status_code == 200
    assert synth.calls[0]["template"] == "tts_instruction" and synth.calls[0]["cfg"] == 4.0


def test_grouping_by_template_and_cfg(voices):
    synth, c = make(voices)
    items = [
        item("clone1"), item("clone2", voice="Bob.wav"),
        item("direct", instruction="Whisper."),
        {"id": "design", "text": "hi", "instruction": "A young girl."},
        item("custom", cfg_scale=2.5),
    ]
    assert post(c, items).status_code == 200
    groups = {(call["template"], call["cfg"]): sorted(call["ids"]) for call in synth.calls}
    assert groups == {
        ("ref_clone_tata", 1.0): ["clone1", "clone2"],
        ("ref_edit_tata", 4.0): ["direct"],
        ("tts_instruction", 4.0): ["design"],
        ("ref_clone_tata", 2.5): ["custom"],
    }


def test_sort_chunk_and_order_preserved(voices):
    synth, c = make(voices, max_batch=3)
    lengths = [50, 10, 40, 20, 30, 60, 5]
    items = [item(f"i{n}", "x" * n) for n in lengths]
    out = post(c, items, seed=100).json()["items"]
    assert [o["id"] for o in out] == [f"i{n}" for n in lengths]
    assert [o["seconds"] for o in out] == pytest.approx([n * 240 / 24000 for n in lengths])
    assert [len(call["ids"]) for call in synth.calls] == [3, 3, 1]
    assert [call["ids"] for call in synth.calls] == [["i5", "i10", "i20"], ["i30", "i40", "i50"], ["i60"]]
    assert [call["seed"] for call in synth.calls] == [100, 101, 102]


def test_max_new_tokens_rule(voices):
    synth, c = make(voices)
    post(c, [item("a", "x" * 150), item("b", "x" * 10)])
    # longest 150 chars: (3 * 150/15 + 3) s = 33 s * 12.5 frames/s
    assert synth.calls[0]["tokens"] == 413
    synth.calls.clear()
    post(c, [item("a", "x" * 5000)])
    assert synth.calls[0]["tokens"] == 1500
    synth.calls.clear()
    synth.frame_rate = None
    post(c, [item("a", "x" * 10)])
    assert synth.calls[0]["tokens"] == 1500


def test_oom_split_and_retry(voices):
    # anything over 2 items does not fit
    synth, c = make(voices, fail=lambda chunk, cfg: OutOfMemory("cuda oom") if len(chunk) > 2 else None)
    items = [item(f"i{n}", "x" * (n + 1)) for n in range(8)]
    out = post(c, items).json()["items"]
    assert all(o["error"] is None and o["wav_b64"] for o in out)
    assert [len(call["ids"]) for call in synth.calls] == [8, 4, 2, 2, 4, 2, 2]


def test_oom_releases_failed_batch_before_cleanup_and_retry(voices, monkeypatch):
    refs = []
    retries = []

    class Resource:
        pass

    class Synth(FakeSynth):
        def generate(self, chunk, cfg, seed, max_new_tokens):
            if len(chunk) > 2:
                resource = Resource()
                refs.append(weakref.ref(resource))
                raise OutOfMemory("cuda oom")
            retries.append([r["id"] for r in chunk])
            return super().generate(chunk, cfg, seed, max_new_tokens)

    def clean_cache():
        assert all(ref() is None for ref in refs)

    monkeypatch.setattr(server, "empty_cuda_cache", clean_cache)
    synth = Synth()
    client = TestClient(create_app(synth, voices_dir=str(voices)))
    items = [item("i3", "xxx"), item("i1", "x"), item("i4", "xxxx"), item("i2", "xx")]

    out = post(client, items).json()["items"]

    assert [o["id"] for o in out] == ["i3", "i1", "i4", "i2"]
    assert all(o["error"] is None and o["wav_b64"] for o in out)
    assert retries == [["i1", "i2"], ["i3", "i4"]]


def test_oom_single_item_fails_others_succeed(voices):
    synth, c = make(voices, fail=lambda chunk, cfg: OutOfMemory("cuda oom")
                    if any(len(r["text"]) > 50 for r in chunk) else None)
    out = post(c, [item("small1", "x" * 5), item("huge", "x" * 80), item("small2", "x" * 6)]).json()["items"]
    by_id = {o["id"]: o for o in out}
    assert by_id["huge"]["wav_b64"] is None and "OutOfMemory" in by_id["huge"]["error"]
    assert by_id["small1"]["error"] is None and by_id["small2"]["error"] is None


def test_other_exception_isolated_to_chunk(voices):
    synth, c = make(voices, max_batch=2, fail=lambda chunk, cfg: ValueError("bad") if "x" * 30 in chunk[0]["text"] else None)
    items = [item("a", "x" * 5), item("b", "x" * 6), item("c", "x" * 30), item("d", "x" * 31)]
    out = {o["id"]: o for o in post(c, items).json()["items"]}
    assert out["a"]["error"] is None and out["b"]["error"] is None
    assert "ValueError" in out["c"]["error"] and "ValueError" in out["d"]["error"]
    assert out["c"]["wav_b64"] is None and out["c"]["seconds"] == 0.0
    assert len(synth.calls) == 2  # no retry for non-OOM errors


def test_speech_endpoint(voices):
    synth, c = make(voices)
    r = c.post("/v1/audio/speech", json={"input": "x" * 100, "voice": "Ann.wav", "ref_text": "clip",
                                         "response_format": "wav"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    assert sf.info(io.BytesIO(r.content)).samplerate == 24000
    assert c.post("/v1/audio/speech", json={"input": "hi", "voice": "../x"}).status_code == 400


def test_generation_lock_serialises_requests(voices):
    synth, c = make(voices, delay=0.05)
    results = []
    threads = [threading.Thread(target=lambda: results.append(post(c, [item("a")]).status_code)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [200] * 4
    assert synth.max_active == 1 and synth.loads == 1 and len(synth.calls) == 4


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_guard_ignores_odd_slow_batches_and_small_chunks():
    guard = server.SlowdownGuard(1.5, full_batch=24, window=4, clock=FakeClock())
    for speed in (3.1, 1.2, 3.0, 1.1, 2.9, 3.3):  # healthy books have up to two slow batches in a row
        guard.record(32, speed)
    for _ in range(10):
        guard.record(8, 0.3)  # a chunk of a few sentences runs well under real time anyway
    assert not guard.reload_due


def test_guard_asks_for_a_reload_when_full_batches_slow_down():
    guard = server.SlowdownGuard(1.5, full_batch=24, window=4, clock=FakeClock())
    for speed in (1.2, 1.6, 1.1, 1.4):  # median 1.3x
        guard.record(30, speed)
    assert guard.reload_due


def test_guard_waits_out_the_cooldown_when_a_reload_did_not_help():
    clock = FakeClock()
    guard = server.SlowdownGuard(1.5, full_batch=24, window=4, cooldown=1800, clock=clock)
    guard.reloaded()
    for _ in range(4):
        guard.record(32, 1.0)
    assert not guard.reload_due
    clock.now = 1800
    for _ in range(4):
        guard.record(32, 1.0)
    assert guard.reload_due


def test_guard_off_at_zero():
    guard = server.SlowdownGuard(0, full_batch=1, window=1)
    guard.record(32, 0.01)
    assert not guard.reload_due


def test_slowdown_reloads_the_model_before_the_next_batch(voices):
    clock = FakeClock()
    guard = server.SlowdownGuard(float("inf"), full_batch=2, window=2, clock=clock)  # every chunk is "slow"
    synth = FakeSynth()
    c = TestClient(create_app(synth, voices_dir=str(voices), max_batch=2, guard=guard))
    items = [item(f"i{n}") for n in range(4)]
    post(c, items)
    assert guard.reload_due and (synth.loads, synth.unloads) == (1, 0)
    out = post(c, items).json()["items"]
    assert all(o["error"] is None and o["wav_b64"] for o in out)
    assert (synth.loads, synth.unloads) == (2, 1)
    post(c, items)  # still slow inside the cooldown: no second reload
    post(c, items)
    assert (synth.loads, synth.unloads) == (2, 1)
    clock.now = server.SLOW_RELOAD_COOLDOWN_SECONDS
    post(c, items)
    post(c, items)
    assert (synth.loads, synth.unloads) == (3, 2)


def test_unload_drops_a_pending_reload(voices):
    guard = server.SlowdownGuard(float("inf"), full_batch=1, window=1)
    synth = FakeSynth()
    c = TestClient(create_app(synth, voices_dir=str(voices), guard=guard))
    post(c, [item("a")])
    assert guard.reload_due
    c.post("/api/unload")
    post(c, [item("a")])
    assert (synth.loads, synth.unloads) == (2, 1) and guard.last_reload is None


def test_reference_cache_encodes_a_clip_once_until_it_changes(tmp_path):
    clip = tmp_path / "Ann.wav"
    clip.write_bytes(b"one")
    encoded = []
    cache = server.ReferenceCache(lambda tokenizer, path: encoded.append(path) or f"codes {len(encoded)}")
    assert [cache("tok", str(clip)) for _ in range(3)] == ["codes 1"] * 3
    clip.write_bytes(b"replaced")  # a new clip under the same name
    assert [cache("tok", str(clip)) for _ in range(2)] == ["codes 2"] * 2
    cache.clear()
    assert cache("tok", str(clip)) == "codes 3"
    assert encoded == [str(clip)] * 3


def test_the_synthesizer_routes_reference_encoding_through_its_cache(monkeypatch, tmp_path):
    import sys
    import types
    calls = []
    templates = types.ModuleType("breeze_infer.templates")
    templates._encode_prompt_audio = lambda tokenizer, path: calls.append(path) or "codes"
    package = types.ModuleType("breeze_infer")
    package.templates = templates
    monkeypatch.setitem(sys.modules, "breeze_infer", package)
    monkeypatch.setitem(sys.modules, "breeze_infer.templates", templates)
    clip = tmp_path / "Ann.wav"
    clip.write_bytes(b"x")
    synth = server.BreezeSynthesizer(model_dir=str(tmp_path))
    synth._cache_references()
    synth._cache_references()  # a reload doesn't wrap the cache in another
    assert isinstance(templates._encode_prompt_audio, server.ReferenceCache)
    assert not isinstance(templates._encode_prompt_audio.encode, server.ReferenceCache)
    for _ in range(3):
        assert templates._encode_prompt_audio("tok", str(clip)) == "codes"
    assert calls == [str(clip)]
    synth.unload()
    templates._encode_prompt_audio("tok", str(clip))
    assert calls == [str(clip)] * 2

