"""Server contract tests with a fake enhancer: no torch, no GPU."""
import io
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from server import create_app


class FakeEnhancer:
    """Returns its input at half volume, labelled with out_rate; records every call."""

    def __init__(self, out_rate=44100):
        self.loaded = False
        self.loads = self.unloads = 0
        self.calls = []
        self.out_rate = out_rate

    def load(self):
        self.loaded, self.loads = True, self.loads + 1

    def unload(self):
        self.loaded, self.unloads = False, self.unloads + 1

    def enhance(self, samples, rate):
        self.calls.append((samples.copy(), rate))
        return samples * 0.5, self.out_rate


def _wav(samples, rate=24000):
    buffer = io.BytesIO()
    sf.write(buffer, samples, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


@pytest.fixture
def served():
    fake = FakeEnhancer()
    return fake, TestClient(create_app(fake))


def test_health_answers_with_the_model_unloaded(served):
    fake, client = served
    assert client.get("/health").json() == {"status": "ok", "loaded": False}


def test_enhance_returns_mono_pcm16_at_the_models_rate_peaking_at_minus_1_dbfs(served):
    fake, client = served
    stereo = np.stack([np.full(2400, 0.2), np.full(2400, 0.4)], axis=1)
    response = client.post("/enhance", content=_wav(stereo))
    assert response.status_code == 200 and response.headers["content-type"] == "audio/wav"
    out, rate = sf.read(io.BytesIO(response.content))
    info = sf.info(io.BytesIO(response.content))
    assert (rate, info.channels, info.subtype) == (44100, 1, "PCM_16")
    assert np.abs(out).max() == pytest.approx(10 ** (-1 / 20), abs=1e-3)
    (samples, sent_rate), = fake.calls
    assert sent_rate == 24000 and samples.ndim == 1 and samples.dtype == np.float32
    assert samples.mean() == pytest.approx(0.3, abs=1e-3)  # the two channels averaged
    assert fake.loads == 1 and client.get("/health").json()["loaded"] is True


def test_the_model_loads_once_for_a_run_of_clips(served):
    fake, client = served
    for _ in range(3):
        assert client.post("/enhance", content=_wav(np.full(2400, 0.1))).status_code == 200
    assert (fake.loads, len(fake.calls)) == (1, 3)


@pytest.mark.parametrize("body", [b"", b"not audio at all"])
def test_empty_or_undecodable_audio_is_refused(served, body):
    fake, client = served
    assert client.post("/enhance", content=body).status_code == 400
    assert fake.calls == [] and fake.loads == 0


def test_unload_frees_the_model(served):
    fake, client = served
    client.post("/enhance", content=_wav(np.full(2400, 0.1)))
    assert client.post("/api/unload").json() == {"loaded": False}
    assert fake.unloads == 1 and client.get("/health").json()["loaded"] is False


def test_an_idle_model_unloads_by_itself(monkeypatch):
    monkeypatch.setenv("RESEMBLE_IDLE_UNLOAD_SECONDS", "0.05")
    fake = FakeEnhancer()
    client = TestClient(create_app(fake))
    client.post("/enhance", content=_wav(np.full(2400, 0.1)))
    deadline = time.time() + 5
    while fake.loaded and time.time() < deadline:
        time.sleep(0.01)
    assert fake.unloads == 1 and not fake.loaded
