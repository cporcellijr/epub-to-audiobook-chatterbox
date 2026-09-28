import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
from openai import APIConnectionError, APIStatusError, OpenAIError

from audiobook_generator.tts_providers.base_tts_provider import get_tts_provider
from audiobook_generator.tts_providers.openai_tts_provider import (
    SERVER_WAIT_TOTAL_SECONDS,
    OpenAITTSProvider,
)
from tests.test_utils import get_openai_config


def _connection_error() -> APIConnectionError:
    return APIConnectionError(request=httpx.Request("POST", "http://chatterbox:8004/v1/audio/speech"))


def _status_error(status_code: int) -> APIStatusError:
    response = httpx.Response(status_code, request=httpx.Request("POST", "http://chatterbox:8004/v1/audio/speech"))
    return APIStatusError("boom", response=response, body=None)


class _FakeClock:
    """A monotonic clock double: each call advances by `step` and returns the new total."""

    def __init__(self, step: float = 0.0):
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


class TestOpenAiTtsProvider(unittest.TestCase):

    def test_missing_env_var_keys(self):
        config = get_openai_config()
        with self.assertRaises(OpenAIError):
            get_tts_provider(config)

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'fake_key'})
    def test_estimate_cost(self):
        config = get_openai_config()
        tts_provider = get_tts_provider(config)
        self.assertIsInstance(tts_provider, OpenAITTSProvider)
        self.assertEqual(tts_provider.estimate_cost(1000000), 15)

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'fake_key'})
    def test_default_args(self):
        config = get_openai_config()
        config.model_name = None
        config.voice_name = None
        config.output_format = None
        tts_provider = get_tts_provider(config)
        self.assertIsInstance(tts_provider, OpenAITTSProvider)
        self.assertEqual(tts_provider.config.model_name, "gpt-4o-mini-tts")
        self.assertEqual(tts_provider.config.voice_name, "alloy")
        self.assertEqual(tts_provider.config.output_format, "mp3")


class TestOpenAiBaseUrl(unittest.TestCase):
    """A per-config base URL (e.g. Kokoro's) must win over the ambient OPENAI_BASE_URL, since a
    container can have both set at once (Chatterbox's env var plus a Kokoro-selected book)."""

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'fake_key', 'OPENAI_BASE_URL': 'http://chatterbox:8004/v1'})
    @patch('audiobook_generator.tts_providers.openai_tts_provider.OpenAI')
    def test_per_config_base_url_overrides_the_ambient_env_var(self, mock_openai):
        config = get_openai_config()
        config.openai_base_url = 'http://kokoro:8880/v1'
        get_tts_provider(config)
        self.assertEqual(mock_openai.call_args.kwargs['base_url'], 'http://kokoro:8880/v1')

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'fake_key'})
    @patch('audiobook_generator.tts_providers.openai_tts_provider.OpenAI')
    def test_no_config_base_url_passes_none_through_unchanged(self, mock_openai):
        config = get_openai_config()
        get_tts_provider(config)
        self.assertIsNone(mock_openai.call_args.kwargs['base_url'])


class TestCreateSpeechRetry(unittest.TestCase):
    """F-02: a unit request must survive Chatterbox being temporarily unavailable (still
    loading its model after a restart) instead of failing the chapter after the SDK's own
    ~7 s of retries."""

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'fake_key'})
    def _provider(self) -> OpenAITTSProvider:
        provider = get_tts_provider(get_openai_config())
        provider.client = MagicMock()
        return provider

    def test_retries_connection_error_until_success(self):
        provider = self._provider()
        provider.client.audio.speech.create.side_effect = [
            _connection_error(), _connection_error(), SimpleNamespace(content=b"ok"),
        ]
        sleeps = []
        result = provider._create_speech(sleep=sleeps.append, clock=_FakeClock(), input="hi")
        self.assertEqual(result.content, b"ok")
        self.assertEqual(provider.client.audio.speech.create.call_count, 3)
        self.assertEqual(sleeps, [2.0, 4.0])

    def test_retries_503_until_success(self):
        provider = self._provider()
        provider.client.audio.speech.create.side_effect = [
            _status_error(503), _status_error(502), SimpleNamespace(content=b"ok"),
        ]
        sleeps = []
        result = provider._create_speech(sleep=sleeps.append, clock=_FakeClock(), input="hi")
        self.assertEqual(result.content, b"ok")
        self.assertEqual(provider.client.audio.speech.create.call_count, 3)
        self.assertEqual(sleeps, [2.0, 4.0])

    def test_does_not_retry_4xx(self):
        provider = self._provider()
        provider.client.audio.speech.create.side_effect = _status_error(400)
        sleeps = []
        with self.assertRaises(APIStatusError):
            provider._create_speech(sleep=sleeps.append, clock=_FakeClock(step=50.0), input="hi")
        self.assertEqual(provider.client.audio.speech.create.call_count, 1)
        self.assertEqual(sleeps, [])

    def test_gives_up_after_total_budget(self):
        provider = self._provider()
        provider.client.audio.speech.create.side_effect = [
            _connection_error(), _connection_error(), _connection_error(),
        ]
        sleeps = []
        with self.assertRaises(APIConnectionError):
            provider._create_speech(sleep=sleeps.append, clock=_FakeClock(step=200.0), input="hi")
        self.assertEqual(provider.client.audio.speech.create.call_count, 3)
        self.assertEqual(sleeps, [2.0, 4.0])
        self.assertLessEqual(sum(sleeps), SERVER_WAIT_TOTAL_SECONDS)


if __name__ == '__main__':
    unittest.main()
