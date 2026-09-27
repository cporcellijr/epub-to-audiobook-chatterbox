import os
import tempfile
import unittest
from unittest.mock import patch

from audiobook_generator.tts_providers.openai_tts_provider import get_openai_supported_voices
from audiobook_generator.ui.web_ui import default_openai_voice, openai_voice_choices, refresh_openai_voices


class TestOpenAiVoiceChoices(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        for name in ("love poem.wav", "Elena.wav", "Teen.mp3", "notes.txt"):
            open(os.path.join(self.tmp.name, name), "wb").close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_voices_listed_from_mounted_folder(self):
        with patch.dict(os.environ, {"TTS_VOICES_DIR": self.tmp.name}):
            choices = openai_voice_choices()
        self.assertEqual(choices, [("Elena", "Elena.wav"), ("love poem", "love poem.wav"), ("Teen", "Teen.mp3")])

    def test_folder_listing_does_not_call_the_tts_server(self):
        env = {"TTS_VOICES_DIR": self.tmp.name, "OPENAI_BASE_URL": "http://busy-server:8004/v1"}
        with patch.dict(os.environ, env), patch("urllib.request.urlopen") as urlopen:
            openai_voice_choices()
        urlopen.assert_not_called()

    def test_stock_voices_when_nothing_configured(self):
        with patch.dict(os.environ, {"TTS_VOICES_DIR": "", "OPENAI_BASE_URL": ""}):
            values = [value for _, value in openai_voice_choices()]
        self.assertEqual(values, list(get_openai_supported_voices()))

    def test_default_voice_prefers_configured_voice(self):
        choices = [("Elena", "Elena.wav"), ("Teen", "Teen.mp3")]
        with patch.dict(os.environ, {"OPENAI_DEFAULT_VOICE": "Teen.mp3"}):
            self.assertEqual(default_openai_voice(choices), "Teen.mp3")
        with patch.dict(os.environ, {"OPENAI_DEFAULT_VOICE": "missing.wav"}):
            self.assertEqual(default_openai_voice(choices), "Elena.wav")

    def test_refresh_picks_up_new_voice_file(self):
        with patch.dict(os.environ, {"TTS_VOICES_DIR": self.tmp.name}):
            open(os.path.join(self.tmp.name, "good morning.wav"), "wb").close()
            update = refresh_openai_voices()
        self.assertIn(("good morning", "good morning.wav"), update["choices"])


if __name__ == "__main__":
    unittest.main()
