"""Dry routing checks through the production provider, without generating audio."""

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from audiobook_generator.core import cast as cast_store
from audiobook_generator.core.dialogue import DIALOGUE, chapter_segments

try:
    import sentencex  # noqa: F401
except ImportError:
    # Routing checks need segment boundaries only; keep this runnable without the optional tokenizer.
    sentencex = types.ModuleType("sentencex")
    sentencex.segment = lambda _language, text: [text]
    sys.modules["sentencex"] = sentencex

try:
    from audiobook_generator.tts_providers.openai_tts_provider import (
        OpenAITTSProvider, adaptive_paragraph_units, adaptive_units,
        voiced_paragraph_units, voiced_units,
    )
    _PROVIDER_IMPORT_ERROR = ""
except ModuleNotFoundError as error:
    OpenAITTSProvider = None
    _PROVIDER_IMPORT_ERROR = str(error)


@unittest.skipIf(OpenAITTSProvider is None, f"provider dependencies unavailable: {_PROVIDER_IMPORT_ERROR}")
class TestProductionCastRouting(unittest.TestCase):
    def _provider(self, cast_path):
        from audiobook_generator.config.general_config import GeneralConfig

        fields = dict(
            tts="openai", model_name="chatterbox", voice_name="SelectedNarrator.wav",
            output_format="mp3", speed=1.0, instructions=None, language="en",
            sentence_pause_ms=100, paragraph_pause_ms=300, paced_unit_mode="sentence",
            voice_mode="cast", dialogue_voice="Fallback.wav", cast_file=str(cast_path),
            openai_base_url=None,
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            return OpenAITTSProvider(GeneralConfig(SimpleNamespace(**fields)))

    def test_narrator_mara_and_minor_routes_survive_saved_cast_in_every_text_mode(self):
        passages = {
            7: ('I waited beside the door. "I can help you now," I said. '
                '"I found Mara\'s note," Mara said. "I can check the chart," said Doctor.'),
            11: ('I checked the room. "I will come back soon," I said. '
                 '"The bag is ready," Mara said. "I will call the doctor," said Nurse.'),
        }
        cast = cast_store.new_cast("audit", "/book.epub", "Book", "Author", "chatterbox", "SelectedNarrator.wav", [7, 11])
        cast["book_tone"] = {"point_of_view": "first", "pov_key": "wren"}
        cast["characters"] = {
            "wren": {"name": "Wren", "voice": None, "lines": 2},
            "mara": {"name": "Mara", "voice": "Mara.wav", "lines": 2},
            "doctor": {"name": "Doctor", "voice": "Doctor.wav", "lines": 1},
            "nurse": {"name": "Nurse", "voice": "Nurse.wav", "lines": 1},
        }
        for number, text in passages.items():
            ids = [segment.line_id for paragraph in chapter_segments(text)
                   for segment in paragraph if segment.kind == DIALOGUE]
            self.assertEqual(ids, [1, 2, 3])
            speakers = ["wren", "mara", "doctor" if number == 7 else "nurse"]
            cast["chapters"][cast_store.text_hash(text)] = {
                "number": number, "narrator": "wren",
                "lines": {str(line): speaker for line, speaker in zip(ids, speakers)},
            }

        with tempfile.TemporaryDirectory() as tmp:
            cast_path = Path(tmp) / "cast.json"
            cast_store.save_cast(str(cast_path), cast)
            self.assertEqual(cast_store.load_cast(str(cast_path))["characters"]["mara"]["voice"], "Mara.wav")
            snapshot = json.loads(json.dumps(cast_store.load_cast(str(cast_path))))
            self.assertEqual(snapshot["characters"]["mara"]["name"], "Mara")
            snapshot_path = Path(tmp) / "queue-snapshot.json"
            snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
            provider = self._provider(snapshot_path)

            for number, text in passages.items():
                voice_of = provider._voice_of(text)
                speaker_of = provider._speaker_of(text)
                dialogue_routes = [
                    (segment.line_id, voice_of(segment), speaker_of(segment))
                    for paragraph in chapter_segments(text)
                    for segment in paragraph if segment.kind == DIALOGUE
                ]
                self.assertEqual(dialogue_routes[0], (1, "SelectedNarrator.wav", None))
                self.assertEqual(dialogue_routes[1], (2, "Mara.wav", "mara"))
                minor = "doctor" if number == 7 else "nurse"
                self.assertEqual(dialogue_routes[2], (3, f"{minor.title()}.wav", minor))

                sentence_units = voiced_units(text, "en", voice_of)
                paragraph_units = voiced_paragraph_units(text, "en", voice_of)
                mood = lambda _piece: "even"
                adaptive = adaptive_units(text, "en", voice_of, mood, speaker_of)
                adaptive_paragraph = adaptive_paragraph_units(text, "en", voice_of, mood, speaker_of)
                for units, voice_index in ((sentence_units, 3), (paragraph_units, 4),
                                           (adaptive, 3), (adaptive_paragraph, 4)):
                    mara_units = [u for u in units if "Mara's note" in u[1] or "bag is ready" in u[1]]
                    minor_units = [u for u in units if "check the chart" in u[1] or "call the doctor" in u[1]]
                    self.assertTrue(mara_units)
                    self.assertTrue(minor_units)
                    self.assertTrue(all(u[voice_index] == "Mara.wav" for u in mara_units))
                    self.assertTrue(all(u[voice_index] == f"{minor.title()}.wav" for u in minor_units))
                self.assertTrue(any(u[-1] == "mara" for u in adaptive))
                self.assertTrue(any(u[-1] == minor for u in adaptive))


if __name__ == "__main__":
    unittest.main()
