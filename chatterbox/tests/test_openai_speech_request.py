"""Pure-Python tests for server.OpenAISpeechRequest's optional delivery overrides (adaptive
delivery, audiobook_generator's Chatterbox-only feature): default to None (today's behaviour,
falling back to the saved generation defaults) and validate their ranges. No GPU or running server
needed -- this only builds the Pydantic model, it never calls engine.synthesize.

Run inside the image (has FastAPI/Pydantic/torch etc.):
    docker run --rm --entrypoint python3 -v <repo>/chatterbox:/app -w /app \
        chatterbox-tts-server:local -m unittest discover -s tests
"""
import os
import sys
import unittest

# tests/ is not a package here; make the chatterbox root (this file's parent's parent) importable
# regardless of how the test runner set up sys.path/top-level-dir.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import ValidationError

from server import OpenAISpeechRequest


def _request(**overrides) -> OpenAISpeechRequest:
    fields = dict(model="chatterbox", input="Hello there.", voice="Elena.wav")
    fields.update(overrides)
    return OpenAISpeechRequest(**fields)


class TestOpenAISpeechRequestDeliveryFields(unittest.TestCase):

    def test_delivery_fields_default_to_none(self):
        request = _request()
        self.assertIsNone(request.exaggeration)
        self.assertIsNone(request.cfg_weight)
        self.assertIsNone(request.temperature)

    def test_values_within_range_are_accepted(self):
        request = _request(exaggeration=1.0, cfg_weight=0.4, temperature=0.7)
        self.assertEqual((request.exaggeration, request.cfg_weight, request.temperature), (1.0, 0.4, 0.7))

    def test_exaggeration_boundaries(self):
        self.assertEqual(_request(exaggeration=0.25).exaggeration, 0.25)
        self.assertEqual(_request(exaggeration=2.0).exaggeration, 2.0)
        with self.assertRaises(ValidationError):
            _request(exaggeration=0.24)
        with self.assertRaises(ValidationError):
            _request(exaggeration=2.01)

    def test_cfg_weight_boundaries(self):
        self.assertEqual(_request(cfg_weight=0.0).cfg_weight, 0.0)
        self.assertEqual(_request(cfg_weight=1.0).cfg_weight, 1.0)
        with self.assertRaises(ValidationError):
            _request(cfg_weight=-0.01)
        with self.assertRaises(ValidationError):
            _request(cfg_weight=1.01)

    def test_temperature_boundaries(self):
        self.assertEqual(_request(temperature=0.05).temperature, 0.05)
        self.assertEqual(_request(temperature=5.0).temperature, 5.0)
        with self.assertRaises(ValidationError):
            _request(temperature=0.04)
        with self.assertRaises(ValidationError):
            _request(temperature=5.01)


if __name__ == "__main__":
    unittest.main()
