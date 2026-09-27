import io
import os
import subprocess
import tempfile
import unittest

from audiobook_generator.utils.utils import merge_audio_segments, pydub_merge_audio_segments


def _chunk_bytes(fmt: str, seconds: float = 1.0) -> bytes:
    """One real, independently-encoded audio chunk in the given format."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, f"c.{fmt}")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                        "-i", f"sine=frequency=300:duration={seconds}", "-ar", "24000", "-ac", "1",
                        path], check=True)
        with open(path, "rb") as f:
            return f.read()


def _duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    ).stdout
    return float(out.strip())


class TestMergeAudioSegmentsFraming(unittest.TestCase):
    """F-08: direct (byte-concatenation) merge only produces a playable file for formats
    whose frames carry their own sync/length. A WAV or FLAC chapter made of several
    independently-encoded chunks concatenated this way plays only the first chunk: ffprobe
    still reports a "valid" file, just a short one.
    """

    def test_wav_chunks_are_pydub_merged_not_silently_truncated(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "out.wav")
            segments = [io.BytesIO(_chunk_bytes("wav")) for _ in range(3)]
            merge_audio_segments(segments, output, "wav", ["a", "b", "c"], use_pydub_merge=False)
            self.assertAlmostEqual(_duration(output), 3.0, delta=0.15)

    def test_flac_chunks_are_pydub_merged_not_silently_truncated(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "out.flac")
            segments = [io.BytesIO(_chunk_bytes("flac")) for _ in range(3)]
            merge_audio_segments(segments, output, "flac", ["a", "b", "c"], use_pydub_merge=False)
            self.assertAlmostEqual(_duration(output), 3.0, delta=0.15)

    def test_mp3_chunks_still_use_the_direct_concatenation_fast_path(self):
        # Unchanged behaviour: mp3 frames are self-framing, so the cheap byte-concatenation
        # path stays in use for it (no re-decode/re-encode cost per chapter).
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "out.mp3")
            segments = [io.BytesIO(_chunk_bytes("mp3")) for _ in range(3)]
            merge_audio_segments(segments, output, "mp3", ["a", "b", "c"], use_pydub_merge=False)
            self.assertAlmostEqual(_duration(output), 3.0, delta=0.5)


class TestPydubMergeDoesNotDoubleDelete(unittest.TestCase):
    """F-09: --use_pydub_merge removed every temp file in the finally loop, then removed
    the last one again outside the loop. The second os.remove always raised
    FileNotFoundError after a successful merge, which propagated out of a successful
    conversion and made process_chapter's own finally block delete the finished chapter.
    """

    def test_pydub_merge_succeeds_without_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            one = os.path.join(tmp, "1.mp3")
            two = os.path.join(tmp, "2.mp3")
            with open(one, "wb") as f:
                f.write(_chunk_bytes("mp3"))
            with open(two, "wb") as f:
                f.write(_chunk_bytes("mp3"))
            output = os.path.join(tmp, "out.mp3")

            pydub_merge_audio_segments([one, two], output, "mp3")  # must not raise

            self.assertTrue(os.path.isfile(output))
            self.assertFalse(os.path.exists(one))
            self.assertFalse(os.path.exists(two))


if __name__ == "__main__":
    unittest.main()
