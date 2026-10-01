"""Per-voice tone matching for Chatterbox narration.

Chatterbox renders some voices much brighter than the reference clip it clones them from. Measured
2026-10-01 with the same two lines per voice against each voice's own clip: Teen came out 7-13 dB
brighter from 4 kHz up, Everett 7-11 dB above 8 kHz and Adrian 14 dB at 10-12 kHz, while Maya stayed
within about 3 dB and Gianna came out slightly darker. That excess is the "treble turned up, faint
hiss" the owner heard; it is on the voice, not in the pauses (every take's background was quieter than
its clip's). One fixed treble cut would dull the darker voices, so each voice is matched to its own
clip instead: its speech is measured as the book is generated, compared with the clip in third-octave
bands from 1.6 kHz up, and turned down only where it is brighter (a cut, never a boost). The owner
picked this by ear over the fixed filter on 2026-10-01.
"""

import logging
import os
import subprocess
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

FRAME = 2048
CORE_HZ = (300, 1600)       # the speech core every band is measured against
LOUD_DB = -30.0             # frames within this of a take's peak count as speech
MAX_CUT_DB = -12.0
MIN_SPEECH_SECONDS = 8.0    # a voice's balance is trusted once this much of its speech is measured
_LOWEST_BAND_HZ, _HIGHEST_HZ = 1600, 11800


def band_edges(rate: int) -> np.ndarray:
    """Third-octave band edges from 1.6 kHz to just under the rate's Nyquist (11.8 kHz at 24 kHz)."""
    top = min(_HIGHEST_HZ, 0.49 * rate)
    edges = _LOWEST_BAND_HZ * 2 ** (np.arange(12) / 3)
    return np.append(edges[edges < top], top)


def speech_spectrum(samples: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
    """Summed power spectrum of the frames within LOUD_DB of the take's peak, and their count."""
    count = len(samples) // FRAME
    peak = float(np.abs(samples).max()) if len(samples) else 0.0
    if count == 0 or peak <= 0:
        return None, 0
    frames = samples[: count * FRAME].reshape(count, FRAME)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    loud = frames[20 * np.log10(rms / peak + 1e-12) > LOUD_DB]
    if not len(loud):
        return None, 0
    return (np.abs(np.fft.rfft(loud * np.hanning(FRAME), axis=1)) ** 2).sum(axis=0), len(loud)


def balance(spectrum: np.ndarray, rate: int) -> np.ndarray:
    """Each band's level in dB relative to the speech core."""
    freqs = np.fft.rfftfreq(FRAME, 1 / rate)
    core = spectrum[(freqs >= CORE_HZ[0]) & (freqs < CORE_HZ[1])].mean()
    edges = band_edges(rate)
    return np.array([10 * np.log10(spectrum[(freqs >= lo) & (freqs < hi)].mean() / core)
                     for lo, hi in zip(edges[:-1], edges[1:])])


def cuts_for(made: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Per-band gain (dB, <= 0) that brings `made` down to `clip` wherever it is brighter, capped at
    MAX_CUT_DB and smoothed across neighbouring bands so the curve has no sharp notches."""
    gain = np.clip(clip - made, MAX_CUT_DB, 0.0)
    return np.convolve(np.pad(gain, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")


def apply(samples: np.ndarray, rate: int, cuts: np.ndarray) -> np.ndarray:
    """Filter one take with the band cuts (zero phase; padded so nothing wraps around its ends)."""
    if not len(samples) or not np.any(cuts):
        return samples
    padded = np.concatenate([np.zeros(FRAME, np.float32), samples, np.zeros(FRAME, np.float32)])
    freqs = np.fft.rfftfreq(len(padded), 1 / rate)
    edges = band_edges(rate)
    centers = np.sqrt(edges[:-1] * edges[1:])
    curve = np.interp(np.log2(np.maximum(freqs, 1)), np.log2(np.r_[1000, centers, rate / 2]),
                      np.r_[0.0, cuts, cuts[-1]])
    filtered = np.fft.irfft(np.fft.rfft(padded) * 10 ** (curve / 20), n=len(padded))
    return filtered[FRAME:FRAME + len(samples)].astype(np.float32)


def read_clip(path: str, rate: int) -> np.ndarray:
    """A reference clip as mono float samples at `rate`. ffmpeg resamples with a proper low-pass;
    pydub's resampler would fold a 44.1 kHz clip's top octave into the bands being compared."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(rate), "pipe:1"],
                         capture_output=True, check=True, timeout=60).stdout
    return np.frombuffer(raw, dtype=np.float32)


class ToneMatcher:
    """Measures each voice's speech across a book run and works out its cuts against its clip."""

    def __init__(self, voices_dir: Optional[str]):
        self.voices_dir = voices_dir
        self._speech: Dict[str, Tuple[np.ndarray, int]] = {}
        self._clips: Dict[Tuple[str, int], Optional[np.ndarray]] = {}

    def add(self, voice: str, samples: np.ndarray) -> None:
        spectrum, frames = speech_spectrum(samples)
        if spectrum is None:
            return
        total, count = self._speech.get(voice, (0.0, 0))
        self._speech[voice] = (total + spectrum, count + frames)

    def cuts(self, voice: str, rate: int) -> Optional[np.ndarray]:
        """This voice's cuts so far, or None until MIN_SPEECH_SECONDS of it are measured or when its
        clip can't be read."""
        spectrum, frames = self._speech.get(voice, (None, 0))
        if spectrum is None or frames * FRAME / rate < MIN_SPEECH_SECONDS:
            return None
        clip = self._clip_balance(voice, rate)
        return None if clip is None else cuts_for(balance(spectrum, rate), clip)

    def _clip_balance(self, voice: str, rate: int) -> Optional[np.ndarray]:
        key = (voice, rate)
        if key not in self._clips:
            self._clips[key] = None
            path = os.path.join(self.voices_dir, voice) if self.voices_dir else None
            if not path or not os.path.isfile(path):
                logger.info("Tone match: no reference clip for %s, left as generated", voice)
                return None
            try:
                spectrum, frames = speech_spectrum(read_clip(path, rate))
                if spectrum is not None:
                    self._clips[key] = balance(spectrum, rate)
            except (subprocess.SubprocessError, OSError) as e:
                logger.warning("Tone match: couldn't read %s (%s), left as generated", path, e)
        return self._clips[key]


def describe(cuts: np.ndarray, rate: int) -> str:
    """'up to 11.0 dB from 4.5 kHz' for the log, or 'none' when the voice needs no cut."""
    if cuts.min() > -0.5:
        return "none"
    edges = band_edges(rate)
    centers = np.sqrt(edges[:-1] * edges[1:])
    first = centers[np.argmax(cuts <= -1.0)] if np.any(cuts <= -1.0) else centers[int(np.argmin(cuts))]
    return f"up to {-cuts.min():.1f} dB from {first / 1000:.1f} kHz"
