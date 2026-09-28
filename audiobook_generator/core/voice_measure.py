"""What each voice sounds like, measured, so cast suggestions can match voices to characters.

A voice is measured from speech the engine makes with it (MEASURE_TEXT), not from its reference
clip. Measured 2026-09-28 on the owner's 33 Chatterbox voices: a clip's pitch predicts the
generated speech's well (r 0.96; the same third of its gender's range for 27 of 33 voices) but its
huskiness doesn't (19 of 33), and the generated speech is what a listener hears. Praat
(parselmouth) gives three numbers per voice:

    f0_median  median pitch, Hz
    f0_range   pitch spread (10th to 90th percentile), semitones: a lively or an even delivery
    hnr        harmonics-to-noise ratio, dB: low sounds husky or breathy, high sounds clear

They are saved in VOICE_FEATURES_FILE (app data, next to voice_genders.json) with the voice file's
size and modification time, so a replaced voice is measured again. For matching, each number
becomes a percentile among the measured voices of the same gender (voice_traits): "low" means low
for a woman, or for a man.
"""
import io
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

MEASURE_TEXT = "I told you already, the boat leaves at dawn, and nobody waits for anyone. Do you understand me?"
VOICE_FEATURES_FILE = "voice_features.json"
FEATURES_VERSION = 1
PITCH_FLOOR_HZ, PITCH_CEILING_HZ = 60, 500


# ---- measuring ----

def measure_audio(audio: bytes) -> dict:
    """{f0_median, f0_range, hnr} of one clip of speech (any format ffmpeg reads).

    Raises ValueError when the clip holds too little voiced speech to measure."""
    import numpy as np
    import parselmouth
    from pydub import AudioSegment

    segment = AudioSegment.from_file(io.BytesIO(audio)).set_channels(1)
    samples = np.array(segment.get_array_of_samples(), dtype=np.float64) / float(1 << (8 * segment.sample_width - 1))
    sound = parselmouth.Sound(samples, sampling_frequency=segment.frame_rate)
    f0 = sound.to_pitch_ac(time_step=0.01, pitch_floor=PITCH_FLOOR_HZ,
                           pitch_ceiling=PITCH_CEILING_HZ).selected_array["frequency"]
    voiced = f0[f0 > 0]
    if len(voiced) < 50:  # half a second
        raise ValueError("too little voiced speech to measure")
    semitones = 12 * np.log2(voiced / np.median(voiced))
    harmonicity = sound.to_harmonicity_cc(time_step=0.01, minimum_pitch=PITCH_FLOOR_HZ).values
    return {
        "f0_median": round(float(np.median(voiced)), 1),
        "f0_range": round(float(np.percentile(semitones, 90) - np.percentile(semitones, 10)), 2),
        "hnr": round(float(np.mean(harmonicity[harmonicity > -200])), 2),
    }


def file_signature(path: str) -> str:
    """Size and modification time of a voice file: a replaced file gets a new signature."""
    stat = os.stat(path)
    return f"{stat.st_size}:{int(stat.st_mtime)}"


# ---- the saved measurements ----

def load_features(path: Optional[str] = None) -> Dict[str, dict]:
    """{voice: measurement} for voices measured with the current MEASURE_TEXT (a changed text
    makes every earlier measurement stale, so none is returned)."""
    try:
        with open(path or VOICE_FEATURES_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("text") != MEASURE_TEXT or not isinstance(data.get("voices"), dict):
        return {}
    return {voice: m for voice, m in data["voices"].items()
            if isinstance(m, dict) and all(isinstance(m.get(k), (int, float)) for k in ("f0_median", "f0_range", "hnr"))}


def _write(voices: Dict[str, dict], path: Optional[str]) -> None:
    path = path or VOICE_FEATURES_FILE
    folder = os.path.dirname(path) or "."
    os.makedirs(folder, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"version": FEATURES_VERSION, "text": MEASURE_TEXT, "voices": voices}, f,
                  ensure_ascii=False, indent=1, sort_keys=True)
    os.replace(tmp, path)


def save_features(voice: str, measurement: dict, signature: str = "", path: Optional[str] = None) -> Dict[str, dict]:
    """Record one voice's measurement; returns every saved measurement."""
    voices = load_features(path)
    voices[voice] = {**measurement, "signature": signature, "measured": datetime.now().strftime("%Y-%m-%d %H:%M")}
    _write(voices, path)
    return voices


def forget_features(voice: str, path: Optional[str] = None) -> None:
    """Drop a deleted voice's measurement (nothing happens when it had none)."""
    voices = load_features(path)
    if voices.pop(voice, None) is not None:
        _write(voices, path)


def voices_to_measure(voice_files: Dict[str, str], features: Dict[str, dict]) -> List[str]:
    """Voices ({voice: file path}) never measured, or whose file changed since, in name order."""
    stale = []
    for voice, file_path in sorted(voice_files.items(), key=lambda kv: kv[0].lower()):
        saved = features.get(voice)
        try:
            signature = file_signature(file_path)
        except OSError:
            continue
        if not saved or saved.get("signature") != signature:
            stale.append(voice)
    return stale


# ---- percentiles and words ----

def _percentiles(values: Dict[str, float]) -> Dict[str, float]:
    """{key: 0..1} rank of each value among the others (ties share their average rank)."""
    ordered = sorted(values.values())
    if len(ordered) < 2:
        return {key: 0.5 for key in values}
    result = {}
    for key, value in values.items():
        first = ordered.index(value)
        last = len(ordered) - 1 - ordered[::-1].index(value)
        result[key] = (first + last) / 2 / (len(ordered) - 1)
    return result


def voice_traits(voices: List[Tuple[str, str]], features: Dict[str, dict]) -> Dict[str, dict]:
    """{voice: {"pitch", "husky", "expressive"}}, each 0..1, for the measured ones of the given
    (voice, gender) pairs. A voice is ranked among the measured voices of its own gender; a neutral
    voice among all of them."""
    measured = [(voice, gender) for voice, gender in voices if voice in features]
    traits: Dict[str, dict] = {}
    for group in ("female", "male", "neutral"):
        members = [voice for voice, gender in measured if gender == group]
        pool = members if group != "neutral" else [voice for voice, _ in measured]
        if not members:
            continue
        pitch = _percentiles({v: features[v]["f0_median"] for v in pool})
        husky = _percentiles({v: -features[v]["hnr"] for v in pool})
        expressive = _percentiles({v: features[v]["f0_range"] for v in pool})
        for voice in members:
            traits[voice] = {"pitch": pitch[voice], "husky": husky[voice], "expressive": expressive[voice]}
    return traits


def _band(value: float, low: str, middle: str, high: str) -> str:
    return low if value < 1 / 3 else high if value > 2 / 3 else middle


def describe(traits: Optional[dict], gender: str = "neutral") -> str:
    """A voice's traits in words: "low for a woman, husky, even"."""
    if not traits:
        return "not measured"
    who = {"female": " for a woman", "male": " for a man"}.get(gender, "")
    words = [f"{_band(traits['pitch'], 'low', 'medium', 'high')}{who}"]
    words += [w for w in (_band(traits["husky"], "clear", "", "husky"),
                          _band(traits["expressive"], "even", "", "expressive")) if w]
    return ", ".join(words)
