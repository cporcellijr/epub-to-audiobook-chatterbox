"""Offline evaluator: score a cast against a source-linked speaker reference.

The reference holds the book's own text, so it stays outside the repository (default: the
stack's data/diagnostics folder; override with --reference or CAST_AUDIT_REFERENCE), and so does a
book's alias file (--aliases: {"the model's name for someone": "reference speaker id"}).
Reported separately: wrong speakers, unresolved lines, identity splits and false merges (pairs of
lines), the narrator's lines and voice, chapter narrators (when the reference lists them), and voice
routes. An unnamed "The Narrator" is judged as whoever most of its lines belong to: one voice, one
person."""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from pathlib import Path


REFERENCE = Path(os.environ.get("CAST_AUDIT_REFERENCE") or Path(__file__).resolve().parents[3] / "data"
                 / "diagnostics" / "apex3_2026-10-01" / "expected_speakers_source_linked.json")
UNKNOWN = {"", "unknown", "null", "none", "unassigned", "unresolved"}
ANONYMOUS_NARRATOR = "the narrator"  # cast_llm.Roster's name for a never-named "I"


def _key(document, line):
    return int(document), int(line)


def _speaker(value, aliases=None):
    if value is None:
        return None
    value = str(value).strip().casefold()
    if value in UNKNOWN:
        return None
    return (aliases or {}).get(value, value)


def _reference_lines(reference):
    return {_key(row["document"], row["line"]): row for row in reference["lines"]}


def evaluate(reference, predictions, voice_routes=None, speaker_aliases=None):
    """Compare line predictions; voice_routes maps canonical speaker IDs to this run's picks."""
    expected = _reference_lines(reference)
    predicted = {_key(row["document"], row["line"]): row for row in predictions}
    if len(predicted) != len(predictions):
        raise ValueError("prediction document/line keys must be unique")

    missing = expected.keys() - predicted.keys()
    extra = predicted.keys() - expected.keys()
    unresolved, misattributions, wrong_routes = [], [], []
    resolved_pairs = []
    narrator = reference["narrator_speaker_id"]
    narrator_raw_ids, narrator_speakers, narrator_voices = set(), set(), set()
    narrator_scored = narrator_voice_coverage = 0
    narrator_wrong_identity = []
    route_coverage = 0

    for key, ref in expected.items():
        row = predicted.get(key)
        if row is None:
            continue
        raw_actual = _speaker(row.get("speaker"))
        actual = _speaker(row.get("speaker"), speaker_aliases)
        if raw_actual is None:
            unresolved.append(key)
        else:
            resolved_pairs.append((ref["speaker_id"], raw_actual))
            if actual != ref["speaker_id"]:
                misattributions.append({
                    "document": key[0], "line": key[1], "expected": ref["speaker_id"],
                    "actual": actual, "confidence": ref["confidence"],
                })
        if ref["speaker_id"] == narrator:
            narrator_scored += raw_actual is not None
            if raw_actual is None:
                narrator_wrong_identity.append({"document": key[0], "line": key[1], "actual": None})
            else:
                narrator_raw_ids.add(raw_actual)
                if actual != narrator:
                    narrator_wrong_identity.append({"document": key[0], "line": key[1], "actual": actual})
            if actual is not None:
                narrator_speakers.add(actual)
            voice = row.get("voice")
            if voice:
                narrator_voice_coverage += 1
                narrator_voices.add(str(voice))
        route = (voice_routes or {}).get(ref["speaker_id"])
        if route is not None:
            route_coverage += 1
            used = row.get("voice")
            if used != route:
                wrong_routes.append({
                    "document": key[0], "line": key[1], "speaker": ref["speaker_id"],
                    "expected_voice": route, "actual_voice": used,
                })

    false_merges = 0
    identity_splits = 0
    # ponytail: pairwise counts are O(n²), fine for this 269-line fixture; use contingency tables if it grows.
    for (gold_a, pred_a), (gold_b, pred_b) in combinations(resolved_pairs, 2):
        if pred_a == pred_b and gold_a != gold_b:
            false_merges += 1
        if gold_a == gold_b and pred_a != pred_b:
            identity_splits += 1

    chapter_narrators = None
    if reference.get("chapter_narrators") is not None:
        told = {}
        for row in predictions:
            told.setdefault(int(row["document"]), _speaker(row.get("chapter_narrator"), speaker_aliases))
        wrong_tellers = [{"document": int(document), "expected": teller, "actual": told.get(int(document))}
                         for document, teller in sorted(reference["chapter_narrators"].items(), key=lambda i: int(i[0]))
                         if int(document) in told and told[int(document)] != teller]
        chapter_narrators = {"checked": sum(int(d) in told for d in reference["chapter_narrators"]),
                             "wrong": wrong_tellers}

    review_counts = {}
    uncertain_reference_lines = []
    for key, row in expected.items():
        status = row.get("review_status", "unspecified")
        review_counts[status] = review_counts.get(status, 0) + 1
        if row.get("uncertainty"):
            uncertain_reference_lines.append({
                "document": key[0], "line": key[1], "speaker": row["speaker_id"],
                "confidence": row["confidence"], "uncertainty": row["uncertainty"],
            })

    return {
        "total_lines": len(expected),
        "predicted_lines": len(predicted),
        "reference_review": {
            "status_counts": review_counts,
            "uncertain_lines": len(uncertain_reference_lines),
            "details": uncertain_reference_lines,
        },
        "missing_lines": [{"document": d, "line": n} for d, n in sorted(missing)],
        "extra_lines": [{"document": d, "line": n} for d, n in sorted(extra)],
        "misattributions": {"count": len(misattributions), "lines": misattributions},
        "identity_splits": {"pair_count": identity_splits},
        "false_merges": {"pair_count": false_merges},
        "unresolved_lines": {
            "count": len(unresolved),
            "lines": [{"document": d, "line": n} for d, n in sorted(unresolved)],
        },
        "narrator_consistency": {
            "expected_lines": sum(row["speaker_id"] == narrator for row in expected.values()),
            "scored_lines": narrator_scored,
            "missing_lines": sum(row["speaker_id"] == narrator for row in expected.values()) - narrator_scored,
            "wrong_identity_lines": narrator_wrong_identity,
            "raw_speaker_ids": sorted(narrator_raw_ids),
            "speaker_ids": sorted(narrator_speakers),
            "speaker_consistent": narrator_scored == sum(row["speaker_id"] == narrator for row in expected.values()) and narrator_speakers == {narrator},
            "voice_coverage": narrator_voice_coverage,
            "missing_voice_lines": sum(row["speaker_id"] == narrator for row in expected.values()) - narrator_voice_coverage,
            "voices": sorted(narrator_voices),
            "voice_consistent": narrator_voice_coverage == sum(row["speaker_id"] == narrator for row in expected.values()) and len(narrator_voices) == 1,
        },
        "chapter_narrators": chapter_narrators,
        "wrong_voice_routes": {
            "checked_lines": route_coverage,
            "count": len(wrong_routes),
            "lines": wrong_routes,
        },
    }


def summary(result) -> str:
    """One line per scored cast: each kind of error on its own."""
    tellers = result.get("chapter_narrators")
    return (f"wrong {result['misattributions']['count']} of {result['total_lines']}, "
            f"unresolved {result['unresolved_lines']['count']}, "
            f"split pairs {result['identity_splits']['pair_count']}, merged pairs {result['false_merges']['pair_count']}"
            + (f", wrong chapter narrators {len(tellers['wrong'])} of {tellers['checked']}" if tellers else "")
            + f", wrong voice routes {result['wrong_voice_routes']['count']}")


def _load_predictions(path, reference, aliases=None):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data, None, None
    if "lines" in data and isinstance(data["lines"], list):
        return data["lines"], data.get("voice_routes"), data.get("speaker_aliases")

    # Saved cast format: chapters contain line→speaker maps and characters own voices.
    characters = data.get("characters", {})
    target_for_alias = {}
    route_candidates = {}
    canonical_ids = {row["speaker_id"] for row in reference["lines"]}
    for key, character in characters.items():
        raw_key = str(key).casefold()
        tokens = [key, character.get("name"), *character.get("aliases", [])]
        direct_matches = {str(token).strip().casefold() for token in tokens if token and str(token).strip().casefold() in canonical_ids}
        if len(direct_matches) != 1:
            continue
        canonical = next(iter(direct_matches))
        target_for_alias[raw_key] = canonical
        if character.get("voice"):
            route_candidates.setdefault(canonical, set()).add(character["voice"])
        for token in tokens:
            if token:
                normalized = str(token).strip().casefold()
                previous = target_for_alias.get(normalized)
                target_for_alias[normalized] = canonical if previous in (None, canonical) else ""
    speaker_aliases = {alias: canonical for alias, canonical in target_for_alias.items() if canonical}
    given = {str(k).strip().casefold(): v for k, v in (aliases or {}).items()}
    for key, character in characters.items():
        tokens = [key, character.get("name"), *character.get("aliases", [])]
        known = next((given[str(t).strip().casefold()] for t in tokens if t and str(t).strip().casefold() in given), None)
        if known:
            speaker_aliases[str(key).casefold()] = known
    narrator_voice = data.get("narrator_voice")
    narrator_id = reference["narrator_speaker_id"]
    if narrator_voice:
        route_candidates.setdefault(narrator_id, set()).add(narrator_voice)
    canonical_routes = {identity: next(iter(voices)) for identity, voices in route_candidates.items() if len(voices) == 1}
    rows = []
    book_tone = data.get("book_tone") or {}
    pov_key = book_tone.get("pov_key") if book_tone.get("point_of_view") == "first" else None
    for chapter_hash, chapter in data.get("chapters", {}).items():
        document = chapter.get("number")
        declared = chapter.get("narrator") if "narrator" in chapter else pov_key
        narrator_character = characters.get(declared) if declared else None
        narrating = declared if narrator_character and not narrator_character.get("voice_picked") else None
        if declared and declared != pov_key:
            chapter_narrator_voice = (narrator_character or {}).get("voice") or narrator_voice
        else:
            chapter_narrator_voice = narrator_voice
        fallback_voice = data.get("dialogue_voice") or chapter_narrator_voice
        for line, speaker in chapter.get("lines", {}).items():
            raw_key = _speaker(speaker)
            character = characters.get(raw_key) if raw_key else None
            if raw_key and raw_key == narrating:
                voice = chapter_narrator_voice
            else:
                voice = (character or {}).get("voice") or fallback_voice
            rows.append({"document": document, "line": line, "speaker": speaker, "voice": voice,
                         "chapter_narrator": declared})
    expected = _reference_lines(reference)
    for key, character in characters.items():
        if str(character.get("name", "")).strip().casefold() == ANONYMOUS_NARRATOR:
            theirs = [expected[k]["speaker_id"] for k in (_key(r["document"], r["line"]) for r in rows
                                                          if r["speaker"] == key) if k in expected]
            if theirs:
                speaker_aliases[str(key).casefold()] = max(set(theirs), key=theirs.count)
    return rows, canonical_routes, speaker_aliases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", help="JSON line predictions or a saved cast JSON")
    parser.add_argument("--reference", default=REFERENCE, type=Path)
    parser.add_argument("--voice-routes", type=Path, help="JSON mapping canonical speaker IDs to this run's selected voices")
    parser.add_argument("--aliases", type=Path, help="JSON mapping the model's names for people to reference speaker IDs")
    parser.add_argument("--summary", action="store_true", help="Print one line instead of the full report")
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    aliases = json.loads(args.aliases.read_text(encoding="utf-8")) if args.aliases else None
    rows, embedded_routes, speaker_aliases = _load_predictions(args.predictions, reference, aliases)
    routes = embedded_routes
    if args.voice_routes:
        routes = json.loads(args.voice_routes.read_text(encoding="utf-8"))
    result = evaluate(reference, rows, routes, speaker_aliases)
    print(summary(result) if args.summary else json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
