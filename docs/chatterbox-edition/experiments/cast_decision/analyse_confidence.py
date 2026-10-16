"""Experiment B (WORKLOG §79): does the model's own probability flag the wrong lines?

Joins per-answer confidences from decisions.jsonl with a private labels file
and compares against an agreement baseline.
"""

import json
import math
import argparse
from pathlib import Path
from collections import defaultdict


THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999)


def load_confidences(path: Path) -> dict[tuple[int, int], dict]:
    """Read decisions.jsonl and return {(chapter, line): {first, last, answers}}."""
    confidences = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if not record.get("ids") or record.get("chapter") is None:
                continue
            chapter = int(record["chapter"])
            for id_str, answer in record.get("answers", {}).items():
                key = (chapter, int(id_str))
                if key not in confidences:
                    confidences[key] = {"first": answer, "last": answer, "answers": 1}
                else:
                    confidences[key]["last"] = answer
                    confidences[key]["answers"] += 1
    return confidences


def confidence(entry: dict, field: str, which: str) -> float:
    """Extract confidence value from entry. Null p_decision counts as 0.0."""
    val = entry[which].get(field)
    if val is None:
        return 0.0
    return float(val)


def auroc(scores_wrong: list[float], scores_right: list[float]) -> float | None:
    """Mann-Whitney U with average ranks (ties count as half). Returns P(wrong < right)."""
    if not scores_wrong or not scores_right:
        return None
    n_wrong = len(scores_wrong)
    n_right = len(scores_right)
    combined = [(s, 1) for s in scores_wrong] + [(s, 0) for s in scores_right]
    combined.sort(key=lambda x: x[0])
    rank_sum = 0.0
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        block_wrong = sum(1 for k in range(i, j) if combined[k][1] == 1)
        avg_rank = (i + 1 + j) / 2.0
        rank_sum += block_wrong * avg_rank
        i = j
    u = rank_sum - n_wrong * (n_wrong + 1) / 2.0  # n_gt + 0.5 * tie_pairs
    return (n_wrong * n_right - u) / (n_wrong * n_right)


def threshold_rows(pairs: list[tuple[float, bool]], thresholds: tuple[float, ...]) -> list[dict]:
    """Generate threshold analysis rows."""
    rows = []
    for t in thresholds:
        flagged = [(c, w) for c, w in pairs if c < t]
        wrong_caught = sum(1 for _, w in flagged if w)
        wrong_total = sum(1 for _, w in pairs if w)
        right_flagged = sum(1 for _, w in flagged if not w)
        right_total = sum(1 for _, w in pairs if not w)
        precision = wrong_caught / len(flagged) if flagged else None
        rows.append({
            "threshold": t,
            "flagged": len(flagged),
            "wrong_caught": wrong_caught,
            "wrong_total": wrong_total,
            "right_flagged": right_flagged,
            "right_total": right_total,
            "precision": precision,
        })
    return rows


def matched(pairs_with_flag: list[tuple[float, bool, bool]]) -> dict:
    """Compare baseline flagging vs confidence at same flag budget."""
    baseline_flagged = [(c, w) for c, w, b in pairs_with_flag if b]
    n_baseline = len(baseline_flagged)
    wrong_caught_baseline = sum(1 for _, w in baseline_flagged if w)
    right_flagged_baseline = sum(1 for _, w in baseline_flagged if not w)

    sorted_pairs = sorted(pairs_with_flag, key=lambda x: x[0])
    confidence_flagged = sorted_pairs[:n_baseline]
    wrong_caught_conf = sum(1 for _, w, _ in confidence_flagged if w)
    right_flagged_conf = sum(1 for _, w, _ in confidence_flagged if not w)

    return {
        "baseline": {
            "flagged": n_baseline,
            "wrong_caught": wrong_caught_baseline,
            "right_flagged": right_flagged_baseline,
        },
        "confidence": {
            "flagged": n_baseline,
            "wrong_caught": wrong_caught_conf,
            "right_flagged": right_flagged_conf,
        },
        "lines": len(pairs_with_flag),
    }


def analyse(run_dir: Path, labels: dict, agree: dict | None, field: str, which: str) -> dict:
    """Analyze all groups and return aggregated results."""
    groups = defaultdict(lambda: {
        "lines": 0, "wrong": 0, "wrong_answered": 0, "right_answered": 0,
        "answered_pairs": [], "matched_pairs": []
    })

    for group_key, label_dict in labels.items():
        kind, book = group_key.split("/", 1)
        decisions_path = run_dir / kind / book / "decisions.jsonl"
        if not decisions_path.exists():
            raise FileNotFoundError(f"Missing decisions.jsonl: {decisions_path}")
        confidences = load_confidences(decisions_path)

        agree_dict = agree.get(group_key, {}) if agree else {}

        for label_key, is_wrong in label_dict.items():
            chapter_str, line_str = label_key.split(":")
            key = (int(chapter_str), int(line_str))
            entry = confidences.get(key)
            answered = entry is not None
            if answered:
                conf = confidence(entry, field, which)
                groups[group_key]["answered_pairs"].append((conf, is_wrong))
                if label_key in agree_dict:
                    groups[group_key]["matched_pairs"].append((conf, is_wrong, agree_dict[label_key]))
            groups[group_key]["lines"] += 1
            if is_wrong:
                groups[group_key]["wrong"] += 1
                if answered:
                    groups[group_key]["wrong_answered"] += 1
            elif answered:
                groups[group_key]["right_answered"] += 1

    aggregates = {}

    masked_groups = [k for k in groups if k.startswith("masked/")]
    keys_groups = [k for k in groups if k.startswith("keys/")]

    def build_aggregate(name: str, group_keys: list[str]) -> dict:
        all_pairs = []
        all_matched = []
        total = {"lines": 0, "wrong": 0, "wrong_answered": 0, "right_answered": 0}
        for k in group_keys:
            g = groups[k]
            all_pairs.extend(g["answered_pairs"])
            all_matched.extend(g["matched_pairs"])
            for key in total:
                total[key] += g[key]
        scores_wrong = [c for c, w in all_pairs if w]
        scores_right = [c for c, w in all_pairs if not w]
        result = {
            "name": name,
            **total,
            "auroc": auroc(scores_wrong, scores_right),
            "thresholds": threshold_rows(all_pairs, THRESHOLDS),
        }
        if all_matched:
            result["matched"] = matched(all_matched)
        return result

    aggregates["masked"] = build_aggregate("masked", masked_groups)
    aggregates["keys"] = build_aggregate("keys", keys_groups)
    for k in keys_groups:
        g = groups[k]
        scores_wrong = [c for c, w in g["answered_pairs"] if w]
        scores_right = [c for c, w in g["answered_pairs"] if not w]
        book_name = k.split("/", 1)[1]
        result = {
            "name": book_name,
            "lines": g["lines"],
            "wrong": g["wrong"],
            "wrong_answered": g["wrong_answered"],
            "right_answered": g["right_answered"],
            "auroc": auroc(scores_wrong, scores_right),
            "thresholds": threshold_rows(g["answered_pairs"], THRESHOLDS),
        }
        if g["matched_pairs"]:
            result["matched"] = matched(g["matched_pairs"])
        aggregates[book_name] = result

    return aggregates


def print_report(aggregates: dict):
    """Print compact plain-text report."""
    for name in ["masked", "keys"]:
        if name not in aggregates:
            continue
        a = aggregates[name]
        auroc_str = f"{a['auroc']:.3f}" if a['auroc'] is not None else "None"
        print(f"{a['name']}: lines {a['lines']}, wrong {a['wrong']} (answered {a['wrong_answered']}), right answered {a['right_answered']}, AUROC {auroc_str}")
        print("  threshold  flagged  wrong caught/total  right flagged/total  precision")
        for row in a["thresholds"]:
            prec = f"{row['precision']:.3f}" if row['precision'] is not None else "None"
            print(f"  {row['threshold']:.3f}        {row['flagged']:6d}   {row['wrong_caught']:3d}/{row['wrong_total']:3d}              {row['right_flagged']:3d}/{row['right_total']:3d}             {prec}")
        if "matched" in a:
            m = a["matched"]
            print(f"  agreement baseline flags {m['baseline']['flagged']}: wrong caught {m['baseline']['wrong_caught']}, right flagged {m['baseline']['right_flagged']} | confidence at the same budget: wrong caught {m['confidence']['wrong_caught']}, right flagged {m['confidence']['right_flagged']} (on {m['lines']} lines)")

    for name, a in aggregates.items():
        if name in ("masked", "keys"):
            continue
        auroc_str = f"{a['auroc']:.3f}" if a['auroc'] is not None else "None"
        print(f"{a['name']}: lines {a['lines']}, wrong {a['wrong']} (answered {a['wrong_answered']}), right answered {a['right_answered']}, AUROC {auroc_str}")
        print("  threshold  flagged  wrong caught/total  right flagged/total  precision")
        for row in a["thresholds"]:
            prec = f"{row['precision']:.3f}" if row['precision'] is not None else "None"
            print(f"  {row['threshold']:.3f}        {row['flagged']:6d}   {row['wrong_caught']:3d}/{row['wrong_total']:3d}              {row['right_flagged']:3d}/{row['right_total']:3d}             {prec}")
        if "matched" in a:
            m = a["matched"]
            print(f"  agreement baseline flags {m['baseline']['flagged']}: wrong caught {m['baseline']['wrong_caught']}, right flagged {m['baseline']['right_flagged']} | confidence at the same budget: wrong caught {m['confidence']['wrong_caught']}, right flagged {m['confidence']['right_flagged']} (on {m['lines']} lines)")


def main():
    parser = argparse.ArgumentParser(description="Analyze model confidence vs labeled wrong lines")
    parser.add_argument("--run", type=Path, required=True, help="Arm folder with masked/ and keys/ subdirs")
    parser.add_argument("--labels", type=Path, required=True, help="JSON labels file")
    parser.add_argument("--agree", type=Path, help="Optional agreement baseline JSON")
    parser.add_argument("--field", choices=["p_seq", "p_decision", "p_first"], default="p_seq")
    parser.add_argument("--which", choices=["last", "first"], default="last")
    parser.add_argument("--json", type=Path, help="Optional output JSON path")
    args = parser.parse_args()

    with args.labels.open(encoding="utf-8") as f:
        labels = json.load(f)

    agree = None
    if args.agree:
        with args.agree.open(encoding="utf-8") as f:
            agree = json.load(f)

    aggregates = analyse(args.run, labels, agree, args.field, args.which)

    print_report(aggregates)

    if args.json:
        with args.json.open("w", encoding="utf-8") as f:
            json.dump(aggregates, f, indent=2, default=str)


if __name__ == "__main__":
    main()