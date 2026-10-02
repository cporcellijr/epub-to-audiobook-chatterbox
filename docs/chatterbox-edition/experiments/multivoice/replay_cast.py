"""Replay a saved private run (evaluate_cast.py) on the current code: every request is answered with
the reply the model gave to that same request, so only the code differs. Use it for changes that
should not change what the model is asked (narrator choice, reconciliation, profiles); a change to
prompts or attribution windows asks new questions, which the replay can't answer: judge those over
repeated live runs.

A request the saved run never made gets an error, which the pipeline treats as the model failing
(a profile or narration question is then skipped); the count is printed and kept in the manifest.
Check first that the code the run was made with reproduces its scores."""

import argparse
import collections
import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location("evaluate_cast", Path(__file__).with_name("evaluate_cast.py"))
evaluate_cast = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate_cast)


class ReplayChat:
    """Answers a request with the next saved reply to exactly the same messages."""

    def __init__(self, request_log: Path):
        self.saved = collections.defaultdict(collections.deque)
        for line in Path(request_log).read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            self.saved[self._key(record["messages"])].append(record["reply"])
        self.unanswered = []

    @staticmethod
    def _key(messages) -> str:
        return json.dumps(messages, ensure_ascii=False, sort_keys=True)

    def __call__(self, messages):
        replies = self.saved.get(self._key(messages))
        if replies:
            return replies.popleft()
        self.unanswered.append(messages[-1]["content"][:60])
        raise RuntimeError("not asked in the saved run (this code asks other questions than the code that made it)")


def replay(input_epub: Path, saved_run: Path, out: Path, analyze=evaluate_cast.analyse_book) -> dict:
    old_cast = json.loads((saved_run / "cast.json").read_text(encoding="utf-8"))
    old_manifest = saved_run / "cast.manifest.json"
    model = json.loads(old_manifest.read_text(encoding="utf-8"))["model"] if old_manifest.is_file() else {}
    chat = ReplayChat(saved_run / "requests.jsonl")
    args = SimpleNamespace(input_epub=input_epub, cast_file=out / "cast.json", request_log=out / "requests.jsonl",
                           model=model.get("name", "replay"), base_url=model.get("base_url", "replay"),
                           chapter_selection=old_cast["chapter_selection"], replayed_from=str(saved_run))
    status = "failed"
    try:
        cast, _ = evaluate_cast.run(args, analyze=analyze, chat_factory=lambda *_: chat, api_key=lambda: "",
                                    model_info=lambda *_: {k: v for k, v in model.items() if k not in ("name", "base_url")})
        status = cast.get("status")
    finally:
        manifest_path = out / "cast.manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["unanswered"] = len(chat.unanswered)
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": status, "unanswered": len(chat.unanswered),
            "unanswered_kinds": dict(collections.Counter(text[:30] for text in chat.unanswered))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-epub", required=True, type=Path)
    parser.add_argument("--saved-run", required=True, type=Path, help="Folder with the run's cast.json and requests.jsonl")
    parser.add_argument("--out", required=True, type=Path, help="New folder for the replayed cast; must not hold one")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(args.out / "analysis.log", encoding="utf-8")])
    print(json.dumps(replay(args.input_epub, args.saved_run, args.out), ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
