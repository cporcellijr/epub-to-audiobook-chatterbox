"""Run the production cast-analysis path privately and save its exact chat exchange log, with a
manifest (<cast>.manifest.json) saying which code, model, input and settings produced the run."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


SRC = Path(__file__).resolve().parents[4]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audiobook_generator.core import cast as cast_store  # noqa: E402
from audiobook_generator.core.cast_analysis import analyse_book  # noqa: E402
from audiobook_generator.core.cast_llm import ChatClient, llm_api_key  # noqa: E402


class RecordingChat:
    def __init__(self, chat, stream):
        self.chat = chat
        self.stream = stream
        self.calls = 0

    def __call__(self, messages):
        reply = self.chat(messages)
        self.calls += 1
        self.stream.write(json.dumps({"call": self.calls, "messages": messages, "reply": reply},
                                     ensure_ascii=False) + "\n")
        self.stream.flush()
        return reply


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_version(src: Path = SRC) -> dict:
    """The commit checked out in `src` (read from .git, so it works where git isn't installed), the
    uncommitted app files when git is available, and a hash of every app source file, which pins
    the exact code either way."""
    tree = hashlib.sha256()
    for path in sorted((src / "audiobook_generator").rglob("*.py")):
        tree.update(path.relative_to(src).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    commit = None
    try:
        head = (src / ".git" / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref = head[5:]
            loose = src / ".git" / ref
            if loose.is_file():
                commit = loose.read_text(encoding="utf-8").strip()
            else:
                packed = (src / ".git" / "packed-refs").read_text(encoding="utf-8").splitlines()
                commit = next((line.split()[0] for line in packed if line.endswith(" " + ref)), None)
        else:
            commit = head
    except OSError:
        pass
    try:
        status = subprocess.run(["git", "-C", str(src), "status", "--porcelain", "--", "audiobook_generator"],
                                capture_output=True, text=True, timeout=20, check=True).stdout
        uncommitted = sorted(line[3:] for line in status.splitlines())
    except (OSError, subprocess.SubprocessError):
        uncommitted = None  # no git here: the tree hash still pins the code
    return {"commit": commit, "uncommitted": uncommitted, "tree_sha256": tree.hexdigest()}


def ollama_model_info(base_url: str, model: str) -> dict:
    """The served model's digest and size from an Ollama server (empty when it isn't one, or can't be
    reached): the same name can be re-pulled as different weights."""
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(root + "/api/tags", timeout=5) as response:
            models = json.loads(response.read()).get("models", [])
    except Exception:  # noqa: BLE001 - provenance is best effort
        return {}
    found = next((m for m in models if m.get("name") == model or m.get("model") == model), None)
    if not found:
        return {}
    details = found.get("details") or {}
    return {"digest": found.get("digest"), "parameter_size": details.get("parameter_size"),
            "quantization": details.get("quantization_level")}


def run(args, *, analyze=analyse_book, chat_factory=ChatClient, api_key=llm_api_key,
        model_info=ollama_model_info):
    input_path = Path(args.input_epub).resolve()
    cast_path = Path(args.cast_file).resolve()
    log_path = Path(args.request_log).resolve()
    manifest_path = cast_path.with_suffix(".manifest.json")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input EPUB does not exist: {input_path}")
    if cast_path == log_path:
        raise ValueError("Cast output and request log must be different files.")
    if cast_path.exists() or log_path.exists() or manifest_path.exists():
        raise FileExistsError("Cast output, request log and manifest paths must not already exist.")
    chapters = list(args.chapter_selection)
    if not chapters or any(number < 1 for number in chapters) or len(set(chapters)) != len(chapters):
        raise ValueError("Chapter selection must contain unique positive chapter numbers.")

    cast_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    chat = chat_factory(args.base_url, args.model, api_key())
    settings = {
        "input_file": str(input_path),
        "chapter_selection": chapters,
        "title_mode": "auto",
        "newline_mode": "double",
        "remove_endnotes": False,
        "remove_reference_numbers": False,
        "search_and_replace_file": None,
        "engine": "chatterbox",
        "voice": None,
        "cast_file": str(cast_path),
        "cast_key": cast_store.cast_key(str(input_path)),
        "auto_pick_voices": False,
    }
    reference = getattr(args, "reference", None)
    manifest = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "code": code_version(),
        "model": {"name": args.model, "base_url": args.base_url, **model_info(args.base_url, args.model)},
        "input": {"file": input_path.name, "sha256": _sha256(input_path)},
        "reference": {"file": Path(reference).name, "sha256": _sha256(reference)} if reference else None,
        "settings": {k: v for k, v in settings.items() if k not in ("input_file", "cast_file")},
        "replayed_from": getattr(args, "replayed_from", None),
    }
    started = time.monotonic()
    cast, recorded = None, None
    try:
        with log_path.open("x", encoding="utf-8") as stream:
            recorded = RecordingChat(chat, stream)
            cast = analyze(settings, chat=recorded)
    finally:
        manifest.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.monotonic() - started, 1),
                        chat_calls=recorded.calls if recorded else 0,
                        status=(cast or {}).get("status", "failed"))
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return cast, recorded.calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-epub", required=True, type=Path)
    parser.add_argument("--cast-file", required=True, type=Path, help="New private cast output; must not exist")
    parser.add_argument("--request-log", required=True, type=Path, help="New JSONL prompt/reply log; must not exist")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--chapter-selection", required=True, nargs="+", type=int)
    parser.add_argument("--reference", type=Path, help="The answer key the run will be scored against (its hash is kept)")
    args = parser.parse_args()
    cast, calls = run(args)
    print(json.dumps({
        "status": cast.get("status"),
        "chapters_done": cast.get("chapters_done"),
        "chapter_selection": cast.get("chapter_selection"),
        "characters": sorted(cast.get("characters", {})),
        "issues": cast.get("issues", []),
        "chat_calls": calls,
        "cast_file": str(args.cast_file.resolve()),
        "request_log": str(args.request_log.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
