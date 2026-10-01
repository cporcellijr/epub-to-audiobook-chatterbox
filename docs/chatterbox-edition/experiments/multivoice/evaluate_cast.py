"""Run the production cast-analysis path privately and save its exact chat exchange log."""

import argparse
import json
import os
import sys
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


def run(args, *, analyze=analyse_book, chat_factory=ChatClient, api_key=llm_api_key):
    input_path = Path(args.input_epub).resolve()
    cast_path = Path(args.cast_file).resolve()
    log_path = Path(args.request_log).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input EPUB does not exist: {input_path}")
    if cast_path == log_path:
        raise ValueError("Cast output and request log must be different files.")
    if cast_path.exists() or log_path.exists():
        raise FileExistsError("Cast output and request log paths must not already exist.")
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
    with log_path.open("x", encoding="utf-8") as stream:
        recorded = RecordingChat(chat, stream)
        cast = analyze(settings, chat=recorded)
    return cast, recorded.calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-epub", required=True, type=Path)
    parser.add_argument("--cast-file", required=True, type=Path, help="New private cast output; must not exist")
    parser.add_argument("--request-log", required=True, type=Path, help="New JSONL prompt/reply log; must not exist")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--chapter-selection", required=True, nargs="+", type=int)
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
