"""The cast analysis job: parse the book, attribute every dialogue line with the local LLM, write
the main characters' profiles (core.cast_profiles), save the cast. Runs from the queue in its own
process (run_cast_analysis), never alongside a book.
"""
import logging
import sys
import time
from typing import Callable, Optional

from audiobook_generator.book_parsers.base_book_parser import get_book_parser
from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.core import cast as cast_store
from audiobook_generator.core import chatterbox_control, engine_gpu
from audiobook_generator.core.cast_llm import ChatClient, Chat, Roster, attribute_chapter, llm_api_key, llm_base_url, \
    llm_model
from audiobook_generator.core.cast_profiles import ChapterText, describe_book, profile_cast
from audiobook_generator.core.dialogue import PARAGRAPH_MARK, chapter_segments
from audiobook_generator.utils.log_handler import setup_logging

logger = logging.getLogger(__name__)


def parsing_config(settings: dict) -> GeneralConfig:
    """The parser settings a book job would use, so chapter numbers and text hashes agree."""
    config = GeneralConfig(None)
    config.input_file = settings["input_file"]
    config.title_mode = settings.get("title_mode", "auto")
    config.newline_mode = settings.get("newline_mode", "double")
    config.remove_endnotes = bool(settings.get("remove_endnotes", False))
    config.remove_reference_numbers = bool(settings.get("remove_reference_numbers", False))
    config.search_and_replace_file = settings.get("search_and_replace_file")
    return config


def analyse_book(settings: dict, chat: Optional[Chat] = None, log: logging.Logger = logger) -> dict:
    """Analyse the selected chapters of one book and write the cast to settings["cast_file"].

    settings: input_file, chapter_selection, the parser options (title_mode, newline_mode,
    remove_endnotes, remove_reference_numbers, search_and_replace_file), engine, voice (the
    narrator), cast_file. The cast file is rewritten after every chapter, so the UI can show
    progress and a crash keeps what was done; the final write marks it done (or failed). Character
    profiles are written after the last chapter; a profile problem never fails the analysis.
    """
    if chat is None:
        chat = ChatClient(llm_base_url(), llm_model(), llm_api_key())
    parser = get_book_parser(parsing_config(settings))
    chapters = [(title, text) for title, text in parser.get_chapters(f" {PARAGRAPH_MARK}") if text.strip()]
    selection = [n for n in settings.get("chapter_selection") or range(1, len(chapters) + 1) if 0 < n <= len(chapters)]
    key = settings.get("cast_key") or cast_store.cast_key(settings["input_file"])
    path = settings["cast_file"]
    previous = cast_store.load_cast(path)  # an earlier analysis of this book: its voice picks carry over
    engine = cast_store.voice_library(settings.get("engine", "chatterbox"))
    if previous and cast_store.voice_library(previous.get("engine", "")) != engine:
        previous = None  # another engine's voices can't be used
    cast = cast_store.new_cast(key, settings["input_file"], parser.get_book_title(), parser.get_book_author(),
                               engine, settings.get("voice"), selection)
    cast_store.save_cast(path, cast)
    roster = Roster()
    stats = cast["stats"]
    analysed = []  # the chapters' text and attributions, for the profiles
    started = time.monotonic()
    try:
        for done, number in enumerate(selection, 1):
            title, text = chapters[number - 1]
            paragraphs = chapter_segments(text)
            line_count = sum(1 for p in paragraphs for s in p if s.kind == "dialogue")
            log.info(f"Cast: chapter {number} ({title}): {line_count} dialogue lines in {len(paragraphs)} paragraphs")
            lines, moods = attribute_chapter(paragraphs, roster, chat, stats, log, label=f" ch{number}")
            analysed.append(ChapterText(number, paragraphs, lines))
            cast["chapters"][cast_store.text_hash(text)] = {
                "number": number, "title": title,
                "lines": {str(line_id): speaker for line_id, speaker in sorted(lines.items())},
                "moods": {str(line_id): mood for line_id, mood in sorted(moods.items())},
                "unknown": sum(1 for speaker in lines.values() if speaker is None),
            }
            cast["characters"] = dict(roster.characters)
            cast_store.carry_voice_choices(previous, cast["characters"],
                                           picked_only=bool(settings.get("auto_pick_voices")))
            cast["chapters_done"] = done
            cast_store.save_cast(path, cast)
            log.info(f"Cast: {done}/{len(selection)} chapters analysed, {len(roster.characters)} characters, "
                     f"{stats['unknown_lines']} of {stats['lines']} lines unknown so far, "
                     f"{stats.get('review_lines', 0)} asked again ({stats.get('review_changed', 0)} changed)")
        profile_cast(cast, analysed, chat, log, save=lambda: cast_store.save_cast(path, cast))
        describe_book(cast, analysed, chat, log)
        cast["status"] = cast_store.STATUS_DONE
    except Exception as e:
        cast["status"], cast["error"] = cast_store.STATUS_FAILED, str(e)
        raise
    finally:
        cast["finished"] = cast_store._now()
        stats["seconds"] = round(time.monotonic() - started, 2)
        cast_store.save_cast(path, cast)
    return cast


def run_cast_analysis(settings: dict, log_file: str, unload_allowed: Callable[[], bool] = chatterbox_control.unload_enabled,
                      unload: Callable[[], bool] = chatterbox_control.unload,
                      reload: Callable[[], bool] = chatterbox_control.reload,
                      unload_breeze: Callable[[], bool] = engine_gpu.unload_breeze_if_loaded,
                      analyse: Callable[..., dict] = analyse_book, exit: Callable[[int], None] = sys.exit) -> None:
    """Process target for the queue: unload Chatterbox and Breeze (when allowed), analyse, then
    reload Chatterbox and wait for the model whether or not the analysis succeeded (Breeze loads
    again when a Breeze book starts, core.engine_gpu). Exit code 0 only on success.

    The Chatterbox and analysis steps are injectable for the queue tests; the queue itself passes
    only (settings, log_file).
    """
    setup_logging(settings.get("log_level", "INFO"), log_file)
    unloaded = False
    succeeded = False
    try:
        if unload_allowed():
            unloaded = unload()
            unload_breeze()
        else:
            logger.info("Cast: LLM_UNLOAD_CHATTERBOX is off; Chatterbox stays loaded during the analysis")
        analyse(settings)
        succeeded = True
    except Exception as e:
        logger.exception(f"Cast analysis failed: {e}")
    finally:
        if unloaded:
            logger.info("Cast: reloading Chatterbox")
            reload()
    exit(0 if succeeded else 1)
