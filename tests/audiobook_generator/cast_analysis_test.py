"""The cast analysis job: the whole book pass on a generated EPUB with a scripted LLM, and the
unload -> analyse -> reload order around it (also when the analysis fails)."""
import hashlib
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from ebooklib import epub

from audiobook_generator.core import cast as cast_store
from audiobook_generator.core import chatterbox_control
from audiobook_generator.core.cast_analysis import analyse_book, drop_empty_placeholders, run_cast_analysis

CHAPTERS = [
    ("One", ['Ada Marsh put the lamp down. "You left the gate open," she said.',
             '"I did not," said Tom. He went on writing.', '"Then who did?"']),
    ("Two", ['"The goats are in the beans," said their mother.', '"Yes, Mrs. Marsh," said Tom.']),
]


def _write_epub(path: str, chapters=CHAPTERS) -> None:
    book = epub.EpubBook()
    book.set_identifier("invented-1")
    book.set_title("An Invented Book")
    book.add_author("Nobody Real")
    items = []
    for n, (title, paragraphs) in enumerate(chapters, 1):
        body = "".join(f"<p>{p}</p>" for p in paragraphs)
        item = epub.EpubHtml(title=title, file_name=f"c{n}.xhtml", lang="en")
        item.content = f"<html><body><h1>{title}</h1>{body}</body></html>"
        book.add_item(item)
        items.append(item)
    book.toc = items
    book.spine = ["nav"] + items
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(path, book)


class ScriptedChat:
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def __call__(self, messages):
        self.prompts.append(messages)
        return json.dumps(self.replies.pop(0))


class TestAnalyseBook(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.book = os.path.join(self.tmp.name, "book.epub")
        _write_epub(self.book)
        self.settings = {"input_file": self.book, "chapter_selection": [1, 2], "title_mode": "auto",
                         "newline_mode": "double", "remove_endnotes": False, "remove_reference_numbers": False,
                         "search_and_replace_file": None, "engine": "chatterbox", "voice": "Narrator.wav",
                         "cast_file": os.path.join(self.tmp.name, "casts", "k.json"), "cast_key": "k"}

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_cast_covers_every_chapter_keyed_by_its_text_hash(self):
        chat = ScriptedChat(
            {"speakers": {"1": "Ada Marsh", "2": "Tom", "3": "Ada Marsh"},
             "characters": [{"name": "Ada Marsh", "gender": "female", "age": "adult"},
                            {"name": "Tom", "gender": "male", "age": "child"}]},
            {"speakers": {"1": "Mrs. Marsh", "2": "Tom"},
             "characters": [{"name": "Mrs. Marsh", "gender": "female", "age": "adult"}]},
        )
        cast = analyse_book(self.settings, chat=chat)
        self.assertEqual(cast["status"], "done")
        self.assertEqual((cast["book"]["title"], cast["book"]["author"]), ("An Invented Book", "Nobody Real"))
        self.assertEqual((cast["chapters_done"], cast["chapters_total"]), (2, 2))
        self.assertEqual({k: c["lines"] for k, c in cast["characters"].items()}, {"ada marsh": 2, "tom": 2, "marsh": 1})
        self.assertEqual(cast["characters"]["tom"]["gender"], "male")
        numbers = sorted(ch["number"] for ch in cast["chapters"].values())
        self.assertEqual(numbers, [1, 2])
        # The chapter keys are SHA-1s of the exact chapter text a book job would hash.
        from audiobook_generator.book_parsers.base_book_parser import get_book_parser
        from audiobook_generator.core.cast_analysis import parsing_config
        chapters = [(t, x) for t, x in get_book_parser(parsing_config(self.settings)).get_chapters(" @BRK#") if x.strip()]
        for _, text in chapters:
            self.assertIn(hashlib.sha1(text.encode("utf-8")).hexdigest(), cast["chapters"])
        saved = cast_store.load_cast(self.settings["cast_file"])
        self.assertEqual(saved["chapters"], cast["chapters"])
        self.assertEqual(saved["stats"]["lines"], 5)
        self.assertEqual(len(chat.prompts), 3)  # two windows, then the book-tone request (left unanswered)

    def test_a_failure_is_recorded_in_the_cast_file_and_raised(self):
        def chat(messages):
            raise ConnectionError("LLM down")
        with self.assertRaises(ConnectionError):
            analyse_book(self.settings, chat=chat)
        saved = cast_store.load_cast(self.settings["cast_file"])
        self.assertEqual((saved["status"], saved["error"], saved["chapters_done"]), ("failed", "LLM down", 0))

    def test_re_analysing_keeps_the_voices_already_picked(self):
        first = ScriptedChat(
            {"speakers": {"1": "Ada Marsh", "2": "Tom", "3": "Ada Marsh"},
             "characters": [{"name": "Ada Marsh", "gender": "female", "age": "adult"}]},
            {"speakers": {"1": "Mrs. Marsh", "2": "Tom"}})
        analyse_book(self.settings, chat=first)
        cast = cast_store.load_cast(self.settings["cast_file"])
        cast["characters"]["ada marsh"]["voice"] = "Picked.wav"
        cast_store.save_cast(self.settings["cast_file"], cast)
        again = ScriptedChat({"speakers": {"1": "Ada Marsh", "2": "Tom", "3": "Ada Marsh"}},
                             {"speakers": {"1": "Mrs. Marsh", "2": "Tom"}})
        redone = analyse_book(self.settings, chat=again)
        self.assertEqual(redone["characters"]["ada marsh"]["voice"], "Picked.wav")
        self.assertIsNone(redone["characters"]["tom"].get("voice"))

    def test_breeze_reanalysis_keeps_picked_and_designed_voices(self):
        settings = dict(self.settings, engine="breeze", auto_pick_voices=True)
        replies = ({"speakers": {"1": "Ada Marsh", "2": "Tom", "3": "Ada Marsh"}},
                   {"speakers": {"1": "Mrs. Marsh", "2": "Tom"}})
        analyse_book(settings, chat=ScriptedChat(*replies))
        cast = cast_store.load_cast(settings["cast_file"])
        cast["engine"] = "breeze"  # Existing casts may store the UI engine name.
        cast["characters"]["ada marsh"].update(voice="Picked.wav", voice_picked=True)
        cast["characters"]["tom"].update(voice="Designed.wav", voice_design={"status": "done"})
        cast_store.save_cast(settings["cast_file"], cast)

        redone = analyse_book(settings, chat=ScriptedChat(*replies))

        self.assertEqual(redone["characters"]["ada marsh"]["voice"], "Picked.wav")
        self.assertEqual(redone["characters"]["tom"]["voice"], "Designed.wav")
        self.assertEqual(redone["characters"]["tom"]["voice_design"], {"status": "done"})

    def test_auto_pick_keeps_only_the_voices_the_owner_saved(self):
        replies = ({"speakers": {"1": "Ada Marsh", "2": "Tom", "3": "Ada Marsh"}}, {"speakers": {"1": "Mrs. Marsh", "2": "Tom"}})
        analyse_book(self.settings, chat=ScriptedChat(*replies))
        cast = cast_store.load_cast(self.settings["cast_file"])
        cast["characters"]["ada marsh"].update(voice="Picked.wav", voice_picked=True)
        cast["characters"]["tom"]["voice"] = "Suggested.wav"
        cast_store.save_cast(self.settings["cast_file"], cast)
        redone = analyse_book(dict(self.settings, auto_pick_voices=True), chat=ScriptedChat(*replies))
        self.assertEqual(redone["characters"]["ada marsh"]["voice"], "Picked.wav")
        self.assertIsNone(redone["characters"]["tom"].get("voice"))  # left for a fresh suggestion

    def test_only_selected_chapters_are_analysed(self):
        self.settings["chapter_selection"] = [2]
        cast = analyse_book(self.settings, chat=ScriptedChat({"speakers": {"1": "Mother", "2": "Tom"}}))
        self.assertEqual([ch["number"] for ch in cast["chapters"].values()], [2])

    @patch("audiobook_generator.core.cast_profiles.PROFILE_MIN_LINES", 2)
    def test_a_gender_contradicted_line_is_left_unattributed_and_noted_but_profiles_still_run(self):
        _write_epub(self.book, [("One", ['"Where is the key?" asked Ivy Lark.', '"Under the mat," she said.',
                                           '"On the hook," he said.', '"Thank you," said Ivy Lark.'])])
        self.settings["chapter_selection"] = [1]
        wrong = {"speakers": {"1": "Ivy Lark", "2": "Ivy Lark", "3": "Ivy Lark", "4": "Ivy Lark"},
                 "characters": [{"name": "Ivy Lark", "gender": "female", "age": "adult"}]}
        tone = {"point_of_view": "third", "tone": "quiet", "narrator_gender": "female"}
        profile = {"role": "protagonist", "gender": "female", "age": "adult", "description": "Looks for a key.",
                   "relationships": "", "voice": "calm"}
        chat = ScriptedChat(wrong, {"speakers": {"3": "Ivy Lark"}}, tone, profile)
        cast = analyse_book(self.settings, chat=chat)
        self.assertEqual(cast["status"], "done")
        chapter = next(iter(cast["chapters"].values()))
        self.assertEqual(chapter["lines"], {"1": "ivy lark", "2": "ivy lark", "3": None, "4": "ivy lark"})
        self.assertEqual(cast["issues"], [{"chapter": 1, "line": 3, "reason": "pronoun gender", "was": "ivy lark"}])
        self.assertEqual((chapter["unknown"], cast["stats"]["unknown_lines"]), (1, 1))
        self.assertEqual(cast["characters"]["ivy lark"]["lines"], 3)
        self.assertEqual(cast["characters"]["ivy lark"]["profile"]["description"], "Looks for a key.")
        profile_prompt = chat.prompts[-1][1]["content"]
        self.assertIn("Under the mat", profile_prompt)
        self.assertNotIn("On the hook", profile_prompt)
        self.assertEqual(cast_store.load_cast(self.settings["cast_file"])["issues"], cast["issues"])

    @patch("audiobook_generator.core.cast_profiles.PROFILE_MIN_LINES", 2)
    def test_profiles_are_written_after_the_chapters_and_an_llm_error_there_still_finishes(self):
        attribution = [{"speakers": {"1": "Ada Marsh", "2": "Tom", "3": "Ada Marsh"},
                        "characters": [{"name": "Tom", "gender": "unknown", "age": "unknown"}]},
                       {"speakers": {"1": "Mrs. Marsh", "2": "Tom"}}]
        profile = {"role": "protagonist", "gender": "female", "age": "adult", "description": "Runs the farm.",
                   "relationships": "Tom's sister", "voice": "firm young woman"}
        tone = {"point_of_view": "third", "tone": "quiet", "narrator_gender": "female"}
        chat = ScriptedChat(*attribution, tone, profile, dict(profile, gender="male", description="Her brother."))
        cast = analyse_book(self.settings, chat=chat)
        self.assertEqual(cast["status"], "done")
        self.assertEqual(len(chat.prompts), 5)  # two windows, book tone, two profiles
        self.assertEqual(cast["characters"]["ada marsh"]["profile"]["voice"], "firm young woman")
        self.assertEqual(cast["characters"]["tom"]["gender"], "male")
        self.assertEqual(cast["characters"]["marsh"]["profile"], {"first_line": {"chapter": 2, "text": '"The goats are in the beans,"'}})
        self.assertEqual(cast_store.load_cast(self.settings["cast_file"])["characters"], cast["characters"])

        class FailingProfiles(ScriptedChat):
            def __call__(self, messages):
                if not self.replies:
                    raise ConnectionError("LLM down")
                return super().__call__(messages)
        cast = analyse_book(self.settings, chat=FailingProfiles(*attribution))
        self.assertEqual((cast["status"], cast["profile_error"], cast["profiles_done"]), ("done", "LLM down", 0))

    @patch("audiobook_generator.core.cast_profiles.PROFILE_MIN_LINES", 2)
    def test_profiles_and_unknown_counts_use_the_resolved_narrators_lines(self):
        _write_epub(self.book, [("One", ["I walked to the harbour, my hands shaking in the cold. " * 20,
                                        '"Sally, wait," said Tom.', '"I am coming," I respond.',
                                        '"We will take the ferry," I say.', '"Meet me there," I read aloud.'])])
        self.settings["chapter_selection"] = [1]
        tone = {"point_of_view": "first", "pov_character": "Sally", "tone": "quiet",
                "narrator_gender": "female"}
        # Explicit first-person tags are attributed deterministically; only tone confirmation and
        # the chapter POV checks need scripted replies. A tone guess alone cannot name the narrator.
        chat = ScriptedChat(tone, tone, {"role": "protagonist", "gender": "female", "age": "adult",
                                         "description": "Tells the story.", "relationships": "", "voice": "calm"})

        cast = analyse_book(self.settings, chat=chat)

        self.assertEqual(cast["status"], "done")
        chapter = next(iter(cast["chapters"].values()))
        self.assertEqual(chapter["lines"], {"1": "tom", "2": "narrator", "3": "narrator", "4": "narrator"})
        self.assertEqual((chapter["unknown"], cast["stats"]["unknown_lines"]), (0, 0))
        self.assertEqual((cast["characters"]["narrator"]["lines"], cast["characters"]["tom"]["lines"]), (3, 1))
        self.assertEqual(cast["issues"], [])
        self.assertEqual(cast["book_tone"]["pov_key"], "narrator")  # the unnamed "I" is the book's narrator
        self.assertEqual(cast["profiles_total"], 1)
        self.assertEqual(cast_store.load_cast(self.settings["cast_file"])["stats"]["unknown_lines"], 0)

    @patch("audiobook_generator.core.cast_llm.ASK_LLM_FOR_MOODS", True)
    def test_moods_are_saved_per_chapter(self):
        chat = ScriptedChat(
            {"speakers": {"1": "Ada Marsh", "2": "Tom", "3": "Ada Marsh"}, "moods": {"1": "excited"}},
            {"speakers": {"1": "Mrs. Marsh", "2": "Tom"}},
        )
        cast = analyse_book(self.settings, chat=chat)
        chapter_one = next(ch for ch in cast["chapters"].values() if ch["number"] == 1)
        # Line 1's rule mood is normal (no cues), so the LLM's "excited" flows through; lines 2/3
        # were never asked (2 is tagged "said Tom", 3 has no cue) so they default to normal.
        self.assertEqual(chapter_one["moods"], {"1": "excited", "2": "normal", "3": "normal"})
        saved = cast_store.load_cast(self.settings["cast_file"])
        self.assertEqual(saved["chapters"], cast["chapters"])


class TestDropEmptyPlaceholders(unittest.TestCase):

    def test_only_unspoken_unnamed_placeholders_nobody_picked_are_dropped(self):
        cast = {"book_tone": {"pov_key": "narrator 3"}, "characters": {
            "narrator": {"name": "The Narrator", "lines": 0},
            "narrator 2": {"name": "The Narrator", "lines": 4},
            "narrator 3": {"name": "The Narrator", "lines": 0},
            "woman": {"name": "unnamed woman", "lines": 0, "reference_scope": "chapter"},
            "guy": {"name": "one of the guys", "lines": 0, "reference_scope": "chapter", "voice_picked": True},
            "ada": {"name": "Ada", "lines": 0},
        }}
        drop_empty_placeholders(cast)
        self.assertEqual(sorted(cast["characters"]), ["ada", "guy", "narrator 2", "narrator 3"])


class TestRunCastAnalysisOrder(unittest.TestCase):
    """The queue's process target: Chatterbox is unloaded before the LLM pass and reloaded after,
    whether or not the pass succeeded."""

    def _run(self, analysis, unload_allowed=True, unload_ok=True):
        events = []

        def analyse(settings):
            events.append("analyse")
            return analysis()
        exits = []
        with tempfile.TemporaryDirectory() as tmp, \
                patch("audiobook_generator.core.cast_analysis.setup_logging"):  # keep the failure's traceback out of the test output
            run_cast_analysis({}, os.path.join(tmp, "log.txt"), unload_allowed=lambda: unload_allowed,
                              unload=lambda: events.append("unload") or unload_ok,
                              reload=lambda: events.append("reload") or True, analyse=analyse, exit=exits.append)
        return events, exits

    def test_unload_then_analyse_then_reload_and_exit_0(self):
        events, exits = self._run(lambda: {"status": "done"})
        self.assertEqual(events, ["unload", "analyse", "reload"])
        self.assertEqual(exits, [0])

    def test_reload_happens_even_when_the_analysis_fails_and_exit_is_1(self):
        def boom():
            raise RuntimeError("LLM exploded")
        events, exits = self._run(boom)
        self.assertEqual(events, ["unload", "analyse", "reload"])
        self.assertEqual(exits, [1])

    def test_no_reload_when_the_setting_kept_chatterbox_loaded(self):
        events, exits = self._run(lambda: {}, unload_allowed=False)
        self.assertEqual(events, ["analyse"])
        self.assertEqual(exits, [0])

    def test_no_reload_when_the_unload_itself_failed(self):
        events, _ = self._run(lambda: {}, unload_ok=False)
        self.assertEqual(events, ["unload", "analyse"])


class TestChatterboxControl(unittest.TestCase):

    def test_unload_setting_defaults_on(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(chatterbox_control.unload_enabled())
        for value in ("off", "0", "false", "No"):
            with patch.dict(os.environ, {"LLM_UNLOAD_CHATTERBOX": value}):
                self.assertFalse(chatterbox_control.unload_enabled())
        with patch.dict(os.environ, {"LLM_UNLOAD_CHATTERBOX": "on"}):
            self.assertTrue(chatterbox_control.unload_enabled())

    def test_model_loaded_reads_model_info_and_none_when_unreachable(self):
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "http://cb:8004/v1"}):
            with patch.object(chatterbox_control, "_get_json", return_value={"loaded": True}):
                self.assertTrue(chatterbox_control.model_loaded())
            with patch.object(chatterbox_control, "_get_json", return_value={"loaded": False}):
                self.assertFalse(chatterbox_control.model_loaded())
            with patch.object(chatterbox_control, "_get_json", side_effect=OSError("down")):
                self.assertIsNone(chatterbox_control.model_loaded())
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(chatterbox_control.model_loaded())

    def test_reload_posts_restart_then_waits_for_loaded(self):
        answers = iter([{"loaded": False}, {"loaded": False}, {"loaded": True}])
        posted = []
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "http://cb:8004/v1"}), \
                patch.object(chatterbox_control, "_post", side_effect=lambda path, timeout: posted.append(path)), \
                patch.object(chatterbox_control, "_get_json", side_effect=lambda path, timeout: next(answers)), \
                patch.object(chatterbox_control.time, "sleep"):
            self.assertTrue(chatterbox_control.reload())
        self.assertEqual(posted, ["/restart_server"])

    def test_wait_gives_up_after_the_timeout(self):
        clock = iter(range(0, 10_000, 100))
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "http://cb:8004/v1"}), \
                patch.object(chatterbox_control, "_get_json", return_value={"loaded": False}):
            self.assertFalse(chatterbox_control.wait_until_loaded(timeout=300, sleep=lambda s: None, clock=lambda: next(clock)))

    def test_unload_posts_and_reports_failure_without_raising(self):
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "http://cb:8004/v1"}):
            with patch.object(chatterbox_control, "_post", return_value=b"{}") as post:
                self.assertTrue(chatterbox_control.unload())
                self.assertEqual(post.call_args.args[0], "/api/unload")
            with patch.object(chatterbox_control, "_post", side_effect=OSError("down")):
                self.assertFalse(chatterbox_control.unload())

    def test_ready_for_book_is_false_while_unloaded_and_kicks_off_one_reload(self):
        started = []
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "http://cb:8004/v1"}), \
                patch.object(chatterbox_control, "model_loaded", return_value=False), \
                patch.object(chatterbox_control.threading, "Thread") as thread:
            thread.return_value.is_alive.return_value = True
            thread.return_value.start.side_effect = lambda: started.append(1)
            chatterbox_control._reload_thread = None
            self.assertFalse(chatterbox_control.ready_for_book())
            self.assertFalse(chatterbox_control.ready_for_book())
        chatterbox_control._reload_thread = None
        self.assertEqual(started, [1])
        with patch.object(chatterbox_control, "model_loaded", return_value=True):
            self.assertTrue(chatterbox_control.ready_for_book())
        with patch.object(chatterbox_control, "model_loaded", return_value=None):
            self.assertTrue(chatterbox_control.ready_for_book())
