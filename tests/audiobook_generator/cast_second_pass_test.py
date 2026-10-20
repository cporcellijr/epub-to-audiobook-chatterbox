import json
import re
import unittest
from unittest.mock import patch

from audiobook_generator.core.cast_profiles import ChapterText
from audiobook_generator.core.dialogue import chapter_segments, PARAGRAPH_MARK as M
from audiobook_generator.core import cast_second_pass
from audiobook_generator.core import cast_llm

FIXTURE_TEXT = (
    f'Intro{M}'
    f'"Hello," Ada said.{M}'
    f'"Hi"{M}'
    f'"I think...," I said.{M}'
    f'"Untagged line."{M}'
)

def build_fixture():
    paragraphs = chapter_segments(FIXTURE_TEXT)
    characters = {
        "ada": {"name": "Ada", "aliases": [], "lines": 2},
        "ben": {"name": "Ben", "aliases": ["Benjamin"], "lines": 1},
        "me": {"name": "Rory", "aliases": [], "lines": 1},
    }
    lines_map = {1: "ada", 2: "ben", 3: "me", 4: "ada"}
    cast = {
        "characters": characters,
        "chapters": {
            "h1": {"number": 1, "narrator": "me", "lines": {str(k): v for k, v in lines_map.items()}}
        }
    }
    chapters = [ChapterText(1, paragraphs, lines_map)]
    return cast, chapters, characters, lines_map

class FakeChat:
    def __init__(self, mapping=None, fail_times=0, raise_exc=None):
        self.mapping = mapping or {}
        self.fail_times = fail_times
        self.calls = 0
        self.prompts = []
        self.raise_exc = raise_exc

    def __call__(self, messages):
        self.calls += 1
        self.prompts.append(messages)
        if self.raise_exc:
            raise self.raise_exc
        if self.calls <= self.fail_times:
            return "not json"
        content = messages[1]["content"]
        ids = [int(i) for i in re.findall(r"\[#(\d+)\] ", content)]
        speakers = {}
        for i in ids:
            name = self.mapping.get(i)
            if name is None:
                name = self.default_name(i)
            speakers[str(i)] = name
        return json.dumps({"speakers": speakers, "characters": []})

    def default_name(self, line_id):
        # fallback to base cast display name
        cast, _, chars, lines = build_fixture()
        key = lines.get(line_id)
        if key and key in chars:
            return chars[key]["name"]
        return "unknown"

class TestCastSecondPass(unittest.TestCase):
    def test_answer_every_line_with_current_speaker_changes_nothing(self):
        cast, chapters, characters, lines_map = build_fixture()
        chat = FakeChat()
        cast_second_pass.second_pass(cast, chapters, chat, "model")
        self.assertEqual(cast["second_pass"]["changed"], 0)
        self.assertEqual(cast["second_pass"]["done"], 1)
        self.assertEqual(cast["second_pass"]["total"], 1)
        self.assertGreater(cast["second_pass"]["asked"], 0)
        for lid, key in lines_map.items():
            self.assertEqual(cast["chapters"]["h1"]["lines"][str(lid)], key)

    def test_different_name_replaces_line_and_refreshes_counts(self):
        cast, chapters, characters, lines_map = build_fixture()
        chat = FakeChat(mapping={2: "Ada"})
        cast_second_pass.second_pass(cast, chapters, chat, "model")
        self.assertEqual(cast["chapters"]["h1"]["lines"]["2"], "ada")
        self.assertEqual(cast["second_pass"]["changed"], 1)
        # counts refreshed
        self.assertEqual(cast["characters"]["ada"]["lines"], 3)
        self.assertEqual(cast["characters"]["ben"]["lines"], 0)

    def test_unknown_or_missing_name_keeps_first_pass(self):
        cast, chapters, _, lines_map = build_fixture()
        chat = FakeChat(mapping={2: "Zed"})
        cast_second_pass.second_pass(cast, chapters, chat, "model")
        self.assertEqual(cast["chapters"]["h1"]["lines"]["2"], "ben")
        cast2, chapters2, _, _ = build_fixture()
        chat2 = FakeChat(mapping={2: "unknown"})
        cast_second_pass.second_pass(cast2, chapters2, chat2, "model")
        self.assertEqual(cast2["chapters"]["h1"]["lines"]["2"], "ben")

    def test_someone_not_heard_in_the_chapter_never_takes_a_line(self):
        cast, chapters, characters, _ = build_fixture()
        characters["cara"] = {"name": "Cara", "aliases": [], "lines": 5}  # speaks only in other chapters
        cast_second_pass.second_pass(cast, chapters, FakeChat(mapping={2: "Cara"}), "model")
        self.assertEqual(cast["chapters"]["h1"]["lines"]["2"], "ben")
        self.assertEqual(cast["second_pass"]["changed"], 0)

    def test_i_resolves_to_narrator(self):
        cast, chapters, _, lines_map = build_fixture()
        chat = FakeChat(mapping={2: "I"})
        cast_second_pass.second_pass(cast, chapters, chat, "model")
        self.assertEqual(cast["chapters"]["h1"]["lines"]["2"], "me")

    def test_tagged_and_i_said_lines_never_asked(self):
        cast, chapters, _, _ = build_fixture()
        chat = FakeChat()
        cast_second_pass.second_pass(cast, chapters, chat, "model")
        for msgs in chat.prompts:
            content = msgs[1]["content"]
            self.assertNotIn("[#1] ", content)
            self.assertNotIn("[#3] ", content)

    def test_prompt_header_and_narrator_line(self):
        cast, chapters, characters, _ = build_fixture()
        # add duplicate display name
        characters["guy1"] = {"name": "Guy", "aliases": [], "lines": 2}
        characters["guy2"] = {"name": "Guy", "aliases": [], "lines": 1}
        cast["characters"] = characters
        chat = FakeChat()
        cast_second_pass.second_pass(cast, chapters, chat, "model")
        content = chat.prompts[0][1]["content"]
        self.assertTrue(content.startswith(cast_second_pass.FIXED_HEADER))
        # each display name once
        self.assertEqual(content.count("Guy"), 1)
        narrator_line = cast_llm.PROMPTS["narrator"].format(name="Rory", aliases="")
        self.assertIn(narrator_line, content)

    def test_resolver_same_name_local_counts(self):
        characters = {
            "a": {"name": "Sam", "lines": 10},
            "b": {"name": "Sam", "lines": 5},
        }
        local = {"a": 1, "b": 3}
        to_key = cast_second_pass.resolver(characters)
        self.assertEqual(to_key("Sam", None, local), "b")
        local2 = {"a": 0, "b": 0}
        to_key2 = cast_second_pass.resolver(characters)
        self.assertEqual(to_key2("Sam", None, local2), "a")

    def test_resolver_alias_and_first_name(self):
        characters = {
            "ben": {"name": "Ben", "aliases": ["Benjamin"], "lines": 1},
            "annlee": {"name": "Ann Lee", "lines": 2},
            "anncole": {"name": "Ann Cole", "lines": 2},
        }
        to_key = cast_second_pass.resolver(characters)
        self.assertEqual(to_key("Benjamin", None, {}), "ben")
        # first name Ann alone ambiguous if both speak
        local_both = {"annlee": 1, "anncole": 1}
        self.assertIsNone(to_key("Ann", None, local_both))
        local_one = {"annlee": 0, "anncole": 1}
        self.assertEqual(to_key("Ann", None, local_one), "anncole")
        # zero lines never returned
        characters["zero"] = {"name": "Zero", "lines": 0}
        to_key3 = cast_second_pass.resolver(characters)
        self.assertIsNone(to_key3("Zero", None, {}))

    def test_unusable_reply_counts(self):
        cast, chapters, _, _ = build_fixture()
        chat = FakeChat(fail_times=2)
        cast_second_pass.second_pass(cast, chapters, chat, "model")
        self.assertEqual(cast["second_pass"]["unusable"], 2)
        cast2, chapters2, _, _ = build_fixture()
        chat2 = FakeChat(fail_times=1)
        cast_second_pass.second_pass(cast2, chapters2, chat2, "model")
        self.assertEqual(cast2["second_pass"]["unusable"], 1)

    def test_exception_propagates_and_leaves_lines(self):
        cast, chapters, _, lines_map = build_fixture()
        chat = FakeChat(raise_exc=ConnectionError("down"))
        with self.assertRaises(ConnectionError):
            cast_second_pass.second_pass(cast, chapters, chat, "model")
        for lid, key in lines_map.items():
            self.assertEqual(cast["chapters"]["h1"]["lines"][str(lid)], key)

    def test_a_failure_in_a_later_chapter_keeps_the_earlier_chapters_changes_out(self):
        cast, chapters, _, lines_map = build_fixture()
        cast["chapters"]["h2"] = {"number": 2, "narrator": "me", "lines": {str(k): v for k, v in lines_map.items()}}
        chapters.append(ChapterText(2, chapter_segments(FIXTURE_TEXT), dict(lines_map)))

        class FailsInChapterTwo(FakeChat):
            def __call__(self, messages):
                if cast["second_pass"]["done"] == 1:  # chapter 1 finished
                    raise ConnectionError("down")
                return super().__call__(messages)

        with self.assertRaises(ConnectionError):
            cast_second_pass.second_pass(cast, chapters, FailsInChapterTwo(mapping={2: "Ada"}), "model")
        self.assertEqual(cast["chapters"]["h1"]["lines"]["2"], "ben")

    def test_model_available(self):
        class Resp:
            def __init__(self, code, data):
                self.status_code = code
                self._data = data
            def json(self):
                return self._data
        with patch("requests.get") as m:
            m.return_value = Resp(200, {"models": [{"name": "qwen2.5-cast6:7b"}]})
            self.assertTrue(cast_second_pass.model_available("http://host", "qwen2.5-cast6:7b"))
            self.assertFalse(cast_second_pass.model_available("http://host", "missing"))
        with patch("requests.get", side_effect=Exception):
            self.assertFalse(cast_second_pass.model_available("http://host", "x"))
        self.assertFalse(cast_second_pass.model_available("", "x"))
        self.assertFalse(cast_second_pass.model_available("http://host", ""))
        with patch("requests.get") as m:
            m.return_value = Resp(200, {"models": [{"name": "qwen2.5-cast6:7b"}]})
            self.assertTrue(cast_second_pass.model_available("http://host/v1", "qwen2.5-cast6:7b"))
            self.assertEqual(m.call_args[0][0], "http://host/api/tags")

    def test_second_pass_model_env(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(cast_second_pass.second_pass_model(), cast_second_pass.DEFAULT_MODEL)
        with patch.dict("os.environ", {"CAST_SECOND_PASS_MODEL": ""}):
            self.assertEqual(cast_second_pass.second_pass_model(), "")
