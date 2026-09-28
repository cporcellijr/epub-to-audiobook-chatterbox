import json
import multiprocessing
import os
import pickle
import tempfile
import unittest

from audiobook_generator.ui.job_queue import (
    BOOK, CAST, DONE, FAILED, QUEUED, RUNNING, STOPPED, JobQueue, job_kind, run_cast_job, run_job,
)


class FakeProcess:
    instances = []

    def __init__(self, target=None, args=()):
        self.target, self.args = target, args
        self.alive, self.exitcode, self.started, self.terminated = False, None, False, False
        FakeProcess.instances.append(self)

    def start(self):
        self.started, self.alive = True, True

    def is_alive(self):
        return self.alive

    def join(self):
        pass

    def terminate(self):
        self.terminated, self.alive, self.exitcode = True, False, -15

    def finish(self, code=0):
        self.alive, self.exitcode = False, code


def _settings(output_dir, **extra):
    settings = {"input_file": "/library/book.epub", "output_dir": output_dir, "voice": "Elena.wav",
                "output_m4b": True, "skip_existing": False}
    settings.update(extra)
    return settings


class TestJobQueue(unittest.TestCase):

    def setUp(self):
        FakeProcess.instances = []
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "queue.json")
        self.built = []
        self.queue = self._queue()

    def tearDown(self):
        self.tmp.cleanup()

    def _queue(self, **kwargs):
        return JobQueue(self.path, lambda **s: self.built.append(s) or s, lambda: "/app/log.txt",
                        process_factory=FakeProcess, **kwargs)

    def _add(self, title, **extra):
        return self.queue.add(title, _settings(os.path.join(self.tmp.name, title), **extra), 3, 600, "Elena.wav")

    def _statuses(self, queue=None):
        return [(j["title"], j["status"]) for j in (queue or self.queue).jobs()]

    def test_books_run_one_at_a_time_in_order(self):
        self.assertEqual(self._add("A"), 1)
        self.assertEqual(self._add("B"), 2)
        self.queue.tick()
        self.assertEqual(self._statuses(), [("A", RUNNING), ("B", QUEUED)])
        self.assertEqual(self.built[0]["output_dir"], os.path.join(self.tmp.name, "A"))
        self.queue.tick()  # A still generating: nothing new starts
        self.assertEqual(len(FakeProcess.instances), 1)
        FakeProcess.instances[0].finish(0)
        self.queue.tick()
        self.assertEqual(self._statuses(), [("A", DONE), ("B", RUNNING)])

    def test_nonzero_exit_marks_failed(self):
        self._add("A")
        self.queue.tick()
        FakeProcess.instances[0].finish(1)
        self.queue.tick()
        self.assertEqual(self._statuses(), [("A", FAILED)])

    def test_paused_queue_starts_nothing(self):
        self._add("A")
        self.queue.set_paused(True)
        self.queue.tick()
        self.assertEqual(FakeProcess.instances, [])
        self.queue.set_paused(False)
        self.queue.tick()
        self.assertEqual(self._statuses(), [("A", RUNNING)])

    def test_stop_current_stops_and_pauses(self):
        self._add("A")
        self._add("B")
        self.queue.tick()
        self.assertTrue(self.queue.stop_current())
        self.assertTrue(FakeProcess.instances[0].terminated)
        self.assertTrue(self.queue.paused)
        self.queue.tick()
        self.assertEqual(self._statuses(), [("A", STOPPED), ("B", QUEUED)])

    def test_stop_current_after_process_already_finished_marks_done_not_stopped(self):
        # F-28: the process can exit (success) in the window before the next tick() notices, while
        # the queue still thinks it is RUNNING. Stop must not overwrite that as STOPPED.
        self._add("A")
        self.queue.tick()
        FakeProcess.instances[0].finish(0)
        self.assertTrue(self.queue.stop_current())
        self.assertFalse(FakeProcess.instances[0].terminated)
        self.assertEqual(self._statuses(), [("A", DONE)])
        self.assertTrue(self.queue.paused)

    def test_stop_current_after_process_already_failed_marks_failed_not_stopped(self):
        self._add("A")
        self.queue.tick()
        FakeProcess.instances[0].finish(1)
        self.assertTrue(self.queue.stop_current())
        job = self.queue.jobs()[0]
        self.assertEqual(job["status"], FAILED)
        self.assertIn("Retry keeps finished chapters", job["note"])

    def test_retry_requeues_at_the_back_and_keeps_finished_chapters(self):
        self._add("A")
        self._add("B")
        self.queue.tick()
        FakeProcess.instances[0].finish(1)
        self.queue.tick()
        failed = self.queue.jobs()[0]
        self.assertTrue(self.queue.retry(failed["id"]))
        self.assertEqual(self._statuses(), [("B", RUNNING), ("A", QUEUED)])
        self.assertTrue(self.queue.jobs()[1]["settings"]["skip_existing"])
        self.assertFalse(self.queue.retry(self.queue.jobs()[0]["id"]))  # running books can't be retried

    def test_retry_scales_estimate_to_chapters_still_to_do(self):
        # F-42a: 1 of 3 chapters is already on disk, so only 2/3 of the original estimate remains.
        self._add("A")
        self.queue.tick()
        job = self.queue.jobs()[0]
        work = os.path.join(job["settings"]["output_dir"], ".chapters")
        os.makedirs(work)
        open(os.path.join(work, "0001_One.mp3"), "w").close()
        FakeProcess.instances[0].finish(1)
        self.queue.tick()
        self.assertTrue(self.queue.retry(self.queue.jobs()[0]["id"]))
        self.assertAlmostEqual(self.queue.jobs()[0]["estimate_seconds"], 600 * 2 / 3)

    def test_restart_resume_scales_estimate_to_chapters_still_to_do(self):
        self._add("A")
        self.queue.tick()
        job = self.queue.jobs()[0]
        work = os.path.join(job["settings"]["output_dir"], ".chapters")
        os.makedirs(work)
        open(os.path.join(work, "0001_One.mp3"), "w").close()
        # The process is left "alive" (RUNNING), exactly as a real restart would find it.
        restarted = self._queue()
        self.assertAlmostEqual(restarted.jobs()[0]["estimate_seconds"], 600 * 2 / 3)

    def test_remove_waiting_but_not_running(self):
        self._add("A")
        self._add("B")
        self.queue.tick()
        running, waiting = self.queue.jobs()
        self.assertFalse(self.queue.remove(running["id"]))
        self.assertTrue(self.queue.remove(waiting["id"]))
        self.assertEqual(self._statuses(), [("A", RUNNING)])

    def test_clear_finished(self):
        self._add("A")
        self._add("B")
        self.queue.tick()
        FakeProcess.instances[0].finish(0)
        self.queue.tick()
        self.assertEqual(self.queue.clear_finished(), 1)
        self.assertEqual(self._statuses(), [("B", RUNNING)])

    def test_queue_survives_restart_and_resumes_interrupted_book(self):
        self._add("A")
        self._add("B")
        self.queue.tick()
        restarted = self._queue()
        self.assertEqual(self._statuses(restarted), [("A", QUEUED), ("B", QUEUED)])
        resumed = restarted.jobs()[0]
        self.assertTrue(resumed["settings"]["skip_existing"])
        self.assertIn("restart", resumed["note"])

    def test_uploaded_copies_are_deleted_with_the_job(self):
        uploads = os.path.join(self.tmp.name, "uploads")
        os.makedirs(uploads)
        copy = os.path.join(uploads, "upload_x.epub")
        open(copy, "w").close()
        queue = self._queue(uploads_dir=uploads)
        queue.add("U", _settings(self.tmp.name, input_file=copy), 1, 60, "Elena.wav")
        self.assertTrue(queue.remove(queue.jobs()[-1]["id"]))
        self.assertFalse(os.path.exists(copy))

    def test_chapters_done_counts_finished_chapter_files(self):
        self._add("A")
        job = self.queue.jobs()[0]
        work = os.path.join(job["settings"]["output_dir"], ".chapters")
        os.makedirs(work)
        for name in ("0001_One.mp3", "0002_Two.mp3", ".0003_Three.mp3.part"):
            open(os.path.join(work, name), "w").close()
        self.assertEqual(JobQueue.chapters_done(job), 2)

    def test_chapters_done_counts_aac_chapters_too(self):
        # F-29: M4B mode is moving to AAC chapters; both extensions must count.
        self._add("A")
        job = self.queue.jobs()[0]
        work = os.path.join(job["settings"]["output_dir"], ".chapters")
        os.makedirs(work)
        for name in ("0001_One.aac", "0002_Two.mp3"):
            open(os.path.join(work, name), "w").close()
        self.assertEqual(JobQueue.chapters_done(job), 2)

    def test_chapters_done_ignores_files_older_than_this_run(self):
        # F-29: numbered files left over from an earlier, unrelated run in the same folder must not
        # be mistaken for this run's own progress.
        self._add("A")
        self.queue.tick()  # RUNNING: records started_ts
        job = self.queue.jobs()[0]
        work = os.path.join(job["settings"]["output_dir"], ".chapters")
        os.makedirs(work)
        stale = os.path.join(work, "0001_Old.mp3")
        open(stale, "w").close()
        old_time = job["started_ts"] - 3600
        os.utime(stale, (old_time, old_time))
        open(os.path.join(work, "0002_New.mp3"), "w").close()  # written just now, after started_ts
        self.assertEqual(JobQueue.chapters_done(job), 1)

    def test_corrupt_queue_file_starts_empty(self):
        with open(self.path, "w") as f:
            f.write("{not json")
        self.assertEqual(self._queue().jobs(), [])
        with open(self.path) as f:
            self.assertEqual(json.load(f), {"paused": False, "preparing": False, "jobs": []})


def _spawn_target(marker_path: str) -> None:
    """Module-level so the spawn start method can pickle a reference to it (a closure or a method
    cannot be)."""
    with open(marker_path, "w") as f:
        f.write("ran")


class TestSpawnProcessFactory(unittest.TestCase):
    """F-19: job processes are forked from a multithreaded server today; a fork can copy another
    thread's lock mid-hold and hang the child forever. These confirm the spawn context this repo
    switches to actually works here, and that the real target/args are picklable for it."""

    def test_spawn_started_process_actually_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = os.path.join(tmp, "marker.txt")
            process = multiprocessing.get_context("spawn").Process(target=_spawn_target, args=(marker,))
            process.start()
            process.join(timeout=60)
            self.assertEqual(process.exitcode, 0)
            with open(marker) as f:
                self.assertEqual(f.read(), "ran")

    def test_run_job_and_a_real_config_are_picklable(self):
        from audiobook_generator.config.general_config import GeneralConfig
        config = GeneralConfig(None)
        config.input_file, config.output_folder, config.voice_name = "book.epub", "out", "Elena.wav"
        config.chapter_selection = [1, 2, 3]
        pickle.dumps(run_job)  # must be a plain module-level function
        pickle.dumps(config)  # must hold only plain data: no open files, locks or threads


if __name__ == "__main__":
    unittest.main()


class TestCastJobs(unittest.TestCase):
    """Cast analyses are queue jobs of their own kind, and a book never starts while Chatterbox
    reports its model unloaded."""

    def setUp(self):
        FakeProcess.instances = []
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "queue.json")
        self.ready = True
        self.asked = []

        def engine_ready(settings):
            self.asked.append(settings.get("engine", "chatterbox"))
            return self.ready
        self.queue = JobQueue(self.path, lambda **s: s, lambda: "/app/log.txt", process_factory=FakeProcess,
                              engine_ready=engine_ready)

    def tearDown(self):
        self.tmp.cleanup()

    def _cast_settings(self):
        return {"input_file": "/library/book.epub", "chapter_selection": [1, 2], "engine": "chatterbox",
                "cast_key": "k", "cast_file": os.path.join(self.tmp.name, "casts", "k.json")}

    def test_a_cast_job_runs_the_analysis_target_not_a_book(self):
        self.queue.add("Cast: Book", self._cast_settings(), 2, 30, "Elena.wav", kind=CAST)
        self.queue.tick()
        process = FakeProcess.instances[0]
        self.assertIs(process.target, run_cast_job)
        self.assertEqual(process.args[0]["cast_key"], "k")
        self.assertEqual(self.asked, [])  # an analysis needs no loaded Chatterbox
        job = self.queue.jobs()[0]
        self.assertEqual((job["kind"], job["status"]), (CAST, RUNNING))

    def test_a_book_waits_while_chatterbox_is_unloaded_and_starts_once_it_is_back(self):
        self.queue.add("Book", _settings(os.path.join(self.tmp.name, "B")), 3, 600, "Elena.wav")
        self.ready = False
        self.queue.tick()
        self.assertEqual(FakeProcess.instances, [])
        self.assertEqual(self.queue.jobs()[0]["status"], QUEUED)
        self.ready = True
        self.queue.tick()
        self.assertEqual(self.queue.jobs()[0]["status"], RUNNING)
        self.assertIs(FakeProcess.instances[0].target, run_job)

    def test_a_book_queued_after_an_analysis_waits_for_the_analysis_to_finish(self):
        self.queue.add("Cast: Book", self._cast_settings(), 2, 30, "Elena.wav", kind=CAST)
        self.queue.add("Book", _settings(os.path.join(self.tmp.name, "B")), 3, 600, "Elena.wav")
        self.queue.tick()
        self.queue.tick()
        self.assertEqual([j["status"] for j in self.queue.jobs()], [RUNNING, QUEUED])
        FakeProcess.instances[0].finish(0)
        self.queue.tick()
        self.assertEqual([j["status"] for j in self.queue.jobs()], [DONE, RUNNING])

    def test_prepare_runs_casts_ahead_of_held_books_until_start(self):
        self.queue.set_preparing(True)
        self.queue.add("Book 1", _settings(os.path.join(self.tmp.name, "B1")), 3, 600, "Elena.wav")
        self.queue.add("Cast 1", self._cast_settings(), 2, 30, "Elena.wav", kind=CAST)
        self.queue.tick()
        self.assertIs(FakeProcess.instances[0].target, run_cast_job)
        self.queue.add("Book 2", _settings(os.path.join(self.tmp.name, "B2")), 3, 600, "Elena.wav")
        self.queue.add("Cast 2", dict(self._cast_settings(), cast_key="k2"), 2, 30, "Elena.wav", kind=CAST)
        FakeProcess.instances[0].finish()
        self.queue.tick()
        self.assertIs(FakeProcess.instances[1].target, run_cast_job)
        FakeProcess.instances[1].finish()
        self.queue.tick()
        self.assertEqual(len(FakeProcess.instances), 2)  # both audiobook jobs still wait
        self.assertTrue(self.queue.preparing)
        self.assertTrue(self._restarted_queue().preparing)
        self.queue.set_preparing(False)
        self.queue.tick()
        self.assertIs(FakeProcess.instances[2].target, run_job)
        self.assertEqual(self.queue.jobs()[0]["status"], RUNNING)

    def _restarted_queue(self):
        return JobQueue(self.path, lambda **s: s, lambda: "/app/log.txt", process_factory=FakeProcess,
                        engine_ready=lambda s: True)

    def test_prepare_waits_for_the_current_book_before_cast_analysis(self):
        self.queue.add("Book", _settings(os.path.join(self.tmp.name, "B")), 3, 600, "Elena.wav")
        self.queue.tick()
        self.queue.set_preparing(True)
        self.queue.add("Cast", self._cast_settings(), 2, 30, "Elena.wav", kind=CAST)
        self.queue.tick()
        self.assertEqual(len(FakeProcess.instances), 1)
        FakeProcess.instances[0].finish()
        self.queue.tick()
        self.assertIs(FakeProcess.instances[1].target, run_cast_job)

    def test_old_jobs_without_a_kind_are_books(self):
        with open(self.path, "w") as f:
            json.dump({"paused": False, "jobs": [{"id": "old1", "title": "Old", "voice": "Elena.wav", "chapters": 1,
                                                   "estimate_seconds": 10, "status": QUEUED, "added": "", "started": None,
                                                   "started_ts": None, "finished": None, "note": "",
                                                   "settings": _settings(os.path.join(self.tmp.name, "Old"))}]}, f)
        queue = JobQueue(self.path, lambda **s: s, lambda: "/app/log.txt", process_factory=FakeProcess,
                         engine_ready=lambda s: True)
        self.assertEqual(job_kind(queue.jobs()[0]), BOOK)
        queue.tick()
        self.assertIs(FakeProcess.instances[0].target, run_job)

    def test_cast_progress_comes_from_the_cast_file(self):
        from audiobook_generator.core import cast as cast_store
        settings = self._cast_settings()
        self.queue.add("Cast: Book", settings, 2, 30, "Elena.wav", kind=CAST)
        self.queue.tick()
        job = self.queue.jobs()[0]
        self.assertEqual(JobQueue.chapters_done(job), 0)
        cast = cast_store.new_cast("k", "/library/book.epub", "T", "A", "chatterbox", "Elena.wav", [1, 2])
        cast["chapters_done"] = 1
        cast_store.save_cast(settings["cast_file"], cast)
        self.assertEqual(JobQueue.chapters_done(job), 1)

    def test_failed_analysis_is_retried_from_scratch_and_the_note_does_not_mention_chapters(self):
        self.queue.add("Cast: Book", self._cast_settings(), 2, 30, "Elena.wav", kind=CAST)
        self.queue.tick()
        FakeProcess.instances[0].finish(1)
        self.queue.tick()
        job = self.queue.jobs()[0]
        self.assertEqual((job["status"], job["note"]), (FAILED, "failed; see the log"))
        self.assertTrue(self.queue.retry(job["id"]))
        job = self.queue.jobs()[0]
        self.assertEqual((job["status"], job["estimate_seconds"]), (QUEUED, 30))
        self.assertNotIn("skip_existing", job["settings"])

    def test_a_cast_snapshot_in_the_uploads_folder_is_deleted_with_the_book(self):
        uploads = os.path.join(self.tmp.name, "uploads")
        os.makedirs(uploads)
        snapshot = os.path.join(uploads, "upload_cast.json")
        open(snapshot, "w").close()
        queue = JobQueue(self.path, lambda **s: s, lambda: "/app/log.txt", process_factory=FakeProcess,
                         uploads_dir=uploads, engine_ready=lambda s: True)
        queue.add("B", _settings(self.tmp.name, cast_file=snapshot, voice_mode="cast"), 1, 60, "Elena.wav")
        self.assertTrue(queue.remove(queue.jobs()[-1]["id"]))
        self.assertFalse(os.path.exists(snapshot))
