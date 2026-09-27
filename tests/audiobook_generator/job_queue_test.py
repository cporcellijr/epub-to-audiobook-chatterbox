import json
import os
import tempfile
import unittest

from audiobook_generator.ui.job_queue import DONE, FAILED, QUEUED, RUNNING, STOPPED, JobQueue


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

    def test_corrupt_queue_file_starts_empty(self):
        with open(self.path, "w") as f:
            f.write("{not json")
        self.assertEqual(self._queue().jobs(), [])
        with open(self.path) as f:
            self.assertEqual(json.load(f), {"paused": False, "jobs": []})


if __name__ == "__main__":
    unittest.main()
