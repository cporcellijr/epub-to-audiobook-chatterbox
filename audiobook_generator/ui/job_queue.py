"""Book queue for the web UI: one job runs at a time, each with its own settings.

The queue is saved to a JSON file, so it survives a container restart; a book that was running
when the server stopped goes back to the front of the line and resumes from its finished chapters.
"""
import json
import logging
import multiprocessing
import os
import sys
import threading
import time
import uuid
from datetime import datetime
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

QUEUED, RUNNING, DONE, FAILED, STOPPED = "queued", "running", "done", "failed", "stopped"
FINISHED = (DONE, FAILED, STOPPED)
# Job kinds: a book to narrate (the default, and what every job queued before kinds existed is),
# or a cast analysis (the LLM pass over a book's dialogue, which must never overlap a book).
BOOK, CAST = "book", "cast"

# Chapter audio files are named "<number>_<title>.<ext>"; a chapter work folder in M4B mode.
_CHAPTER_EXTENSIONS = (".mp3", ".aac")
_CHAPTER_WORK_FOLDER = ".chapters"
# Settings keys that may point at a private copy in the uploads folder, deleted with the job.
UPLOAD_KEYS = ("input_file", "search_and_replace_file", "cast_file")


def run_job(config, log_file: str) -> None:
    """Process target: generate one book; exit code 0 = every chapter (and the M4B) succeeded."""
    from main import main
    sys.exit(0 if main(config, log_file) else 1)


def run_cast_job(settings: dict, log_file: str) -> None:
    """Process target: analyse one book's cast (unloading and reloading Chatterbox around the LLM
    pass); exit code 0 = the cast was written."""
    from audiobook_generator.core.cast_analysis import run_cast_analysis
    run_cast_analysis(settings, log_file)


def job_kind(job: dict) -> str:
    return job.get("kind") or BOOK


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _chatterbox_ready(settings: dict) -> bool:
    """Default engine_ready: a Chatterbox book waits while the model is unloaded; Kokoro has no
    unload and never waits."""
    if settings.get("engine", "chatterbox") == "kokoro":
        return True
    from audiobook_generator.core.chatterbox_control import ready_for_book
    return ready_for_book()


class JobQueue:
    def __init__(self, path: str, build_config: Callable[..., object], log_file: Callable[[], str],
                 process_factory: Callable[..., object] = multiprocessing.Process, uploads_dir: str = "",
                 engine_ready: Callable[[dict], bool] = None):
        """engine_ready(settings) says whether a book may start now (default: not while Chatterbox
        reports its model unloaded after a cast analysis; see core.chatterbox_control).

        on_done(job), when set, is called with a copy of each job that finishes successfully (the
        web UI queues a book after its cast analysis); a string it returns becomes the job's note."""
        self.on_done: Optional[Callable[[dict], Optional[str]]] = None
        self.path = path
        self.uploads_dir = os.path.abspath(uploads_dir) if uploads_dir else ""
        self._build_config = build_config
        self._log_file = log_file
        self._process_factory = process_factory
        self._engine_ready = engine_ready or _chatterbox_ready
        self._lock = threading.RLock()
        self._process = None
        self._running_id: Optional[str] = None
        self._data = self._load()
        for job in self._data["jobs"]:
            if job["status"] == RUNNING:  # the server stopped mid-book: resume it first
                job["status"] = QUEUED
                if job_kind(job) == BOOK:
                    job["settings"]["skip_existing"] = True
                job["note"] = "resumes after restart"
                job["estimate_seconds"] *= self._remaining_chapter_fraction(job)
        self._save()

    # ---- persistence ----

    def _load(self) -> dict:
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("jobs"), list):
                data.setdefault("paused", False)
                data.setdefault("preparing", False)
                return data
        except (OSError, ValueError):
            pass
        return {"paused": False, "preparing": False, "jobs": []}

    def _save(self) -> None:
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)

    def _find(self, job_id: Optional[str]) -> Optional[dict]:
        return next((job for job in self._data["jobs"] if job["id"] == job_id), None)

    # ---- queries ----

    @property
    def paused(self) -> bool:
        return bool(self._data["paused"])

    @property
    def preparing(self) -> bool:
        return bool(self._data["preparing"])

    def jobs(self) -> List[dict]:
        with self._lock:
            return [dict(job) for job in self._data["jobs"]]

    def running_job(self) -> Optional[dict]:
        with self._lock:
            job = self._find(self._running_id)
            return dict(job) if job else None

    @staticmethod
    def _chapter_folder(settings: dict) -> str:
        """Where this job's chapter audio lives (the hidden work folder in M4B mode)."""
        folder = settings["output_dir"]
        if settings.get("output_m4b"):
            folder = os.path.join(folder, _CHAPTER_WORK_FOLDER)
        return folder

    # A mounted filesystem's mtime clock and time.time() are not always perfectly in step (observed:
    # a file written after started_ts was recorded can still read a few ms earlier on a Windows bind
    # mount). This tolerance absorbs that without reopening the stale-file bug it guards against --
    # a file from a genuinely unrelated earlier run is stale by minutes or hours, not milliseconds.
    _STALE_FILE_TOLERANCE_SECONDS = 2.0

    @staticmethod
    def _count_chapter_files(folder: str, since: Optional[float] = None) -> int:
        """Numbered chapter audio files (.mp3 or .aac) in folder; with `since`, only ones modified
        at or after that time (epoch seconds, less a small clock-skew tolerance), so files left
        over from an unrelated earlier run are not counted."""
        try:
            names = os.listdir(folder)
        except OSError:
            return 0
        count = 0
        cutoff = since - JobQueue._STALE_FILE_TOLERANCE_SECONDS if since is not None else None
        for name in names:
            if name.startswith(".") or not name[:4].isdigit() or not name.lower().endswith(_CHAPTER_EXTENSIONS):
                continue
            if cutoff is not None:
                try:
                    if os.path.getmtime(os.path.join(folder, name)) < cutoff:
                        continue
                except OSError:
                    continue
            count += 1
        return count

    @staticmethod
    def chapters_done(job: dict) -> int:
        """Finished chapter files on disk for this job's current run (all of them once it is done).

        Only files written at or after this run started count, so numbered chapter files left over
        from an unrelated earlier job in the same output folder can never be mistaken for this job's
        own progress. A cast analysis reports the chapters its cast file says are analysed.
        """
        if job["status"] == DONE:
            return job["chapters"]
        if job_kind(job) == CAST:
            from audiobook_generator.core.cast import analysis_progress, load_cast
            done, _ = analysis_progress(load_cast(job["settings"].get("cast_file")))
            return min(done, job["chapters"])
        count = JobQueue._count_chapter_files(JobQueue._chapter_folder(job["settings"]), since=job.get("started_ts"))
        return min(count, job["chapters"])

    @staticmethod
    def _remaining_chapter_fraction(job: dict) -> float:
        """Fraction of a job's chapters not yet sitting on disk: what 'Skip chapters already made'
        still has to generate, regardless of which run produced the ones already there. A cast
        analysis always starts over."""
        total = job["chapters"]
        if total <= 0 or job_kind(job) == CAST:
            return 1.0
        done = JobQueue._count_chapter_files(JobQueue._chapter_folder(job["settings"]))
        return max(0.0, (total - min(done, total)) / total)

    # ---- changes ----

    def add(self, title: str, settings: dict, chapters: int, estimate_seconds: float, voice: str,
            kind: str = BOOK) -> int:
        """Queue a book (or, kind=CAST, a cast analysis); returns its position among jobs still
        waiting (1 = next)."""
        with self._lock:
            self._data["jobs"].append({
                "id": uuid.uuid4().hex[:12], "kind": kind, "title": title, "voice": voice, "chapters": chapters,
                "estimate_seconds": estimate_seconds, "status": QUEUED, "added": _now(), "started": None,
                "started_ts": None, "finished": None, "note": "", "settings": settings,
            })
            self._save()
            return sum(1 for job in self._data["jobs"] if job["status"] == QUEUED)

    def remove(self, job_id: str) -> bool:
        """Drop a book that hasn't started (or has finished). The running book must be stopped first."""
        with self._lock:
            job = self._find(job_id)
            if not job or job["status"] == RUNNING:
                return False
            self._data["jobs"].remove(job)
            self._delete_uploads(job)
            self._save()
            return True

    def retry(self, job_id: str) -> bool:
        """Re-queue a failed or stopped book at the back of the line, keeping finished chapters."""
        with self._lock:
            job = self._find(job_id)
            if not job or job["status"] not in (FAILED, STOPPED):
                return False
            self._data["jobs"].remove(job)
            job["estimate_seconds"] *= self._remaining_chapter_fraction(job)
            if job_kind(job) == CAST:
                job.update(status=QUEUED, started=None, started_ts=None, finished=None, note="retry")
            else:
                job.update(status=QUEUED, started=None, started_ts=None, finished=None,
                           note="retry: finished chapters kept")
                job["settings"]["skip_existing"] = True
            self._data["jobs"].append(job)
            self._save()
            return True

    def _delete_uploads(self, job: dict) -> None:
        """Remove this job's private copies of uploaded files."""
        if not self.uploads_dir:
            return
        for key in UPLOAD_KEYS:
            path = job["settings"].get(key)
            if path and os.path.abspath(path).startswith(self.uploads_dir + os.sep) and os.path.isfile(path):
                os.remove(path)

    def clear_finished(self) -> int:
        with self._lock:
            before = len(self._data["jobs"])
            for job in self._data["jobs"]:
                if job["status"] in FINISHED:
                    self._delete_uploads(job)
            self._data["jobs"] = [job for job in self._data["jobs"] if job["status"] not in FINISHED]
            self._save()
            return before - len(self._data["jobs"])

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self._data["paused"] = bool(paused)
            self._save()

    def set_preparing(self, preparing: bool) -> None:
        """While preparing, run queued cast analyses but hold audiobook jobs."""
        with self._lock:
            self._data["preparing"] = bool(preparing)
            self._save()

    def _finish_from_exitcode(self, job: dict, exitcode: int) -> None:
        """DONE/FAILED bookkeeping for a job whose process has actually exited: shared by tick()
        and by a stop_current() that raced a book which had already finished (F-28)."""
        job["status"] = DONE if exitcode == 0 else FAILED
        job["finished"] = _now()
        if job["status"] == FAILED:
            job["note"] = ("failed; see the log" if job_kind(job) == CAST
                           else "failed; see the log (Retry keeps finished chapters)")
        elif self.on_done:
            try:
                note = self.on_done(dict(job))
            except Exception as e:
                logger.exception(f"Queue: after '{job['title']}': {e}")
                note = "done; see the log"
            if note:
                job["note"] = note

    def stop_current(self) -> bool:
        """Stop the running book and pause the queue (Resume starts the next one)."""
        with self._lock:
            self._data["paused"] = True
            job = self._find(self._running_id)
            process = self._process
            if process is not None and process.is_alive():
                process.terminate()
                process.join()
                if job:
                    job["status"] = STOPPED
                    job["finished"] = _now()
                    job["note"] = "stopped; finished chapters kept"
            elif process is not None:
                # The process had already exited on its own -- possibly in the last couple of
                # seconds, before the next tick() could notice -- so there was nothing left to
                # terminate. Record its real outcome instead of overwriting a finished book as
                # STOPPED (F-28).
                process.join()
                if job and job["status"] == RUNNING:
                    self._finish_from_exitcode(job, process.exitcode)
                    logger.info(f"Queue: '{job['title']}' had already finished when Stop was "
                               f"pressed ({job['status']})")
            self._process, self._running_id = None, None
            self._save()
            return job is not None

    def tick(self) -> None:
        """Advance the queue: record a finished book, then start the next one unless paused."""
        with self._lock:
            if self._process is not None:
                if self._process.is_alive():
                    return
                self._process.join()
                job = self._find(self._running_id)
                if job and job["status"] == RUNNING:
                    self._finish_from_exitcode(job, self._process.exitcode)
                self._process, self._running_id = None, None
                self._save()
            if self.paused:
                return
            # Cast analyses go ahead of every waiting book, so all the picked books are analysed (and
            # join the queue) before the long generating starts, however early Start was pressed.
            waiting = [j for j in self._data["jobs"] if j["status"] == QUEUED]
            job = next((j for j in waiting if job_kind(j) == CAST), None)
            if job is None and not self.preparing:
                job = next(iter(waiting), None)
            if job is None:
                return
            if job_kind(job) == CAST:
                process = self._process_factory(target=run_cast_job, args=(dict(job["settings"]), self._log_file()))
            else:
                # Never start a book while Chatterbox is unloaded (a cast analysis freed it and has
                # not brought it back yet): its requests would all fail with 503.
                if not self._engine_ready(job["settings"]):
                    return
                config = self._build_config(**job["settings"])
                process = self._process_factory(target=run_job, args=(config, self._log_file()))
            process.start()
            job["status"], job["note"] = RUNNING, job.get("note", "")
            if not job.get("started"):
                # A brand-new job or a Retry (both cleared "started" first): this is genuinely the
                # first file this run will write, so chapters_done() should count from now. A
                # restart-resume left "started" alone, since it is the same run continuing -- its
                # chapters made before the restart still count as this run's own progress.
                job["started"], job["started_ts"] = _now(), time.time()
            self._process, self._running_id = process, job["id"]
            self._save()
            logger.info(f"Queue: started '{job['title']}'")

    def start_worker(self, interval: float = 2.0) -> None:
        def loop():
            while True:
                try:
                    self.tick()
                except Exception as e:
                    logger.exception(f"Queue worker error: {e}")
                time.sleep(interval)
        threading.Thread(target=loop, name="book-queue", daemon=True).start()
