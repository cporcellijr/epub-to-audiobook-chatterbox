"""Book queue for the web UI: books run one at a time, in order, each with its own settings.

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


def run_job(config, log_file: str) -> None:
    """Process target: generate one book; exit code 0 = every chapter (and the M4B) succeeded."""
    from main import main
    sys.exit(0 if main(config, log_file) else 1)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


class JobQueue:
    def __init__(self, path: str, build_config: Callable[..., object], log_file: Callable[[], str],
                 process_factory: Callable[..., object] = multiprocessing.Process, uploads_dir: str = ""):
        self.path = path
        self.uploads_dir = os.path.abspath(uploads_dir) if uploads_dir else ""
        self._build_config = build_config
        self._log_file = log_file
        self._process_factory = process_factory
        self._lock = threading.RLock()
        self._process = None
        self._running_id: Optional[str] = None
        self._data = self._load()
        for job in self._data["jobs"]:
            if job["status"] == RUNNING:  # the server stopped mid-book: resume it first
                job["status"] = QUEUED
                job["settings"]["skip_existing"] = True
                job["note"] = "resumes after restart"
        self._save()

    # ---- persistence ----

    def _load(self) -> dict:
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("jobs"), list):
                data.setdefault("paused", False)
                return data
        except (OSError, ValueError):
            pass
        return {"paused": False, "jobs": []}

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

    def jobs(self) -> List[dict]:
        with self._lock:
            return [dict(job) for job in self._data["jobs"]]

    def running_job(self) -> Optional[dict]:
        with self._lock:
            job = self._find(self._running_id)
            return dict(job) if job else None

    @staticmethod
    def chapters_done(job: dict) -> int:
        """Finished chapter files on disk for a job (all of them once it is done)."""
        if job["status"] == DONE:
            return job["chapters"]
        settings = job["settings"]
        folder = settings["output_dir"]
        if settings.get("output_m4b"):
            folder = os.path.join(folder, ".chapters")
        try:
            names = os.listdir(folder)
        except OSError:
            return 0
        count = sum(1 for n in names if not n.startswith(".") and n[:4].isdigit() and not n.endswith(".txt"))
        return min(count, job["chapters"])

    # ---- changes ----

    def add(self, title: str, settings: dict, chapters: int, estimate_seconds: float, voice: str) -> int:
        """Queue a book; returns its position among books still waiting (1 = next)."""
        with self._lock:
            self._data["jobs"].append({
                "id": uuid.uuid4().hex[:12], "title": title, "voice": voice, "chapters": chapters,
                "estimate_seconds": estimate_seconds, "status": QUEUED, "added": _now(), "started": None,
                "finished": None, "note": "", "settings": settings,
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
            job.update(status=QUEUED, started=None, finished=None, note="retry: finished chapters kept")
            job["settings"]["skip_existing"] = True
            self._data["jobs"].append(job)
            self._save()
            return True

    def _delete_uploads(self, job: dict) -> None:
        """Remove this job's private copies of uploaded files."""
        if not self.uploads_dir:
            return
        for key in ("input_file", "search_and_replace_file"):
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

    def stop_current(self) -> bool:
        """Stop the running book and pause the queue (Resume starts the next one)."""
        with self._lock:
            self._data["paused"] = True
            job = self._find(self._running_id)
            if self._process is not None and self._process.is_alive():
                self._process.terminate()
                self._process.join()
            if job:
                job["status"] = STOPPED
                job["finished"] = _now()
                job["note"] = "stopped; finished chapters kept"
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
                    job["status"] = DONE if self._process.exitcode == 0 else FAILED
                    job["finished"] = _now()
                    if job["status"] == FAILED:
                        job["note"] = "some chapters failed; see log (re-add with Skip chapters already made)"
                self._process, self._running_id = None, None
                self._save()
            if self.paused:
                return
            job = next((j for j in self._data["jobs"] if j["status"] == QUEUED), None)
            if job is None:
                return
            config = self._build_config(**job["settings"])
            process = self._process_factory(target=run_job, args=(config, self._log_file()))
            process.start()
            job["status"], job["started"], job["note"] = RUNNING, _now(), job.get("note", "")
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
