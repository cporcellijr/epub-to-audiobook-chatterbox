"""Searchable list of the EPUBs in a mounted ebook library, for the web UI's book picker.

Titles and authors come from each EPUB's own metadata (OPF), which is slow to read for a whole
library, so results are cached in a JSON file and only new or changed files are re-read.

Environment:
    EBOOK_LIBRARY_DIR   Library folder to list (read-only mount), searched recursively
    EBOOK_INDEX_FILE    Cache file (default: library_index.json in the working directory)
"""
import html
import json
import logging
import os
import re
import threading
import zipfile
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

_lock = threading.Lock()


def library_dir() -> str:
    return os.environ.get("EBOOK_LIBRARY_DIR", "")


def index_file() -> str:
    return os.environ.get("EBOOK_INDEX_FILE", "library_index.json")


def read_epub_metadata(path: str) -> Tuple[str, str]:
    """(title, author) from the EPUB's OPF, or empty strings if unreadable."""
    try:
        with zipfile.ZipFile(path) as z:
            container = z.read("META-INF/container.xml").decode("utf-8", "replace")
            rootfile = re.search(r'full-path="([^"]+)"', container)
            if not rootfile:
                return "", ""
            opf = z.read(rootfile.group(1)).decode("utf-8", "replace")
    except Exception:
        return "", ""

    def first(tag: str) -> str:
        match = re.search(rf"<dc:{tag}\b[^>]*>(.*?)</dc:{tag}>", opf, re.S | re.I)
        return html.unescape(re.sub(r"\s+", " ", match.group(1))).strip() if match else ""

    return first("title"), first("creator")


def load_index() -> Dict[str, dict]:
    """The cached index ({path: {mtime, size, title, author}}), or {} if there is none yet."""
    try:
        with open(index_file(), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_index(index: Dict[str, dict]) -> None:
    tmp = f"{index_file()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False)
        os.replace(tmp, index_file())
    except OSError as e:
        logger.warning(f"Could not save library index: {e}")


def refresh_index() -> Dict[str, dict]:
    """Walk the library, re-reading metadata only for new or changed files; drop deleted ones."""
    root = library_dir()
    if not root or not os.path.isdir(root):
        return {}
    with _lock:
        cached = load_index()
        index: Dict[str, dict] = {}
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if not name.lower().endswith(".epub"):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    stat = os.stat(path)
                except OSError:
                    continue
                entry = cached.get(path)
                if not entry or entry.get("mtime") != stat.st_mtime or entry.get("size") != stat.st_size:
                    title, author = read_epub_metadata(path)
                    entry = {"mtime": stat.st_mtime, "size": stat.st_size, "title": title, "author": author}
                index[path] = entry
        if index != cached:
            _save_index(index)
        return index


def book_label(path: str, entry: dict) -> str:
    title = entry.get("title") or os.path.splitext(os.path.basename(path))[0] or os.path.basename(path)
    return f"{title} — {entry['author']}" if entry.get("author") else title


def book_choices(index: Dict[str, dict]) -> List[Tuple[str, str]]:
    """(label, path) pairs sorted by label."""
    return sorted(((book_label(p, e), p) for p, e in index.items()), key=lambda c: c[0].lower())


def book_title(path: str, index: Dict[str, dict]) -> str:
    entry = index.get(path) or {}
    return entry.get("title") or os.path.splitext(os.path.basename(path))[0]


def warm_up_in_background() -> None:
    """Build/refresh the cache at server start so the first page load already has the list."""
    threading.Thread(target=refresh_index, name="library-index", daemon=True).start()
