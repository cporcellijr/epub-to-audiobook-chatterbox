"""Searchable list of the EPUBs in a mounted ebook library, for the web UI's book picker.

Titles and authors come from each EPUB's own metadata (OPF), which is slow to read for a whole
library, so results are cached in a JSON file and only new or changed files are re-read.

Environment:
    EBOOK_LIBRARY_DIR   Library folder to list (read-only mount), searched recursively
    EBOOK_INDEX_FILE    Cache file (default: library_index.json in the working directory)
"""
import json
import logging
import os
import re
import threading
import xml.etree.ElementTree as ET
import zipfile
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

_lock = threading.Lock()

# EPUB3 title-type values (besides "main") that mark a <dc:title> as a non-primary variant,
# plus the ad hoc "sort"/"alt" wording some tools use for the same purpose.
_NON_MAIN_TITLE_TYPES = {"subtitle", "short", "collection", "edition", "expanded", "sort", "alt", "alternate"}


def library_dir() -> str:
    return os.environ.get("EBOOK_LIBRARY_DIR", "")


def index_file() -> str:
    return os.environ.get("EBOOK_INDEX_FILE", "library_index.json")


def _local_name(tag: str) -> str:
    """An element or attribute name with its namespace URI stripped."""
    return tag.rpartition("}")[2]


def _local_attr(element: ET.Element, name: str) -> str:
    """An attribute's value, matched by local name regardless of its namespace prefix ('' if absent)."""
    for key, value in element.attrib.items():
        if _local_name(key) == name:
            return value
    return ""


def _element_text(element: ET.Element) -> str:
    """All text inside an element (CDATA and nested markup included), whitespace-normalized."""
    return re.sub(r"\s+", " ", "".join(element.itertext())).strip()


def _refinements(root: ET.Element) -> Dict[str, List[Tuple[str, str]]]:
    """id -> [(property, value), ...] from every `<meta refines="#id" property="...">value</meta>`."""
    refinements: Dict[str, List[Tuple[str, str]]] = {}
    for meta in root.iter():
        if _local_name(meta.tag) != "meta":
            continue
        refines = _local_attr(meta, "refines")
        if not refines.startswith("#"):
            continue
        prop = _local_attr(meta, "property").strip().lower()
        refinements.setdefault(refines[1:], []).append((prop, _element_text(meta)))
    return refinements


def _pick_main_title(titles: List[ET.Element], refinements: Dict[str, List[Tuple[str, str]]]) -> str:
    """The display title: the one marked title-type "main", else the first not refined as a
    sort/alternate variant, else simply the first (a plain single-title EPUB2 file)."""
    if not titles:
        return ""
    unmarked = []
    for title in titles:
        props = refinements.get(_local_attr(title, "id"), [])
        title_types = {value.strip().lower() for prop, value in props if prop == "title-type"}
        if "main" in title_types:
            return _element_text(title)
        has_sort_marker = any(prop == "file-as" for prop, _ in props)
        if not has_sort_marker and not (title_types & _NON_MAIN_TITLE_TYPES):
            unmarked.append(title)
    return _element_text(unmarked[0] if unmarked else titles[0])


def _pick_author(creators: List[ET.Element], refinements: Dict[str, List[Tuple[str, str]]]) -> str:
    """The creator with role "aut" (EPUB2 opf:role or EPUB3 <meta property="role">), else the first."""
    if not creators:
        return ""
    for creator in creators:
        role = _local_attr(creator, "role").strip().lower()
        if not role:
            props = refinements.get(_local_attr(creator, "id"), [])
            role = next((value.strip().lower() for prop, value in props if prop == "role"), "")
        if role == "aut":
            return _element_text(creator)
    return _element_text(creators[0])


def read_epub_metadata(path: str) -> Tuple[str, str]:
    """(title, author) from the EPUB's OPF, or empty strings if unreadable.

    Namespace-aware (xml.etree.ElementTree): prefers the title not refined as a sort/alternate
    variant (EPUB3 <meta refines property="title-type"|"file-as">) and the creator with role
    "aut" (EPUB2 opf:role or EPUB3 <meta property="role">), falling back to the first of each
    when no such marker is present -- so a plain single-title/single-author EPUB2 file (the
    common case) behaves exactly as before. Any read or parse failure (bad zip, missing
    container, malformed OPF) yields ("", ""), same as previously.
    """
    try:
        with zipfile.ZipFile(path) as z:
            container = z.read("META-INF/container.xml").decode("utf-8", "replace")
            rootfile = re.search(r'full-path="([^"]+)"', container)
            if not rootfile:
                return "", ""
            opf = z.read(rootfile.group(1))
        root = ET.fromstring(opf)
    except Exception:
        return "", ""

    titles = [el for el in root.iter() if _local_name(el.tag) == "title"]
    creators = [el for el in root.iter() if _local_name(el.tag) == "creator"]
    refinements = _refinements(root)
    return _pick_main_title(titles, refinements), _pick_author(creators, refinements)


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
