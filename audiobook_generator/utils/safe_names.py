"""Shared filename/foldername sanitizing helpers (F-34).

Three call sites used to each reimplement "make this title safe as a file or folder
name" slightly differently: core/m4b.py's safe_book_file_name, ui/web_ui.py's
safe_folder_name, and utils/filename_sanitizer.py's per-chapter file names. This module
is the one place that knows which characters are illegal on Windows/Linux, that
truncation must happen before trailing space/dot stripping (not after, or truncating a
long title can reintroduce a trailing space or dot), and which plain names Windows
reserves for devices.
"""
import re

_FORBIDDEN_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# The same set, usable for a char-by-char scan (utils/filename_sanitizer.py replaces
# each forbidden character individually rather than stripping them).
FORBIDDEN_CHARS = frozenset('<>:"/\\|?*') | frozenset(chr(c) for c in range(0x20))

WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


def strip_forbidden_chars(name: str) -> str:
    """Remove characters illegal in a Windows or Linux file/folder name."""
    return _FORBIDDEN_CHARS_RE.sub("", name or "")


def avoid_reserved_name(name: str, suffix: str = "_") -> str:
    """Append `suffix` if `name` is a Windows-reserved device name (CON, NUL, COM1, ...).

    Compares case-insensitively and ignores any extension, since Windows reserves the
    name whether or not one is present (e.g. "nul.txt" is still the NUL device).
    """
    if not name:
        return name
    stem = name.split(".", 1)[0].upper()
    if stem in WINDOWS_RESERVED_NAMES:
        return f"{name}{suffix}"
    return name


def sanitize_display_name(name: str, max_length: int = 150, fallback: str = "") -> str:
    """A title usable as a file or folder name on Windows and Linux, keeping spaces.

    Order matters: truncate to max_length before stripping a trailing space/dot, or the
    cut itself can reintroduce one at the new end of the string (F-34).
    """
    name = strip_forbidden_chars(name)
    name = re.sub(r"\s+", " ", name).strip()
    name = name[:max_length].rstrip(" .")
    name = avoid_reserved_name(name)
    return name or fallback
