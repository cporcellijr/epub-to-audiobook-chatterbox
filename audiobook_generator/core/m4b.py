"""Merge finished chapter files into one .m4b audiobook with chapter markers, cover and tags."""
import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import List, Optional, Tuple

from audiobook_generator.utils.safe_names import sanitize_display_name

logger = logging.getLogger(__name__)

# ffmpeg's MP4 muxer only accepts these as an attached picture with "-c:v copy" (F-04);
# any other cover format (GIF, WebP, TIFF, SVG, ...) must be re-encoded.
_COPY_COVER_EXTS = frozenset({"jpg", "jpeg", "png"})


class BadChapterFileError(RuntimeError):
    """A chapter audio file could not be read while building the M4B: empty, corrupt, or
    missing. `path` and `title` identify which chapter, so the caller can delete it and
    let a resume regenerate just that one (F-33).
    """

    def __init__(self, path: str, title: str, cause: Exception):
        self.path = path
        self.title = title
        super().__init__(f"Chapter '{title}' audio file is unreadable or corrupt: {path} ({cause})")


def safe_book_file_name(title: str) -> str:
    """Book title usable as a file name on Windows and Linux (spaces kept)."""
    return sanitize_display_name(title, max_length=150, fallback="audiobook")


def _cover_video_codec(cover_path: str) -> str:
    """The ffmpeg '-c:v' value for embedding cover_path as the M4B's attached picture."""
    ext = os.path.splitext(cover_path)[1].lstrip(".").lower()
    return "copy" if ext in _COPY_COVER_EXTS else "mjpeg"


def _check_ffmpeg_available() -> None:
    """Fail once, with one clear message, if ffmpeg/ffprobe aren't on PATH, instead of a
    raw FileNotFoundError from whichever call happens to need them first (F-33)."""
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(f"{' and '.join(missing)} not found on PATH; required to build the M4B.")


def _duration_seconds(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def _escape_metadata(value: str) -> str:
    return re.sub(r"([=;#\\\n])", r"\\\1", value or "")


def _escape_concat_path(path: str) -> str:
    return path.replace("'", "'\\''")


def build_m4b(chapters: List[Tuple[str, str]], output_path: str, title: str, author: str,
              cover_path: Optional[str] = None, bitrate: str = "64k") -> None:
    """Encode (chapter title, audio file) pairs, in order, into output_path.

    Writes to a hidden temporary file in the same folder and renames it into place, so a
    library scanner never sees a half-written book.
    """
    if not chapters:
        raise ValueError("No chapters to merge")
    _check_ffmpeg_available()
    folder = os.path.dirname(output_path) or "."
    temp_output = os.path.join(folder, f".{os.path.basename(output_path)}.part")

    with tempfile.TemporaryDirectory() as tmp:
        concat_list = os.path.join(tmp, "chapters.txt")
        metadata = os.path.join(tmp, "metadata.txt")
        with open(concat_list, "w", encoding="utf-8") as f:
            f.write("ffconcat version 1.0\n")
            for _, path in chapters:
                f.write(f"file '{_escape_concat_path(os.path.abspath(path))}'\n")

        lines = [";FFMETADATA1", f"title={_escape_metadata(title)}", f"album={_escape_metadata(title)}",
                 f"artist={_escape_metadata(author)}", f"album_artist={_escape_metadata(author)}",
                 "genre=Audiobook"]
        start_ms = 0
        for chapter_title, path in chapters:
            try:
                duration = _duration_seconds(path)
            except (subprocess.CalledProcessError, OSError, ValueError) as e:
                raise BadChapterFileError(path, chapter_title, e) from e
            end_ms = start_ms + int(round(duration * 1000))
            lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={start_ms}", f"END={end_ms}",
                      f"title={_escape_metadata(chapter_title)}"]
            start_ms = end_ms
        with open(metadata, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "concat", "-safe", "0", "-i", concat_list, "-i", metadata]
        if cover_path and os.path.isfile(cover_path):
            command += ["-i", cover_path, "-map", "0:a", "-map", "2:v",
                        "-c:v", _cover_video_codec(cover_path),
                        "-disposition:v:0", "attached_pic"]
        else:
            command += ["-map", "0:a"]
        audio_ext = os.path.splitext(chapters[0][1])[1].lstrip(".").lower()
        if audio_ext == "aac":
            # Chapters are already ADTS AAC (F-06): remux with a stream copy instead of a
            # second lossy pass. MP4 doesn't use ADTS framing, hence the bitstream filter.
            audio_args = ["-c:a", "copy", "-bsf:a", "aac_adtstoasc"]
        else:
            audio_args = ["-c:a", "aac", "-b:a", bitrate]
        command += ["-map_metadata", "1", "-map_chapters", "1", *audio_args,
                    "-movflags", "+faststart", "-f", "mp4", temp_output]
        logger.info(f"Building M4B from {len(chapters)} chapters: {output_path}")
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            if os.path.exists(temp_output):
                os.remove(temp_output)
            raise RuntimeError(f"ffmpeg could not build the M4B: {result.stderr.strip()[-500:]}")
    os.replace(temp_output, output_path)
