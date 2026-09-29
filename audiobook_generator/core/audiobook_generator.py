import hashlib
import json
import logging
import mimetypes
import multiprocessing
import os
import shutil

from audiobook_generator.book_parsers.base_book_parser import get_book_parser
from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.core.audio_tags import AudioTags
from audiobook_generator.core.m4b import BadChapterFileError, build_m4b, safe_book_file_name
from audiobook_generator.tts_providers.base_tts_provider import get_tts_provider
from audiobook_generator.utils.log_handler import setup_logging
from audiobook_generator.utils.filename_sanitizer import make_safe_filename

logger = logging.getLogger(__name__)

CHAPTER_WORK_FOLDER = ".chapters"
# Chapter work folder manifest (F-11): output file name -> the original chapter number
# and a hash of the text that produced it, so a later run with a changed chapter
# selection can tell a same-named leftover file apart from the chapter it now names.
CHAPTER_MANIFEST_FILENAME = ".manifest.json"


def _set_output_owner(path: str) -> None:
    """Let a library manager with a different container UID manage finished output."""
    uid = os.environ.get("AUDIOBOOK_OUTPUT_UID")
    if uid and hasattr(os, "chown"):
        try:
            os.chown(path, int(uid), -1)
        except (OSError, ValueError) as exc:
            logger.warning("Could not set audiobook output owner for %s: %s", path, exc)


_MIME_TO_EXT = {
    'image/jpeg': 'jpg',
    'image/png': 'png',
    'image/gif': 'gif',
    'image/webp': 'webp',
    'image/svg+xml': 'svg',
    'image/tiff': 'tiff',
    'image/bmp': 'bmp',
}


def _ext_for_mime(mime: str) -> str:
    """Return a safe file extension for an image MIME type, defaulting to 'jpg'."""
    if not mime:
        return 'jpg'
    if mime in _MIME_TO_EXT:
        return _MIME_TO_EXT[mime]
    # Fall back to mimetypes stdlib (strips the leading dot)
    ext = mimetypes.guess_extension(mime)
    if ext:
        return ext.lstrip('.')
    return 'jpg'


def confirm_conversion():
    logger.info("Do you want to continue? (y/n)")
    answer = input()
    if answer.lower() != "y":
        logger.info("Aborted.")
        exit(0)


def get_total_chars(chapters):
    total_characters = 0
    for title, text in chapters:
        total_characters += len(text)
    return total_characters


class AudiobookGenerator:
    def __init__(self, config: GeneralConfig):
        self.config = config
        self.cover = None
        self.book_title = None
        self.book_author = None
        # Final chapter idx -> original chapter number, set by run() (F-11). Direct
        # callers of process_chapter (tests, previews) get idx == original number.
        self.original_numbers = {}

    def __str__(self) -> str:
        return f"{self.config}"

    def output_m4b(self) -> bool:
        return getattr(self.config, "output_m4b", None) is True

    def chapter_folder(self) -> str:
        """Where chapter audio goes: a hidden working folder when merging into an M4B, so a
        library scanner never picks up a half-made book; otherwise the output folder itself."""
        if self.output_m4b():
            return os.path.join(self.config.output_folder, CHAPTER_WORK_FOLDER)
        return self.config.output_folder

    def chapter_audio_path(self, idx: int, title: str, extension: str) -> str:
        folder = self.chapter_folder()
        name = make_safe_filename(title=title, idx=idx, output_dir=folder, ext=extension, collision_check=False)
        return os.path.join(folder, name)

    def _manifest_path(self) -> str:
        return os.path.join(self.chapter_folder(), CHAPTER_MANIFEST_FILENAME)

    def _load_manifest(self) -> dict:
        """The chapter work folder's manifest: output file name -> the original chapter
        number and a hash of the text that produced it (F-11).

        A missing or unreadable manifest means a book started before this manifest
        existed, or a file this book never recorded; callers treat that as "trust the
        existing file" rather than forcing a full regeneration of an in-flight book on
        every upgrade.
        """
        try:
            with open(self._manifest_path(), "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _update_manifest(self, filename: str, chapter_number: int, text_hash: str) -> None:
        """Record which original chapter number and text produced `filename`, written
        atomically so a crash mid-write can't corrupt the manifest (F-11)."""
        manifest = self._load_manifest()
        manifest[filename] = {"chapter_number": chapter_number, "text_hash": text_hash}
        folder = self.chapter_folder()
        os.makedirs(folder, exist_ok=True)
        tmp_path = os.path.join(folder, f"{CHAPTER_MANIFEST_FILENAME}.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)
        os.replace(tmp_path, self._manifest_path())

    def process_chapter(self, idx, title, text):
        """Process a single chapter: write text (if needed) and convert to audio."""
        try:
            logger.info(f"Processing chapter {idx}: {title}")
            tts_provider = get_tts_provider(self.config)

            # Save chapter text if required
            if self.config.output_text:
                safe_txt_name = make_safe_filename(
                    title=title,
                    idx=idx,
                    output_dir=self.config.output_folder,
                    ext=".txt",
                    collision_check=False,
                )
                text_file = os.path.join(self.config.output_folder, safe_txt_name)
                with open(text_file, "w", encoding="utf-8") as f:
                    f.write(text)

            # Skip audio generation in preview mode
            if self.config.preview:
                return True

            # Generate audio file (safe, length-limited, cross-platform)
            output_file = self.chapter_audio_path(idx, title, "." + tts_provider.get_output_file_extension())
            safe_audio_name = os.path.basename(output_file)
            original_chapter_number = self.original_numbers.get(idx, idx)
            text_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()

            if self.config.skip_existing and os.path.isfile(output_file):
                manifest_entry = self._load_manifest().get(safe_audio_name)
                if manifest_entry is None or (
                    manifest_entry.get("chapter_number") == original_chapter_number
                    and manifest_entry.get("text_hash") == text_hash
                ):
                    logger.info(
                        f"⏭️  Skipping chapter {idx}: {title}, output file already exists: {output_file}"
                    )
                    return True
                logger.warning(
                    f"Chapter {idx}: {title}: existing file {output_file} was made for a "
                    f"different chapter selection; regenerating instead of reusing it."
                )

            audio_tags = AudioTags(
                title, self.book_author, self.book_title, idx,
                self.cover,
            )
            # Generate into a hidden partial file and rename into place only once it is complete
            # (audio + tags), so an interrupted run never leaves a truncated or untagged chapter
            # that skip_existing would then keep, or that a library scanner would pick up.
            partial_file = os.path.join(os.path.dirname(output_file), f".{safe_audio_name}.part")
            try:
                tts_provider.text_to_speech(text, partial_file, audio_tags)
                os.replace(partial_file, output_file)
            finally:
                if os.path.exists(partial_file):
                    os.remove(partial_file)
            self._update_manifest(safe_audio_name, original_chapter_number, text_hash)

            logger.info(f"✅ Converted chapter {idx}: {title}, output file: {output_file}")

            return True
        except Exception as e:
            logger.exception(f"Error processing chapter {idx}, error: {e}")
            return False

    def process_chapter_wrapper(self, args):
        """Wrapper for process_chapter to handle unpacking args for imap."""
        idx, title, text = args
        return idx, self.process_chapter(idx, title, text)

    def run(self) -> bool:
        """Generate the book; True when every selected chapter (and the M4B, if asked) succeeded."""
        succeeded = False
        try:
            logger.info("Starting audiobook generation...")
            book_parser = get_book_parser(self.config)
            if self.output_m4b() and getattr(self.config, "tts", None) == "openai":
                # Chapters become AAC (ADTS) so the M4B can be built with a stream copy
                # instead of a second lossy re-encode (F-06); loose-file output is unaffected.
                self.config.output_format = "aac"
            tts_provider = get_tts_provider(self.config)

            # Preview writes nothing unless chapter text was asked for, so previewing into a
            # library folder doesn't leave an empty or cover-only "book" behind.
            if not self.config.preview or self.config.output_text:
                os.makedirs(self.config.output_folder, exist_ok=True)
                _set_output_owner(self.config.output_folder)

            # Log and save book metadata
            self.book_title = book_parser.get_book_title()
            self.book_author = book_parser.get_book_author()
            logger.info(f"Book title: {self.book_title}")
            logger.info(f"Book author: {self.book_author}")

            if not self.config.preview:
                os.makedirs(self.chapter_folder(), exist_ok=True)

            self.cover = book_parser.get_book_cover()
            cover_path = None
            if self.cover and not self.config.preview:
                ext = _ext_for_mime(self.cover.mime)
                # In M4B mode the cover goes into the hidden work folder while the book is
                # still generating, so a library scanner never sees a cover-only "book"; it
                # is copied out next to the finished .m4b only once the book completes
                # (F-23). Loose-file output keeps today's behaviour: straight into the
                # visible output folder.
                cover_dir = self.chapter_folder() if self.output_m4b() else self.config.output_folder
                cover_path = os.path.join(cover_dir, f"cover.{ext}")
                with open(cover_path, 'wb') as f:
                    f.write(self.cover.data)
                logger.info(f"Cover saved: {cover_path}")

            chapters = book_parser.get_chapters(tts_provider.get_break_string())
            # Filter out empty or very short chapters
            chapters = [(title, text) for title, text in chapters if text.strip()]

            logger.info(f"Chapters count: {len(chapters)}.")

            # Check chapter start and end args
            if self.config.chapter_start < 1 or self.config.chapter_start > len(chapters):
                raise ValueError(
                    f"Chapter start index {self.config.chapter_start} is out of range. Check your input."
                )
            if self.config.chapter_end < -1 or self.config.chapter_end > len(chapters):
                raise ValueError(
                    f"Chapter end index {self.config.chapter_end} is out of range. Check your input."
                )
            if self.config.chapter_end == -1:
                self.config.chapter_end = len(chapters)
            if self.config.chapter_start > self.config.chapter_end:
                raise ValueError(
                    f"Chapter start index {self.config.chapter_start} is larger than chapter end index {self.config.chapter_end}. Check your input."
                )

            logger.info(
                f"Converting chapters from {self.config.chapter_start} to {self.config.chapter_end}."
            )

            # Prepare chapters for processing. Tasks carry only per-chapter data; book
            # metadata travels on self (resolved once above) rather than passing the
            # whole book_parser, which would re-pickle the parsed EPUB per chapter.
            chapters_to_process = chapters[self.config.chapter_start - 1 : self.config.chapter_end]
            numbered = list(enumerate(chapters_to_process, start=self.config.chapter_start))
            if self.config.chapter_selection:
                selected = {int(number) for number in self.config.chapter_selection}
                numbered = [(idx, chapter) for idx, chapter in numbered if idx in selected]
                logger.info(f"Selected chapters: {[idx for idx, _ in numbered]}")
                # Renumber the output 1..n so the finished audiobook's files and tracks
                # have no gaps, but remember each position's original chapter number
                # (F-11): a later run with a changed selection must not trust a same-named
                # leftover file just because a new chapter landed on the same position.
                original_by_new_idx = {new_idx: old_idx for new_idx, (old_idx, _) in
                                       enumerate(numbered, start=1)}
                numbered = list(enumerate((chapter for _, chapter in numbered), start=1))
                if not numbered:
                    raise ValueError("None of the selected chapters exist in this book.")
            else:
                original_by_new_idx = {idx: idx for idx, _ in numbered}
            tasks = [(idx, title, text) for idx, (title, text) in numbered]
            titles_by_idx = {idx: title for idx, title, _ in tasks}
            self.original_numbers = original_by_new_idx

            total_characters = get_total_chars([(title, text) for _, title, text in tasks])
            logger.info(f"Total characters in selected book chapters: {total_characters}")
            rough_price = tts_provider.estimate_cost(total_characters)
            logger.info(f"Estimate book voiceover would cost you roughly: ${rough_price:.2f}\n")

            # Prompt user to continue if not in preview mode
            if self.config.no_prompt:
                logger.info("Skipping prompt as passed parameter no_prompt")
            elif self.config.preview:
                logger.info("Skipping prompt as in preview mode")
            else:
                confirm_conversion()

            # Track failed chapters
            failed_chapters = []

            # worker_count == 1 (always true in the UI) processes chapters directly in
            # this process: terminating this process (e.g. "Stop current book") then has
            # nothing left running, instead of orphaning a Pool worker that keeps
            # generating after the job is already reported stopped (F-01). worker_count
            # > 1 (CLI only) keeps the multiprocessing Pool.
            if self.config.worker_count == 1:
                results = [self.process_chapter_wrapper(task) for task in tasks]
            else:
                with multiprocessing.Pool(
                    processes=self.config.worker_count,
                    initializer=setup_logging,
                    initargs=(self.config.log, self.config.log_file, True)
                ) as pool:
                    results = list(pool.imap_unordered(self.process_chapter_wrapper, tasks))

            # Check for failed chapters
            for idx, success in results:
                if not success:
                    failed_chapters.append((idx, titles_by_idx[idx]))

            if failed_chapters:
                logger.warning("The following chapters failed to convert:")
                for idx, title in failed_chapters:
                    logger.warning(f"  - Chapter {idx}: {title}")
                logger.info(f"Conversion completed with {len(failed_chapters)} failed chapters. Check your output directory: {self.config.output_folder} and log file: {self.config.log_file} for more details.")
                if self.output_m4b() and not self.config.preview:
                    logger.warning("M4B not built because chapters failed. Finished chapters are kept; start the "
                                   "book again with 'Skip chapters already made' to redo only the missing ones.")
            elif self.output_m4b() and not self.config.preview:
                logger.info(f"All chapters converted successfully. Check your output directory: {self.config.output_folder}")
                try:
                    self._merge_into_m4b(tasks, tts_provider.get_output_file_extension(), cover_path)
                    succeeded = True
                except Exception as e:
                    # Distinguish this from "chapters failed" (above): every chapter is
                    # fine, only the M4B mux step itself failed (F-04).
                    logger.error(
                        f"M4B build failed even though every chapter converted successfully: {e}. "
                        f"Finished chapters are kept; fix the cause and retry with 'Skip chapters "
                        f"already made' to rebuild just the M4B.", exc_info=True)
            else:
                logger.info(f"All chapters converted successfully. Check your output directory: {self.config.output_folder}")
                succeeded = True

        except KeyboardInterrupt:
            logger.info("Audiobook generation process interrupted by user (Ctrl+C).")
        except Exception as e:
            logger.exception(f"Error during audiobook generation: {e}")
        finally:
            logger.debug("AudiobookGenerator.run() method finished.")
        return succeeded

    def _merge_into_m4b(self, tasks, extension: str, cover_path) -> None:
        """Merge the finished chapters into <title>.m4b and remove the working folder."""
        chapters = [(title.replace("_", " "), self.chapter_audio_path(idx, title, "." + extension))
                    for idx, title, _ in sorted(tasks)]
        output_path = os.path.join(self.config.output_folder, f"{safe_book_file_name(self.book_title)}.m4b")
        try:
            build_m4b(chapters, output_path, self.book_title or "", self.book_author or "", cover_path)
        except BadChapterFileError as e:
            # Name the bad chapter and delete it, so a retry with skip_existing
            # regenerates just that one instead of failing the same way forever (F-33).
            if os.path.exists(e.path):
                os.remove(e.path)
            logger.error(
                f"Chapter '{e.title}' audio file ({e.path}) was unreadable or corrupt; "
                f"deleted it so a retry with 'Skip chapters already made' regenerates "
                f"just that chapter."
            )
            raise
        if cover_path and os.path.isfile(cover_path):
            # The owner keeps a folder cover beside the finished M4B, like the rest of
            # their library (F-23); the working copy is removed below with the rest of
            # the hidden chapter folder.
            final_cover = os.path.join(self.config.output_folder, os.path.basename(cover_path))
            shutil.copyfile(cover_path, final_cover)
            _set_output_owner(final_cover)
        _set_output_owner(output_path)
        shutil.rmtree(self.chapter_folder(), ignore_errors=True)
        logger.info(f"✅ M4B saved: {output_path}")

