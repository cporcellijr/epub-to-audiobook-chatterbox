import logging
import re
from typing import List, Optional, Tuple

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub

from audiobook_generator.book_parsers.base_book_parser import BaseBookParser
from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.core.cover_image import CoverImage

logger = logging.getLogger(__name__)


class EpubBookParser(BaseBookParser):
    def __init__(self, config: GeneralConfig):
        super().__init__(config)
        self.book = epub.read_epub(self.config.input_file, {"ignore_ncx": True})

    def __str__(self) -> str:
        return super().__str__()

    def validate_config(self):
        if self.config.input_file is None:
            raise ValueError("Epub Parser: Input file cannot be empty")
        if not self.config.input_file.endswith(".epub"):
            raise ValueError(f"Epub Parser: Unsupported file format: {self.config.input_file}")

    def get_book(self):
        return self.book

    def get_book_title(self) -> str:
        if self.book.get_metadata('DC', 'title'):
            return self.book.get_metadata("DC", "title")[0][0]
        return "Untitled"

    def get_book_author(self) -> str:
        if self.book.get_metadata('DC', 'creator'):
            return self.book.get_metadata("DC", "creator")[0][0]
        return "Unknown"

    def get_book_cover(self) -> Optional[CoverImage]:
        """Return a CoverImage, or None if no cover is found."""
        # 1. Items explicitly typed as cover
        for item in self.book.get_items_of_type(ebooklib.ITEM_COVER):
            return CoverImage(data=item.get_content(), mime=item.media_type)

        # 2. Item with id 'cover' that is an image
        cover_item = self.book.get_item_with_id('cover')
        if cover_item and cover_item.media_type.startswith('image/'):
            return CoverImage(data=cover_item.get_content(), mime=cover_item.media_type)

        # 3. OPF metadata <meta name="cover" content="<id>"/>
        meta = self.book.get_metadata('OPF', 'cover')
        if meta:
            cover_id = meta[0][1].get('content')
            if cover_id:
                cover_item = self.book.get_item_with_id(cover_id)
                if cover_item:
                    return CoverImage(data=cover_item.get_content(), mime=cover_item.media_type)

        # 4. Fallback: first image whose name contains 'cover'
        for item in self.book.get_items_of_type(ebooklib.ITEM_IMAGE):
            if 'cover' in item.file_name.lower():
                return CoverImage(data=item.get_content(), mime=item.media_type)

        logger.warning("No cover image found in EPUB")
        return None

    @staticmethod
    def _is_nav_document(item) -> bool:
        """True for the EPUB3 navigation (table of contents) document."""
        return isinstance(item, epub.EpubNav) or "nav" in (getattr(item, "properties", None) or [])

    # A guide-referenced "toc" page this long is probably real content a publisher mistagged,
    # not a contents listing (some publishers point <guide type="toc"> at a real chapter) --
    # keep it. Chosen so an ordinary front-matter contents list (a short list of headings) is
    # still dropped while a genuine chapter (thousands of characters of prose) is kept.
    _GUIDE_TOC_MAX_CHARS = 5000

    def _guide_toc_href(self) -> Optional[str]:
        """href (fragment stripped) of an EPUB2 `<guide><reference type="toc">`, or None."""
        for entry in getattr(self.book, "guide", None) or []:
            if (entry.get("type") or "").strip().lower() == "toc":
                href = (entry.get("href") or "").split("#", 1)[0]
                if href:
                    return href
        return None

    @classmethod
    def _is_long_guide_toc_page(cls, item) -> bool:
        """True when a guide-referenced toc page reads as real content, not a contents listing."""
        text_length = len(BeautifulSoup(item.get_content(), "lxml-xml").get_text(strip=True))
        return text_length > cls._GUIDE_TOC_MAX_CHARS

    def _reading_order_documents(self) -> list:
        """Content documents in spine (reading) order.

        The manifest is an unordered resource list, so iterating it can put chapters out of
        order and picks up the navigation document (a table of contents read aloud). Follow
        the spine instead, skipping the nav document and non-linear (auxiliary) items, and the
        EPUB2 `<guide type="toc">` page -- a secondary signal, since it is only excluded when
        it also reads short (see `_GUIDE_TOC_MAX_CHARS`). Falls back to manifest order if the
        spine resolves to nothing.
        """
        documents = []
        guide_toc_href = self._guide_toc_href()
        for entry in self.book.spine:
            idref, linear = (entry, "yes") if isinstance(entry, str) else (entry[0], entry[1])
            item = self.book.get_item_with_id(idref)
            if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT or self._is_nav_document(item):
                continue
            if str(linear).lower() in ("no", "false"):
                continue
            if guide_toc_href and item.get_name() == guide_toc_href and not self._is_long_guide_toc_page(item):
                continue
            documents.append(item)
        if documents:
            return documents
        logger.warning("EPUB spine yielded no content documents; falling back to manifest order")
        return [item for item in self.book.get_items_of_type(ebooklib.ITEM_DOCUMENT)
                if not self._is_nav_document(item)]

    _BLOCK_TAGS = ["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "pre", "tr", "dt", "dd",
                   "section", "article", "header", "footer", "aside", "figcaption"]
    _CELL_TAGS = ["td", "th"]

    @classmethod
    def _mark_paragraphs(cls, soup) -> None:
        """End every block element with a blank line so paragraph breaks survive text extraction,
        and every table cell with a space so tightly packed cells don't run together as one word.

        Many EPUBs put a single newline (or none) between <p> elements, which "double" newline
        mode can't see; the HTML structure is the reliable signal. A global `get_text(separator=...)`
        would fix cells too, but it would also split inline tags like <em> inside a word.
        """
        for tag in soup.find_all(cls._BLOCK_TAGS):
            tag.append("\n\n")
        for tag in soup.find_all(cls._CELL_TAGS):
            tag.append(" ")
        for line_break in soup.find_all("br"):
            line_break.replace_with("\n")

    def get_chapters(self, break_string) -> List[Tuple[str, str]]:
        chapters = []
        search_and_replaces = self.get_search_and_replaces()
        for item in self._reading_order_documents():
            content = item.get_content()
            soup = BeautifulSoup(content, "lxml-xml")
            self._mark_paragraphs(soup)
            raw = soup.get_text(strip=False)
            logger.debug("Raw text: <%s>", raw)

            # Replace excessive whitespaces and newline characters based on the mode
            if self.config.newline_mode == "single":
                cleaned_text = re.sub(r"[\n]+", break_string, raw.strip())
            elif self.config.newline_mode == "double":
                cleaned_text = re.sub(r"[\n]{2,}", break_string, raw.strip())
            elif self.config.newline_mode == "none":
                cleaned_text = re.sub(r"[\n]+", " ", raw.strip())
            else:
                raise ValueError(f"Invalid newline mode: {self.config.newline_mode}")

            logger.debug("Cleaned text step 1: <%s>", cleaned_text)
            cleaned_text = re.sub(r"\s+", " ", cleaned_text)
            logger.debug("Cleaned text step 2: <%s>", cleaned_text[:100])

            # Removes end-note numbers: 1-3 digits glued onto a lowercase letter or closing
            # punctuation, and not themselves followed by a letter or digit. The lowercase-only
            # requirement (and the 1-3 digit cap) keeps acronym/product compounds like "COVID19",
            # "B2B" or a quoted "1990s" intact -- a real inline marker almost always ends an
            # ordinary (lowercase) word or clause, not an all-caps token or a 4+ digit number.
            if self.config.remove_endnotes:
                cleaned_text = re.sub(r'(?<=[a-z.,!?;”")])\d{1,3}(?![a-zA-Z0-9])', "", cleaned_text)
                logger.debug("Cleaned text step 4: <%s>", cleaned_text[:100])

            # Removes references numbers like [1] or [2.3]
            if self.config.remove_reference_numbers:
                cleaned_text = re.sub(r'\[\d+(\.\d+)?\]', '', cleaned_text)
                logger.debug("Cleaned text step 4.1 (removed brackets): <%s>", cleaned_text[:100])

            # Does user defined search and replaces
            for search_and_replace in search_and_replaces:
                cleaned_text = re.sub(search_and_replace['search'], search_and_replace['replace'], cleaned_text)
            logger.debug("Cleaned text step 5: <%s>", cleaned_text[:100])

            # Get proper chapter title
            if self.config.title_mode == "auto":
                title = ""
                title_levels = ['title', 'h1', 'h2', 'h3']
                for level in title_levels:
                    if soup.find(level):
                        title = soup.find(level).text
                        break
                if title.strip() == "" or re.match(r'^\d{1,3}$', title.strip()) is not None:
                    title = cleaned_text.replace(break_string, " ")[:60]
            elif self.config.title_mode == "tag_text":
                title = ""
                title_levels = ['title', 'h1', 'h2', 'h3']
                for level in title_levels:
                    if soup.find(level):
                        title = soup.find(level).text
                        break
                if title.strip() == "":
                    title = "<blank>"
            elif self.config.title_mode == "first_few":
                title = cleaned_text.replace(break_string, " ")[:60]
            else:
                raise ValueError("Unsupported title_mode")
            logger.debug("Raw title: <%s>", title)
            title = self._display_title(title, break_string)
            logger.debug("Display title: <%s>", title)

            chapters.append((title, cleaned_text))
            soup.decompose()
        return chapters

    def get_search_and_replaces(self) -> List[dict]:
        """Load search-and-replace rules from `config.search_and_replace_file`.

        Each non-blank, non-comment ('#') line is "search==replace" (an empty replace deletes
        the match; maxsplit=1 so a literal "==" inside the replacement text is kept intact).
        Patterns are compiled here, on load, so a bad one is reported with its line number
        instead of raising deep inside per-chapter text cleaning and aborting the whole run.
        """
        search_and_replaces: List[dict] = []
        if not self.config.search_and_replace_file:
            return search_and_replaces
        with open(self.config.search_and_replace_file, encoding="utf-8") as fp:
            lines = fp.read().splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not line or line.startswith('#') or '==' not in line:
                continue
            search, replace = line.split('==', 1)
            if not search:
                continue
            try:
                re.compile(search)
            except re.error as exc:
                raise ValueError(
                    f"Search-and-replace file {self.config.search_and_replace_file}, "
                    f"line {line_number}: bad pattern {search!r}: {exc}"
                ) from exc
            search_and_replaces.append({'search': search, 'replace': replace})
        return search_and_replaces

    @staticmethod
    def _display_title(title: str, break_string: str) -> str:
        """A human-readable chapter title for M4B markers, tags, the UI table and logs.

        Unlike `_sanitize_title` (still used for file-name-safe stems, e.g. by callers that
        need a slug), this keeps punctuation and normal spacing; it only removes the paragraph
        break marker inserted by `_mark_paragraphs`/`newline_mode` and collapses whitespace runs
        to single spaces. Actual file-name safety is applied later by `make_safe_filename`.
        """
        title = title.replace(break_string, " ")
        return re.sub(r"\s+", " ", title).strip()

    @staticmethod
    def _sanitize_title(title, break_string) -> str:
        # replace MAGIC_BREAK_STRING with a blank space
        # strip incase leading bank is missing
        title = title.replace(break_string, " ")
        sanitized_title = re.sub(r"[^\w\s]", "", title, flags=re.UNICODE)
        sanitized_title = re.sub(r"\s+", "_", sanitized_title.strip())
        return sanitized_title
