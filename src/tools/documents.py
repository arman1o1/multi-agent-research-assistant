"""Document parsing tools: web pages and PDFs."""

from __future__ import annotations

import logging

from src.config import get_settings

logger = logging.getLogger(__name__)


def read_webpage(url: str) -> str:
    """Extract clean text content from a web page.

    Args:
        url: The URL of the web page to read.

    Returns:
        Extracted text content from the page, truncated to the configured limit.
    """
    try:
        import trafilatura

        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            return f"Error: Could not fetch URL: {url}"

        text = trafilatura.extract(
            downloaded,
            include_links=True,
            include_tables=True,
            favor_recall=True,
        )

        if not text:
            return f"Error: Could not extract text from: {url}"

        max_chars = get_settings().max_content_chars
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[... truncated at {max_chars} characters]"

        return text

    except Exception as e:
        logger.error(f"Failed to read webpage {url}: {e}")
        return f"Error reading webpage: {e}"


def read_pdf(file_path: str) -> str:
    """Extract text content from a PDF file.

    Args:
        file_path: Path to the PDF file on disk.

    Returns:
        Extracted text content from the PDF, truncated to the configured limit.
    """
    try:
        from pathlib import Path

        # Prevent path traversal by validating containment in workspace
        workspace_dir = Path(__file__).parent.parent.parent.resolve()
        resolved_path = Path(file_path).resolve()
        try:
            resolved_path.relative_to(workspace_dir)
        except ValueError:
            return "Error: Access denied. File path must be within the project workspace."

        import pymupdf

        doc = pymupdf.open(str(resolved_path))
        pages = []

        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                pages.append(f"--- Page {page_num + 1} ---\n{text}")

        doc.close()

        if not pages:
            return f"Error: No text content found in PDF: {file_path}"

        full_text = "\n\n".join(pages)
        max_chars = get_settings().max_content_chars
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars] + f"\n\n[... truncated at {max_chars} characters]"

        return full_text

    except Exception as e:
        logger.error(f"Failed to read PDF {file_path}: {e}")
        return f"Error reading PDF: {e}"
