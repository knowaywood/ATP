"""Local PDF extraction helpers.

Despite the legacy filename, this module now uses `pypdf` rather than an OCR
service. It is intended for text-based PDFs.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def clean_pdf_text(text: str) -> str:
    """Light cleanup for extracted PDF text."""
    normalized = text.replace("-\n", "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in normalized.splitlines()]
    compact: list[str] = []
    for line in lines:
        if not line:
            if compact and compact[-1] != "":
                compact.append("")
            continue
        compact.append(" ".join(line.split()))
    return "\n".join(compact).strip()


def pdf_to_markdown(pdf_path: str, output_path: str | None = None) -> str:
    """Extract text from a PDF and optionally save it as Markdown."""
    reader = PdfReader(pdf_path)
    pages: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        page_text = clean_pdf_text(page.extract_text() or "")
        if page_text:
            pages.append(f"## Page {index}\n\n{page_text}")
    markdown = "\n\n".join(pages).strip()

    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(markdown, encoding="utf-8")

    return markdown
