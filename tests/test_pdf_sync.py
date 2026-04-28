from pathlib import Path

from atp.tools.OCR import pdf_to_markdown
from atp.tools.VectorStore import extract_pdf_dir_to_markdown


def test_pdf_to_markdown_extracts_text():
    markdown = pdf_to_markdown("tests/docs/pdf/test.pdf")
    assert isinstance(markdown, str)
    assert markdown.strip() != ""


def test_extract_pdf_dir_to_markdown_creates_md_file(tmp_path: Path):
    pdf_dir = tmp_path / "pdf"
    md_dir = tmp_path / "md"
    pdf_dir.mkdir()
    md_dir.mkdir()

    source_pdf = Path("tests/docs/pdf/test.pdf")
    target_pdf = pdf_dir / "test.pdf"
    target_pdf.write_bytes(source_pdf.read_bytes())

    created = extract_pdf_dir_to_markdown(str(pdf_dir), str(md_dir))

    assert created == ["test.md"]
    markdown_path = md_dir / "test.md"
    assert markdown_path.exists()
    assert markdown_path.read_text(encoding="utf-8").strip() != ""
