from pathlib import Path

from src.ingestion.chunker import chunk_document
from src.ingestion.parser import parse_document

FIXTURE = Path(__file__).resolve().parent.parent / "data" / "raw" / "freud_unconscious.md"


def test_parse_document_extracts_psychoanalytic_metadata() -> None:
    parsed = parse_document(FIXTURE)

    assert parsed.author == "Sigmund Freud"
    assert parsed.school == "Freud"
    assert "unconscious" in parsed.concepts
    assert parsed.source == "freud_unconscious.md"
    assert "压抑" in parsed.text


def test_filename_heuristic_infers_lacan_school(tmp_path: Path) -> None:
    path = tmp_path / "lacan_notes.txt"
    path.write_text("能指先于主体。", encoding="utf-8")

    parsed = parse_document(path)

    assert parsed.school == "Lacan"
    assert parsed.author == "Lacan"


def test_chunk_document_attaches_metadata_to_every_chunk() -> None:
    parsed = parse_document(FIXTURE)
    chunks = chunk_document(parsed, chunk_size=80, chunk_overlap=20)

    assert chunks
    for chunk in chunks:
        assert chunk.metadata["author"] == "Sigmund Freud"
        assert chunk.metadata["school"] == "Freud"
        assert chunk.metadata["source_document"] == "freud_unconscious.md"
        assert "unconscious" in chunk.metadata["concepts"]
        assert chunk.text
