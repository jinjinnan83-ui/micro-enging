from pathlib import Path

from src.ingestion.chunking import extract_main_body, extract_metadata, load_source

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "raw" / "sample_self_efficacy.txt"


def test_extract_main_body_drops_non_body_sections() -> None:
    raw = load_source(SAMPLE)
    body = extract_main_body(raw)

    assert "自我效能不是一般自信" in body
    assert "效能预期决定应对行为" in body
    assert "摘要：" not in body
    assert "关键词：" not in body
    assert "目录" not in body
    assert "参考文献" not in body
    assert "作者简介" not in body
    assert "ISBN" not in body
    assert "[1]" not in body
    assert "第 3 页" not in body


def test_extract_metadata_uses_author_school_concepts_filename() -> None:
    raw = load_source(SAMPLE)
    body = extract_main_body(raw)
    meta = extract_metadata(SAMPLE, raw, body)

    assert meta.author == "Albert Bandura"
    assert meta.school == "自我效能"
    assert meta.file_name == "sample_self_efficacy.txt"
    assert "自我效能" in meta.core_concepts
    assert "掌握经验" in meta.core_concepts


def test_chunk_source_keeps_metadata() -> None:
    from src.ingestion.chunking import chunk_source

    chunks = chunk_source(SAMPLE, splitter="sentence", chunk_size=180, chunk_overlap=40)

    assert chunks
    for chunk in chunks:
        assert chunk.metadata["author"] == "Albert Bandura"
        assert chunk.metadata["school"] == "自我效能"
        assert chunk.metadata["file_name"] == "sample_self_efficacy.txt"
        assert "自我效能" in chunk.metadata["core_concepts"]
        assert "参考文献" not in chunk.text
        assert "摘要" not in chunk.text


def test_long_paragraph_is_windowed() -> None:
    from src.ingestion.chunker import _window_paragraphs

    sentence = "Mastery experience is the strongest source of self-efficacy. "
    text = sentence * 40
    windows = _window_paragraphs([text], size=180, overlap=40)
    assert len(windows) > 1
    assert all(len(window) <= 220 for window in windows)
