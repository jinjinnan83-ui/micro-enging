from pathlib import Path

import pytest

from src.ingestion.chunking import extract_main_body, extract_metadata, load_source

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "raw" / "sample_repression.txt"


def test_extract_main_body_drops_non_body_sections() -> None:
    raw = load_source(SAMPLE)
    body = extract_main_body(raw)

    assert "无意识不是一个可以随意打开的储藏室" in body
    assert "移情现场往往最先暴露压抑失败后的返回" in body
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

    assert meta.author == "Sigmund Freud"
    assert meta.school == "精神分析"
    assert meta.file_name == "sample_repression.txt"
    assert "压抑" in meta.core_concepts
    assert "无意识" in meta.core_concepts
    assert "移情" in meta.core_concepts

def test_chunk_source_uses_llamaindex_and_keeps_metadata() -> None:
    pytest.importorskip("llama_index.core")
    from src.ingestion.chunking import chunk_source

    chunks = chunk_source(SAMPLE, splitter="sentence", chunk_size=180, chunk_overlap=40)

    assert chunks
    for chunk in chunks:
        assert chunk.metadata["author"] == "Sigmund Freud"
        assert chunk.metadata["school"] == "精神分析"
        assert chunk.metadata["file_name"] == "sample_repression.txt"
        assert "压抑" in chunk.metadata["core_concepts"]
        assert "参考文献" not in chunk.text
        assert "摘要" not in chunk.text
