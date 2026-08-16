"""Extract main-body chunks from data/raw and index them into the knowledge base."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_settings
from src.ingestion.chunking import chunk_source, iter_source_files


def build(raw_dir: Path | None = None) -> dict:
    settings = get_settings()
    source_dir = raw_dir or settings.data_raw_dir
    files = iter_source_files(source_dir)
    chunks = []
    report = []

    for path in files:
        try:
            file_chunks = chunk_source(path)
        except ValueError as exc:
            report.append({"file": path.name, "status": "skipped", "reason": str(exc), "chunks": 0})
            continue
        except Exception as exc:  # noqa: BLE001
            report.append({"file": path.name, "status": "error", "reason": str(exc), "chunks": 0})
            continue
        if not file_chunks:
            report.append({"file": path.name, "status": "empty", "reason": "no chunks", "chunks": 0})
            continue
        meta = file_chunks[0].metadata
        report.append(
            {
                "file": path.name,
                "status": "ok",
                "chunks": len(file_chunks),
                "author": meta.get("author"),
                "school": meta.get("school"),
                "core_concepts": meta.get("core_concepts"),
                "body_chars": sum(len(chunk.text) for chunk in file_chunks),
            }
        )
        chunks.extend(file_chunks)

    processed = settings.data_processed_dir
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "ingest_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    records = [chunk.to_dict() for chunk in chunks]
    (processed / "chunks.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in records),
        encoding="utf-8",
    )
    (processed / "bm25_corpus.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in records),
        encoding="utf-8",
    )

    index_result: dict = {"status": "skipped", "reason": "not attempted"}
    try:
        from src.retrieval.hybrid_engine import HybridEngine

        index_result = HybridEngine(settings).index_documents(chunks)
        index_result["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        index_result = {"status": "index_skipped", "reason": str(exc)}

    summary = {
        "files_seen": len(files),
        "files_ok": sum(1 for row in report if row["status"] == "ok"),
        "chunks": len(chunks),
        "index": index_result,
        "report_path": str(processed / "ingest_report.json"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    build()
