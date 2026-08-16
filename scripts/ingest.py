"""CLI: ingest raw psychoanalytic files into Qdrant.

Usage (from project root):
    python -m scripts.ingest
    python -m scripts.ingest --path data/raw/freud_unconscious.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_settings
from src.ingestion.pipeline import IngestionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest psychoanalysis sources into Qdrant.")
    parser.add_argument("--path", default=None, help="File or directory. Defaults to data/raw/")
    args = parser.parse_args()

    settings = get_settings()
    source = Path(args.path) if args.path else settings.data_raw_dir
    chunks = IngestionPipeline(settings).ingest_path(source)
    print(f"Ingested {len(chunks)} chunks into collection `{settings.qdrant_collection}`.")


if __name__ == "__main__":
    main()
