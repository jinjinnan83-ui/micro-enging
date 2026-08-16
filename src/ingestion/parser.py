"""Parse source files and attach Bandura metadata.

Supported formats: Markdown, plain text, PDF (via LlamaIndex).
Metadata is taken from YAML frontmatter first, then filename heuristics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SCHOOLS = ("社会认知理论", "自我效能", "观察学习", "道德疏离", "目标与自我调节", "干预与测量")

SCHOOL_ALIASES = {
    "bandura": "社会认知理论",
    "班杜拉": "社会认知理论",
    "self-efficacy": "自我效能",
    "自我效能": "自我效能",
    "observational": "观察学习",
    "观察学习": "观察学习",
    "moral": "道德疏离",
    "道德疏离": "道德疏离",
    "proximal": "目标与自我调节",
    "目标": "目标与自我调节",
}

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


@dataclass
class ParsedDocument:
    """A source document with domain metadata preserved for retrieval."""

    text: str
    source: str
    author: str = "unknown"
    school: str = "unknown"
    concepts: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def metadata(self) -> dict[str, Any]:
        return {
            "author": self.author,
            "school": self.school,
            "concepts": self.concepts,
            "source_document": self.source,
            **self.extra,
        }


def parse_document(path: str | Path) -> ParsedDocument:
    """Read one file and extract Author / School / Core Concepts / Source."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Source file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        text = _read_pdf(file_path)
    else:
        text = file_path.read_text(encoding="utf-8")

    frontmatter, body = _split_frontmatter(text)
    inferred = _infer_from_filename(file_path.stem)
    merged = {**inferred, **frontmatter}

    concepts = merged.get("concepts") or merged.get("core_concepts") or []
    if isinstance(concepts, str):
        concepts = [item.strip() for item in concepts.split(",") if item.strip()]

    school = str(merged.get("school") or "unknown")
    school = SCHOOL_ALIASES.get(school.lower(), school)

    return ParsedDocument(
        text=body.strip(),
        source=file_path.name,
        author=str(merged.get("author") or "unknown"),
        school=school,
        concepts=list(concepts),
        extra={
            k: v
            for k, v in merged.items()
            if k not in {"author", "school", "concepts", "core_concepts"}
        },
    )


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    raw = yaml.safe_load(match.group(1)) or {}
    if not isinstance(raw, dict):
        return {}, text[match.end() :]
    return raw, text[match.end() :]


def _infer_from_filename(stem: str) -> dict[str, Any]:
    lowered = stem.lower()
    for alias, school in SCHOOL_ALIASES.items():
        if alias in lowered or alias in stem:
            return {"school": school}
    return {}


def _read_pdf(path: Path) -> str:
    from llama_index.core import SimpleDirectoryReader

    documents = SimpleDirectoryReader(input_files=[str(path)]).load_data()
    return "\n\n".join(doc.text for doc in documents)
