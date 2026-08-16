"""LlamaIndex chunking for Bandura PDF / TXT / MD sources.

Only the article body is kept. Front matter, TOC, abstracts, running
headers, footnotes, and reference lists are dropped so mechanism sentences and boundary conditions is not mixed with bibliographic noise.

Every chunk carries: author, school, core_concepts, file_name.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import get_settings
from src.ingestion.chunker import TextChunk

SplitterName = Literal["sentence", "semantic"]

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}

SCHOOL_LABELS = {
    "社会认知理论": "社会认知理论",
    "social cognitive": "社会认知理论",
    "social learning": "观察学习",
    "自我效能": "自我效能",
    "self-efficacy": "自我效能",
    "self efficacy": "自我效能",
    "观察学习": "观察学习",
    "observational learning": "观察学习",
    "modeling": "观察学习",
    "道德疏离": "道德疏离",
    "moral disengagement": "道德疏离",
    "目标与自我调节": "目标与自我调节",
    "proximal": "目标与自我调节",
    "self-regulation": "目标与自我调节",
    "干预与测量": "干预与测量",
    "guided mastery": "干预与测量",
    "bandura": "社会认知理论",
    "班杜拉": "社会认知理论",
}

AUTHOR_ALIASES = {
    "班杜拉": "Albert Bandura",
    "bandura": "Albert Bandura",
    "albert bandura": "Albert Bandura",
    "schunk": "Dale H. Schunk",
    "dale h. schunk": "Dale H. Schunk",
    "dale schunk": "Dale H. Schunk",
    "usher": "Ellen L. Usher",
    "ellen l. usher": "Ellen L. Usher",
    "pajares": "Frank Pajares",
    "frank pajares": "Frank Pajares",
    "ozer": "Elizabeth M. Ozer",
    "artino": "Anthony R. Artino Jr.",
    "ashford": "Stefanie Ashford",
    "egele": "Viktoria S. Egele",
    "panadero": "Ernesto Panadero",
    "zakariya": "Yusuf F. Zakariya",
}

CONCEPT_LEXICON = (
    "自我效能", "效能预期", "结果预期", "掌握经验", "亲身掌握", "表现成就",
    "替代经验", "观察学习", "社会劝说", "言语劝说", "生理状态", "情绪状态",
    "引导掌握", "参与示范", "近端目标", "远端目标", "三元交互", "相互决定",
    "人类能动性", "道德疏离", "应对榜样", "掌握榜样", "同伴榜样", "归因反馈",
    "self-efficacy", "outcome expectancy", "mastery experience",
    "performance accomplishments", "vicarious experience", "social persuasion",
    "verbal persuasion", "physiological states", "guided mastery",
    "participant modeling", "proximal goal", "reciprocal determinism",
    "moral disengagement", "coping model", "peer model", "human agency",
)

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
PAGE_LINE_RE = re.compile(r"^(?:[-—–]\s*)?(?:第\s*)?\d{1,4}(?:\s*页)?(?:\s*[-—–])?$")
RUNNING_HEADER_RE = re.compile(
    r"^(Psychological Review|American Psychologist|Journal of Personality|"
    r"Educational Psychologist|Review of Educational Research|"
    r"Annual Review of Psychology|Cognitive Therapy and Research|"
    r"社会认知|自我效能研究|班杜拉文选).*$"
)
NOISE_LINE_RE = re.compile(
    r"^(copyright|©|isbn|issn|doi\s*:|all rights reserved|版权所有|"
    r"作者简介|通讯作者|基金项目|收稿日期|修回日期|"
    r"to cite this article|to link to this article|published online|"
    r"submit your article|view related articles|view crossmark|"
    r"full terms & conditions|article views:|citing articles:|"
    r"page \d+ of \d+|received:|accepted:|correspondence:|"
    r"中图分类号|文献标识码|文章编号|http[s]?://)",
    re.IGNORECASE,
)
BODY_END_RE = re.compile(
    r"^(参考文献|参考书目|注释|注\s*释|附录|致谢|作者简介|"
    r"notes|endnotes|references|bibliography|acknowledgements?)"
    r"(?:\s*$|\s*[.·…．])",
    re.IGNORECASE,
)
BODY_START_RE = re.compile(
    r"^(?:[一二三四五六七八九十]+、|[0-9]+\.\s*|第[一二三四五六七八九十]+[章节]|引言|导论|正文|Introduction)\b"
)
TOC_LINE_RE = re.compile(r"[.·…．]{3,}\s*\d+\s*$")
FOOTNOTE_LINE_RE = re.compile(r"^\[\d+\]\s+\S+")
INLINE_FOOTNOTE_RE = re.compile(r"\[\d+\]")


@dataclass
class DocumentMeta:
    author: str = "unknown"
    school: str = "社会认知理论"
    core_concepts: list[str] = field(default_factory=list)
    file_name: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "author": self.author,
            "school": self.school,
            "core_concepts": self.core_concepts,
            "file_name": self.file_name,
            "source_document": self.file_name,
        }


def load_source(path: str | Path) -> str:
    """Read a PDF, TXT, or Markdown file as raw text."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Source file not found: {file_path}")
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Only PDF/TXT/MD are supported, got: {suffix}")
    if suffix == ".pdf":
        return _read_pdf(file_path)
    return file_path.read_text(encoding="utf-8")


def extract_main_body(raw_text: str) -> str:
    """Keep the central article body; drop everything around it."""
    _, without_frontmatter = _split_frontmatter(raw_text)
    text = _normalize_layout(without_frontmatter)
    text = _cut_reference_tail(text)
    text = _cut_front_matter_head(text)
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line and not _is_noise_line(line)]
    lines = _drop_repeated_headers(lines)

    start = _find_body_start(lines)
    end = _find_body_end(lines, start)
    body_lines = lines[start:end]
    body_lines = [INLINE_FOOTNOTE_RE.sub("", line).strip() for line in body_lines]
    body_lines = [line for line in body_lines if line and not FOOTNOTE_LINE_RE.match(line)]
    return _collapse_blank_paragraphs("\n".join(body_lines))


def extract_metadata(path: str | Path, raw_text: str, body: str) -> DocumentMeta:
    """Build Author / School / Core Concepts / file_name from file + body."""
    from src.ingestion.catalog import lookup_catalog

    file_path = Path(path)
    catalog = lookup_catalog(file_path.name)
    frontmatter, _ = _split_frontmatter(raw_text)
    keywords = _extract_keywords(raw_text)
    filename_hits = _infer_from_filename(file_path.stem)

    author = (
        _normalize_author(frontmatter.get("author"))
        or _normalize_author(catalog.get("author"))
        or _extract_author_line(raw_text)
        or filename_hits.get("author")
        or _infer_author_from_text(f"{file_path.stem}\n{body}")
        or "unknown"
    )
    school = (
        _normalize_school(frontmatter.get("school"))
        or _normalize_school(catalog.get("school"))
        or filename_hits.get("school")
        or _infer_school(f"{author} {file_path.stem} {body}")
        or "社会认知理论"
    )
    concepts = _merge_concepts(
        frontmatter.get("core_concepts") or frontmatter.get("concepts"),
        catalog.get("core_concepts"),
        keywords,
        _match_concepts(body),
    )
    return DocumentMeta(
        author=author,
        school=school,
        core_concepts=concepts,
        file_name=file_path.name,
    )


def chunk_source(
    path: str | Path,
    splitter: SplitterName = "sentence",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[TextChunk]:
    """Parse one source, keep the body, then split with LlamaIndex."""
    settings = get_settings()
    file_path = Path(path)
    raw = load_source(file_path)
    body = extract_main_body(raw)
    if not body:
        raise ValueError(f"No main body extracted from: {file_path}")
    meta = extract_metadata(file_path, raw, body)
    nodes = _split_with_llamaindex(
        body,
        metadata=meta.as_dict(),
        splitter=splitter,
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
    )
    chunks: list[TextChunk] = []
    for index, node in enumerate(nodes):
        payload = {**meta.as_dict(), "chunk_index": index}
        chunks.append(TextChunk(text=node, metadata=payload, chunk_index=index))
    return chunks


def iter_source_files(directory: str | Path | None = None) -> list[Path]:
    root = Path(directory) if directory else get_settings().data_raw_dir
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and " (1)" not in path.stem
    )


def _split_with_llamaindex(
    body: str,
    metadata: dict[str, Any],
    splitter: SplitterName,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    try:
        from llama_index.core import Document

        document = Document(text=body, metadata=metadata)
        parser = _build_splitter(splitter, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nodes = parser.get_nodes_from_documents([document])
        return [node.get_content().strip() for node in nodes if node.get_content().strip()]
    except ImportError:
        from src.ingestion.chunker import _window_paragraphs

        paragraphs = [part.strip() for part in body.split("\n\n") if part.strip()]
        return _window_paragraphs(paragraphs, size=chunk_size, overlap=chunk_overlap)


def _build_splitter(splitter: SplitterName, chunk_size: int, chunk_overlap: int):
    from llama_index.core.node_parser import SentenceSplitter

    if splitter == "sentence":
        return SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            paragraph_separator="\n\n",
        )
    try:
        from llama_index.core.node_parser import SemanticSplitterNodeParser
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        settings = get_settings()
        embed_model = HuggingFaceEmbedding(model_name=settings.embedding_model)
        return SemanticSplitterNodeParser(
            embed_model=embed_model,
            buffer_size=1,
            breakpoint_percentile_threshold=90,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[chunking] SemanticSplitter unavailable ({exc}); using SentenceSplitter.")
        return SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if _looks_like_article_text(text):
        return text
    raise ValueError(f"No extractable article text in PDF (scanned or empty): {path.name}")


def _looks_like_article_text(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 200:
        return False
    if stripped.startswith("%PDF") or "endobj" in stripped[:800]:
        return False
    letters = sum(ch.isalpha() or ("\u4e00" <= ch <= "\u9fff") for ch in stripped)
    return letters / len(stripped) > 0.25


def _cut_reference_tail(text: str) -> str:
    return re.split(
        r"\n\s*(参考文献|参考书目|References|REFERENCES|Bibliography|BIBLIOGRAPHY)\s*\n",
        text,
        maxsplit=1,
    )[0]


def _cut_front_matter_head(text: str) -> str:
    keyword = re.search(r"(?:关键词|Keywords?|Key\s*Words)\s*[：:]\s*.+", text, re.I)
    if keyword:
        return text[keyword.end() :]
    abstract = re.search(r"(?:摘要|Abstract|ABSTRACT)\s*[：:]?\s*", text)
    if not abstract:
        return text
    rest = text[abstract.end() :]
    parts = re.split(r"\n\s*\n", rest, maxsplit=1)
    return parts[1] if len(parts) > 1 else rest


def _drop_repeated_headers(lines: list[str]) -> list[str]:
    counts = Counter(lines)
    return [line for line in lines if counts[line] < 3 or len(line) < 40]


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    try:
        import yaml

        raw = yaml.safe_load(match.group(1)) or {}
    except Exception:
        raw = {}
    if not isinstance(raw, dict):
        return {}, text[match.end() :]
    return raw, text[match.end() :]


def _normalize_layout(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def _is_toc_line(line: str) -> bool:
    return line in {"目录", "目 录", "Contents"} or bool(TOC_LINE_RE.search(line))


def _is_noise_line(line: str) -> bool:
    return bool(
        PAGE_LINE_RE.match(line)
        or RUNNING_HEADER_RE.match(line)
        or NOISE_LINE_RE.match(line)
        or FOOTNOTE_LINE_RE.match(line)
        or _is_toc_line(line)
    )


def _find_body_start(lines: list[str]) -> int:
    hint = 0
    for index, line in enumerate(lines):
        if re.match(r"^(关键词|keywords?|key\s*words)\s*[：:]?", line, re.I):
            hint = index + 1
            break
        if re.match(r"^(摘要|abstract)\s*[：:]?", line, re.I):
            hint = index + 1
    for index, line in enumerate(lines[hint:], start=hint):
        if _is_toc_line(line) or PAGE_LINE_RE.match(line):
            continue
        if BODY_START_RE.match(line) or len(line) > 40:
            return index
    return hint


def _find_body_end(lines: list[str], start: int) -> int:
    for index, line in enumerate(lines[start:], start=start):
        # PDF line wraps can start with "references." mid-sentence; only
        # treat short heading-like lines as the bibliography boundary.
        if (
            BODY_END_RE.match(line)
            and len(line) <= 40
            and not TOC_LINE_RE.search(line)
        ):
            return index
    return len(lines)


def _collapse_blank_paragraphs(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _extract_keywords(raw_text: str) -> list[str]:
    match = re.search(r"^(?:关键词|keywords?|key\s*words)\s*[：:]\s*(.+)$", raw_text, re.I | re.M)
    if not match:
        return []
    return [item.strip() for item in re.split(r"[，,;；/|]", match.group(1)) if item.strip()]


def _extract_author_line(raw_text: str) -> str | None:
    cite = re.search(r"To cite this article:\s*(.+?)\s*\(\d{4}\)", raw_text, re.I)
    if cite:
        return _normalize_author(cite.group(1).split(",")[0])
    match = re.search(r"^(?:作者|author)\s*[：:]\s*(.+)$", raw_text, re.I | re.M)
    if not match:
        return None
    return _normalize_author(match.group(1).split("（")[0].split("(")[0])


def _infer_from_filename(stem: str) -> dict[str, str]:
    lowered = stem.lower()
    found: dict[str, str] = {}
    for alias, author in AUTHOR_ALIASES.items():
        if alias in lowered or alias in stem:
            found["author"] = author
            break
    school = _infer_school(stem)
    if school:
        found["school"] = school
    return found


def _infer_author_from_text(text: str) -> str | None:
    lowered = text.lower()
    for alias, author in AUTHOR_ALIASES.items():
        if alias in lowered or alias in text:
            return author
    return None


def _infer_school(text: str) -> str | None:
    lowered = text.lower()
    for alias, school in SCHOOL_LABELS.items():
        if alias in lowered or alias in text:
            return school
    return None


def _normalize_author(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    return AUTHOR_ALIASES.get(text.lower(), text) or None


def _normalize_school(value: Any) -> str | None:
    if not value:
        return None
    raw = str(value).strip()
    return SCHOOL_LABELS.get(raw.lower()) or SCHOOL_LABELS.get(raw) or raw


def _match_concepts(body: str) -> list[str]:
    lowered = body.lower()
    return [term for term in CONCEPT_LEXICON if term.lower() in lowered or term in body]


def _merge_concepts(*groups: Iterable[Any] | Any) -> list[str]:
    seen: list[str] = []
    for group in groups:
        values = group if isinstance(group, (list, tuple, set)) else [group]
        for item in values:
            if not item:
                continue
            if isinstance(item, str) and ("," in item or "，" in item):
                parts = [part.strip() for part in re.split(r"[，,]", item) if part.strip()]
            else:
                parts = [str(item).strip()]
            for part in parts:
                if part and part not in seen:
                    seen.append(part)
    return seen


def _default_sample_path() -> Path:
    raw_dir = get_settings().data_raw_dir
    preferred = raw_dir / "sample_self_efficacy.txt"
    if preferred.exists():
        return preferred
    files = iter_source_files(raw_dir)
    if files:
        return files[0]
    raise FileNotFoundError(f"No PDF/TXT sample under {raw_dir}")


def preview_chunks(path: str | Path, splitter: SplitterName = "sentence") -> None:
    chunks = chunk_source(path, splitter=splitter)
    print(f"file: {path}")
    print(f"chunks: {len(chunks)}")
    print("-" * 60)
    for chunk in chunks:
        meta = chunk.metadata
        print(f"[chunk {chunk.chunk_index}] {len(chunk.text)} chars")
        print(f"  author         : {meta.get('author')}")
        print(f"  school         : {meta.get('school')}")
        print(f"  core_concepts  : {meta.get('core_concepts')}")
        print(f"  file_name      : {meta.get('file_name')}")
        print(f"  text           : {chunk.text}")
        print("-" * 60)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Preview Bandura PDF/TXT chunks.")
    parser.add_argument("--path", default=None)
    parser.add_argument("--splitter", choices=("sentence", "semantic"), default="sentence")
    args = parser.parse_args(argv)
    target = Path(args.path) if args.path else _default_sample_path()
    preview_chunks(target, splitter=args.splitter)


if __name__ == "__main__":
    main()
