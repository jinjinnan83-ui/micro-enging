# 四理论知识库开发规范

本文件从 `/Users/isa/Desktop/精神分析` 的建库方法同步而来，作为弗洛伊德、班杜拉、罗杰斯、斯金纳四库的共同规矩。班杜拉库必须按此实现，后续两库不得另起炉灶。

这是理论检索与 Agent 对话脚手架，**不是心理治疗或诊断工具**。

## 1. 技术栈（与精神分析库一致）

| 层 | 规定 |
|---|---|
| 语言 | Python 3.10+ |
| RAG | LlamaIndex；切块优先 `SentenceSplitter` |
| 向量库 | Qdrant；Docker 优先，未启动时回落 `data/qdrant_storage/` |
| 嵌入 | `BAAI/bge-large-zh-v1.5`（1024 维，Cosine） |
| 重排 | `BAAI/bge-reranker-v2-m3` |
| 词法检索 | `rank_bm25` + 领域词最长匹配分词 |
| API | FastAPI：`GET /health` `POST /query` `POST /ingest` |
| 校验 | Pydantic v2；配置用 `pydantic-settings` |

禁止改成另一套框架（例如只用 LangChain、换 embedding 品牌）而不先改本规范。

## 2. 目录（四库同构）

```
data/raw/                 原始文献与理论笔记（PDF / TXT / MD）
data/processed/           chunks.jsonl、bm25_corpus.jsonl、ingest_report.json
data/qdrant_storage/      本地嵌入式 Qdrant
src/ingestion/            解析、元数据、切块
src/vectorstore/          Qdrant 集合管理
src/retrieval/            混合检索 + BGE rerank
src/api/                  FastAPI
scripts/                  build_knowledge_base.py
tests/                    切块、入库、API
.cursor/skills/<name>/    对话 Skill：先检索，后开口
```

集合名：精神分析 `psychoanalysis`；班杜拉 `bandura`；罗杰斯 `rogers`；斯金纳 `skinner`。

## 3. 源文件规范

理论笔记必须带 YAML frontmatter，字段与精神分析库相同：

```yaml
---
author: Albert Bandura
school: 自我效能
core_concepts:
  - 自我效能
  - 掌握经验
doc_type: theory_note
sources:
  - "Bandura, A. (1977). Self-efficacy: Toward a unifying theory of behavioral change. Psychological Review, 84(2), 191-215. https://doi.org/10.1037/0033-295X.84.2.191"
---
```

规则：

- 只写论证正文。摘要、目录、页眉、脚注、参考文献列表在入库时剔除。
- 主张必须能回到班杜拉原典或已核验综述；操作译法（微小胜利、案例匹配）不得写成 1977 年原词。
- 文件名：`{author_slug}_{topic}.md`，例如 `bandura_four_sources.md`。
- 扫描 PDF 抽不出正文则跳过，并在 `ingest_report.json` 记 `skipped`。

## 4. 元数据（每条切片必带）

与精神分析库相同的四元组，外加文件名：

| 字段 | 含义 | 班杜拉受控值 |
|---|---|---|
| `author` | 作者规范名 | `Albert Bandura` `Dale H. Schunk` `Ellen L. Usher` `Frank Pajares` `Elizabeth M. Ozer` |
| `school` | 理论模块（对应精神分析的流派） | `社会认知理论` `自我效能` `观察学习` `道德疏离` `目标与自我调节` `干预与测量` |
| `core_concepts` | 中英核心概念 | 见领域词表 |
| `source_document` / `file_name` | 源文件名 | 与 `data/raw/` 一致 |
| `chunk_index` | 切片序号 | 从 0 计 |

优先级：frontmatter → `src/ingestion/catalog.py` → 文件名启发式 → 正文匹配。不得只靠模型事后猜测。

## 5. 切块

- `chunk_size=512`，`chunk_overlap=80`
- 先去 front matter / 摘要 / 参考文献，再切
- 优先按段落窗口，避免把一个机制句与其限定条件切开
- 每条 chunk 必须带上节全部元数据

## 6. 检索

- 对话默认：**BM25**（快）。脚本：`.cursor/skills/.../scripts/query_kb.py`
- 需要语义贴近时才用：**稠密向量 + BM25 + RRF + BGE 重排**，Top-20 → 返回 3–5 条
- 分词必须最长匹配领域词，例如 `自我效能` 不可拆成 `自我` + `效能`
- 查询扩展必须带中英别名，例如 `掌握经验` ↔ `mastery experience` ↔ `performance accomplishments`
- 可用 `--school` / `--author` 过滤
- 输出 JSON：`hits[].text|score|source|author|school|core_concepts|file_name`

## 7. Agent Skill 接线

精神分析库的规矩是：**先调库，再开口**。班杜拉教练 Skill 必须同样：

1. 从用户原话压一条检索句，再写一条理论改写句，至少查 1 次，最好 2 次
2. 只用 1–2 条最贴的命中，不要罗列
3. 再按班杜拉教练四步回复（效能重构、微步骤、归因、榜样）
4. 高风险（自伤/伤人）先转安全，不继续布置掌握任务

## 8. 入库与测试

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m scripts.build_knowledge_base
.venv/bin/python .cursor/skills/bandura-self-efficacy-coach/scripts/query_kb.py "掌握经验" --top-n 5
python -m pytest tests -q
```

无 embedding 环境时：必须仍写出 `data/processed/chunks.jsonl`，BM25 对话可用；向量索引记入报告为 `index_skipped`，不得假装已建密向量。

## 9. 质量门

- 切块测试：正文保留，摘要/参考文献/页码被丢掉；元数据含 author/school/concepts
- 检索测试：`自我效能` 能命中 1977 机制切片；`--school 观察学习` 不串到道德疏离
- API 测试：`/query` 返回 3–5 条，payload 含 metadata
- 不把未核验的百科原句写成 Bandura 直接引文
