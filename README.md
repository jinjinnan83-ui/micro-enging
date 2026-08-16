# 精神分析知识库

微距黑客松 Pro 第五组（[micro-enging](https://github.com/jinjinnan83-ui/micro-enging)）

基于 Python + LlamaIndex 的精神分析 RAG 知识库：解析切块、Qdrant 向量检索、BM25 混合检索、BGE 重排，以及可供 Agent 调用的 FastAPI 与 Cursor Skill。

这是理论检索与对话脚手架，**不是心理治疗或诊断工具**。

## 当前库容

本地已索引 **645** 条切片（`data/processed/chunks.jsonl` 与 `data/qdrant_storage` 一致）。

| 流派 | 切片约数 | 代表内容 |
|---|---:|---|
| 精神分析 | 251 | 弗洛伊德结构模型、俄狄浦斯、移情；霍妮基本焦虑；自我心理学；动力取向疗效综述 |
| 客体关系 | 127 | 克莱因位置与投射性认同；温尼科特抱持/过渡客体；TFP |
| 自体心理学 | 61 | 科胡特自体客体；抑郁症视角综述 |
| 关系精神分析 | 56 | 关系学派综述；Sullivan 人际/主体间 |
| 个体心理学 | 49 | 阿德勒个体心理学 |
| 心智化 | 48 | MBT 与边缘型人格综述 |
| 分析心理学 | 48 | 荣格集体无意识、原型；荣格治疗疗效研究 |
| 拉康派 | 5 | 镜像阶段、三界、父亲之名 |

原始文献在 `data/raw/`（PDF + 理论笔记）。两份扫描 PDF 抽不出正文，未入库。

## 目录

```
data/raw/                 原始文献与理论笔记
data/processed/           chunks.jsonl、BM25 语料、入库报告
data/qdrant_storage/      本地嵌入式 Qdrant（645 向量）
src/ingestion/            解析、元数据、切块
src/vectorstore/          Qdrant 集合管理
src/retrieval/            混合检索 + BGE rerank
src/api/                  FastAPI：/health /query /ingest
scripts/                  入库与构建脚本
.cursor/skills/psychoanalysis-dialogue/   对话 Skill（先检索再开口）
```

每条切片带 `author`、`school`、`core_concepts`、`source_document`。

## 环境

- Python 3.10+
- 可选：Docker（远程 Qdrant）。未启动 Docker 时，自动回落到 `data/qdrant_storage/`

```bash
git clone https://github.com/jinjinnan83-ui/micro-enging.git
cd micro-enging
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

可选启动独立 Qdrant：

```bash
docker compose up -d
```

## 入库

仓库已包含切好的切片和本地向量库，克隆后一般不必重跑。新增 `data/raw/` 文件后再构建：

```bash
python -m scripts.build_knowledge_base
```

只预览切块：

```bash
python -m src.ingestion.chunking --path data/raw/freud_structural_model.md
```

## 检索

快速 BM25（对话默认）：

```bash
.venv/bin/python .cursor/skills/psychoanalysis-dialogue/scripts/query_kb.py "投射性认同" --top-n 5
```

按流派过滤：

```bash
.venv/bin/python .cursor/skills/psychoanalysis-dialogue/scripts/query_kb.py "父亲之名" --school 拉康派
```

`--school` 可选：`精神分析` `拉康派` `客体关系` `分析心理学` `自体心理学` `关系精神分析` `个体心理学` `心智化`

混合检索（稠密向量 + BM25 + BGE 重排，较慢）：

```bash
python -m src.retrieval.hybrid_engine query "死本能" --top-n 5
# 或
.venv/bin/python .cursor/skills/psychoanalysis-dialogue/scripts/query_kb.py "死本能" --hybrid
```

## API

```bash
uvicorn src.api.main:app --reload --port 8000
```

- `GET /health`
- `POST /query`  body: `{"query": "...", "school": "拉康派", "top_n": 5, "rerank": true}`
- `POST /ingest`  body: `{"path": "data/raw"}`（可选）

## 对话 Skill

`.cursor/skills/psychoanalysis-dialogue/`：西格蒙德·弗洛伊德——兼具精神分析底蕴与顶级执行官冷峻气场的深度分析师。先调库，再极简切开显意、指出被挡住的结构。不骂人、不说教、不说「你应该」。

在 Cursor 打开本仓库后，直接说梦、关系重复、移情或「从精神分析听听我」即可触发。出现自伤/伤人等风险时，应先转向现实支持（如 `12356`，紧急时 `120`/`110`），不要继续常规解剖。

## 测试

```bash
.venv/bin/python -m pytest tests -q
```

## 技术栈

LlamaIndex · Qdrant · FastAPI · `bge-large-zh-v1.5` · `bge-reranker-v2-m3` · rank_bm25
