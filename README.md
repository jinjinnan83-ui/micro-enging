# 班杜拉社会认知知识库与教练 Skill

> **本分支内容为班杜拉（Albert Bandura）。**  
> 微距黑客松 Pro 第五组（[micro-enging](https://github.com/jinjinnan83-ui/micro-enging)）  
> 检出本分支：`git checkout bandura`

基于 Python + LlamaIndex 的班杜拉 RAG 知识库，加上可供 Cursor Agent 调用的自我效能教练 Skill。建库方法与同仓库精神分析分支对齐：解析切块、Qdrant 向量检索、BM25 混合检索、BGE 重排、FastAPI。

这是理论检索与教练对话脚手架，**不是心理治疗或诊断工具**。

## 两块东西

| 模块 | 路径 | 做什么 |
|---|---|---|
| 知识库 | `data/` `src/` `scripts/` | 班杜拉原文与后续研究的切块检索 |
| 教练 Skill | `.cursor/skills/bandura-self-efficacy-coach/` | 人设「正义赋能者」：先调库，再按四源机制回话 |

## 教练 Skill：正义赋能者

热血行动派导师。光明、直爽、永不言弃；乐观必须有步骤，禁止无脑打鸡血。

理论四步（每条回复都要出现）：

1. **效能重构**：你不是不行，只是缺乏一套行之有效的路径。
2. **微小胜利**：只给一个 5–15 分钟、做完即有证据的微步骤。
3. **归因调整**：「我太差了」改成暂时被困难限制发挥，把掌控权抢回来。
4. **替代经验**：起点相似的人如何走前一两步，不用名人逆袭。

在 Cursor 打开本仓库后，用户谈困惑、焦虑、拖延、「做不到」时触发。出现自伤/伤人等风险时，停止教练流程，转向现实支持（如 `12356`，紧急时 `120` / `110`）。

## 当前库容

已切 **2606** 条（`data/processed/chunks.jsonl`）。对话默认走 BM25。稠密向量需安装 `sentence-transformers` 后重跑入库。

| 模块 | 切片 | 代表内容 |
|---|---:|---|
| 自我效能 | 379 | 四源、Artino 教学转译、Schunk 成就行为 |
| 社会认知理论 | 928 | Bandura 1978 相互决定、1982 效能机制、2001 能动性视角 |
| 观察学习 | 179 | 示范、自我示范、同伴/应对榜样 |
| 目标与自我调节 | 632 | 近端目标、社会起源、自我调节综述 |
| 干预与测量 | 487 | 四源相对权重、效能改变元分析 |
| 道德疏离 | 1 | 目前仅有理论笔记，原文 PDF 待补 |

`school` 受控值：`社会认知理论` `自我效能` `观察学习` `道德疏离` `目标与自我调节` `干预与测量`

接入记录见 [literature/oa-ingest.md](literature/oa-ingest.md)。

## 目录

```
data/raw/                 理论笔记与原始文献
data/processed/           chunks.jsonl、BM25 语料、入库报告
src/ingestion/            解析、元数据、切块
src/vectorstore/          Qdrant 集合管理
src/retrieval/            混合检索 + BGE rerank
src/api/                  FastAPI：/health /query /ingest
scripts/                  入库脚本
literature/               题录与接入记录
.cursor/skills/bandura-self-efficacy-coach/
```

每条切片带 `author`、`school`、`core_concepts`、`source_document`。

## 环境

- Python 3.10+
- 可选 Docker（远程 Qdrant）；未启动时回落到本地存储

```bash
git clone -b bandura https://github.com/jinjinnan83-ui/micro-enging.git
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

仓库已包含切好的切片，克隆后对话检索可直接用。新增 `data/raw/` 文件后：

```bash
python -m scripts.build_knowledge_base
```

只预览切块：

```bash
python -m src.ingestion.chunking --path data/raw/bandura_self_efficacy_unifying.md
```

## 检索

默认 BM25：

```bash
.venv/bin/python .cursor/skills/bandura-self-efficacy-coach/scripts/query_kb.py "掌握经验" --top-n 5
```

按理论模块过滤：

```bash
.venv/bin/python .cursor/skills/bandura-self-efficacy-coach/scripts/query_kb.py "应对榜样" --school 观察学习
```

混合检索（需先装 embedding 并建向量索引）：

```bash
python -m src.retrieval.hybrid_engine query "自我效能 四源" --top-n 5
```

## API

```bash
uvicorn src.api.main:app --reload --port 8001
```

- `GET /health`
- `POST /query`  body: `{"query": "近端目标", "school": "目标与自我调节", "top_n": 5}`
- `POST /ingest`  body: `{"path": "data/raw"}`

## 测试

```bash
.venv/bin/python -m pytest tests -q
```

## 技术栈

LlamaIndex · Qdrant · FastAPI · `bge-large-zh-v1.5` · `bge-reranker-v2-m3` · rank_bm25
