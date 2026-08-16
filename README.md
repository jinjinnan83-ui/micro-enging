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
.cursor/skills/psychoanalysis-dialogue/   对话 Skill（基础检索；必要时追加解释性检索）
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
- `POST /dialogue`  body: `{"letter": "...", "include_trace": false}`
- `POST /ingest`  body: `{"path": "data/raw"}`（可选）

### 动力学回信 Agent

`/dialogue` 不是模板拼接。每封来信通常会经过两次独立的 DeepSeek 调用：

1. 完成安全预检，为所有非风险来信生成 1–2 条基础检索词；若检测到明确或潜在的知识理解需求，再生成 1–2 条解释性检索词；同时判断用户是否疲惫或想暂停交流。
2. 基础检索结果用于约束精神动力学理解；解释性检索结果用于寻找能安慰、减少自责或澄清问题的知识点。
3. 将原信、完整 `SKILL.md` 和两类检索上下文送入最终写作 prompt，最终只生成一封中文回信。

第一步不是心理分类器，不输出动力学剖析，也不选择多个 skill。第二步直接以 `SKILL.md` 作为理解与文风规范。若任一步返回的 JSON 不符合 Pydantic contract，服务仅追加一次格式修复调用；这不是内容 critic。

常规响应返回 `reply` 和可选的 `interaction`。调试或评估时设置 `include_trace: true`，可分别查看 `grounding_knowledge`、`explanatory_knowledge`、两组检索词、交互判断和 token usage。只有明确表达当前自杀/自残意图、计划或行动的来信才会暂停两类检索并进入安全回复；“不想死也不想活”、空虚或无望等被动表达仍走普通回信流程。

### 暂停交流识别信号

当来信明确表示「很累」「撑不动了」「想停一下」「不想继续说」等当下疲惫或退出语言的状态时，响应会额外包含最小化的 `interaction` 识别信号，并在回信末尾追加固定提示语。后端不选择训练类型，也不提供呼吸方法、冥想步骤、时长或轮次。

请求示例：

```json
{
  "letter": "我真的很累，不想再说了。",
  "include_trace": true
}
```

响应示例：

```json
{
  "reply": "你已经说了很多，现在可以先停在这里。\n\n不说话也没关系，也许呼吸或者冥想更适合现在的状态？",
  "interaction": {
    "type": "breathing_or_meditation_offer",
    "message": "不说话也没关系，也许呼吸或者冥想更适合现在的状态？"
  },
  "trace": null
}
```

前端实现约定：

1. `interaction.type` 仅表示后端识别到用户可能希望暂停说话，当前不代表任何训练协议。
2. `interaction=null` 时只显示回信。
3. 明确自杀/自残安全分流时，后端固定返回 `interaction=null`。
4. 未来如需启动前端呼吸或冥想功能，应由独立的用户操作和前端协议负责，不从本响应推断具体方法。

仅运行 BM25 + DeepSeek + API 时可安装轻量依赖：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dialogue.txt
cp .env.example .env
.venv/bin/uvicorn src.api.main:app --reload --port 8000
```

`.env` 中配置：

```dotenv
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-v4-flash
LLM_API_KEY=
LLM_BASE_URL=https://api.deepseek.com
LLM_THINKING_TYPE=disabled
```

示例：

```bash
curl -sS http://127.0.0.1:8000/dialogue \
  -H 'Content-Type: application/json' \
  -d '{"letter":"我一靠近朋友就想退开。","include_trace":true}'
```

## 对话 Skill

`.cursor/skills/psychoanalysis-dialogue/`：西格蒙德·弗洛伊德——兼具精神分析底蕴与顶级执行官冷峻气场的深度分析师。普通来信先用知识库约束理解；检测到知识理解需求时追加解释性检索，并把知识融进同一文风。不骂人、不说教、不说「你应该」。

在 Cursor 打开本仓库后，直接说梦、关系重复、移情或「从精神分析听听我」即可触发。出现自伤/伤人等风险时，应先转向现实支持（如 `12356`，紧急时 `120`/`110`），不要继续常规解剖。

## 测试

```bash
.venv/bin/python -m pytest tests -q
```

## 技术栈

LlamaIndex · Qdrant · FastAPI · `bge-large-zh-v1.5` · `bge-reranker-v2-m3` · rank_bm25
