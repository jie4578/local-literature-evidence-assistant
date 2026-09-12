# Local-First Literature Evidence Assistant

一个支持离线 PDF 提取、科研事实识别、原文页码追溯、本地全文检索和多格式导出的文献证据整理工具。

## 核心能力

- 默认使用 Local Offline，无需 API Key，不上传论文；
- 按页提取 PDF，并按 chunk 处理长文；
- 元数据和章节识别；
- 保守的规则型科研事实提取；
- `raw_text` / `display_text` 双层文本；
- `verified` evidence 和 PDF 物理页码追溯；
- SQLite FTS5 本地全文检索；
- 带原文证据的摘取式摘要；
- 多论文结构化对比；
- JSON、CSV、Excel、Word 导出；
- 可选 DeepSeek AI 分析模式。

## 隐私边界

Local Offline 模式只读取本地文件，不访问网络，也不上传论文。

DeepSeek AI 是可选模式；主动选择后，论文文本会发送至 DeepSeek API。请勿上传涉密、敏感或未授权材料。

`verified=true` 仅表示证据原文与 PDF 文本匹配，不表示科研结论本身已经被验证。这个工具不能替代科研人员的专业判断。

## 安装

Windows PowerShell：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip check
```

如已有其他 Python 安装，可将第一条命令中的解释器替换为本机 Python。项目不会自动安装依赖。

## Windows 启动

推荐双击 `start_local.bat`，或在项目根目录运行：

```powershell
.venv\Scripts\python.exe paper_claude.py
```

服务默认绑定 `127.0.0.1`，浏览器访问 `http://127.0.0.1:7860`。启动后保持默认的 `Local Offline` 模式即可无 Key 使用。

## 使用流程

1. 启动程序并确认运行模式为 `Local Offline`。
2. 上传一篇或多篇文字版 PDF。
3. 点击“开始处理”，查看分页提取、章节、事实和证据页码。
4. 使用本地证据搜索查找关键词或统计表达式。
5. 下载 JSON、CSV、Excel 或 Word 结构化结果。
6. 重要结果根据 PDF 页码回查原文。

## 项目结构

```text
paper_claude.py       Gradio 兼容入口和界面
paper_pipeline.py     分页提取、chunk 和 DeepSeek 长文管线
local_extractor.py    本地 PDF 提取和元数据
section_parser.py     保守章节识别
scientific_facts.py   规则型科研事实与证据
local_search.py       SQLite FTS5 本地搜索
local_summary.py      多论文对比和摘取式摘要
exporters.py          JSON/CSV/Excel/Word 导出
tests/                离线自动化测试
docs/images/          README 截图位置
```

## 测试

```powershell
.venv\Scripts\python.exe -m pytest -q
```

当前基线：47 passed。测试不调用真实 DeepSeek API。

## 技术亮点

- 可追溯的原文证据和 PDF 物理页码；
- 本地优先、AI 可选的隐私设计；
- 长文 Map-Reduce AI 管线；
- 请求预算和硬上限；
- 文档提示注入隔离；
- Local Offline 与 DeepSeek 双模式；
- 自动化测试和真实 PDF 离线验收。

## 已知限制

- 当前不包含 OCR；
- 复杂表格的正文/表格标题/表格内容区分不稳定；
- 规则提取不等于语义理解；
- PDF 文本层可能包含固有噪声；
- 摘取式摘要不是生成式综述；
- 重要结果必须根据页码核对原文；
- 本地规则不能自动证明科研事实或结论真实可靠。

本地规则提取暂不能稳定区分正文、表格标题和表格内容，重要结果请根据 PDF 页码回查原文。

## 截图

发布展示时，可将脱敏后的真实运行截图放入 `docs/images/`：

- `main-ui.png`：主界面截图；
- `offline-result.png`：离线处理结果截图；
- `evidence-search.png`：证据搜索截图；
- `export-preview.png`：Excel/Word 导出截图。

当前版本不包含真实论文截图。请勿提交含有本地绝对路径、用户名、敏感文件名或 API Key 的截图。

## 后续路线

后续可独立评估 OCR、表格识别、本地 embedding/RAG、人工反馈工作流和可选本地语言模型；这些功能不属于当前 Local Pilot 版本。
