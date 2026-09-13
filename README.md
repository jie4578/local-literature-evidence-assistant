# Local-First Literature Evidence Assistant

当前版本：`0.3.0-bilingual-report-preview`

一个支持离线 PDF 提取、科研事实识别、原文页码追溯、本地全文检索和多格式导出的文献证据整理工具。

## 功能状态

| 功能 | 状态 |
| --- | --- |
| Local Offline | Limited Pilot |
| DeepSeek synthetic test | Live tested |
| DeepSeek long real paper | Experimental |
| OpenAI | Mock tested |
| Ollama | Mock tested |
| Custom Endpoint | Mock tested |

Local Offline 已通过两篇公开真实 PDF 验收，适合 Limited Local Pilot。AI Provider Preview 的 DeepSeek 合成 PDF 受控测试通过；26 页公开论文测试尚未完整通过，后续输出契约、Fail-Fast 和动态规划问题已离线修复，但修复后未再次进行真实调用。

## 核心能力

- Local Offline 默认运行，无 API Key、默认不联网；
- PyMuPDF 分页提取、页面边界保留和长文 chunk；
- 元数据、章节和保守规则型科研事实识别；
- `raw_text` / `display_text` 双层文本；
- `verified` evidence 与 PDF 物理页码追溯；
- SQLite FTS5 本地全文检索；
- 只摘取原句的摘取式摘要；
- 多论文结构化对比；
- JSON、CSV、Excel、Word 导出；
- 可选 DeepSeek、OpenAI、Ollama 和 Custom OpenAI-Compatible Provider。

## 双语报告与语言模式

- `Local Offline` 是默认模式，不需要 API Key；报告使用中文字段名，同时保留英文证据原文、PDF 物理页码和 `verified` 状态；
- `原文` 模式适合直接回查英文论文；
- `中文摘要＋英文证据` 和 `中英对照` 只在用户主动选择并使用 AI Provider 时提供可选机器翻译；
- 中文译文仅供阅读，不改变英文原文、页码或 `verified`；翻译失败时保留英文；
- Word 报告包含基础信息、结构化概览、摘取式摘要和关键证据，可用于人工复核和求职展示；
- Local Offline 报告会明确标注“未启用语义翻译”，不会把规则提取描述为 AI 综述。

## 隐私与可信边界

Local Offline 只读取本地文件，不访问网络，也不上传论文。选择 AI Provider 后，论文文本会发送至对应服务商；建议只使用公开、脱敏且获授权的材料。

`verified=true` 只表示候选证据文本与 PDF 原文匹配，不表示科研事实或结论真实。AI 输出是不可信输入，页码和 verified 由程序处理。AI 模式默认 Fail-Fast，动态 Request Plan 不保证初始估算就是最终请求数；用户设定的 hard limit 不会自动扩大。

## 安装

Windows PowerShell：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip check
```

项目不会自动安装依赖，也不会自动配置代理或 Provider。

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
6. 如主动选择 AI Provider，填写服务商、模型和配置，并确认论文文本会离开本机。
7. 重要结果根据 PDF 页码回查原文。

## 项目结构

```text
paper_claude.py       Gradio 兼容入口和界面
paper_pipeline.py     分页、chunk 和 AI Map-Reduce 管线
providers/            Provider 接口、注册和适配器
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

当前版本测试结果：116 passed。测试不调用真实 DeepSeek API。

## 技术亮点

- 可追溯的原文证据和 PDF 物理页码；
- 本地优先、AI 可选的隐私设计；
- 长文 Map-Reduce AI 管线；
- Provider 会话隔离、输出 Token 预算和请求硬上限；
- 文档提示注入隔离；
- 全局 Fail-Fast 与动态归并规划；
- 自动化测试和真实 PDF 离线验收。

## 已知限制

- 当前不包含 OCR；
- 复杂表格的正文/表格标题/表格内容区分不稳定；
- 规则提取不等于语义理解；
- PDF 文本层可能包含固有噪声；
- 摘取式摘要不是生成式综述；
- 重要结果必须根据页码核对原文；
- 26 页公开论文的 AI Provider 分析尚未达到生产稳定性；
- 不应把本工具当作科研人员判断或完整事实核验的替代品。

## 截图

发布展示时，可将脱敏后的真实运行截图放入 `docs/images/`：

- `main-ui.png`：主界面截图；
- `offline-result.png`：离线处理结果截图；
- `evidence-search.png`：证据搜索截图；
- `export-preview.png`：Excel/Word 导出截图。

本版本不包含真实论文截图。请勿提交含有本地绝对路径、用户名、敏感文件名或 API Key 的截图。

## 后续路线

后续可独立评估 OCR、表格识别、本地 embedding/RAG、人工反馈工作流和可选本地语言模型；这些功能不属于当前 Provider Preview。
