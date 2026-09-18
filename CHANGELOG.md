# Changelog

## [0.3.0-bilingual-corpus-preview]

预览版本，已发布为 GitHub Pre-release `v0.3.0-bilingual-corpus-preview`。不是稳定版，也不是科研生产级版本。

### Added

- 多报告模式：自动选择、仅单篇报告、简洁对比、文献库汇总，以及仅导出 Excel/JSON；
- 大文献库批处理：4 篇以上使用一篇论文一行的纵向汇总，列数不随论文数量增长；
- 文件指纹和稳定 `document_id`：按内容 SHA-256 聚合，basename 只用于展示，不参与去重；
- 输入/文档两层身份模型：同名不同内容分别处理，同内容不同文件名识别为重复输入；
- 暂停续跑：`pause_after`、`resume_local_batch`、`pause_local_batch` 和逐篇结果缓存复用；
- 原子 manifest 和缓存：`batch_manifest.json`、`status.json` 和逐篇结果均原子写入；
- 独立论文报告：`export_individual_reports` 为每篇成功论文生成 Word 报告；
- 双语展示入口：中文字段名配合英文证据原文、PDF 物理页码和 `verified` 状态；
- 完整 Excel 输入映射和技术明细工作表，披露输入标识、内容指纹、状态、选择原因和独立报告路径。

### Changed

- Word 大语料布局改为纵向汇总，不再把论文数量映射为表格列数；
- 11 篇以上只展示有记录选择原因的代表论文，最多 10 篇，其余通过结构化文件保留；
- 研究方法和实验条件使用独立结构化字段；
- 样本量输出改为简洁值；
- Word/Excel 字段中文化和可读性优化。

### Fixed

- 同名不同内容文件错误合并；
- 重复输入身份复用；
- 失败论文错误生成独立报告；
- 方法与实验条件重复；
- `研究方法：研究方法`；
- 异常标点；
- Unicode 科研符号导出问题。

### Security / Privacy

- Local Offline 保持无网络，不创建 Provider 客户端；
- Provider 仍采用会话隔离；
- 不持久化 API Key，密钥不写入结果、SQLite、日志或 `.env`；
- output、PDF、数据库和用户文件继续被 Git 忽略。

### Known limitations

- 仅支持文字版 PDF，扫描件无 OCR；复杂表格、公式和图像识别不稳定；
- 规则型事实提取不等于科研事实核验，`verified=true` 只代表证据原文可在对应 PDF 页找到，不证明研究结论真实或可靠，也不替代科研人员判断；
- 大文献库验收主要使用合成 PDF，真实论文 Local Offline 验证规模有限；
- DeepSeek 长论文真实 Provider 测试尚未完整端到端通过；
- OpenAI、Ollama 和 Custom Endpoint 主要完成 Mock 验证；
- AI 长论文分析仍属于实验功能。

## v0.3.0-bilingual-report-preview

- 完善 Local Offline 与中英对照报告展示；
- 增加中文字段名、英文证据原文、PDF 物理页码和 verified 状态的双语呈现边界；
- 改进摘取式摘要、主要结果、局限性和结论的内容筛选；
- 改进 Word 表格布局、固定列宽、表格分页保护、英文断词控制和中文字体标记；
- 增加作者贡献、出版信息和模型拟合证据的章节级过滤与分类测试；
- 支持 JSON、CSV、Excel 和 Word 结构化结果导出；
- 当前版本为 Bilingual Report Preview，不是 Production Ready。

## v0.2.0-provider-preview

- 增加可插拔 Provider 架构：DeepSeek、OpenAI、Ollama 和 Custom OpenAI-Compatible；
- 增加 Provider 会话隔离、Base URL 安全验证和密钥脱敏；
- 增加 chunk、reduce、final 和 health check 的输出 Token 预算；
- 增加 bounded chunk output contract，拒绝无界数组、重复事实和过长证据；
- 增加统一 ProviderResponse 元数据；
- 增加全局 Fail-Fast、部分结果状态和动态 reduce 计划；
- 保留请求硬上限，并在归并前检查剩余预算；
- DeepSeek 合成 PDF 受控真实测试通过，公开长论文测试仍为 Experimental；
- 当前版本不是 Production Ready，详见 `docs/provider_validation.md`。

## v0.1.0-local-pilot

- 增加 Local Offline 默认模式，无 API Key 也可运行；
- 支持 PDF 分页提取和长文 chunk；
- 增加元数据、章节和规则型科研事实识别；
- 保留 evidence 原文、verified 状态和 PDF 物理页码；
- 增加 SQLite FTS5 本地全文搜索；
- 增加摘取式摘要和多论文结构化对比；
- 支持 JSON、CSV、Excel、Word 导出；
- 保留可选 DeepSeek AI 分析模式；
- 增加请求预算、提示注入隔离和离线自动化测试；
- 当前限制包括无 OCR、复杂表格识别不稳定、规则提取不等于语义理解。
