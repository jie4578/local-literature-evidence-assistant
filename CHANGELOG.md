# Changelog

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
