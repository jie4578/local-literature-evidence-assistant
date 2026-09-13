# Provider Validation Notes

本文件记录 `v0.2.0-provider-preview` 的 Provider 验收边界。所有密钥、请求头、完整模型响应和本地绝对路径均不写入仓库。

## 已完成验证

- 合成 PDF 受控真实 DeepSeek 测试：PASS；
- OpenAI、Ollama、Custom OpenAI-Compatible：Mock 验证；
- Local Offline：两篇公开真实 PDF 离线验收通过。

## 公开长论文测试记录

- 第一次公开抗体论文测试：在第 4 个 chunk 出现 `output_limit_reached`；
- 第二次公开抗体论文测试：出现 `invalid_provider_output`，并暴露全局 Fail-Fast 未生效的问题；
- Phase 6E：离线加入 bounded chunk output contract；
- Phase 6G：离线加入全局 Fail-Fast、统一 ProviderResponse 和动态请求规划；
- 修复后未执行第三次真实 API 调用。

## 当前结论

这是 Provider Preview，不是 Production Ready。AI Provider 会把论文文本发送给对应服务商；建议仅使用公开、脱敏、获授权论文。模型输出仍是不可信输入，`verified=true` 只表示证据文本与 PDF 原文匹配，不代表科研结论真实。AI 模式默认 Fail-Fast，达到 hard limit 或发生输出/契约错误时应停止后续请求。

## 测试边界

离线测试使用 mock，不调用真实服务商。真实测试仅覆盖明确授权的合成材料和有限公开论文范围，未完成的长论文稳定性验证不能被表述为已通过。
