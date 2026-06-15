# Task Card 05: Agent JSON Schema 与基础 LLM Client

## 1. 任务目标

定义所有 agent 的结构化 JSON schema，并实现统一 Qwen3-8B OpenAI-compatible client，使所有方法共享同一模型、endpoint 选择、generation config、日志字段和错误处理入口。

## 2. 背景与设计约束

方案规定所有 agent 使用同一个 Qwen3-8B，只通过 prompt 和 JSON schema 区分角色。每次 agent 调用必须记录 prompt、completion、token usage、latency、raw output、parsed output。baseline 与 PRISM-Tutor 必须共享 client，避免框架差异影响公平性。

## 3. 前置依赖

- [ ] 依赖 Task Card 03 的 endpoint 配置和 health check。
- [ ] 依赖 Task Card 04 的统一样本 schema。
- [ ] 依赖 `configs/default.yaml` 的 generation config。
- [ ] 不依赖 LangGraph runtime 已完成。

## 4. 需要新增或修改的文件

```text
prism_tutor/agents/schemas.py
prism_tutor/agents/base_client.py
prism_tutor/agents/types.py
prism_tutor/serving/endpoints.py
tests/test_agent_schemas.py
tests/test_base_client.py
```

## 5. 具体执行步骤

- [ ] Step 1: 用 Pydantic 定义 Solver、Misconception、Pedagogy、Hint、Verifier、State Manager、Final Tutor 的 JSON schema。
- [ ] Step 2: 为每个 schema 定义必填字段、类型范围和 confidence 取值 `[0,1]`。
- [ ] Step 3: 实现 `BaseLLMClient`，统一读取 endpoints、model name、temperature、top_p、top_k、max_tokens 和 `enable_thinking=false`。
- [ ] Step 4: 在 client 中实现 round-robin endpoint 选择，并记录 sample_id、method、agent_name 和 endpoint。
- [ ] Step 5: 每次调用返回统一 `LLMCallRecord`，包含 prompt、raw_completion、parsed_output、usage、latency、error、parse_success。
- [ ] Step 6: 为 timeout、HTTP error、empty response、usage missing 定义标准错误码。

## 6. 边界情况与失败处理

- [ ] Endpoint timeout：按配置 retry 请求级别错误，但不得跨样本共享 completion。
- [ ] usage 字段缺失：记录 `usage_missing=true`，用 tokenizer fallback 估算 token，并标记来源。
- [ ] raw completion 为空：记录错误并返回 `parse_success=false` 的 call record。
- [ ] schema 版本升级：日志中写入 `schema_version`，避免旧日志无法解释。
- [ ] 多 endpoint 不一致：记录实际 `served_model_name`，发现非 Qwen3-8B 时直接失败。

## 7. 验收标准

- [ ] 所有 agent schema 均有单元测试覆盖 required fields 与非法值。
- [ ] client 请求会传入 `chat_template_kwargs={"enable_thinking": false}`。
- [ ] 每个 call record 包含 prompt、raw output、parsed output、token usage、latency 和 endpoint。
- [ ] 使用同一 client 可服务 baseline 和 PRISM-Tutor。
- [ ] 不会因为单次 parse 失败删除样本。

## 8. 不允许做的事情

- [ ] 不允许为不同方法使用不同 base model 或不同 generation config。
- [ ] 不允许在 agent 层隐藏 prompt 或 raw completion。
- [ ] 不允许把 DeepSeek judge client 混入 generator client。
- [ ] 不允许把 API key 或服务凭据写入代码。

## 9. 完成后产物

```text
prism_tutor/agents/schemas.py
prism_tutor/agents/base_client.py
tests/test_agent_schemas.py
tests/test_base_client.py
```
