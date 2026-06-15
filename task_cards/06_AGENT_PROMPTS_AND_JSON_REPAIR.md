# Task Card 06: 基础 Agent Prompt 与结构化输出解析

## 1. 任务目标

为七类 agent 编写教学任务 prompt、JSON-only 输出约束和解析失败恢复流程，确保所有 agent 输出可记录、可校验、可失败留痕。

## 2. 背景与设计约束

方案要求所有 agent 必须输出 JSON；JSON parse 失败时 retry 一次，retry 后仍失败则 JSON repair，repair 后仍失败记录 `parse_success=false`，不能删除样本。最终给 judge 和人工审计的回复必须剥离 `<think>`，但 token cost 使用原始 usage。

## 3. 前置依赖

- [ ] 依赖 Task Card 05 的 schema 和 base client。
- [ ] 依赖 Task Card 04 的统一数据字段。
- [ ] 依赖 Task Card 03 的 `<think>` 剥离约束。
- [ ] 不依赖完整 LangGraph runtime。

## 4. 需要新增或修改的文件

```text
prism_tutor/agents/prompts.py
prism_tutor/agents/parser.py
prism_tutor/agents/json_repair.py
prism_tutor/agents/solver.py
prism_tutor/agents/misconception.py
prism_tutor/agents/pedagogy.py
prism_tutor/agents/hint.py
prism_tutor/agents/verifier.py
prism_tutor/agents/state_manager.py
prism_tutor/agents/final_tutor.py
tests/test_json_repair.py
```

## 5. 具体执行步骤

- [ ] Step 1: 为每个 agent 编写 system prompt，说明角色、输入字段、禁止输出非 JSON 文本。
- [ ] Step 2: 为 Final Tutor prompt 加入 student-facing 约束，不允许直接泄露最终答案或关键步骤。
- [ ] Step 3: 实现 `parse_agent_json`，先剥离 `<think>` 块，再提取 JSON，再按 Pydantic schema 校验。
- [ ] Step 4: 实现 parse 失败 retry：第二次 prompt 必须包含上次错误摘要和严格 JSON schema。
- [ ] Step 5: 实现 deterministic JSON repair，处理尾逗号、markdown fence、单引号、额外前后缀等常见问题。
- [ ] Step 6: repair 后仍失败时，返回标准失败对象，写入 raw output、error message 和 `parse_success=false`。

## 6. 边界情况与失败处理

- [ ] 输出含 markdown code fence：剥离 fence 后解析。
- [ ] 输出含多个 JSON 对象：优先选择第一个完整对象，并记录 warning。
- [ ] 输出含 `<think>`：剥离后进入下游，但 raw output 和 token usage 保留。
- [ ] confidence 越界：repair 不擅自截断，schema 校验失败后触发 retry。
- [ ] Final response 泄露答案：交给 Verifier 与 leakage detector 标记，不在 parser 层删除内容。

## 7. 验收标准

- [ ] 每个 agent prompt 明确 JSON schema 和禁止事项。
- [ ] JSON parse 失败、retry、repair、最终失败路径都有测试。
- [ ] 日志能区分 `raw_output`、`stripped_output`、`parsed_output` 和 `parse_success`。
- [ ] repair 后仍失败的样本不会被删除。
- [ ] 最终 student-facing response 不包含 `<think>` 内容。

## 8. 不允许做的事情

- [ ] 不允许让 agent 输出自由文本作为正式结果。
- [ ] 不允许 silent fallback 到空 JSON。
- [ ] 不允许手动编辑失败样本 completion。
- [ ] 不允许把 judge score 用作 parser 成功标准。

## 9. 完成后产物

```text
prism_tutor/agents/prompts.py
prism_tutor/agents/parser.py
prism_tutor/agents/json_repair.py
prism_tutor/agents/*.py
tests/test_json_repair.py
```
