# Task Card 07: LangGraph Runtime Graph 状态结构

## 1. 任务目标

设计 PRISM-Tutor 与所有 baseline 共用的 LangGraph/runtime 状态结构、节点接口、边条件和执行记录格式，确保不同方法之间的工程框架公平。

## 2. 背景与设计约束

方案明确要求统一用 LangGraph 实现，LangGraph 负责 graph execution、state transitions、conditional edges、loops、checkpoint 和 logging。PRISM-Tutor 与 baseline 的差异应来自路由、通信和状态机制，而不是不同运行框架。

## 3. 前置依赖

- [ ] 依赖 Task Card 05 的 client 和 schema。
- [ ] 依赖 Task Card 06 的 agent prompt 与 parser。
- [ ] 依赖 Task Card 04 的统一样本字段。
- [ ] 不依赖 PRISM 核心模块已实现。

## 4. 需要新增或修改的文件

```text
prism_tutor/runtime/graph_state.py
prism_tutor/runtime/node_base.py
prism_tutor/runtime/graph_builder.py
prism_tutor/runtime/checkpointing.py
prism_tutor/runtime/errors.py
tests/test_graph_state.py
```

## 5. 具体执行步骤

- [ ] Step 1: 定义 `TutorGraphState`，包含 sample、method、messages、agent_outputs、risk_scores、selected_agents、rounds、state_before、state_after、errors。
- [ ] Step 2: 定义 student state schema：weak_skills、active_misconceptions、preferred_feedback、recent_failures、tentative_updates。
- [ ] Step 3: 定义统一 node 接口，输入 state，输出 state patch 和 `LLMCallRecord`。
- [ ] Step 4: 实现 checkpoint 写入策略，支持每个样本每轮保存状态快照。
- [ ] Step 5: 定义 graph error 策略，agent 失败时写入 errors，按方法策略决定继续、fallback 或终止。
- [ ] Step 6: 提供 baseline 和 PRISM graph builder 可复用的节点注册机制。

## 6. 边界情况与失败处理

- [ ] Agent 输出失败：state 保留失败记录，Verifier 或 Final Tutor 根据可用信息继续或返回失败状态。
- [ ] 循环超出 max rounds：强制退出并标记 `termination_reason=max_rounds`。
- [ ] token 超预算：强制停止后续 deliberation，并记录 `termination_reason=token_budget`。
- [ ] state patch 冲突：不直接覆盖，交给 Task Card 09 的 state commit 处理。
- [ ] checkpoint 写入失败：输出 stderr 错误并终止该样本，防止不可复现。

## 7. 验收标准

- [ ] Baseline 与 PRISM-Tutor 都能通过同一 runtime state 执行。
- [ ] 每个节点的输入、输出和错误格式统一。
- [ ] checkpoint 包含 round、selected_agents、agent_outputs、state_before/after。
- [ ] runtime 不依赖具体实验脚本，可被 runner 调用。
- [ ] 单元测试覆盖 max rounds、token budget 和 agent failure。

## 8. 不允许做的事情

- [ ] 不允许为某个 baseline 绕过统一 runtime。
- [ ] 不允许在 graph state 中保存不可序列化对象作为正式日志。
- [ ] 不允许静默覆盖 student state。
- [ ] 不允许把 LangGraph 源码纳入项目修改。

## 9. 完成后产物

```text
prism_tutor/runtime/graph_state.py
prism_tutor/runtime/node_base.py
prism_tutor/runtime/graph_builder.py
tests/test_graph_state.py
```
