# Task Card 09: PRISM-Tutor 核心 Runtime 模块实现

## 1. 任务目标

实现 PRISM-Tutor 的四个核心模块：Pedagogical Risk Estimator、Pedagogical QoS Router、Budgeted Deliberation Controller 和 Student State Commit，并组合成 M1、M2、M3 三个方法。

## 2. 背景与设计约束

方案将 PRISM-Tutor 定位为 inference-time runtime，不训练模型。核心比较对象是教学风险感知的动态路由、预算讨论和状态提交。风险权重和阈值必须来自配置，不能根据 test set 人工调整。

## 3. 前置依赖

- [ ] 依赖 Task Card 04 的数据 schema。
- [ ] 依赖 Task Card 07 的 runtime state 与 graph builder。
- [ ] 依赖 Task Card 05/06 的 agent 输出。
- [ ] 依赖 `configs/default.yaml` 中的 risk weights、thresholds 和 budget。

## 4. 需要新增或修改的文件

```text
prism_tutor/runtime/risk_estimator.py
prism_tutor/runtime/qos_router.py
prism_tutor/runtime/budget_controller.py
prism_tutor/runtime/state_commit.py
prism_tutor/runtime/prism_graph.py
configs/default.yaml
tests/test_prism_modules.py
```

## 5. 具体执行步骤

- [ ] Step 1: 实现 Risk Estimator，输出 answer_uncertainty、misconception_risk、pedagogy_risk、leakage_risk、state_conflict_risk、estimated_difficulty 和 recommended_mode。
- [ ] Step 2: 用配置权重计算总风险 `R`，并保存每个子风险和最终风险分桶。
- [ ] Step 3: 实现 QoS Router，根据 low/medium/high risk、misconception、leakage、uncertainty、state conflict 选择 agent 集合。
- [ ] Step 4: 实现 Budgeted Deliberation Controller，按 verifier issue 增加 Hint、Pedagogy、Solver、Misconception 或 State Manager，直到 approval、max_rounds 或 max_tokens。
- [ ] Step 5: 实现 State Commit：两阶段提议、Verifier conflict check、confidence-weighted commit、tentative、reject。
- [ ] Step 6: 组合 M1 Ours Routing Only、M2 Routing+Budget、M3 Full。

## 6. 边界情况与失败处理

- [ ] Risk Estimator 输出 JSON 失败：记录失败并使用保守 medium risk fallback。
- [ ] 风险字段缺失：拒绝路由并写入 schema error，不默默置零。
- [ ] Budget 超限：停止新增 agent，保留当前最好 final response 和 termination reason。
- [ ] Verifier 标记 leakage：必须优先加入 Hint/Pedagogy 修复路径。
- [ ] State update 冲突：进入 tentative 或 reject，不直接覆盖 committed state。
- [ ] 阈值变更：只能通过 config snapshot 记录，不允许实验后手工改结果。

## 7. 验收标准

- [ ] M1、M2、M3 都可通过统一 runner 执行。
- [ ] risk_scores、selected_agents、num_rounds、state updates 写入 generation logs。
- [ ] 单元测试覆盖 low/medium/high routing、leakage block、state conflict 和 token budget。
- [ ] M1 不使用 budget loop 和 state commit；M2 不使用 state commit；M3 完整使用四模块。
- [ ] 所有阈值来自 config snapshot。

## 8. 不允许做的事情

- [ ] 不允许训练或微调 Qwen3-8B。
- [ ] 不允许在 test set 上人工调 risk threshold。
- [ ] 不允许将 Oracle 标签用于真实路由。
- [ ] 不允许跳过 Verifier 直接 commit state。

## 9. 完成后产物

```text
prism_tutor/runtime/risk_estimator.py
prism_tutor/runtime/qos_router.py
prism_tutor/runtime/budget_controller.py
prism_tutor/runtime/state_commit.py
prism_tutor/runtime/prism_graph.py
```
