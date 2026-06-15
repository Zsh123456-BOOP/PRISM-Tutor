# Task Card 08: Baseline 方法实现

## 1. 任务目标

实现 PRISM-Tutor 论文需要的 baseline：Single Tutor、Fixed 2-Agent Reflection、Fixed 4-Agent Full Communication、Multi-Agent Debate、Generic Sparse MAS 和 Difficulty Routing。

## 2. 背景与设计约束

方案要求所有 baseline 使用同一个 Qwen3-8B、同一 generation config、同一批样本、同一 LangGraph/runtime 封装。Generic Sparse 不能使用教育风险信号；Difficulty Routing 只能使用难度信号；Oracle Routing 只能作为上界，不参与真实部署比较。

## 3. 前置依赖

- [ ] 依赖 Task Card 04 的数据 split。
- [ ] 依赖 Task Card 07 的 runtime graph。
- [ ] 依赖 Task Card 05/06 的 agent client、schema 和 prompts。
- [ ] 不依赖 PRISM-Tutor 核心模块。

## 4. 需要新增或修改的文件

```text
prism_tutor/baselines/single_tutor.py
prism_tutor/baselines/fixed_2_reflection.py
prism_tutor/baselines/fixed_4_full.py
prism_tutor/baselines/debate.py
prism_tutor/baselines/generic_sparse.py
prism_tutor/baselines/difficulty_routing.py
prism_tutor/baselines/oracle_routing.py
tests/test_baseline_fairness.py
```

## 5. 具体执行步骤

- [ ] Step 1: 实现 B0 Single Tutor，单次调用 Final Tutor 或等价 tutor prompt，记录 agent_calls=1。
- [ ] Step 2: 实现 B1 Fixed 2-Agent Reflection：Tutor -> Critic -> Tutor revise。
- [ ] Step 3: 实现 B2 Fixed 4-Agent Full Communication：Solver -> Misconception -> Pedagogy -> Verifier -> Final Tutor，所有样本固定全调用。
- [ ] Step 4: 实现 B3 Debate：三个 diagnosis agent、debate round、judge summary、Final Tutor。
- [ ] Step 5: 实现 B4 Generic Sparse：用 confidence、novelty、redundancy 选择 top-k，不读取 misconception/leakage/state risk。
- [ ] Step 6: 实现 B5 Difficulty Routing：仅用 problem length、Solver 估计步数、solver confidence 判断难度。
- [ ] Step 7: 可选实现 Oracle Routing，仅标记为 upper bound，不纳入真实部署主比较。

## 6. 边界情况与失败处理

- [ ] Agent parse failure：按 Task Card 06 标准记录，不删除样本。
- [ ] Generic Sparse 相似度依赖不可用：先用 char overlap 或 ROUGE-L fallback。
- [ ] Difficulty Routing Solver 失败：使用 problem length fallback，并记录 `difficulty_fallback=true`。
- [ ] Fixed 全通信超 token：记录超预算，不为其单独放宽预算。
- [ ] Oracle 需要 gold label：只在有 gold/pseudo-gold 的评估中运行，缺失时跳过并记录。

## 7. 验收标准

- [ ] B0-B5 都能被统一 runner 调用。
- [ ] 所有 baseline 日志字段与 PRISM-Tutor 兼容。
- [ ] 公平性测试确认 base model、generation config、sample ids 一致。
- [ ] Generic Sparse 不读取教育风险字段。
- [ ] Difficulty Routing 不读取 misconception/leakage/state risk。

## 8. 不允许做的事情

- [ ] 不允许为 baseline 使用更强模型或不同温度。
- [ ] 不允许让 baseline 使用 PRISM risk estimator。
- [ ] 不允许把 Oracle Routing 当作可部署方法。
- [ ] 不允许手动剔除 baseline 失败样本。

## 9. 完成后产物

```text
prism_tutor/baselines/*.py
tests/test_baseline_fairness.py
```
