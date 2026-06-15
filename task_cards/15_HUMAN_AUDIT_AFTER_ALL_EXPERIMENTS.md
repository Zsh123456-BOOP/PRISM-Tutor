# Task Card 15: 全部实验结束后的正式人工抽样审计

## 1. 任务目标

在所有自动实验、自动指标、LLM judge 和统计表图完成后，抽取 200 条样本进行 blind human audit，并计算 human-judge agreement 与 Ours vs strongest baseline 的人工偏好结果。

## 2. 背景与设计约束

方案明确不做实验前人工 pilot；人工审计只在全部实验跑完后进行。盲评表不得显示 method name、selected_agents、risk score，候选顺序必须随机打乱。抽样包含 50% random stratified samples 和 50% hard samples。

## 3. 前置依赖

- [ ] 依赖 Task Card 13 全部实验完成。
- [ ] 依赖 Task Card 12 judge scores。
- [ ] 依赖 Task Card 14 表格和统计结果。
- [ ] 依赖可用人工标注流程，但不依赖模型或 API。

## 4. 需要新增或修改的文件

```text
scripts/07_sample_human_audit.py
scripts/08_human_agreement.py
prism_tutor/eval/human_audit_sampler.py
prism_tutor/eval/human_agreement.py
outputs/human_audit/human_audit_blind.csv
outputs/human_audit/human_audit_labeled.csv
outputs/human_audit/human_agreement_report.json
```

## 5. 具体执行步骤

- [ ] Step 1: 验证 Exp0-Exp6、metrics、judge scores、tables 均已完成；未完成则拒绝抽样。
- [ ] Step 2: 从 MathDial 抽 80、Bridge 抽 80、Misconception 抽 40。
- [ ] Step 3: 50% 使用 random stratified samples，按 dataset、risk bucket、method coverage 分层。
- [ ] Step 4: 50% 使用 hard samples，包括 judge/rule leakage 不一致、ours 与 strongest baseline 差距大、high-risk、state conflict、judge 方差大。
- [ ] Step 5: 生成 blind CSV，只包含 sample_id、dataset、problem、student_answer、ground_truth、dialogue_context、candidate_response 和标注字段。
- [ ] Step 6: 人工标注完成后运行 agreement 脚本，计算 Cohen's kappa、Spearman correlation 和 human preference win rate。

## 6. 边界情况与失败处理

- [ ] 样本不足 200：按可用样本输出实际 n，并记录缺口原因。
- [ ] blind 字段泄漏 method：脚本必须检测并失败。
- [ ] 候选顺序未随机：拒绝输出 blind CSV。
- [ ] 标注文件缺列：agreement 脚本输出 schema error。
- [ ] 多人标注不一致：按预定义规则计算 agreement，不人工回改原始标注。

## 7. 验收标准

- [ ] 只有在全部自动实验完成后才能生成 `human_audit_blind.csv`。
- [ ] blind CSV 不包含 method name、selected_agents、risk score。
- [ ] 输出包含 200 条目标样本或明确样本不足原因。
- [ ] agreement report 包含 kappa、Spearman、preference win rate。
- [ ] Human audit 结果不用于回改实验阈值或主结果。

## 8. 不允许做的事情

- [ ] 不允许实验前人工 pilot。
- [ ] 不允许人工标注者看到方法名或风险分数。
- [ ] 不允许根据人工审计结果重跑或调参后替换主实验。
- [ ] 不允许删除分歧样本。

## 9. 完成后产物

```text
outputs/human_audit/human_audit_blind.csv
outputs/human_audit/human_audit_labeled.csv
outputs/human_audit/human_agreement_report.json
outputs/human_audit/sampling_manifest.json
```
