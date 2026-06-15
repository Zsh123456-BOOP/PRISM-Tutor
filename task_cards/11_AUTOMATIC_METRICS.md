# Task Card 11: 自动指标计算

## 1. 任务目标

实现 PRISM-Tutor 所需自动指标计算，包括 token cost、agent calls、rounds、latency、internal correctness、misconception F1、routing F1、state conflict rate 和 leakage rule detector。

## 2. 背景与设计约束

方案要求自动指标与 LLM judge 指标分开；有标准答案或 gold label 的指标优先自动评估；Answer leakage 必须使用 rule detector + LLM judge 双重判断。本任务只计算自动指标和可从 raw logs 推导的指标，不调用 judge。

## 3. 前置依赖

- [ ] 依赖 Task Card 10 的 generation JSONL。
- [ ] 依赖 Task Card 04 的 processed/split 数据和 gold labels。
- [ ] 依赖 Task Card 06 的 parsed output 和 parse_success 字段。
- [ ] 不依赖 DeepSeek judge API。

## 4. 需要新增或修改的文件

```text
scripts/04_compute_metrics.py
prism_tutor/eval/token_counter.py
prism_tutor/eval/correctness.py
prism_tutor/eval/misconception_metrics.py
prism_tutor/eval/routing_metrics.py
prism_tutor/eval/leakage_detector.py
prism_tutor/eval/state_metrics.py
prism_tutor/eval/aggregate.py
outputs/metrics/*.csv
outputs/metrics/*.json
```

## 5. 具体执行步骤

- [ ] Step 1: 读取 generation JSONL 和 dataset gold labels，按 sample_id、dataset、method 对齐。
- [ ] Step 2: 统计 Total Tokens、Agent Calls、Rounds、Latency、parse_success_rate。
- [ ] Step 3: 实现 Internal Correctness：优先 exact/normalized match，复杂开放式结果标记 coverage，不强行 judge。
- [ ] Step 4: 实现 Misconception Precision/Recall/F1，支持 Bridge 和 Misconception Benchmark gold label。
- [ ] Step 5: 实现 Routing Precision/Recall/F1，使用 gold/pseudo-gold agent need 映射。
- [ ] Step 6: 实现 State Conflict Rate、Incorrect Commit Rate 和 Tentative Update Rate。
- [ ] Step 7: 实现 leakage rule detector：final answer match、直接答案短语、完整 solution chain、关键步骤泄露。
- [ ] Step 8: 输出 metric coverage report，说明哪些指标因 gold label 缺失被跳过。

## 6. 边界情况与失败处理

- [ ] gold label 缺失：该指标输出 `missing_gold_count`，不使用 judge 替代。
- [ ] token usage 缺失：使用 tokenizer fallback 并标记 token_source。
- [ ] parse_success=false：计入失败率，相关结构化指标按 missing 处理，不删除样本。
- [ ] leakage rule 误报风险：保留 matched_rule 和 evidence span，最终 leakage 由 Task Card 12 合并 judge。
- [ ] 样本对齐失败：输出 orphan generation 与 missing sample report。

## 7. 验收标准

- [ ] `python scripts/04_compute_metrics.py` 可生成主 metrics CSV。
- [ ] 自动指标文件不包含 LLM judge 打分列，除非显式合并阶段。
- [ ] 每个指标有 numerator、denominator 和 coverage。
- [ ] leakage rule detector 保存规则命中证据。
- [ ] 所有指标可从 raw logs 和 dataset files 重新计算。

## 8. 不允许做的事情

- [ ] 不允许手动改结果表。
- [ ] 不允许用 LLM judge 替代有 gold label 的自动指标。
- [ ] 不允许忽略 parse failure 对指标覆盖率的影响。
- [ ] 不允许把 rule leakage 当作唯一最终 leakage 结论。

## 9. 完成后产物

```text
outputs/metrics/main_auto_metrics.csv
outputs/metrics/routing_metrics.csv
outputs/metrics/state_metrics.csv
outputs/metrics/leakage_rule_hits.jsonl
outputs/metrics/metric_coverage_report.json
```
