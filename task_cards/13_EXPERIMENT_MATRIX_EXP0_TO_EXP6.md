# Task Card 13: 实验矩阵 Exp0 到 Exp6

## 1. 任务目标

把 PRISM-Tutor 论文实验矩阵组织成可复现的执行配置，覆盖 Exp0 problem diagnosis、Exp1 routing、Exp2 budgeted deliberation、Exp3 state commit、Exp4 end-to-end、Exp5 ablation 和 Exp6 robustness。

## 2. 背景与设计约束

方案要求每个实验写清楚输入、方法、指标、输出表格；所有实验必须可从 raw logs 重新生成结果；不允许手动改结果表；每个实验失败时必须保留失败日志。鲁棒性实验必须明确 noisy agent 注入方式和随机种子；Ablation 必须逐项移除模块。

## 3. 前置依赖

- [ ] 依赖 Task Card 10 的 generation runner。
- [ ] 依赖 Task Card 11 自动指标。
- [ ] 依赖 Task Card 12 judge scores。
- [ ] 依赖 Task Card 08 baseline 和 Task Card 09 PRISM 方法。

## 4. 需要新增或修改的文件

```text
configs/experiments.yaml
scripts/run_exp0_problem_diagnosis.sh
scripts/run_exp1_routing.sh
scripts/run_exp2_budget.sh
scripts/run_exp3_state_commit.sh
scripts/run_exp4_end_to_end.sh
scripts/run_exp5_ablation.sh
scripts/run_exp6_robustness.sh
prism_tutor/experiments/experiment_matrix.py
outputs/experiments/*/manifest.json
```

## 5. 具体执行步骤

- [ ] Step 1: 在 `configs/experiments.yaml` 中为 Exp0-Exp6 定义 datasets、splits、methods、metrics、output_table 和 seeds。
- [ ] Step 2: Exp0 使用 MathDial test、Bridge test、Misconception all，比较 B0-B4，输出通信冗余、泄露和状态冲突诊断。
- [ ] Step 3: Exp1 比较 Random Routing、Fixed All、Difficulty、Generic Contribution、Oracle、Ours QoS，输出 routing F1 与质量/成本。
- [ ] Step 4: Exp2 比较 single/fixed rounds/debate/generic early stopping/ours budget，输出质量-token Pareto 与 risk bucket 分析。
- [ ] Step 5: Exp3 比较 No Memory、Naive Shared Memory、Single Writer、Two-Phase、Ours confidence-weighted commit。
- [ ] Step 6: Exp4 跑 B0-B5 与 M1-M3 end-to-end 主实验。
- [ ] Step 7: Exp5 逐项 ablation：risk estimator、QoS routing、budget、leakage risk、misconception risk、state conflict risk、state commit、confidence-weighted commit 等。
- [ ] Step 8: Exp6 注入 noisy agents，设置 p=0.2/0.4 和 max_tokens_per_case=1000/2000/4000，固定随机种子。

## 6. 边界情况与失败处理

- [ ] 某实验部分方法失败：保留 failure logs，结果表显示 missing/failed，不手动补值。
- [ ] Judge 未完成：自动指标可先生成 provisional 表，但正式论文表必须标记 judge coverage。
- [ ] Oracle 缺 gold label：只在可映射数据上运行，并记录 denominator。
- [ ] Noisy agent 随机性：保存 seed、注入位置、注入概率和实际扰动记录。
- [ ] Ablation 互相混淆：每次只移除或替换一个模块，其他配置保持不变。

## 7. 验收标准

- [ ] Exp0-Exp6 每个都有 manifest，列出输入、方法、指标和输出路径。
- [ ] 每个实验可通过单独脚本启动。
- [ ] 所有输出表格均从 raw logs、metrics 和 judge scores 聚合。
- [ ] 失败样本和失败方法有日志。
- [ ] 主实验样本集和 generation config 对所有方法一致。

## 8. 不允许做的事情

- [ ] 不允许手动编辑实验结果表。
- [ ] 不允许实验失败后删除失败日志。
- [ ] 不允许 ablation 同时移除多个模块后当作单项结论。
- [ ] 不允许把人工审计提前用于实验调参。

## 9. 完成后产物

```text
configs/experiments.yaml
scripts/run_exp*.sh
outputs/experiments/exp0_problem_diagnosis/
outputs/experiments/exp1_routing/
outputs/experiments/exp2_budget/
outputs/experiments/exp3_state_commit/
outputs/experiments/exp4_end_to_end/
outputs/experiments/exp5_ablation/
outputs/experiments/exp6_robustness/
```
