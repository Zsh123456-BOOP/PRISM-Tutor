# Task Card 14: 统计检验、表格与图生成

## 1. 任务目标

实现 paired bootstrap、Wilcoxon signed-rank、McNemar test、effect size 计算，并从 metrics 自动生成论文表格和图，不手动改结果。

## 2. 背景与设计约束

方案要求输出 mean、std、95% CI、p-value、effect size，并比较 Ours Full vs Fixed 4-Agent、Ours Full vs Generic Sparse、Ours Full vs Debate、Ours Routing+Budget vs Difficulty Routing。必须生成主结果表、routing/budget/state/ablation/robustness 表和系统图、Pareto 图、risk bucket 图等。

## 3. 前置依赖

- [ ] 依赖 Task Card 11 自动 metrics。
- [ ] 依赖 Task Card 12 judge scores。
- [ ] 依赖 Task Card 13 实验 manifest。
- [ ] 不依赖 human audit 已完成。

## 4. 需要新增或修改的文件

```text
scripts/05_make_tables.py
scripts/06_make_figures.py
prism_tutor/eval/significance.py
prism_tutor/eval/table_builder.py
prism_tutor/eval/figure_builder.py
outputs/tables/*.csv
outputs/tables/*.tex
outputs/figures/*.pdf
outputs/metrics/significance_tests.json
```

## 5. 具体执行步骤

- [ ] Step 1: 对每个 metric 按 sample_id 对齐方法结果，确保 paired tests 使用同一批样本。
- [ ] Step 2: 实现 paired bootstrap 95% CI，支持小数据集 bootstrap。
- [ ] Step 3: 对连续或 ordinal 指标实现 Wilcoxon signed-rank；对 binary leakage 实现 McNemar test。
- [ ] Step 4: 输出 effect size，并在缺少 paired data 时记录无法计算原因。
- [ ] Step 5: 生成 Table 1-6：main、routing、budget、state commit、ablation、robustness。
- [ ] Step 6: 生成 Figure 1-5：system overview、quality-token Pareto、risk bucket、agent call distribution、state conflict case study。
- [ ] Step 7: 所有表格同时输出 machine-readable CSV 和 paper-ready LaTeX。

## 6. 边界情况与失败处理

- [ ] paired sample 不足：输出 warning 和可用样本数，不做无意义显著性声明。
- [ ] judge coverage 不全：表格标注 coverage，不用缺失值补平均。
- [ ] Bootstrap CI 异常宽：保留结果并在 limitations 中提示小样本限制。
- [ ] 图生成缺列：失败并提示缺失 metric，不生成空图。
- [ ] 多重比较风险：输出原始 p-value，并可额外输出 Holm 校正列。

## 7. 验收标准

- [ ] `python scripts/05_make_tables.py` 生成 CSV 和 TeX 表。
- [ ] `python scripts/06_make_figures.py` 生成 PDF 图。
- [ ] 所有表图可追溯到 metrics 和 experiment manifest。
- [ ] significance JSON 包含比较对象、metric、n、CI、p-value、effect size。
- [ ] 没有手工改表或手工填数。

## 8. 不允许做的事情

- [ ] 不允许手动编辑论文结果表。
- [ ] 不允许在样本未 paired 时伪造 paired test。
- [ ] 不允许只报告显著结果、隐藏失败或不显著结果。
- [ ] 不允许把 judge 指标与自动 gold 指标混成一个未解释总分。

## 9. 完成后产物

```text
outputs/tables/table1_main_results.csv
outputs/tables/table1_main_results.tex
outputs/tables/table2_routing.csv
outputs/tables/table3_budget.csv
outputs/tables/table4_state_commit.csv
outputs/tables/table5_ablation.csv
outputs/tables/table6_robustness.csv
outputs/figures/*.pdf
outputs/metrics/significance_tests.json
```
