# Task Card 16: 论文 Artifact 导出与 Reproducibility Checklist

## 1. 任务目标

导出论文写作所需的表格、图、实验摘要、case study、experiment manifest 和 reproducibility checklist，并执行最终复现完整性检查。

## 2. 背景与设计约束

方案要求最终所有图表和表格能从 raw logs 自动生成，所有实验日志可复现，保留 config snapshot、git commit hash、Python/package 版本，并输出 experiment manifest 和 reproducibility checklist。PRISM-Tutor 是 inference-time runtime，不训练模型。

## 3. 前置依赖

- [ ] 依赖 Task Card 14 的 tables、figures 和 significance results。
- [ ] 依赖 Task Card 15 的 human audit report。
- [ ] 依赖 Task Card 10 的 raw logs 与 manifest。
- [ ] 依赖 Task Card 01 的复现信息采集工具。

## 4. 需要新增或修改的文件

```text
scripts/09_export_paper_artifacts.py
prism_tutor/export/artifact_exporter.py
prism_tutor/export/reproducibility_checklist.py
outputs/paper_artifacts/experiment_summary.md
outputs/paper_artifacts/reproducibility_checklist.md
outputs/paper_artifacts/experiment_manifest.json
outputs/paper_artifacts/artifact_index.md
```

## 5. 具体执行步骤

- [ ] Step 1: 汇总 Exp0-Exp6 manifest，生成全局 `experiment_manifest.json`，列出数据、split、方法、模型、参数、输出路径。
- [ ] Step 2: 导出 `experiment_summary.md`，按论文结构总结每个实验目的、方法、指标、主要表格和图。
- [ ] Step 3: 导出 reproducibility checklist，检查 seed、config snapshot、git commit、dirty status、package versions、GPU、judge metadata、raw logs。
- [ ] Step 4: 收集 tables 和 figures，生成 artifact index，列出每个文件来源脚本和输入 metrics。
- [ ] Step 5: 导出 case study 候选，但不得泄露 blind human audit 中的方法顺序映射，除非审计已结束。
- [ ] Step 6: 运行完整性检查，确保 raw logs、judge raw responses、metrics、tables、figures、human audit 和 checklist 都存在。

## 6. 边界情况与失败处理

- [ ] 缺少某实验输出：checklist 标记 failed，并指出缺失路径，不生成“全通过”结论。
- [ ] Git dirty：记录 dirty 文件，允许论文草稿继续，但复现 checklist 标注。
- [ ] Judge metadata 缺失：阻止 final artifact export，要求补跑或补记录。
- [ ] 表格无法追溯 raw logs：导出失败，避免不可复现数字进入论文。
- [ ] 明文密钥检测命中：立即失败并要求清理，不打包 artifacts。

## 7. 验收标准

- [ ] `python scripts/09_export_paper_artifacts.py` 可生成 paper artifacts。
- [ ] `reproducibility_checklist.md` 包含 seed、config、git、package、GPU、model、judge、data、raw logs。
- [ ] `experiment_manifest.json` 覆盖 Exp0-Exp6。
- [ ] artifact index 能说明每张表和图由哪个脚本生成。
- [ ] 没有手工修改结果表、没有明文密钥、没有遗漏失败日志。

## 8. 不允许做的事情

- [ ] 不允许导出伪造或手工填写结果。
- [ ] 不允许把 raw data、大模型权重或 API key 打包进论文 artifacts。
- [ ] 不允许忽略失败实验并声称完整完成。
- [ ] 不允许把人工审计用于回调实验阈值。

## 9. 完成后产物

```text
outputs/paper_artifacts/experiment_summary.md
outputs/paper_artifacts/reproducibility_checklist.md
outputs/paper_artifacts/experiment_manifest.json
outputs/paper_artifacts/artifact_index.md
outputs/tables/*.tex
outputs/figures/*.pdf
```
