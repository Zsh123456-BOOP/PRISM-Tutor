# Task Card 04: 数据集下载、读取、统一 Schema 与 Split

## 1. 任务目标

为 MathDial、Bridge 和 Math Misconception Benchmark 建立数据读取、统一 schema、split 与 dataset report 流程，保证后续所有方法使用同一批样本。

## 2. 背景与设计约束

方案要求 MathDial 支持 tutor response、scaffolding 和 leakage；Bridge 支持 student error、remediation strategy 和 teacher intention；Misconception Benchmark 支持 misconception diagnosis。官方 split 不存在时使用 seed=42，且禁止查看 test set 后手动调阈值。

## 3. 前置依赖

- [ ] 依赖 Task Card 01 的目录和 config 规范。
- [ ] 依赖 Task Card 02 的 Python 数据处理环境。
- [ ] 依赖数据源可下载，或用户已将 raw 文件放入 `data/raw/`。
- [ ] 不依赖 Qwen3 服务或 judge API。

## 4. 需要新增或修改的文件

```text
data/loaders/mathdial_loader.py
data/loaders/bridge_loader.py
data/loaders/misconception_loader.py
data/loaders/foundational_assist_loader.py
data/build_dataset.py
scripts/01_build_datasets.py
configs/datasets.yaml
data/processed/*.jsonl
data/splits/*.jsonl
outputs/logs/dataset_report.json
```

## 5. 具体执行步骤

- [ ] Step 1: 在 `configs/datasets.yaml` 中定义 raw 路径、processed 路径、split 策略和 seed=42。
- [ ] Step 2: 实现三个核心 loader，将原始字段映射到统一 schema，并保留 `raw_record_id`。
- [ ] Step 3: 对缺失字段使用 `null`、空列表或 `missing_fields` 标记，不删除样本。
- [ ] Step 4: MathDial 若有官方 train/dev/test 则使用官方 split；否则按 conversation 级别 80/10/10 split。
- [ ] Step 5: Bridge 按 20% dev / 80% test split；可用标签时按 error 或 remediation stratified split。
- [ ] Step 6: Misconception Benchmark 全部 220 条作为主评估，并准备 bootstrap confidence interval 所需 sample index。
- [ ] Step 7: 输出 dataset report，包含样本数、字段完整率、label 分布、空字段比例和 split hash。

## 6. 边界情况与失败处理

- [ ] 数据集缺失字段：写入 `missing_fields`，下游指标按可用字段计算 coverage。
- [ ] 官方 split 与本地 split 冲突：优先官方 split，并在 report 中记录来源。
- [ ] 小数据集 CI 不稳定：使用 bootstrap 并报告样本数限制，不夸大显著性。
- [ ] 数据下载失败：保存失败日志；允许本地下载后放入 `data/raw/`，但不得伪造数据。
- [ ] 重复样本：按 dataset 内 id 去重，并保留 duplicate report。

## 7. 验收标准

- [ ] `python scripts/01_build_datasets.py` 可生成 processed 和 split JSONL。
- [ ] 三个核心数据集均符合统一 schema。
- [ ] `outputs/logs/dataset_report.json` 包含字段完整率、label 分布和 split 策略。
- [ ] 所有下游实验读取同一 split 文件。
- [ ] 没有人工查看 test set 后调整阈值或标签。

## 8. 不允许做的事情

- [ ] 不允许删除缺失字段样本来制造干净结果。
- [ ] 不允许手动修改 test set 标签或 split。
- [ ] 不允许把 FoundationalASSIST 纳入第一版主实验。
- [ ] 不允许把 raw data、下载缓存或大文件提交到 Git。

## 9. 完成后产物

```text
data/processed/mathdial.jsonl
data/processed/bridge.jsonl
data/processed/misconception.jsonl
data/splits/*.jsonl
outputs/logs/dataset_report.json
```
